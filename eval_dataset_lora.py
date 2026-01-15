# file: eval_dataset_lora.py

import argparse
import json
import os
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel


# ==========================================
# 1. Args
# ==========================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, required=True, help="평가할 manifest.jsonl 경로")
    p.add_argument("--base_model", type=str, default="openai/whisper-small", help="베이스 모델")
    p.add_argument("--lora_dir", type=str, required=True, help="학습된 LoRA adapter 경로 (checkpoint-XXXX 포함 가능)")
    p.add_argument("--output_csv", type=str, default="comparison_results.csv", help="결과 저장 CSV 파일")
    p.add_argument("--language", type=str, default="ko")
    p.add_argument("--task", type=str, default="transcribe")

    p.add_argument("--max_samples", type=int, default=200, help="랜덤 샘플 수 (0이면 전체)")
    p.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    p.add_argument("--max_new_tokens", type=int, default=128)

    # 디버깅/진단 옵션
    p.add_argument("--debug_first_n", type=int, default=0, help="처음 N개 샘플은 출력(Ref/Base/LoRA) 보여주기")
    p.add_argument("--disable_adapter", action="store_true", help="LoRA 어댑터를 끈 상태로만 평가 (진단용)")
    return p.parse_args()


# ==========================================
# 2. Utils
# ==========================================
def load_audio_16k(wav_path: str):
    """오디오 로드 후 16kHz mono float32"""
    if not os.path.exists(wav_path):
        alt = os.path.join(os.getcwd(), wav_path)
        if os.path.exists(alt):
            wav_path = alt
        else:
            return None

    try:
        audio, sr = sf.read(wav_path)
        if isinstance(audio, np.ndarray) and audio.ndim == 2:
            audio = audio.mean(axis=1)  # stereo -> mono

        if sr != 16000:
            from scipy.signal import resample_poly
            import math

            g = math.gcd(sr, 16000)
            audio = resample_poly(audio, 16000 // g, sr // g)

        return audio.astype(np.float32)
    except Exception as e:
        print(f"[load_audio_16k] Error loading {wav_path}: {e}")
        return None


def normalize_text_ko(s: str) -> str:
    """한국어 텍스트 정규화 (CER용)"""
    s = str(s).strip()
    s = re.sub(r"\([^)]*\)", "", s)  # 괄호 제거
    s = re.sub(r"[^0-9A-Za-z가-힣\s]", "", s)  # 특수문자 제거
    s = re.sub(r"\s+", " ", s).strip()
    return s


def levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[n][m]


def calc_cer(ref: str, hyp: str) -> float:
    ref = ref.replace(" ", "")
    hyp = hyp.replace(" ", "")
    if len(ref) == 0:
        return 1.0 if len(hyp) > 0 else 0.0
    dist = levenshtein(ref, hyp)
    return dist / len(ref)


def read_manifest(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def resolve_lora_dir(lora_dir: str) -> str:
    """
    LoRA 경로가 루트(outputs/xxx)인지 checkpoint-xxxx인지 모를 때,
    루트면 가장 마지막 checkpoint- 폴더를 자동 선택(있으면).
    """
    p = Path(lora_dir)
    if not p.exists():
        raise FileNotFoundError(f"LoRA path not found: {lora_dir}")

    # checkpoint-* 가 있으면 가장 큰 step 선택
    ckpts = sorted([x for x in p.glob("checkpoint-*") if x.is_dir()], key=lambda x: x.name)
    if ckpts:
        chosen = ckpts[-1]
        print(f"🔎 LoRA dir has checkpoints. Auto-selected: {chosen}")
        return str(chosen)

    return str(p)


def check_adapter_files(lora_dir: str):
    p = Path(lora_dir)
    candidates = [
        p / "adapter_model.safetensors",
        p / "adapter_model.bin",
        p / "adapter_config.json",
    ]
    exists = {c.name: c.exists() for c in candidates}
    print(f"🧩 LoRA adapter files in {lora_dir}: {exists}")


# ==========================================
# 3. Main
# ==========================================
def main():
    args = parse_args()
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"🚀 Evaluation Start on {device.upper()}")
    print(f"📂 Manifest: {args.manifest}")

    # --- Load dataset
    dataset = read_manifest(args.manifest)
    if args.max_samples and args.max_samples > 0 and len(dataset) > args.max_samples:
        dataset = random.sample(dataset, args.max_samples)
    print(f"📊 Processing {len(dataset)} samples...")

    # --- Load models
    print("⏳ Loading Models...")
    processor = WhisperProcessor.from_pretrained(args.base_model)

    base_model = WhisperForConditionalGeneration.from_pretrained(args.base_model, dtype=dtype).to(device)
    base_model.eval()

    lora_path = resolve_lora_dir(args.lora_dir)
    check_adapter_files(lora_path)

    base_for_lora = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=dtype
    )
    lora_model = PeftModel.from_pretrained(base_for_lora, lora_path).to(device)
    lora_model.eval()

    print("✅ Models Loaded Successfully.")

    # --- Whisper prompt 강제 (핵심)
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language,
        task=args.task,
    )

    results = []
    total_cer_base = 0.0
    total_cer_lora = 0.0
    count = 0

    # --- Inference loop
    for idx, item in enumerate(tqdm(dataset)):
        audio_path = item.get("audio")
        ref_text = item.get("text", "")

        audio = load_audio_16k(audio_path)
        if audio is None:
            continue

        # attention_mask 포함 (경고 제거 + 안정성)
        proc_out = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        # input_features = proc_out.input_features.to(device)
        # attention_mask = proc_out.attention_mask.to(device)
        input_features = proc_out.input_features.to(device=device, dtype=base_model.dtype)
        attention_mask = proc_out.attention_mask.to(device)

        with torch.no_grad():
            
            gen_kwargs= dict(
                max_new_tokens=args.max_new_tokens,
                forced_decoder_ids=forced_decoder_ids,
                # repetition_penalty=1.2,  <-- 주석 처리
                # no_repeat_ngram_size=3,  <-- 주석 처리
                # 대신 아래 옵션 추가 고려
                temperature=0.0, # Greedy Decoding 명시
                # condition_on_prev_tokens=False # 문맥 의존성 줄이기 (선택사항)
            )
                        
            # Base
            gen_base = base_model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                forced_decoder_ids=forced_decoder_ids,
            )
            text_base = processor.batch_decode(gen_base, skip_special_tokens=True)[0]

            # LoRA (혹은 disable_adapter 진단)
            if args.disable_adapter:
                with lora_model.disable_adapter():
                    gen_lora = lora_model.generate(
                        input_features=input_features,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        forced_decoder_ids=forced_decoder_ids,
                    )
            else:
                gen_lora = lora_model.generate(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    forced_decoder_ids=forced_decoder_ids,
                )

            text_lora = processor.batch_decode(gen_lora, skip_special_tokens=True)[0]

        # Normalize & CER
        norm_ref = normalize_text_ko(ref_text)
        norm_base = normalize_text_ko(text_base)
        norm_lora = normalize_text_ko(text_lora)

        cer_base = calc_cer(norm_ref, norm_base)
        cer_lora = calc_cer(norm_ref, norm_lora)

        total_cer_base += cer_base
        total_cer_lora += cer_lora
        count += 1

        results.append({
            "audio": audio_path,
            "Reference": norm_ref,
            "Base_Pred": norm_base,
            "LoRA_Pred": norm_lora,
            "Base_CER": round(cer_base, 4),
            "LoRA_CER": round(cer_lora, 4),
            "Improved": cer_lora < cer_base,
        })

        # 디버그 출력
        if args.debug_first_n and idx < args.debug_first_n:
            print("\n" + "-" * 60)
            print(f"[DEBUG #{idx}] {audio_path}")
            print(f"REF : {norm_ref}")
            print(f"BASE: {norm_base}  (CER={cer_base:.4f})")
            print(f"LORA: {norm_lora}  (CER={cer_lora:.4f})")
            print("-" * 60)

    # --- Summary
    if count == 0:
        print("No valid samples processed.")
        return

    avg_base = total_cer_base / count
    avg_lora = total_cer_lora / count
    improvement = avg_base - avg_lora
    rel = (improvement / avg_base * 100.0) if avg_base > 1e-12 else 0.0

    print("\n" + "=" * 40)
    print(" 📢 FINAL EVALUATION REPORT")
    print("=" * 40)
    print(f" Samples Evaluated : {count}")
    print(f" Average CER (Base): {avg_base:.4f} ({avg_base * 100:.2f}%)")
    print(f" Average CER (LoRA): {avg_lora:.4f} ({avg_lora * 100:.2f}%)")
    print(f" 📈 Improvement    : {improvement:+.4f} ({rel:+.2f}% relative)")
    print("=" * 40)

    df = pd.DataFrame(results)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"💾 Detailed results saved to: {out_path}")

    # 샘플 개선 사례
    better_df = df[df["Improved"] == True].head(3)
    if not better_df.empty:
        print("\n[Sample Improvements]")
        for _, row in better_df.iterrows():
            print(f"Ref : {row['Reference']}")
            print(f"Base: {row['Base_Pred']}")
            print(f"LoRA: {row['LoRA_Pred']}")
            print("-" * 30)


if __name__ == "__main__":
    main()
