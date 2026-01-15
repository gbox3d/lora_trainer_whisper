import argparse
import json
import os
import re
import time
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import soundfile as sf
from pathlib import Path
import random

# PyTorch (Base)
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# CTranslate2 (CT2)
from faster_whisper import WhisperModel

# ==========================================
# 1. 유틸리티 함수 (기존 유지)
# ==========================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, default="datasets/Sample/manifest.jsonl", help="평가할 데이터셋 경로")
    p.add_argument("--base_model", type=str, default="openai/whisper-small", help="비교할 PyTorch 베이스 모델")
    p.add_argument("--ct2_dir", type=str, default="outputs/ct2_small", help="변환된 CT2 모델 경로")
    p.add_argument("--output_csv", type=str, default="compare_ct2_result.csv", help="결과 저장 파일명")
    p.add_argument("--language", default="ko")
    p.add_argument("--max_samples", type=int, default=200)
    return p.parse_args()

def load_audio_16k(wav_path: str):
    """PyTorch 모델용 오디오 로딩 (Numpy Array)"""
    if not os.path.exists(wav_path):
        return None
    try:
        audio, sr = sf.read(wav_path)
        if isinstance(audio, np.ndarray) and audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample_poly
            import math
            g = math.gcd(sr, 16000)
            audio = resample_poly(audio, 16000 // g, sr // g)
        return audio.astype(np.float32)
    except Exception as e:
        print(f"Error loading {wav_path}: {e}")
        return None

def normalize_text_ko(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r'\([^)]*\)', '', s)
    s = re.sub(r"[^0-9A-Za-z가-힣\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def levenshtein(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[n][m]

def calc_cer(ref, hyp):
    ref = ref.replace(" ", "")
    hyp = hyp.replace(" ", "")
    if len(ref) == 0: return 1.0 if len(hyp) > 0 else 0.0
    return levenshtein(ref, hyp) / len(ref)

# ==========================================
# 2. 메인 평가 로직
# ==========================================

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Speed & Accuracy Evaluation Start on {device.upper()}")
    print(f"📂 Manifest: {args.manifest}")
    print(f"🤖 Base Model: {args.base_model}")
    print(f"⚡ CT2 Model : {args.ct2_dir}")

    # --- 1. 모델 로딩 ---
    print("\n⏳ Loading Models...")
    
    # (1) Base Model (PyTorch)
    processor = WhisperProcessor.from_pretrained(args.base_model, language=args.language, task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained(args.base_model).to(device)
    base_model.eval()

    # (2) CT2 Model (Faster-Whisper)
    # compute_type: float16 (GPU) or int8 (CPU/GPU) -> 성능 최적화를 위해 float16 권장 (양자화 모델이면 알아서 처리됨)
    ct2_model = WhisperModel(args.ct2_dir, device=device, compute_type="float16")
    
    print("✅ Models Loaded Successfully.")

    # --- 2. 데이터 로딩 ---
    dataset = []
    if not os.path.exists(args.manifest):
        print("❌ Manifest file not found.")
        return

    with open(args.manifest, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
            
    if len(dataset) > args.max_samples:
        dataset = random.sample(dataset, args.max_samples)
    
    print(f"📊 Processing {len(dataset)} samples...")
    
    

    # --- 3. Warmup (GPU 예열) ---
    print("🔥 Warming up models...")
    if len(dataset) > 0:
        dummy_audio = load_audio_16k(dataset[0]['audio'])
        if dummy_audio is not None:
            # Base Warmup
            inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            base_model.generate(inputs, max_new_tokens=10)
            # CT2 Warmup
            list(ct2_model.transcribe(dataset[0]['audio'], language=args.language))
    print("🔥 Warmup Done.\n")

    # --- 4. 추론 루프 ---
    results = []
    total_time_base = 0
    total_time_ct2 = 0
    total_cer_base = 0
    total_cer_ct2 = 0
    count = 0

    for item in tqdm(dataset):
        audio_path = item['audio']
        ref_text = item['text']
        
        # 오디오 파일 존재 확인
        if not os.path.exists(audio_path):
            continue

        # --- [A] Base Model Inference ---
        # 오디오 로드 (IO 시간 제외를 위해 미리 로드)
        audio_np = load_audio_16k(audio_path)
        if audio_np is None: continue
        inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt").input_features.to(device)

        torch.cuda.synchronize() if device == "cuda" else None
        start_t = time.time()
        
        with torch.no_grad():
            gen_base = base_model.generate(
                input_features=inputs, 
                max_new_tokens=128,
                language=args.language, 
                task="transcribe"
            )
        
        torch.cuda.synchronize() if device == "cuda" else None
        end_t = time.time()
        time_base = end_t - start_t
        text_base = processor.batch_decode(gen_base, skip_special_tokens=True)[0]


        # --- [B] CT2 Model Inference ---
        # faster-whisper는 파일 경로를 직접 받는 것이 가장 효율적임 (내부 전처리 포함)
        torch.cuda.synchronize() if device == "cuda" else None
        start_t = time.time()
        
        segments, _ = ct2_model.transcribe(
            audio_path, 
            language=args.language, 
            beam_size=1  # 속도 최적화를 위해 beam_size=1 사용
        )
        # Generator이므로 리스트로 변환해야 실제 추론이 완료됨
        text_ct2 = "".join([s.text for s in segments])
        
        torch.cuda.synchronize() if device == "cuda" else None
        end_t = time.time()
        time_ct2 = end_t - start_t


        # --- Metrics Calculation ---
        norm_ref = normalize_text_ko(ref_text)
        norm_base = normalize_text_ko(text_base)
        norm_ct2 = normalize_text_ko(text_ct2)

        cer_base = calc_cer(norm_ref, norm_base)
        cer_ct2 = calc_cer(norm_ref, norm_ct2)

        total_cer_base += cer_base
        total_cer_ct2 += cer_ct2
        total_time_base += time_base
        total_time_ct2 += time_ct2
        count += 1

        results.append({
            "audio": audio_path,
            "Reference": norm_ref,
            "Base_Pred": norm_base,
            "CT2_Pred": norm_ct2,
            "Base_CER": round(cer_base, 4),
            "CT2_CER": round(cer_ct2, 4),
            "Base_Time(s)": round(time_base, 3),
            "CT2_Time(s)": round(time_ct2, 3),
            "Speedup": round(time_base / time_ct2, 2) if time_ct2 > 0 else 0
        })

    # --- 5. 결과 리포트 ---
    if count > 0:
        avg_cer_base = total_cer_base / count
        avg_cer_ct2 = total_cer_ct2 / count
        avg_time_base = total_time_base / count
        avg_time_ct2 = total_time_ct2 / count
        
        speedup_ratio = avg_time_base / avg_time_ct2

        print("\n" + "="*50)
        print(" 🚀 FINAL SPEED & ACCURACY REPORT")
        print("="*50)
        print(f" Samples Evaluated  : {count}")
        print("-" * 50)
        print(f" [Accuracy / CER] (Lower is better)")
        print(f"   - Base Model     : {avg_cer_base:.4f}")
        print(f"   - CT2 (LoRA)     : {avg_cer_ct2:.4f}")
        print(f"   - CER Diff       : {avg_cer_ct2 - avg_cer_base:+.4f}")
        print("-" * 50)
        print(f" [Latency / Time] (Lower is better)")
        print(f"   - Base Avg Time  : {avg_time_base:.3f} sec/sample")
        print(f"   - CT2 Avg Time   : {avg_time_ct2:.3f} sec/sample")
        print(f"   ⚡ Speedup Factor : {speedup_ratio:.2f}x Faster")
        print("="*50)

        df = pd.DataFrame(results)
        df.to_csv(args.output_csv, index=False, encoding='utf-8-sig')
        print(f"💾 Detailed results saved to: {args.output_csv}")
    else:
        print("No samples processed.")

if __name__ == "__main__":
    main()