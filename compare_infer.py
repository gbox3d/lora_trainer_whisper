import argparse
from pathlib import Path
import re
import numpy as np
import torch

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True, help="input wav path")
    p.add_argument("--base_model", default="openai/whisper-small")
    p.add_argument("--lora_dir", default="outputs/small_lora")
    p.add_argument("--language", default="ko")
    p.add_argument("--task", default="transcribe")
    p.add_argument("--max_new_tokens", type=int, default=128)

    # optional reference (for CER/WER)
    p.add_argument("--ref", default=None, help="reference text (ground truth)")
    p.add_argument("--ref_txt", default=None, help="path to a txt file containing reference text")

    # printing / normalization
    p.add_argument("--normalize", action="store_true", help="apply light normalization before metrics/print")
    p.add_argument("--show_tokens", action="store_true", help="show token length info")
    return p.parse_args()


def load_audio_any_sr(wav_path: str):
    p = Path(wav_path)
    if not p.exists():
        raise FileNotFoundError(f"파일이 없습니다: {wav_path}")

    import soundfile as sf
    audio, sr = sf.read(str(p))

    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = audio.mean(axis=1)  # stereo -> mono
    audio = np.asarray(audio, dtype=np.float32)
    return audio, int(sr)


def resample_to_16k(audio: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    if sr == target_sr:
        return audio

    # Prefer scipy (lightweight & common)
    try:
        from scipy.signal import resample_poly
        import math
        g = math.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        return resample_poly(audio, up, down).astype(np.float32)
    except Exception:
        # Fallback to torchaudio if available
        try:
            import torchaudio
            x = torch.from_numpy(audio).unsqueeze(0)
            y = torchaudio.functional.resample(x, sr, target_sr)
            return y.squeeze(0).numpy().astype(np.float32)
        except Exception as e:
            raise RuntimeError(
                f"리샘플링 필요: sr={sr} -> 16000.\n"
                f"해결: `uv add scipy` 또는 `uv add torchaudio`\n"
                f"원인: {e}"
            )


def normalize_text_ko(s: str) -> str:
    # 아주 가벼운 정규화(원하시면 더 강하게 가능)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)

    # 일반적으로 의미 없는 기호 일부 제거(원하면 옵션화 가능)
    s = s.replace("…", "...")

    # 한글/숫자/영문/공백/기본 구두점 정도만 남기기(너무 과격하면 주석 처리)
    s = re.sub(r"[^0-9A-Za-z가-힣\s\.\,\?\!\-:;\"'()]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@torch.no_grad()
def generate_text(model, processor, audio_16k: np.ndarray, device: str, max_new_tokens: int) -> str:
    inputs = processor(audio_16k, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    pred_ids = model.generate(
        input_features=input_features,
        max_new_tokens=max_new_tokens,
    )
    return processor.batch_decode(pred_ids, skip_special_tokens=True)[0]


def levenshtein(a, b):
    # a, b: list of tokens (chars or words)
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost     # substitution
            )
            prev = cur
    return dp[m]


def cer(ref: str, hyp: str) -> float:
    # Character Error Rate
    r = list(ref)
    h = list(hyp)
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    dist = levenshtein(r, h)
    return dist / len(r)


def wer(ref: str, hyp: str) -> float:
    # Word Error Rate (space tokenized)
    r = ref.split()
    h = hyp.split()
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    dist = levenshtein(r, h)
    return dist / len(r)


def read_ref_text(args) -> str | None:
    if args.ref is not None:
        return args.ref
    if args.ref_txt is not None:
        p = Path(args.ref_txt)
        if not p.exists():
            raise FileNotFoundError(f"ref_txt 파일이 없습니다: {args.ref_txt}")
        return p.read_text(encoding="utf-8").strip()
    return None


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    audio, sr = load_audio_any_sr(args.wav)
    print(f"[WAV] {args.wav}  (sr={sr})")
    audio_16k = resample_to_16k(audio, sr, 16000)

    # Processor: LoRA dir에 저장된 processor 우선 사용(훈련과 같은 설정 유지)
    processor = WhisperProcessor.from_pretrained(args.lora_dir, language=args.language, task=args.task)

    # Base model
    base = WhisperForConditionalGeneration.from_pretrained(args.base_model).to(device)
    base.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    base.config.use_cache = False
    base.eval()

    # LoRA model (base 위에 adapter)
    lora_model = PeftModel.from_pretrained(base, args.lora_dir).to(device)
    lora_model.eval()

    base_text = generate_text(base, processor, audio_16k, device, args.max_new_tokens)
    lora_text = generate_text(lora_model, processor, audio_16k, device, args.max_new_tokens)

    ref_text = read_ref_text(args)

    if args.normalize:
        base_text_n = normalize_text_ko(base_text)
        lora_text_n = normalize_text_ko(lora_text)
        ref_text_n = normalize_text_ko(ref_text) if ref_text is not None else None
    else:
        base_text_n, lora_text_n, ref_text_n = base_text, lora_text, ref_text

    print("\n==================== BASE ====================")
    print(base_text_n)
    print("\n==================== LoRA ====================")
    print(lora_text_n)

    if args.show_tokens:
        # 토큰 길이(대충) 확인: tokenizer로 인코딩 길이 출력
        bt = processor.tokenizer(base_text_n).input_ids
        lt = processor.tokenizer(lora_text_n).input_ids
        print("\n[INFO] token_len: base=", len(bt), " lora=", len(lt))

    if ref_text_n is not None and len(ref_text_n) > 0:
        base_cer = cer(ref_text_n, base_text_n)
        lora_cer = cer(ref_text_n, lora_text_n)
        base_wer = wer(ref_text_n, base_text_n)
        lora_wer = wer(ref_text_n, lora_text_n)

        print("\n==================== METRICS (vs REF) ====================")
        print("[REF]")
        print(ref_text_n)
        print("\n[CER]  base={:.4f}  lora={:.4f}  (delta={:+.4f})".format(
            base_cer, lora_cer, (lora_cer - base_cer)
        ))
        print("[WER]  base={:.4f}  lora={:.4f}  (delta={:+.4f})".format(
            base_wer, lora_wer, (lora_wer - base_wer)
        ))
    else:
        print("\n(REF가 없어서 CER/WER는 생략했습니다. --ref 또는 --ref_txt로 넣으면 자동 계산됩니다.)")


if __name__ == "__main__":
    main()
