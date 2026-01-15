import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="openai/whisper-small")
    p.add_argument("--lora_dir", default="outputs/small_lora")
    p.add_argument("--wav", required=True)
    p.add_argument("--language", default="ko")
    p.add_argument("--task", default="transcribe")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--compare_base", action="store_true")
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

    # 가볍고 설치 쉬운 방식: scipy
    # (이미 환경에 없으면 uv add scipy 로 추가)
    try:
        from scipy.signal import resample_poly
        import math
        g = math.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        return resample_poly(audio, up, down).astype(np.float32)
    except Exception:
        # torch 기반 대안: torchaudio (설치되어 있으면 사용)
        try:
            import torchaudio
            x = torch.from_numpy(audio).unsqueeze(0)
            y = torchaudio.functional.resample(x, sr, target_sr)
            return y.squeeze(0).numpy().astype(np.float32)
        except Exception as e:
            raise RuntimeError(
                f"리샘플링이 필요합니다: sr={sr} -> 16000.\n"
                f"해결: `uv add scipy` 또는 `uv add torchaudio` 후 다시 실행하세요.\n"
                f"원인: {e}"
            )


@torch.no_grad()
def run_generate(model, processor, audio_16k, device, max_new_tokens: int):
    inputs = processor(audio_16k, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    pred_ids = model.generate(
        input_features=input_features,
        max_new_tokens=max_new_tokens,
    )
    return processor.batch_decode(pred_ids, skip_special_tokens=True)[0]


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = WhisperProcessor.from_pretrained(args.lora_dir, language=args.language, task=args.task)

    base = WhisperForConditionalGeneration.from_pretrained(args.base_model).to(device)
    base.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    base.config.use_cache = False

    model = PeftModel.from_pretrained(base, args.lora_dir).to(device)
    model.eval()

    audio, sr = load_audio_any_sr(args.wav)
    print(f"[WAV] {args.wav}, sr={sr}")

    audio_16k = resample_to_16k(audio, sr, 16000)

    lora_text = run_generate(model, processor, audio_16k, device, args.max_new_tokens)
    print("\n==== LoRA RESULT ====\n", lora_text, "\n")

    if args.compare_base:
        base.eval()
        base_text = run_generate(base, processor, audio_16k, device, args.max_new_tokens)
        print("\n==== BASE RESULT ====\n", base_text, "\n")


if __name__ == "__main__":
    main()
