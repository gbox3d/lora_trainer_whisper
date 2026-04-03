import argparse
import time
from pathlib import Path

from faster_whisper import WhisperModel


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="outputs/ct2_small")
    p.add_argument(
        "--audio-path",
        default="datasets/Sample/wav/SPK014/SPK014KBSCU001/SPK014KBSCU001F001.wav",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-type", default="float16")
    p.add_argument("--language", default="ko")
    p.add_argument("--beam-size", type=int, default=5)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print(f"🚀 Loading CT2 Model from {args.model_path}...")
    model = WhisperModel(
        args.model_path,
        device=args.device,
        compute_type=args.compute_type,
    )

    print("🎤 Transcribing...")
    start = time.time()
    segments, info = model.transcribe(
        args.audio_path,
        language=args.language,
        beam_size=args.beam_size,
    )

    print(
        f"\n[Detected Language]: {info.language} "
        f"(Probability: {info.language_probability:.2f})"
    )
    print("-" * 30)

    full_text = ""
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        full_text += segment.text

    print("-" * 30)
    elapsed = time.time() - start
    print(f"✅ Total Time: {elapsed:.4f} sec")
    return {
        "model_path": str(Path(args.model_path).resolve()),
        "audio_path": str(Path(args.audio_path).resolve()),
        "device": args.device,
        "compute_type": args.compute_type,
        "language": info.language,
        "language_probability": info.language_probability,
        "text": full_text.strip(),
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    main()
