from __future__ import annotations

import argparse
import json
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Whisper dataset preparation tools")
    sub = p.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest", help="manifest.jsonl 생성")
    manifest.add_argument("--root", required=True)
    manifest.add_argument("--wav-dir", default="wav")
    manifest.add_argument("--label-dir", default="lb")
    manifest.add_argument("--output", default="manifest.jsonl")
    manifest.add_argument("--audio-path-mode", choices=("relative", "absolute"), default="relative")
    manifest.add_argument("--workers", type=int, default=16)
    manifest.add_argument("--no-progress", action="store_true")

    validate = sub.add_parser("validate", help="데이터셋 유효성 검사")
    validate.add_argument("root", help="데이터셋 루트 경로")
    validate.add_argument("--manifest", default=None, help="검사할 manifest.jsonl 경로")
    validate.add_argument("--check-audio", action="store_true", help="WAV 헤더 검사")
    validate.add_argument("--clean", action="store_true", help="누락/오류 행 제거")
    validate.add_argument("--max-errors", type=int, default=20)
    validate.add_argument("--no-progress", action="store_true")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.command == "manifest":
        from dataset_tools.make_manifest import main as manifest_main

        result = manifest_main(
            [
                "--root", args.root,
                "--wav_dir", args.wav_dir,
                "--label_dir", args.label_dir,
                "--output", args.output,
                "--audio_path_mode", args.audio_path_mode,
                "--workers", str(args.workers),
                *(["--no-progress"] if args.no_progress else []),
            ]
        )
        print(json.dumps(result, ensure_ascii=False))
        return result

    elif args.command == "validate":
        from dataset_tools.validate_dataset import main as validate_main

        return validate_main(
            [
                args.root,
                *(["--manifest", args.manifest] if args.manifest else []),
                *(["--check-audio"] if args.check_audio else []),
                *(["--clean"] if args.clean else []),
                "--max-errors", str(args.max_errors),
                *(["--no-progress"] if args.no_progress else []),
            ]
        )

    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
