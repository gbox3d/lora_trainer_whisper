import json
import os
import re
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from manifest_utils import relativize_audio_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="데이터셋 매니페스트 생성 스크립트")
    parser.add_argument("--root", type=str, default="datasets/Sample", help="데이터셋 루트 경로")
    parser.add_argument("--wav_dir", type=str, default="wav", help="오디오 폴더명")
    parser.add_argument("--label_dir", type=str, default="lb", help="라벨 폴더명")
    parser.add_argument("--output", type=str, default="manifest.jsonl", help="출력 파일명")
    parser.add_argument(
        "--audio_path_mode",
        "--audio-path-mode",
        dest="audio_path_mode",
        choices=("relative", "absolute"),
        default="relative",
        help="manifest에 저장할 audio 경로 형식",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, max(4, (os.cpu_count() or 4))),
        help="라벨 JSON 읽기 병렬 worker 수",
    )
    return parser.parse_args(argv)

def normalize_text(t: str) -> str:
    # (A)/(B) -> B 패턴 처리
    t = re.sub(r"\(([^()]+?)\)/\(([^()]+?)\)", r"\2", t)
    # 남은 괄호 제거
    t = t.replace("(", "").replace(")", "")
    # 공백 정리
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_manifest_row(label_path: Path, lab_root: Path, wav_root: Path, out_path: Path, audio_path_mode: str):
    obj = json.loads(label_path.read_text(encoding="utf-8"))

    if "script" not in obj or "text" not in obj["script"]:
        return None

    text = normalize_text(obj["script"]["text"])
    rel = label_path.relative_to(lab_root).with_suffix(".wav")
    wav_path = wav_root / rel

    if not wav_path.exists():
        return None

    audio_path = (
        relativize_audio_path(wav_path, out_path)
        if audio_path_mode == "relative"
        else str(wav_path.resolve())
    )
    return {"audio": audio_path, "text": text}


def safe_build_manifest_row(label_path: Path, lab_root: Path, wav_root: Path, out_path: Path, audio_path_mode: str):
    try:
        return label_path, build_manifest_row(label_path, lab_root, wav_root, out_path, audio_path_mode), None
    except Exception as error:
        return label_path, None, error

def main(argv=None):
    args = parse_args(argv)
    
    root = Path(args.root)
    wav_root = root / args.wav_dir
    lab_root = root / args.label_dir
    out_path = root / args.output

    if not lab_root.exists():
        print(f"❌ Error: 라벨 폴더를 찾을 수 없습니다: {lab_root}")
        raise FileNotFoundError(f"라벨 폴더를 찾을 수 없습니다: {lab_root}")

    print(f"📂 Root: {root}")
    print(f"   Searching labels in: {lab_root}")
    print(f"   Matching wavs in:    {wav_root}")

    count = 0
    label_paths = sorted(lab_root.rglob("*.json"))
    with out_path.open("w", encoding="utf-8") as out:
        if args.workers <= 1:
            for label_path in label_paths:
                try:
                    row = build_manifest_row(label_path, lab_root, wav_root, out_path, args.audio_path_mode)
                    if row is None:
                        continue
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
                except Exception as e:
                    print(f"Error processing {label_path}: {e}")
                    continue
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for label_path, row, error in executor.map(
                    lambda current_path: safe_build_manifest_row(
                        current_path,
                        lab_root,
                        wav_root,
                        out_path,
                        args.audio_path_mode,
                    ),
                    label_paths,
                ):
                    if error is not None:
                        print(f"Error processing {label_path}: {error}")
                        continue
                    if row is None:
                        continue
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1

    print("-" * 30)
    print(f"✅ Created: {out_path}")
    print(f"📊 Total Rows: {count}")
    return {
        "root": str(root.resolve()),
        "wav_dir": args.wav_dir,
        "label_dir": args.label_dir,
        "manifest_path": str(out_path.resolve()),
        "row_count": count,
        "audio_path_mode": args.audio_path_mode,
        "workers": args.workers,
    }

if __name__ == "__main__":
    main()
