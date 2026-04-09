from __future__ import annotations

import argparse
import json
import os
import wave
from pathlib import Path

from tqdm import tqdm


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="데이터셋 유효성 검사")
    p.add_argument("root", type=str, help="데이터셋 루트 경로 (manifest.jsonl이 있는 상위)")
    p.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="검사할 manifest.jsonl 경로 (미지정 시 root 아래 자동 탐색)",
    )
    p.add_argument("--check-audio", action="store_true", help="WAV 파일 헤더 읽기 검사")
    p.add_argument("--clean", action="store_true", help="누락/오류 행을 제거한 manifest를 덮어쓰기")
    p.add_argument("--max-errors", type=int, default=20, help="출력할 최대 오류 수")
    p.add_argument("--no-progress", action="store_true", help="진행 막대 비활성화")
    return p.parse_args(argv)


def read_wav_info(path: Path) -> dict:
    """WAV 헤더를 읽어 sample rate, channels, duration 반환."""
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        duration = frames / rate if rate > 0 else 0.0
        return {
            "sample_rate": rate,
            "channels": channels,
            "sample_width": sampwidth,
            "frames": frames,
            "duration_sec": round(duration, 3),
        }


def find_manifests(root: Path, max_depth: int = 4) -> list[Path]:
    """root 아래 manifest.jsonl 파일들을 탐색 (최대 max_depth 단계까지)."""
    found = []
    _find_manifests_recursive(root, root, 0, max_depth, found)
    return sorted(found)


def _find_manifests_recursive(
    current: Path, root: Path, depth: int, max_depth: int, found: list
) -> None:
    if depth > max_depth:
        return
    try:
        for entry in current.iterdir():
            try:
                if entry.is_symlink() and not entry.exists():
                    continue  # 깨진 심볼릭 링크 무시
                if entry.is_file() and entry.name == "manifest.jsonl":
                    found.append(entry)
                elif entry.is_dir() and entry.name not in ("wav", "lb", ".git", "__pycache__"):
                    _find_manifests_recursive(entry, root, depth + 1, max_depth, found)
            except (PermissionError, OSError):
                continue
    except (PermissionError, OSError):
        pass


def resolve_audio_path(audio_field: str, root: Path) -> Path:
    """manifest의 audio 필드를 실제 파일 경로로 변환.

    선행 / 제거 후 root 기준으로 해석하되,
    경로 선두가 root 말단 디렉토리와 중복되면 자동 제거.
    예: root=/data/130/Validation, audio=/Validation/wav/... → /data/130/Validation/wav/...
    """
    if not audio_field:
        return Path(audio_field)

    p = Path(audio_field)
    if p.is_absolute() and p.exists():
        return p

    stripped = audio_field.lstrip("/")
    candidate = root / stripped

    if not candidate.exists():
        parts = Path(stripped).parts
        for i in range(1, min(len(parts), 4)):
            shorter = root / Path(*parts[i:])
            if shorter.exists():
                return shorter

    return candidate


def format_size(nbytes: int) -> str:
    """바이트를 사람이 읽기 쉬운 단위로 변환."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:,.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:,.1f} PB"


def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}h {m}m {s:.1f}s"


def pct(part: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{part / total * 100:.2f}%"


def validate_manifest(
    manifest_path: Path,
    root: Path,
    check_audio: bool,
    clean: bool,
    max_errors: int,
    show_progress: bool,
) -> dict:
    """단일 manifest.jsonl 검사. 결과 dict 반환."""
    errors: list[str] = []
    missing_files: list[str] = []
    valid_lines: list[str] = []

    stats = {
        "manifest": str(manifest_path),
        "total_rows": 0,
        "valid_rows": 0,
        "missing_audio": 0,
        "empty_text": 0,
        "bad_json": 0,
        "audio_read_errors": 0,
        "total_file_size": 0,
        "cleaned": False,
        "removed_rows": 0,
        "duration_total_sec": 0.0,
        "duration_min_sec": float("inf"),
        "duration_max_sec": 0.0,
        "sample_rates": set(),
        "text_len_min": float("inf"),
        "text_len_max": 0,
        "text_len_total": 0,
    }

    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    stats["total_rows"] = len([ln for ln in lines if ln.strip()])

    progress = tqdm(
        lines,
        desc=f"Validate {manifest_path.parent.name}/{manifest_path.name}",
        unit="row",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    for line in progress:
        line = line.strip()
        if not line:
            continue

        # JSON 파싱
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            stats["bad_json"] += 1
            if len(errors) < max_errors:
                errors.append(f"JSON parse error: {e}")
            continue

        audio_field = record.get("audio", "")
        text = record.get("text", "")

        # 텍스트 체크
        if not text or not text.strip():
            stats["empty_text"] += 1
            if len(errors) < max_errors:
                errors.append(f"Empty text: audio={audio_field}")
            continue

        # 오디오 파일 존재 체크
        audio_path = resolve_audio_path(audio_field, root)
        if not audio_path.exists():
            stats["missing_audio"] += 1
            missing_files.append(audio_field)
            if len(errors) < max_errors:
                errors.append(f"Missing: {audio_path}")
            continue

        # 파일 크기
        try:
            stats["total_file_size"] += audio_path.stat().st_size
        except OSError:
            pass

        # 텍스트 길이 통계
        tlen = len(text)
        stats["text_len_total"] += tlen
        if tlen < stats["text_len_min"]:
            stats["text_len_min"] = tlen
        if tlen > stats["text_len_max"]:
            stats["text_len_max"] = tlen

        # WAV 헤더 검사
        if check_audio:
            try:
                info = read_wav_info(audio_path)
                stats["sample_rates"].add(info["sample_rate"])
                dur = info["duration_sec"]
                stats["duration_total_sec"] += dur
                if dur < stats["duration_min_sec"]:
                    stats["duration_min_sec"] = dur
                if dur > stats["duration_max_sec"]:
                    stats["duration_max_sec"] = dur
            except Exception as e:
                stats["audio_read_errors"] += 1
                if len(errors) < max_errors:
                    errors.append(f"Audio read error: {audio_path} -> {e}")
                continue

        stats["valid_rows"] += 1
        valid_lines.append(line)
        progress.set_postfix(
            valid=stats["valid_rows"],
            miss=stats["missing_audio"],
            refresh=False,
        )

    progress.close()

    # --clean: 유효한 행만 남기고 manifest 덮어쓰기
    removed = stats["total_rows"] - stats["valid_rows"]
    if clean and removed > 0:
        manifest_path.write_text(
            "\n".join(valid_lines) + "\n",
            encoding="utf-8",
        )
        stats["cleaned"] = True
        stats["removed_rows"] = removed

    # inf → 0 처리
    if stats["duration_min_sec"] == float("inf"):
        stats["duration_min_sec"] = 0.0
    if stats["text_len_min"] == float("inf"):
        stats["text_len_min"] = 0

    stats["sample_rates"] = sorted(stats["sample_rates"])
    stats["errors"] = errors
    stats["missing_files"] = missing_files
    return stats


def print_report(stats: dict, check_audio: bool):
    total = stats["total_rows"]
    valid = stats["valid_rows"]
    missing = stats["missing_audio"]
    empty = stats["empty_text"]
    bad = stats["bad_json"]
    audio_err = stats["audio_read_errors"]
    issue_count = missing + empty + bad + audio_err

    print()
    print(f"{'=' * 60}")
    print(f"  {stats['manifest']}")
    print(f"{'=' * 60}")

    # --- 기본 통계 ---
    print(f"  Total rows:        {total:>12,}")
    print(f"  Valid rows:        {valid:>12,}  ({pct(valid, total)})")
    print(f"  Total file size:   {format_size(stats['total_file_size']):>12}")

    # --- 텍스트 통계 ---
    if valid > 0:
        avg_tlen = stats["text_len_total"] / valid
        print(f"  Text length:       min={stats['text_len_min']}, max={stats['text_len_max']}, avg={avg_tlen:.1f} chars")

    # --- 오류 통계 ---
    print()
    print(f"  [Issues]")
    print(f"  Missing audio:     {missing:>12,}  ({pct(missing, total)})")
    print(f"  Empty text:        {empty:>12,}  ({pct(empty, total)})")
    print(f"  Bad JSON:          {bad:>12,}  ({pct(bad, total)})")
    if check_audio:
        print(f"  Audio read errors: {audio_err:>12,}  ({pct(audio_err, total)})")

    # --- 오디오 상세 (--check-audio) ---
    if check_audio and valid > 0:
        dur_total = stats["duration_total_sec"]
        dur_avg = dur_total / valid
        print()
        print(f"  [Audio]")
        print(f"  Sample rates:      {stats['sample_rates']}")
        print(f"  Total duration:    {format_duration(dur_total)} ({dur_total:,.1f}s)")
        print(f"  Duration:          min={stats['duration_min_sec']:.3f}s, max={stats['duration_max_sec']:.3f}s, avg={dur_avg:.3f}s")

    # --- 결과 ---
    print()
    if issue_count == 0:
        print(f"  Result: PASS")
    else:
        print(f"  Result: FAIL ({issue_count:,} issues)")

    # --- Clean 결과 ---
    if stats["cleaned"]:
        print(f"  Cleaned: {stats['removed_rows']:,} rows removed -> {valid:,} rows remaining")

    # --- 누락 파일 목록 ---
    missing_files = stats.get("missing_files", [])
    if missing_files:
        show_n = min(len(missing_files), len(stats["errors"]))
        print()
        print(f"  [Missing files] (showing {show_n} of {len(missing_files):,})")
        for f in missing_files[:show_n]:
            print(f"    - {f}")
        if len(missing_files) > show_n:
            print(f"    ... and {len(missing_files) - show_n:,} more")

    # --- 기타 오류 ---
    other_errors = [e for e in stats["errors"] if not e.startswith("Missing:")]
    if other_errors:
        print()
        print(f"  [Other errors] (first {len(other_errors)})")
        for e in other_errors:
            print(f"    - {e}")


def run(argv=None) -> list[dict]:
    """프로그래밍 API. 검사 결과 리스트 반환."""
    args = parse_args(argv)
    root = Path(args.root).resolve()

    if not root.exists():
        print(f"Error: 경로가 존재하지 않습니다: {root}")
        raise SystemExit(1)

    # manifest 목록 결정
    if args.manifest:
        manifests = [Path(args.manifest).resolve()]
    else:
        print(f"Searching manifest.jsonl under: {root}")
        manifests = find_manifests(root)
        print(f"Found: {len(manifests)}")

    if not manifests:
        print(f"Error: manifest.jsonl을 찾을 수 없습니다: {root}")
        raise SystemExit(1)

    # --clean 시 오디오 헤더 검사 자동 활성화
    check_audio = args.check_audio or args.clean

    print(f"Dataset root: {root}")
    print(f"Manifests found: {len(manifests)}")
    for m in manifests:
        print(f"  - {m.relative_to(root)}")
    if check_audio:
        print("Audio header check: ON")
    if args.clean:
        print("Clean mode: ON (invalid rows will be removed)")

    all_stats = []
    for manifest_path in manifests:
        stats = validate_manifest(
            manifest_path,
            root,
            check_audio=check_audio,
            clean=args.clean,
            max_errors=args.max_errors,
            show_progress=not args.no_progress,
        )
        all_stats.append(stats)
        print_report(stats, check_audio)

    # 전체 요약
    if len(all_stats) > 1:
        total_rows = sum(s["total_rows"] for s in all_stats)
        total_valid = sum(s["valid_rows"] for s in all_stats)
        total_missing = sum(s["missing_audio"] for s in all_stats)
        total_empty = sum(s["empty_text"] for s in all_stats)
        total_bad = sum(s["bad_json"] for s in all_stats)
        total_audio_err = sum(s["audio_read_errors"] for s in all_stats)
        total_issues = total_missing + total_empty + total_bad + total_audio_err
        total_size = sum(s["total_file_size"] for s in all_stats)

        print()
        print(f"{'=' * 60}")
        print(f"  TOTAL SUMMARY ({len(all_stats)} manifests)")
        print(f"{'=' * 60}")
        print(f"  Total rows:        {total_rows:>12,}")
        print(f"  Valid rows:        {total_valid:>12,}  ({pct(total_valid, total_rows)})")
        print(f"  Total file size:   {format_size(total_size):>12}")
        print()
        print(f"  Missing audio:     {total_missing:>12,}  ({pct(total_missing, total_rows)})")
        print(f"  Empty text:        {total_empty:>12,}  ({pct(total_empty, total_rows)})")
        print(f"  Bad JSON:          {total_bad:>12,}  ({pct(total_bad, total_rows)})")
        if check_audio:
            print(f"  Audio read errors: {total_audio_err:>12,}  ({pct(total_audio_err, total_rows)})")
            dur = sum(s["duration_total_sec"] for s in all_stats)
            print(f"  Total duration:    {format_duration(dur)}")
        print()
        if total_issues == 0:
            print(f"  Result: PASS")
        else:
            print(f"  Result: FAIL ({total_issues:,} issues)")
        if args.clean:
            total_removed = sum(s["removed_rows"] for s in all_stats)
            if total_removed > 0:
                print(f"  Cleaned: {total_removed:,} rows removed")

    return all_stats


def main(argv=None):
    """CLI 진입점. exit code 0=성공, 1=문제 발견."""
    all_stats = run(argv)
    total_issues = sum(
        s["missing_audio"] + s["empty_text"] + s["bad_json"] + s["audio_read_errors"]
        for s in all_stats
    )
    raise SystemExit(1 if total_issues > 0 else 0)


if __name__ == "__main__":
    main()
