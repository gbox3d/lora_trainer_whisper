from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def resolve_manifest_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def relativize_audio_path(audio_path: str | Path, manifest_path: str | Path) -> str:
    manifest_dir = resolve_manifest_path(manifest_path).parent
    resolved_audio_path = Path(audio_path).expanduser().resolve()
    return os.path.relpath(resolved_audio_path, manifest_dir).replace("\\", "/")


def resolve_audio_path(audio_path: Any, manifest_path: str | Path) -> Any:
    if not isinstance(audio_path, str) or not audio_path.strip():
        return audio_path

    candidate = Path(audio_path).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    manifest_dir = resolve_manifest_path(manifest_path).parent
    return str((manifest_dir / candidate).resolve())


def read_manifest_records(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = resolve_manifest_path(path)
    records: list[dict[str, Any]] = []

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            if isinstance(record, dict) and "audio" in record:
                record["audio"] = resolve_audio_path(record.get("audio"), manifest_path)
            records.append(record)

    return records
