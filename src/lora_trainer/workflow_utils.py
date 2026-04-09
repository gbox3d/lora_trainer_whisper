from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, payload: dict[str, Any]) -> str:
    target = ensure_parent(path)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(target)


def checkpoint_names(output_dir: str | Path) -> list[str]:
    root = Path(output_dir)
    if not root.exists():
        return []
    return sorted(
        [entry.name for entry in root.glob("checkpoint-*") if entry.is_dir()],
        key=lambda name: int(name.replace("checkpoint-", "")) if name.replace("checkpoint-", "").isdigit() else -1,
    )
