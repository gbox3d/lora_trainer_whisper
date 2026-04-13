"""
Cached Whisper data pipeline for multi-worker-friendly LoRA training.

Provides CachedWhisperDataset and helpers for ``--data-pipeline cached`` mode.

Key design decisions:
- No dependency on datasets.Audio; audio is read directly via soundfile.
- WhisperFeatureExtractor is initialized lazily per worker process so that
  multi-worker DataLoader does not hang on process-global state.
- Feature cache key includes: absolute audio path, file size, file mtime,
  model name, and max_audio_sec, so stale cache entries are automatically
  bypassed after audio files change or settings differ.
- Manifest row cache key includes: manifest absolute path, file size, mtime.
- Default cache root is <manifest_dir>/.lora_trainer_cache when
  --feature-cache-dir is not specified.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------

def _manifest_cache_key(manifest_path: Path) -> str:
    stat = manifest_path.stat()
    raw = f"{manifest_path}|{stat.st_size}|{stat.st_mtime:.6f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _audio_cache_key(audio_path: str, model_name: str, max_audio_sec: float) -> str:
    try:
        s = os.stat(audio_path)
        size, mtime = s.st_size, s.st_mtime
    except OSError:
        size, mtime = 0, 0.0
    raw = f"{audio_path}|{size}|{mtime:.6f}|{model_name}|{max_audio_sec}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Manifest row cache
# ---------------------------------------------------------------------------

def _load_manifest_rows(manifest_path: Path, cache_dir: Path) -> List[Dict[str, str]]:
    """Parse manifest.jsonl with an optional disk cache keyed on path+size+mtime."""
    cache_key = _manifest_cache_key(manifest_path)
    cache_file = cache_dir / f"manifest_{cache_key}.pkl"

    if cache_file.exists():
        with cache_file.open("rb") as f:
            return pickle.load(f)

    rows: List[Dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    cache_dir.mkdir(parents=True, exist_ok=True)
    with cache_file.open("wb") as f:
        pickle.dump(rows, f)

    return rows


def _load_resolved_rows(manifest_path: Path, cache_dir: Path) -> List[Dict[str, str]]:
    """
    Return manifest rows with audio paths fully resolved to absolute paths,
    using a disk cache keyed on manifest path+size+mtime.

    This avoids repeating resolve_audio_path() for every row on every startup
    when the manifest has not changed (large-manifest bottleneck prevention).
    """
    from dataset_tools.manifest_utils import resolve_audio_path

    cache_key = _manifest_cache_key(manifest_path)
    cache_file = cache_dir / f"resolved_{cache_key}.pkl"

    if cache_file.exists():
        with cache_file.open("rb") as f:
            return pickle.load(f)

    raw_rows = _load_manifest_rows(manifest_path, cache_dir)
    resolved = [
        {"audio": resolve_audio_path(r["audio"], manifest_path), "text": r["text"]}
        for r in raw_rows
    ]

    cache_dir.mkdir(parents=True, exist_ok=True)
    with cache_file.open("wb") as f:
        pickle.dump(resolved, f)

    return resolved


def _save_array_atomic(cache_file: Path, input_features: np.ndarray) -> None:
    """
    Save a numpy array atomically so parallel readers never observe a partial file.
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=cache_file.parent,
            prefix=f"{cache_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            np.save(temp_file, input_features)
            temp_path = Path(temp_file.name)
        os.replace(temp_path, cache_file)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CachedWhisperDataset(Dataset):
    """
    PyTorch Dataset that decodes audio and extracts Whisper input_features,
    caching the result to disk per sample.

    WhisperFeatureExtractor and the audio reader are intentionally NOT
    initialized in __init__; they are created lazily on first __getitem__
    call inside each worker process so that forked DataLoader workers do not
    inherit process-global model or device state from the main process.

    Each item returns:
        {"input_features": np.ndarray (80, 3000), "text": str}
    """

    def __init__(
        self,
        rows: List[Dict[str, str]],
        model_name: str,
        cache_dir: Path,
        max_audio_sec: float = 20.0,
    ) -> None:
        self._rows = rows
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._max_audio_sec = max_audio_sec
        # Worker-local — stays None until first __getitem__ in this process
        self._feature_extractor = None

    def __len__(self) -> int:
        return len(self._rows)

    def _get_feature_extractor(self):
        if self._feature_extractor is None:
            from transformers import WhisperFeatureExtractor
            self._feature_extractor = WhisperFeatureExtractor.from_pretrained(
                self._model_name
            )
        return self._feature_extractor

    @staticmethod
    def _load_audio(audio_path: str, max_audio_sec: float) -> np.ndarray:
        """Read only the needed audio prefix as float32 mono at 16 kHz."""
        import soundfile as sf

        with sf.SoundFile(audio_path) as audio_file:
            sr = audio_file.samplerate
            max_source_frames = -1
            if max_audio_sec > 0:
                max_source_frames = max(1, int(math.ceil(max_audio_sec * sr)))
            audio = audio_file.read(
                frames=max_source_frames,
                dtype="float32",
                always_2d=False,
            )

        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            except ImportError:
                from scipy.signal import resample as _resample
                n = int(len(audio) * 16000 / sr)
                audio = _resample(audio, n).astype(np.float32)

        if max_audio_sec > 0:
            max_len = int(max_audio_sec * 16000)
            if len(audio) > max_len:
                audio = audio[:max_len]

        return audio

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._rows[idx]
        audio_path: str = row["audio"]
        text: str = row["text"]

        cache_key = _audio_cache_key(audio_path, self._model_name, self._max_audio_sec)
        cache_file = self._cache_dir / f"feat_{cache_key}.npy"

        if cache_file.exists():
            input_features = np.load(str(cache_file), allow_pickle=False)
        else:
            audio = self._load_audio(audio_path, self._max_audio_sec)
            fe = self._get_feature_extractor()
            result = fe(audio, sampling_rate=16000, return_tensors="np")
            input_features = result["input_features"][0]  # (80, 3000)
            _save_array_atomic(cache_file, input_features)

        return {"input_features": input_features, "text": text}


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

@dataclass
class CachedDataCollator:
    """
    Collator for CachedWhisperDataset.

    Stacks pre-computed input_features into a tensor and tokenizes labels.
    Audio decoding and feature extraction do NOT happen here.
    """
    processor: Any  # WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = torch.from_numpy(
            np.stack([f["input_features"] for f in features])
        ).float()

        labels = self.processor.tokenizer(
            [f["text"] for f in features],
            return_tensors="pt",
            padding=True,
        ).input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_features": input_features,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_cached_datasets(
    manifest_path_str: str,
    eval_manifest_path_str: str,
    eval_ratio: float,
    seed: int,
    model_name: str,
    cache_dir: Optional[Path],
    max_audio_sec: float,
) -> Tuple[CachedWhisperDataset, CachedWhisperDataset]:
    """
    Build train and eval CachedWhisperDatasets from manifest paths.

    When eval_manifest_path_str is empty, an eval split is carved from the
    train manifest using eval_ratio and seed, matching legacy split semantics.

    cache_dir defaults to <manifest_dir>/.lora_trainer_cache when None.
    """
    from dataset_tools.manifest_utils import resolve_manifest_path

    manifest_path = resolve_manifest_path(manifest_path_str)
    if cache_dir is None:
        cache_dir = manifest_path.parent / ".lora_trainer_cache"

    resolved_rows = _load_resolved_rows(manifest_path, cache_dir)

    if eval_manifest_path_str:
        eval_manifest_path = resolve_manifest_path(eval_manifest_path_str)
        resolved_eval_rows = _load_resolved_rows(eval_manifest_path, cache_dir)
        train_rows = resolved_rows
    else:
        import random

        rng = random.Random(seed)
        indices = list(range(len(resolved_rows)))
        rng.shuffle(indices)
        n_eval = max(1, int(round(len(resolved_rows) * eval_ratio)))
        if len(resolved_rows) - n_eval < 1:
            n_eval = len(resolved_rows) - 1
        eval_set = set(indices[:n_eval])
        train_rows = [resolved_rows[i] for i in range(len(resolved_rows)) if i not in eval_set]
        resolved_eval_rows = [
            resolved_rows[i] for i in range(len(resolved_rows)) if i in eval_set
        ]

    train_ds = CachedWhisperDataset(train_rows, model_name, cache_dir, max_audio_sec)
    eval_ds = CachedWhisperDataset(resolved_eval_rows, model_name, cache_dir, max_audio_sec)
    return train_ds, eval_ds
