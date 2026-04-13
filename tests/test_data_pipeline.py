import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lora_trainer import data_pipeline, train_whisper_lora


class _FakeFeatureExtractor:
    def __call__(self, audio, sampling_rate, return_tensors):
        self.last_audio_len = len(audio)
        return {"input_features": np.full((1, 80, 8), len(audio), dtype=np.float32)}


def _fake_get_feature_extractor(_self):
    return _FakeFeatureExtractor()


class _FakeSoundFile:
    last_frames = None

    def __init__(self, _audio_path):
        self.samplerate = 16000

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, frames=-1, dtype="float32", always_2d=False):
        del dtype, always_2d
        _FakeSoundFile.last_frames = frames
        return np.ones(frames, dtype=np.float32)


class CachedDataPipelineTests(unittest.TestCase):
    def test_load_audio_reads_only_requested_prefix(self):
        with mock.patch("soundfile.SoundFile", _FakeSoundFile):
            audio = data_pipeline.CachedWhisperDataset._load_audio(
                "ignored.wav",
                max_audio_sec=0.5,
            )

        self.assertEqual(_FakeSoundFile.last_frames, 8000)
        self.assertEqual(audio.shape, (8000,))

    def test_cached_dataloader_kwargs_reduce_prefetch_for_multi_worker(self):
        self.assertEqual(
            train_whisper_lora._cached_dataloader_kwargs(True, 0),
            {},
        )
        self.assertEqual(
            train_whisper_lora._cached_dataloader_kwargs(False, 2),
            {},
        )
        self.assertEqual(
            train_whisper_lora._cached_dataloader_kwargs(True, 1),
            {
                "dataloader_persistent_workers": True,
                "dataloader_prefetch_factor": 2,
            },
        )
        self.assertEqual(
            train_whisper_lora._cached_dataloader_kwargs(True, 2),
            {
                "dataloader_persistent_workers": True,
                "dataloader_prefetch_factor": 1,
            },
        )

    def test_cached_multiprocessing_context_uses_spawn_for_cached_workers(self):
        self.assertIsNone(train_whisper_lora._cached_multiprocessing_context(True, 0))
        self.assertIsNone(train_whisper_lora._cached_multiprocessing_context(False, 4))
        self.assertEqual(
            train_whisper_lora._cached_multiprocessing_context(True, 1),
            "spawn",
        )
        self.assertEqual(
            train_whisper_lora._cached_multiprocessing_context(True, 4),
            "spawn",
        )

    @unittest.skipUnless(sys.platform.startswith("linux"), "fork-based worker smoke test requires Linux")
    def test_cached_dataset_smoke_with_multiple_workers(self):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            audio_dir = temp_dir / "audio"
            cache_dir = temp_dir / "cache"
            audio_dir.mkdir()

            rows = []
            base_wave = np.linspace(-0.2, 0.2, 16000, dtype=np.float32)
            for index in range(8):
                audio_path = audio_dir / f"sample_{index}.wav"
                sf.write(audio_path, base_wave, 16000)
                rows.append({"audio": str(audio_path), "text": f"text {index}"})

            dataset = data_pipeline.CachedWhisperDataset(
                rows=rows,
                model_name="dummy-whisper",
                cache_dir=cache_dir,
                max_audio_sec=0.5,
            )

            with mock.patch.object(
                data_pipeline.CachedWhisperDataset,
                "_get_feature_extractor",
                new=_fake_get_feature_extractor,
            ):
                loader = DataLoader(
                    dataset,
                    batch_size=4,
                    num_workers=2,
                    persistent_workers=True,
                    prefetch_factor=1,
                    multiprocessing_context="fork",
                )
                iterator = iter(loader)
                batch = next(iterator)
                del iterator
                del loader

            self.assertEqual(batch["input_features"].shape, torch.Size([4, 80, 8]))
            self.assertGreaterEqual(len(list(cache_dir.glob("feat_*.npy"))), 4)


if __name__ == "__main__":
    unittest.main()
