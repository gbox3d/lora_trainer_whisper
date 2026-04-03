# Inventory

## Repository

- Name: `lora_trainer_whisper`
- Path: `/home/miso/work/lora_trainer_whisper`

## Top-level structure

- `_forAI/`: local AI-facing notes, plans, and maintenance log
- `datasets/`: input datasets and generated `manifest.jsonl` files; sample data exists under `datasets/Sample`
- `outputs/`: training checkpoints, merged models, TensorBoard runs, and CT2 export targets
- `.venv/`: local `uv` virtual environment
- `README.md`: setup instructions and example training/evaluation commands
- `pyproject.toml`: project metadata, Python dependencies, and custom `uv` source config
- `uv.lock`: locked dependency resolution for the current environment

## Important scripts

- `train_whisper_lora.py`: main training entrypoint for Whisper LoRA fine-tuning
- `make_manifest.py`: builds `manifest.jsonl` from dataset label/audio structure
- `eval_dataset_lora.py`: compares LoRA output against a base Whisper model on a manifest
- `infer_lora.py`: single-file inference using a base model plus LoRA adapter
- `compare_infer.py`: side-by-side base vs LoRA inference with optional normalization and metrics
- `merge_peft.py`: merges a LoRA adapter into the base model for export
- `eval_dataset_ct2.py`: compares PyTorch Whisper and Faster-Whisper/CT2 outputs
- `test_run_ct2.py`: manual smoke test for a converted CT2 model
- `check.py`: local environment and CUDA/torchcodec verification script
- `setup.sh`: `uv`-based environment bootstrap for DGX-style machines
- `main.py`: placeholder console entrypoint, not the operational training path

## Entrypoints

- Environment setup: `sh setup.sh`
- Environment check: `python check.py`
- Manifest build: `python make_manifest.py --root ./datasets/Sample --wav_dir wav --label_dir lb`
- Single-GPU training: `python train_whisper_lora.py --model_name openai/whisper-small --manifest datasets/Sample/manifest.jsonl --output_dir outputs/small_lora`
- Multi-GPU training: `torchrun --nproc_per_node=2 train_whisper_lora.py ...`
- LoRA evaluation: `python eval_dataset_lora.py --manifest ... --base_model ... --lora_dir ...`
- Single-file inference: `python infer_lora.py --wav <path> --base_model ... --lora_dir ...`
- LoRA merge: `python merge_peft.py`
- CT2 evaluation: `python eval_dataset_ct2.py --manifest ... --ct2_dir ...`

## Tests

- No formal `pytest` or CI-based test suite is present in the repository root.
- `test_run_ct2.py` is a manual smoke test for converted CT2/Faster-Whisper inference.
- `check.py` is also a manual validation tool for local environment readiness.
- Current confidence comes from ad hoc script execution rather than repeatable automated tests.

## Notes

- The project uses `uv` and pins PyTorch packages through a custom CUDA 13.0 index in `pyproject.toml`.
- `train_whisper_lora.py` supports either a dedicated `--eval_manifest` or an automatic train/eval split via `--eval_ratio`.
- Most scripts assume Korean transcription defaults: `language=ko`, `task=transcribe`, and 16 kHz audio input.
- Several operational examples in `README.md` use machine-specific absolute paths; scripts should be preferred over copied paths.
- `datasets/*` and `outputs/` are git-ignored, so repository state alone is not enough to reconstruct experiments.
