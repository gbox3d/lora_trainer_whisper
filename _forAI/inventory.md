# Inventory

## Repository

- Name: `lora_trainer_whisper`
- Path: `/home/miso/work/lora_trainer_whisper`

## Top-level structure

- `src/dataset_tools/`: 데이터셋 준비 패키지 (manifest 생성, 유효성 검사). ML 의존 없음
- `src/lora_trainer/`: LoRA 학습/평가/추론/병합/CT2 변환 패키지. ML 의존 (torch, transformers, peft 등)
- `_forAI/`: local AI-facing notes, plans, and maintenance log
- `datasets/`: input datasets and generated `manifest.jsonl` files; sample data exists under `datasets/Sample`
- `outputs/`: training checkpoints, merged models, TensorBoard runs, and CT2 export targets
- `.venv/`: local `uv` virtual environment
- `README.md`: setup instructions and example training/evaluation commands
- `pyproject.toml`: project metadata, Python dependencies, hatchling 빌드, console_scripts 정의
- `uv.lock`: locked dependency resolution for the current environment
- 루트 스크립트 (`runner_cli.py`, `train_whisper_lora.py` 등): `src/` 패키지로 위임하는 얇은 래퍼. dalus_server 호환용

## src/dataset_tools/ (경량 패키지)

- `manifest_utils.py`: manifest 경로 해석 공용 유틸. relative `audio` 경로를 manifest 위치 기준 절대경로로 복원
- `make_manifest.py`: builds `manifest.jsonl` from dataset label/audio structure; 상대경로 `audio` 저장과 병렬 label scan 지원
- `validate_dataset.py`: manifest 유효성 검사 (오디오 존재, 텍스트 비어있음, JSON 파싱, WAV 헤더 검사, `--clean`으로 불량 행 제거)
- `cli.py`: `whisper-dataset` 콘솔 엔트리포인트 (manifest / validate 서브커맨드)

## src/lora_trainer/ (ML 패키지)

- `runner_cli.py`: 통합 CLI 진입점 (dalus_server가 subprocess로 호출). subcommand: manifest, train, eval, infer, merge, ct2-export, validate
- `workflow_utils.py`: 공유 유틸리티 (JSON 쓰기, 타임스탬프, 체크포인트 탐색)
- `train_whisper_lora.py`: main training entrypoint for Whisper LoRA fine-tuning
- `eval_dataset_lora.py`: compares LoRA output against a base Whisper model on a manifest
- `infer_lora.py`: single-file inference using a base model plus LoRA adapter
- `compare_infer.py`: side-by-side base vs LoRA inference with optional normalization and metrics
- `merge_peft.py`: merges a LoRA adapter into the base model for export
- `eval_dataset_ct2.py`: compares PyTorch Whisper and Faster-Whisper/CT2 outputs
- `test_run_ct2.py`: manual smoke test for a converted CT2 model
- `check.py`: local environment and CUDA/torchcodec verification script

## Other files

- `setup.sh`: `uv`-based environment bootstrap for DGX-style machines
- `main.py`: placeholder console entrypoint, not the operational training path

## Entrypoints

- Environment setup: `sh setup.sh`
- Environment check: `uv run python check.py`
- Unified CLI (래퍼): `uv run python runner_cli.py <subcommand> [args]` (manifest, train, eval, infer, merge, ct2-export, validate)
- Unified CLI (엔트리포인트): `uv run runner-cli <subcommand> [args]`
- Dataset CLI (엔트리포인트): `uv run whisper-dataset <subcommand> [args]` (manifest, validate)
- Dataset validate: `uv run python runner_cli.py validate <root> [--check-audio] [--clean]`
- Single-GPU training: `uv run python train_whisper_lora.py --model_name openai/whisper-large-v3 --manifest datasets/Sample/manifest.jsonl --output_dir outputs/large-v3_lora`
- Multi-GPU training: `uv run torchrun --nproc_per_node=2 train_whisper_lora.py ...`
- LoRA evaluation: `uv run python eval_dataset_lora.py --manifest ... --base_model ... --lora_dir ...`
- Single-file inference: `uv run python infer_lora.py --wav <path> --base_model ... --lora_dir ...`
- LoRA merge: `uv run python merge_peft.py`
- CT2 evaluation: `uv run python eval_dataset_ct2.py --manifest ... --ct2_dir ...`

## Tests

- No formal `pytest` or CI-based test suite is present in the repository root.
- `test_run_ct2.py` is a manual smoke test for converted CT2/Faster-Whisper inference.
- `check.py` is also a manual validation tool for local environment readiness.
- Current confidence comes from ad hoc script execution rather than repeatable automated tests.

## Notes

- The project uses `uv` and pins PyTorch packages through a custom CUDA 13.0 index in `pyproject.toml`.
- 2026-04-09: `src/` 패키지 구조로 전환. `dataset_tools` (경량)과 `lora_trainer` (ML) 두 패키지로 분리. 루트 스크립트는 얇은 래퍼로 유지하여 dalus_server 호환성 보존.
- `pyproject.toml`에 hatchling 빌드 시스템과 `[project.scripts]` 콘솔 엔트리포인트 추가 (`runner-cli`, `whisper-dataset`, `validate-dataset`).
- dalus_server는 여전히 `uv run python runner_cli.py <cmd>` 형식으로 호출하며, 루트 래퍼가 이를 `lora_trainer.runner_cli`로 위임.
- `validate_dataset.py`가 `dataset_tools`에 추가됨 (dev_dataset_tools 브랜치에서 통합). manifest 유효성 검사, WAV 헤더 검사, 불량 행 제거 기능.
- Manifest `audio` 필드는 2026-04-03부터 기본적으로 `manifest.jsonl` 기준 상대경로를 사용한다.
- 학습/평가 로더는 relative/absolute 두 형식을 모두 읽도록 맞춰져 있어, 기존 manifest와 새 manifest를 모두 소비할 수 있다.
- `train_whisper_lora.py` collator는 `datasets`의 legacy dict payload와 newer `torchcodec` `AudioDecoder` payload를 둘 다 처리한다.
- `train_whisper_lora.py` supports either a dedicated `--eval_manifest` or an automatic train/eval split via `--eval_ratio`.
- 기본 모델이 `openai/whisper-large-v3`로 변경됨 (2026-04-03). 모든 스크립트 CLI 기본값 일괄 변경.
- `runner_cli.py` 기준으로 `infer`는 `--result-json` 옵션이 있을 때만 JSON 파일을 별도 저장한다.
- `runner_cli.py` 기준으로 상태 파일 이름은 학습 `run_status.json`, 평가 `eval_status.json`으로 서로 다르다.
- Most scripts assume Korean transcription defaults: `language=ko`, `task=transcribe`, and 16 kHz audio input.
- `datasets/*` and `outputs/` are git-ignored, so repository state alone is not enough to reconstruct experiments.
