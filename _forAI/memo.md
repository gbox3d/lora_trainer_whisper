# Memo

## Version management

- Use `major.minor.patch` versioning.
- Current project version in `pyproject.toml` is `0.1.0`.
- Increase `major` for large structural changes or compatibility-breaking changes.
- Increase `minor` for new features or meaningful improvements.
- Increase `patch` for bug fixes or small changes.
- Record version-by-version changes in `dev_log.md`.

## Open questions

- Should helper scripts be promoted into a small package/CLI, or is a flat script layout still the right tradeoff?
- Which training/evaluation path should be treated as the official baseline for future comparisons: LoRA-on-base, merged model, or CT2 export?
- Do we want `_forAI/` committed with the repo, or kept purely as a local workspace note set?

## Decision criteria

- Prefer reproducible CLI workflows over notebook-style or one-off command edits.
- Prefer removing hard-coded machine paths before adding new experiment helpers.
- Prefer manifest-relative audio paths over machine-specific absolute paths, especially when datasets live behind symlinks, external disks, or network mounts.
- Prefer changes that keep DGX/ARM64 environment setup explicit, since build issues are a recurring operational risk.
- Favor lightweight smoke tests that verify wiring without requiring long GPU training jobs.

## Short notes

- `manifest_utils.resolve_audio_path()`는 이제 manifest-relative 경로뿐 아니라, 실제로는 존재하지 않는 pseudo-absolute 경로도 manifest 상위 디렉터리를 기준으로 복구 시도한다.
- `runner_cli.py train` and `train_whisper_lora.py` now support `--num-epochs`; `0` means keep the existing `max_steps`-driven behavior.
- `runner_cli.py`는 train 시작 전에 기존 `train_summary.json`을 지워서 UI가 이전 run 요약을 잘못 보여주지 않게 한다.
- `train_whisper_lora.py` manifest audio path remap은 `load_from_cache_file=False`로 강제 재계산한다.
- `make_manifest.py` now defaults to relative `audio` paths and exposes `--audio-path-mode` plus `--workers` for large manifest rebuilds on slower storage.
- `train_whisper_lora.py` can auto-split train/eval if `--eval_manifest` is not supplied.
- `train_whisper_lora.py`, `eval_dataset_lora.py`, and `eval_dataset_ct2.py` resolve manifest-relative audio paths before reading files.
- Current `datasets` releases may return `torchcodec` audio decoders instead of the older `{"array": ...}` dict shape; the collator now has compatibility handling for both.
- `merge_peft.py` and `test_run_ct2.py` are now argparse-based, but they still keep convenience defaults and are best treated as operator tools rather than a stable public API.
- Repository documentation should prefer `uv run ...` examples even when plain `python ...` would work inside an activated environment.

## Doc audit: source vs docs

- `README.md` had several examples still using `python ...`; source and 운영 기준은 `uv` 중심이므로 `uv run ...` 쪽이 더 정확하다.
- `README.md` 일부 예시와 옵션 표는 예전 기본값인 `openai/whisper-small`을 가리켰지만, 실제 코드 기본값은 현재 `openai/whisper-large-v3`다.
- 상태 파일 설명은 대체로 맞았지만, 실제 소스에서는 학습과 평가가 서로 다른 파일명(`run_status.json`, `eval_status.json`)을 쓴다.
- `infer`는 항상 결과 파일을 남기지 않고, `runner_cli.py infer --result-json`을 넘겼을 때만 JSON을 기록한다.
