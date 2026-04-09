# Dev Log

## Entries

- 2026-04-01: Initialized `_forAI/` with `README.md`, `inventory.md`, `plan.md`, `memo.md`, and `dev_log.md` using the `forai-scaffold` skill workflow.
- 2026-04-01: Replaced default TODO sections with project-specific notes covering repository structure, main scripts, workflow entrypoints, and operational risks.
- 2026-04-01: Confirmed `_forAI/` is currently untracked rather than git-ignored; decide later whether to commit it or keep it local-only.
- 2026-04-03: Switched manifest generation toward manifest-relative `audio` paths, added shared manifest path resolution helpers, and updated train/eval consumers to resolve relative paths at read time.
- 2026-04-03: Added `make_manifest.py` options for `--audio-path-mode` and parallel `--workers`, and documented the new operational rule to avoid broken absolute paths behind symlinks or network-mounted storage.
- 2026-04-03: Added compatibility handling in `train_whisper_lora.py` for both legacy dataset audio dict payloads and newer `torchcodec` `AudioDecoder` payloads observed in the local environment.
- 2026-04-03: Rebuilt `/home/miso/datasets/Sample/manifest.jsonl` with the new manifest workflow, verified the file now stores `wav/...` relative audio paths, and confirmed `_load_and_cast_manifest` plus `DataCollatorSpeechSeq2Seq` can read the regenerated sample dataset end-to-end.
- 2026-04-09: Audited source vs documentation and updated repo docs to reflect the current `uv`-first workflow, the actual default base model (`openai/whisper-large-v3`), and the current status/result file behavior in `runner_cli.py`.
- 2026-04-09: Added pseudo-absolute manifest path recovery in `manifest_utils.py`, added `--num-epochs` support through `runner_cli.py` and `train_whisper_lora.py`, cleared stale `train_summary.json` before new runs, and forced manifest audio path remap to bypass stale dataset cache.
- 2026-04-09: `src/` 패키지 구조로 전환. `src/dataset_tools/` (manifest, validate) + `src/lora_trainer/` (train, eval, infer, merge, ct2) 두 패키지로 분리. 루트에 얇은 래퍼 유지하여 dalus_server 호환. `dev_dataset_tools` 브랜치에서 `validate_dataset.py`를 통합하고 `runner_cli.py`에 `validate` 서브커맨드 추가. `pyproject.toml`에 hatchling 빌드와 콘솔 엔트리포인트(`runner-cli`, `whisper-dataset`, `validate-dataset`) 추가. 버전 0.1.0 → 0.2.0.
- 2026-04-09: `runner_cli.py`가 `train.log` 시작부에 실제 사용 파라미터 JSON 블록을 남기고 progress 로그를 읽기 좋게 정규화하도록 보강. 루트 README와 `_forAI/memo.md`의 버전/운영 메모도 현재 상태에 맞게 정리. 버전 0.2.1 → 0.2.2.
