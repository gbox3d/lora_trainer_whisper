# Dev Log

## Entries

- 2026-04-13: 문서 정리 및 커밋. `README.md`·`_forAI/memo.md`의 버전 표기를 `0.3.0` → `0.3.1`로 수정 (`pyproject.toml` 기준 불일치 해소). `src/lora_trainer/data_pipeline.py`(신규, cached 파이프라인 + spawn worker + atomic write + resolved-manifest cache)와 `tests/test_data_pipeline.py` 포함 전체 변경사항을 단일 커밋으로 통합. 버전 0.2.2 → 0.3.1.

- 2026-04-01: Initialized `_forAI/` with `README.md`, `inventory.md`, `plan.md`, `memo.md`, and `dev_log.md` using the `forai-scaffold` skill workflow.
- 2026-04-01: Replaced default TODO sections with project-specific notes covering repository structure, main scripts, workflow entrypoints, and operational risks.
- 2026-04-01: At that time `_forAI/` was treated as a local workspace candidate rather than an established repo-tracked collaboration surface. This note is kept as historical context only and is no longer current.
- 2026-04-03: Switched manifest generation toward manifest-relative `audio` paths, added shared manifest path resolution helpers, and updated train/eval consumers to resolve relative paths at read time.
- 2026-04-03: Added `make_manifest.py` options for `--audio-path-mode` and parallel `--workers`, and documented the new operational rule to avoid broken absolute paths behind symlinks or network-mounted storage.
- 2026-04-03: Added compatibility handling in `train_whisper_lora.py` for both legacy dataset audio dict payloads and newer `torchcodec` `AudioDecoder` payloads observed in the local environment.
- 2026-04-03: Rebuilt `/home/miso/datasets/Sample/manifest.jsonl` with the new manifest workflow, verified the file now stores `wav/...` relative audio paths, and confirmed `_load_and_cast_manifest` plus `DataCollatorSpeechSeq2Seq` can read the regenerated sample dataset end-to-end.
- 2026-04-09: Audited source vs documentation and updated repo docs to reflect the current `uv`-first workflow, the actual default base model (`openai/whisper-large-v3`), and the current status/result file behavior in `runner_cli.py`.
- 2026-04-09: Added pseudo-absolute manifest path recovery in `manifest_utils.py`, added `--num-epochs` support through `runner_cli.py` and `train_whisper_lora.py`, cleared stale `train_summary.json` before new runs, and forced manifest audio path remap to bypass stale dataset cache.
- 2026-04-09: `src/` 패키지 구조로 전환. `src/dataset_tools/` (manifest, validate) + `src/lora_trainer/` (train, eval, infer, merge, ct2) 두 패키지로 분리. 루트에 얇은 래퍼 유지하여 dalus_server 호환. `dev_dataset_tools` 브랜치에서 `validate_dataset.py`를 통합하고 `runner_cli.py`에 `validate` 서브커맨드 추가. `pyproject.toml`에 hatchling 빌드와 콘솔 엔트리포인트(`runner-cli`, `whisper-dataset`, `validate-dataset`) 추가. 버전 0.1.0 → 0.2.0.
- 2026-04-09: `runner_cli.py`가 `train.log` 시작부에 실제 사용 파라미터 JSON 블록을 남기고 progress 로그를 읽기 좋게 정규화하도록 보강. 루트 README와 `_forAI/memo.md`의 버전/운영 메모도 현재 상태에 맞게 정리. 버전 0.2.1 → 0.2.2.
- 2026-04-10: 버전 0.2.2 → 0.3.0. `cached` 데이터 파이프라인, resolved-manifest cache, `dalus_server` 옵션 연동, 그리고 관련 문서/검증 결과를 현재 기준으로 정리했다.
- 2026-04-10: Adopted `_forAI` as the official repo-tracked Codex/Claude collaboration workspace. Updated `README.md` with explicit agent roles and read order, revised `plan.md` to be the primary decision-complete handoff surface, added durable governance rules to `memo.md`, and corrected the stale local-only interpretation of `_forAI`.
- 2026-04-10: Benchmark 4종 완료 (model=whisper-small, manifest=datasets/Sample/manifest.jsonl, batch=2, grad_accum=1, max_steps=30, seed=42, fresh output dirs).

  | run | pipeline | workers | steps 1-10 | steps 11-20 | steps 21-30 | total_wall |
  |-----|----------|---------|-----------|------------|------------|------------|
  | 1 | legacy | 0 | 3.5s (0.35s/step) | 2.8s (0.28s/step) | 2.8s (0.28s/step) | 24,970ms |
  | 2 | cached | 0 (cold cache) | 10.6s (1.06s/step) | 2.8s (0.28s/step) | 2.8s (0.28s/step) | 31,720ms |
  | 3 | cached | 1 (warm cache) | 3.1s (0.31s/step) | 2.5s (0.25s/step) | 2.5s (0.25s/step) | 22,981ms |
  | 4 | cached | 2 (warm cache) | 3.1s (0.31s/step) | 2.5s (0.25s/step) | 2.5s (0.25s/step) | 24,084ms |

  Key findings: (1) cached+workers≥1 warm cache: ~10-12% faster per step vs legacy+workers=0. (2) No hang on cached+workers=1 or workers=2. (3) workers=1 vs workers=2 shows no difference on this dataset/model size. (4) Cold-start overhead (run 2): first 10 steps took 10.6s while feature cache was built for 327 samples; subsequent runs warm. (5) datasets/Sample/manifest.jsonl regenerated with correct manifest-relative paths (wav/...) and now resolves correctly.

- 2026-04-10: Added `_load_resolved_rows()` to `data_pipeline.py`. Caches `resolve_audio_path()` results keyed on manifest path+size+mtime (`resolved_{key}.pkl`). `build_cached_datasets()` no longer re-runs path resolution on every startup when manifest is unchanged. Removed prefetch_factor residual-risk note (confirmed valid in transformers 4.57.5).
- 2026-04-10: Implemented `--data-pipeline {legacy,cached}` + `--feature-cache-dir` options. Created `src/lora_trainer/data_pipeline.py` (CachedWhisperDataset, CachedDataCollator, build_cached_datasets). Modified `train_whisper_lora.py` and `runner_cli.py`. legacy mode unchanged. cached mode: no datasets.Audio dependency, lazy worker-local feature extractor init, disk-keyed feature cache, auto persistent_workers+prefetch_factor for workers>0. Benchmark results are recorded above.
- 2026-04-10: Completed `_forAI`-Centered Claude Collaboration Protocol refactor. `plan.md` Active Handoff cleared (protocol adoption completed). `README.md` updated with `data_prep.md` update scope rule and task format requirement. `data_prep.md` memory frontmatter removed (it is a `_forAI` collaboration document, not a Claude memory file). `plan.md` Active Handoff section reset to empty state for next Codex-assigned task.
- 2026-04-10: Reviewed Claude's `_forAI` cleanup and accepted the empty `plan.md` handoff state plus the `data_prep.md` frontmatter removal. Added a new decision-complete Active Handoff for the multi-worker training data pipeline refactor, and created operator-approved temporary files `_tmp_claude_start_prompt.md` and `_tmp_claude_channel.md` to start the implementation cycle.
- 2026-04-10: Temporary `_tmp_claude_*` coordination files were retired after their benchmark notes, scope decisions, and final outcomes were absorbed into `plan.md`, `memo.md`, and `dev_log.md`.
- 2026-04-10: Real-dataset direct debugging pass for the worker issue (manifest=`/home/miso/datasets/extrahdd/sdc1/prepared_datasets/130/Training/manifest.jsonl`, model=`openai/whisper-large-v3`).

  Direct smoke runs executed by Codex:

  | run | pipeline | workers | batch | grad_accum | gc | result |
  |-----|----------|---------|-------|------------|----|--------|
  | `/home/miso/datasets/outputs/smoke_largev3_cached_w0_bs1` | cached | 0 | 1 | 1 | on | pass |
  | `/home/miso/datasets/outputs/smoke_largev3_cached_w1_bs1` | cached | 1 | 1 | 1 | on | pass |
  | `/home/miso/datasets/outputs/smoke_largev3_cached_w2_bs1` | cached | 2 | 1 | 1 | on | pass |
  | `/home/miso/datasets/outputs/smoke_largev3_cached_w0_bs4` | cached | 0 | 4 | 1 | on | pass |
  | `/home/miso/datasets/outputs/smoke_largev3_cached_w1_bs4` | cached | 1 | 4 | 1 | on | pass |
  | `/home/miso/datasets/outputs/smoke_largev3_cached_w2_bs4` | cached | 2 | 4 | 1 | on | hang before first step; interrupted after ~111s |

  Additional observations:

  - In the failing `cached + workers=2 + batch=4` run, GPU stayed at 0%, `train.log` remained at `0/1`, and the feature cache directory did not produce any new `feat_*.npy` files beyond the earlier smoke artifacts.
  - This narrowed the current real-data boundary to: `workers=2` is not categorically impossible, but it is not stable once batch size increased from 1 to 4 on `large-v3`.
  - The next debugging pass should start in `src/lora_trainer/data_pipeline.py`, especially `_load_audio()`, because the current implementation reads the full audio file before truncating to `max_audio_sec`.

- 2026-04-10: User-created dalus_server profile `parm_1` applied correctly and launched a fresh run with:
  - `batch_size=4`
  - `grad_accum=8`
  - `use_gradient_checkpointing=true`
  - `dataloader_workers=0`
  - `data_pipeline=legacy`
  - output dir: `/home/miso/datasets/outputs/large-v3_lora`

  This run progressed past the first steps, confirming that the stable behavior observed there came from the safer `worker=0` configuration rather than any `worker>0` fix.

- 2026-04-10: Reusing `/home/miso/datasets/outputs/large-v3_lora` caused older `worker=2` append-log blocks and the newer `worker=0` run to coexist in one `train.log`. This directly caused confusion during diagnosis. Treat fresh output directories as mandatory for future worker debugging.
