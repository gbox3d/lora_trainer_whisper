# Plan

## How To Use This File

- `plan.md` is the primary Claude handoff document.
- Codex owns scope, task framing, and acceptance.
- Claude implements only after the active task here is decision complete.
- Claude may update this file while working to mark progress, narrow remaining work, and record blockers.
- When a task changes repository facts or entrypoints, `inventory.md` must be updated in the same task.
- When a task produces a durable rule, copy the conclusion into `memo.md`.
- When a task includes performance claims, add baseline and comparison results to `dev_log.md`.

## Active Task Template

Use this compact structure for each Claude-facing task:

- Goal
- Success criteria
- In scope
- Out of scope
- Expected files or subsystems
- Verification
- Risks or known constraints

## Active Handoff

- Goal
  - Fix or decisively bound the `worker > 0` instability on the real dataset path (`manifest=/home/miso/datasets/extrahdd/sdc1/prepared_datasets/130/Training/manifest.jsonl`) for `openai/whisper-large-v3`.
- Success criteria
  - Reproduce the current failure mode with fresh output directories and exact settings.
  - Identify the blocking stage with code-level evidence, not guesswork.
  - Land either:
    - a code fix that makes at least `cached + workers=1` reliable in a practical large-v3 setting, or
    - an explicit runtime guardrail/error message if a safe fix is not yet ready.
  - Update `memo.md` and `dev_log.md` with the verified limits and the final recommendation.
- In scope
  - `src/lora_trainer/data_pipeline.py`
  - `src/lora_trainer/train_whisper_lora.py`
  - `src/lora_trainer/runner_cli.py`
  - `_forAI/plan.md`, `_forAI/memo.md`, `_forAI/dev_log.md`
- Out of scope
  - New model architecture work
  - Large product/UI changes in `dalus_server` unless behavior or wording must be corrected
  - Full-dataset offline precompute pipeline
- Expected files or subsystems
  - Cached dataset path, audio loading path, feature cache write path, DataLoader settings, timing/logging around first batch
- Verification
  - Use fresh output directories for every run.
  - Record `run_config.json`, relevant `train.log` block, whether `feat_*.npy` files start appearing, and whether GPU utilization leaves 0%.
  - Minimum matrix to keep in view:
    - `cached + workers=0 + batch=4`
    - `cached + workers=1 + batch=4`
    - `cached + workers=2 + batch=4`
    - `legacy + workers=0 + batch=4 + grad_accum=8`
- Risks or known constraints
  - `train.log` is append-based; reusing an output directory causes false conclusions.
  - Cold-start cached runs pay manifest/resolved cache cost first.
  - On this machine, GB10 reports a CUDA capability warning with the current PyTorch build.
  - Current real-manifest evidence says `cached + workers=2 + batch>=4` can stall before the first feature-cache write.

## Current Debug Snapshot

- Real dataset + large-v3:
  - `legacy + workers=0 + batch=4 + grad_accum=8 + gradient_checkpointing=true` is currently progressing in `/home/miso/datasets/outputs/large-v3_lora`.
  - `cached + workers=0/1/2 + batch=1 + max_steps=1` all completed.
  - `cached + workers=0/1 + batch=4 + max_steps=1` completed.
  - `cached + workers=2 + batch=4 + max_steps=1` stalled at `0/1` for ~111s with GPU at 0% and no new `feat_*.npy` files; run was manually interrupted.
- Immediate suspicion to investigate first:
  - `data_pipeline.py` currently reads the full audio file before truncating to `max_audio_sec`; this may amplify worker-side I/O and memory pressure.

## Current Project Priorities

- Keep the `src/` package split stable and preserve dalus_server compatibility through the CLI wrappers.
- Maintain the existing CLI and output contracts unless a coordinated change is documented in both code and `_forAI`.
- Use this active task to drive the next direct Codex debugging pass on the worker issue.

## Backlog Seeds

- Add a benchmark protocol for comparing `dataloader_workers=0/1/2` under fixed training settings.
- Clarify which evaluation path should be treated as the official regression baseline.
