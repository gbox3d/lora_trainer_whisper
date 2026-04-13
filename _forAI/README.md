# _forAI Guide

## One-line summary

This directory is the repo-tracked collaboration workspace for `lora_trainer_whisper`.
Codex manages scope, review, and acceptance here, and Claude reads/writes the approved
working documents here while implementing tasks.

## Read order

Claude must read `_forAI` in this order before coding:

1. `README.md`
2. `inventory.md`
3. `memo.md`
4. `data_prep.md`
5. `dev_log.md`
6. `plan.md`

## Roles

- Codex
  - owns intake, scoping, review, and acceptance
  - turns user intent into decision-complete tasks in `plan.md`
  - checks Claude output against code, logs, and observed runtime behavior
- Claude
  - reads `_forAI` in the order above before implementation
  - updates `plan.md`, `memo.md`, and `dev_log.md` while working
  - updates `inventory.md` only when repository facts or entrypoints actually change

## File roles

- `README.md`: stable entrypoint for `_forAI`; defines read order, roles, and update rules
- `inventory.md`: factual source of truth for repository structure, entrypoints, and stable runtime behavior; edit only for factual repo changes
- `memo.md`: durable rules, constraints, performance findings, and decisions that should not be rediscovered
- `data_prep.md`: dataset-specific operational memory; only update when dataset preparation conventions change
- `dev_log.md`: append-only log of completed work, experiments, corrections, and measured outcomes
- `plan.md`: active execution spec and primary Claude handoff document

## Update rules

- Keep `_forAI` repo-tracked. Do not treat it as temporary local scratch space.
- Do not add ad hoc coordination files unless the standard `_forAI` set becomes clearly insufficient.
- If code, logs, or runtime observations disagree with `_forAI`, code and evidence win.
- When correcting stale `_forAI` text, record the correction in `dev_log.md`; do not silently rewrite history.
- For performance work, record baseline and comparison results in `dev_log.md`.
- For architectural work, copy the durable conclusion into `memo.md`.
- For repository shape or entrypoint changes, update `inventory.md` in the same task.
- `data_prep.md` changes are for dataset preparation convention changes only; do not write general observations or transient experiment notes here.
- Each active task in `plan.md` must be decision-complete before Claude implements: Goal, Success criteria, In scope / Out of scope, Expected files or subsystems, Verification, Risks.

## Working assumptions

- The repository is a script-first Python project rather than a packaged application.
- Core workflow is: prepare env -> build manifest -> train LoRA -> evaluate -> merge -> convert to CT2.
- `datasets/` and `outputs/` are runtime artifact locations and are already git-ignored.
- Korean ASR is the default use case in most scripts unless CLI args override it.

## Repository

- Name: `lora_trainer_whisper`
- Path: `/home/miso/work/lora_trainer_whisper`
