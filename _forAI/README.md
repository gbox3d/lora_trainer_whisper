# _forAI Guide

## One-line summary

This directory stores AI-facing project notes for `lora_trainer_whisper`, a Whisper LoRA training and evaluation workspace.

## Read order

1. `README.md`
2. `inventory.md`
3. `memo.md`
4. `data_prep.md`
5. `dev_log.md`
6. `plan.md`

## File roles

- `inventory.md`: list what actually exists in the repository
- `memo.md`: keep short notes, open questions, and decision criteria
- `data_prep.md`: 데이터셋별 원본 구조·정리 상태·manifest 생성 요령 (AI 장기 메모리)
- `dev_log.md`: log `_forAI` maintenance and AI-assisted changes
- `plan.md`: capture remaining work and next project steps

## Working assumptions

- The repository is a script-first Python project rather than a packaged application.
- Core workflow is: prepare env -> build manifest -> train LoRA -> evaluate -> merge -> convert to CT2.
- `datasets/` and `outputs/` are runtime artifact locations and are already git-ignored.
- Korean ASR is the default use case in most scripts unless CLI args override it.

## Repository

- Name: `lora_trainer_whisper`
- Path: `/home/miso/work/lora_trainer_whisper`
