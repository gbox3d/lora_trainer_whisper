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
- Prefer changes that keep DGX/ARM64 environment setup explicit, since build issues are a recurring operational risk.
- Favor lightweight smoke tests that verify wiring without requiring long GPU training jobs.

## Short notes

- `make_manifest.py` expects paired label/audio trees and writes JSONL rows with `audio` and `text`.
- `train_whisper_lora.py` can auto-split train/eval if `--eval_manifest` is not supplied.
- `merge_peft.py` currently uses fixed defaults for `BASE_MODEL`, `LORA_DIR`, and `MERGED_DIR`; it is a good refactor target.
- `test_run_ct2.py` is a manual check script and currently assumes a GPU-capable environment and sample audio path.
