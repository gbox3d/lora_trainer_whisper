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

- `make_manifest.py` now defaults to relative `audio` paths and exposes `--audio-path-mode` plus `--workers` for large manifest rebuilds on slower storage.
- `train_whisper_lora.py` can auto-split train/eval if `--eval_manifest` is not supplied.
- `train_whisper_lora.py`, `eval_dataset_lora.py`, and `eval_dataset_ct2.py` resolve manifest-relative audio paths before reading files.
- Current `datasets` releases may return `torchcodec` audio decoders instead of the older `{"array": ...}` dict shape; the collator now has compatibility handling for both.
- `merge_peft.py` currently uses fixed defaults for `BASE_MODEL`, `LORA_DIR`, and `MERGED_DIR`; it is a good refactor target.
- `test_run_ct2.py` is a manual check script and currently assumes a GPU-capable environment and sample audio path.
