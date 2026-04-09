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
