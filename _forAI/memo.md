# Memo

## Version management

- Use `major.minor.patch` versioning.
- Current project version in `pyproject.toml` is `0.3.1`.
- Increase `major` for large structural changes or compatibility-breaking changes.
- Increase `minor` for new features or meaningful improvements.
- Increase `patch` for bug fixes or small changes.
- Record version-by-version changes in `dev_log.md`.

## Open questions

- Should helper scripts be promoted into a small package/CLI, or is a flat script layout still the right tradeoff?
- Which training/evaluation path should be treated as the official baseline for future comparisons: LoRA-on-base, merged model, or CT2 export?

## Decision criteria

- Prefer reproducible CLI workflows over notebook-style or one-off command edits.
- Prefer removing hard-coded machine paths before adding new experiment helpers.
- Prefer manifest-relative audio paths over machine-specific absolute paths, especially when datasets live behind symlinks, external disks, or network mounts.
- Prefer changes that keep DGX/ARM64 environment setup explicit, since build issues are a recurring operational risk.
- Favor lightweight smoke tests that verify wiring without requiring long GPU training jobs.

## README / memo 분리 원칙

- `README.md`는 실제 학습을 돌리는 사용자가 빠르게 판단해야 하는 내용 위주로 유지한다.
- 머신/런타임 의존적인 worker 병목, benchmark 수치, 파이프라인 내부 구현 메모는 `memo.md`에 둔다.
- 세부 실험 이력과 날짜별 변경 기록은 `dev_log.md`에 남기고, README에는 decision-complete 사용 가이드만 남긴다.

## Short notes

- `_forAI/`는 repo-tracked collaboration workspace로 운영한다. Codex가 범위/검토를 맡고 Claude는 `_forAI` 문서를 읽고 `plan.md`, `memo.md`, `dev_log.md`를 갱신하면서 구현한다.
- Claude는 구현 전에 `_forAI/README.md` -> `inventory.md` -> `memo.md` -> `data_prep.md` -> `dev_log.md` -> `plan.md` 순서로 읽는다.
- `plan.md`의 active task는 Claude에게 넘기는 decision-complete spec이다. 각 task는 goal, success criteria, in scope/out of scope, expected files or subsystems, verification, risks를 포함해야 한다.
- `inventory.md`는 factual document다. 저장소 구조, 엔트리포인트, 운영 사실이 실제로 바뀌었을 때만 수정한다.
- `_forAI` 내용과 코드/로그가 충돌하면 코드와 관측 결과를 우선한다. 문서 correction은 `dev_log.md`에 명시적으로 남긴다.
- 성능 작업은 `dev_log.md`에 baseline과 comparison을 함께 남긴다.
- 재사용 output directory는 실험 로그 해석을 꼬이게 할 수 있다. `runner_cli.py`의 `train.log`가 append 모드이므로 성능 비교나 hang 분석에서는 새 output directory를 쓴다.
- `manifest_utils.resolve_audio_path()`는 이제 manifest-relative 경로뿐 아니라, 실제로는 존재하지 않는 pseudo-absolute 경로도 manifest 상위 디렉터리를 기준으로 복구 시도한다.
- `runner_cli.py train` and `train_whisper_lora.py` now support `--num-epochs`; `0` means keep the existing `max_steps`-driven behavior.
- `runner_cli.py`는 train 시작 전에 기존 `train_summary.json`을 지워서 UI가 이전 run 요약을 잘못 보여주지 않게 한다.
- `runner_cli.py`는 `train.log` 시작부에 실제 사용한 training parameters JSON 블록을 기록하고, 진행바 carriage return 출력을 사람이 읽기 쉬운 줄바꿈으로 정규화한다.
- `train_whisper_lora.py` manifest audio path remap은 `load_from_cache_file=False`로 강제 재계산한다.
- `make_manifest.py` now defaults to relative `audio` paths and exposes `--audio-path-mode` plus `--workers` for large manifest rebuilds on slower storage.
- `train_whisper_lora.py` can auto-split train/eval if `--eval_manifest` is not supplied.
- `train_whisper_lora.py`, `eval_dataset_lora.py`, and `eval_dataset_ct2.py` resolve manifest-relative audio paths before reading files.
- Current `datasets` releases may return `torchcodec` audio decoders instead of the older `{"array": ...}` dict shape; the collator now has compatibility handling for both.
- `merge_peft.py` and `test_run_ct2.py` are now argparse-based, but they still keep convenience defaults and are best treated as operator tools rather than a stable public API.
- Repository documentation should prefer `uv run ...` examples even when plain `python ...` would work inside an activated environment.

## 데이터 파이프라인 선택 기준

- `--data-pipeline legacy` (기본): 기존 `datasets.Audio` 경로. 현재도 가장 보수적인 baseline은 `legacy + dataloader_workers=0`이다.
- `--data-pipeline cached`: `datasets.Audio` 미사용. 오디오 디코딩 + feature 추출을 worker process 내에서 지연 초기화하고 디스크 캐시에 저장한다. 멀티워커 실전 경로는 이쪽으로 본다.
- `cached` 모드에서 `dataloader_workers > 0`이면 `dataloader_persistent_workers=True`가 자동 적용된다.
- `cached` 모드의 prefetch는 `workers=1`이면 `2`, `workers>=2`이면 `1`로 제한한다. cold-cache 시작 시 과도한 선행 적재를 줄이기 위한 선택이다.
- `cached` 멀티워커는 `spawn` context로 DataLoader worker를 띄워 CUDA/Accelerate 초기화 후 `fork`로 인한 정체를 피한다.
- resolved-manifest 캐시 키: manifest 절대경로 + size + mtime. `resolve_audio_path()` 결과를 `resolved_{key}.pkl`에 저장하여 대용량 manifest 재기동 비용을 없앤다. manifest 내용이 바뀌면 자동으로 무효화된다.
- feature 캐시 키: 오디오 절대경로 + file size + mtime + model_name + max_audio_sec. 파일이 바뀌면 자동으로 재계산된다.
- 캐시 기본 위치: `<manifest_dir>/.lora_trainer_cache`. `--feature-cache-dir`로 변경 가능.
- `data_pipeline.py`는 이제 파일 전체를 먼저 읽지 않고 `max_audio_sec`에 필요한 길이만 읽는다. cold-start I/O 압력을 줄이기 위한 수정이다.
- feature cache 저장은 atomic write로 바꿨다. 멀티워커가 같은 샘플을 건드릴 때 partial `.npy`를 읽는 경쟁을 피하려는 목적이다.
- 대용량 데이터셋에서 전체 사전 계산은 이 파이프라인의 설계 범위 밖이다; 학습 중 on-demand 계산 + 캐시 축적 방식으로 운영한다.
- cached 모드 cold-start: 327샘플 기준 steps 1-10에 ~10.6s 오버헤드 발생 (feature cache 쓰기). 두 번째 실행부터 warm. 대용량 데이터셋은 첫 run이 느리다.
- GB10(CUDA 12.1) + whisper-small 기준 workers=1과 workers=2는 성능 차이 없음.
- 2026-04-10 real dataset + large-v3 direct tests:
  - `cached + workers=0/1/2 + batch=1 + max_steps=1` 통과
  - `cached + workers=0/1 + batch=4 + max_steps=1` 통과
  - historical failure before the spawn fix: `cached + workers=2 + batch=4 + max_steps=1`가 `0/1`에서 정체했고 GPU 0%, 새 `feat_*.npy` 생성 없음
  - after prefix-read + atomic cache write + spawn-worker fix: `cached + workers=4 + batch=4 + max_steps=1` 통과 (`/home/miso/datasets/outputs/smoke_largev3_cached_w4_bs4_spawnfix`, `train_runtime≈22.2s`)
- 현재 실전 가이드는 다음과 같다:
  - 멀티워커가 필요하면 `cached`를 우선 고려한다.
  - `legacy + workers>0`는 작은 샘플 로더 테스트는 통과할 수 있어도 real full run에서는 다시 `0/step` 정체가 날 수 있다.
  - 가장 보수적인 baseline은 `legacy + workers=0`, 멀티워커 baseline은 검증된 `cached` profile이다.
- `large-v3_lora` 출력 디렉터리를 재사용하면서 old `worker=2` 로그와 new `worker=0` 로그가 섞여 판단이 흔들렸다. real-run diagnosis에서는 output dir 재사용 금지 규칙을 더 엄격하게 적용한다.
- datasets/Sample/manifest.jsonl은 manifest 디렉터리 기준 상대경로 (`wav/...`)로 재생성됨. resolve_audio_path()와 호환 확인.

## Dalus UI 설정 스냅샷

- Dalus Training UI에서 worker 관련 핵심 옵션은 `Advanced > Data pipeline` 섹션에 있다.
- UI 레이블:
  - `Pipeline mode`
  - `Feature cache dir` (`cached` 선택 시에만 보임)
  - `Workers`
  - `Grad checkpointing`
- 기본값은 여전히 `data_pipeline=legacy`, `feature_cache_dir=""`다. UI에서 따로 바꾸지 않으면 cached 경로는 사용되지 않는다.
- Training profile을 불러오면 profile의 `params`가 현재 UI state에 그대로 merge된다. 따라서 profile에 `data_pipeline: legacy`가 들어 있으면, 사용자가 직전에 UI에서 `cached`로 바꿔놨더라도 다시 `legacy`로 덮일 수 있다.
- 다음 컨텍스트에서 Dalus로 worker 문제를 재현할 때는 profile 로드 직후 아래 값을 다시 확인한다:
  - `Pipeline mode = cached`
  - `Feature cache dir = /home/miso/datasets/cache/whisper_features_130`
  - `Workers = 1`, `2`, 또는 검증된 `4`
  - `Grad checkpointing = Enabled`
  - `Output dir = fresh directory`
- 2026-04-10 사용자 profile `parm_1`의 실제 내용:
  - `batch_size=4`
  - `grad_accum=8`
  - `use_gradient_checkpointing=true`
  - `dataloader_workers=0`
  - `data_pipeline=legacy`
  - `feature_cache_dir=""`
  - 즉 `parm_1`은 worker 디버그용 cached profile이 아니라, 안정성 우선의 legacy profile이다.
- 2026-04-10 builtin multi-worker example profile:
  - `dalus_server/training_profiles/builtin/130-large-v3-cached-w4.yaml`
  - `batch_size=4`
  - `grad_accum=8`
  - `dataloader_workers=4`
  - `data_pipeline=cached`
  - `feature_cache_dir=""`

## Doc audit: source vs docs

- `README.md` had several examples still using `python ...`; source and 운영 기준은 `uv` 중심이므로 `uv run ...` 쪽이 더 정확하다.
- `README.md` 일부 예시와 옵션 표는 예전 기본값인 `openai/whisper-small`을 가리켰지만, 실제 코드 기본값은 현재 `openai/whisper-large-v3`다.
- 상태 파일 설명은 대체로 맞았지만, 실제 소스에서는 학습과 평가가 서로 다른 파일명(`run_status.json`, `eval_status.json`)을 쓴다.
- `infer`는 항상 결과 파일을 남기지 않고, `runner_cli.py infer --result-json`을 넘겼을 때만 JSON을 기록한다.
