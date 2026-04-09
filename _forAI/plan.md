# Plan

## 현재 목표

- 현재 `dalus_server` Whisper 브랜치가 이 저장소를 안정적으로 호출할 수 있도록, 이미 추가된 CLI/산출물 계약을 유지하면서 문서와 실제 소스의 차이를 줄인다.
- 학습/평가/추론/병합/CT2 변환의 실제 로직은 이 저장소가 계속 책임지고, 웹 UI는 이 저장소의 표준 CLI와 산출물 규약만 믿고 동작하게 만든다.

## 이번 계획에서 고정할 것

- 웹 UI를 이 저장소 안에 만들지 않는다.
- 기존 학습 로직의 중심은 `train_whisper_lora.py`에 둔다.
- TensorBoard 로그 위치는 현재처럼 `output_dir/runs`를 기본으로 유지한다.
- `dalus_server`는 이 저장소를 직접 import 하기보다 CLI 프로세스로 호출하는 것을 1차 기준으로 둔다.

## 현재 기준선

- `dalus_server`가 아래 작업을 CLI로 시작할 수 있다.
  - manifest 생성
  - train 시작
  - eval 시작
  - 단일 파일 infer
  - LoRA merge
  - CT2 export
- 각 작업은 성공/실패를 명확한 종료 코드로 반환한다.
- 각 run/output 디렉토리에는 웹 UI가 읽을 수 있는 메타 파일이 남는다.
- 경로 규약이 정해져서 절대 경로 예시를 UI가 그대로 복사하지 않아도 된다.
- 문서는 `uv` 기반 실행 방식과 실제 기본값을 기준으로 유지한다.

## 구현 원칙

- 기존 스크립트를 전면 재작성하지 않고, 먼저 "얇은 공통 진입점"을 추가한다.
- 스크립트별 argparse 이름은 가능한 범위에서 맞추되, 급하면 wrapper에서 흡수한다.
- 사람이 읽는 stdout 로그와 UI가 읽는 JSON 메타데이터를 분리한다.
- 실패 시 traceback만 남기지 말고, 요약 에러 메시지를 메타 파일에도 남긴다.

## 1단계. 공통 실행 진입점 만들기

### 대상 파일

- `train_whisper_lora.py`
- `eval_dataset_lora.py`
- `infer_lora.py`
- `merge_peft.py`
- `eval_dataset_ct2.py`
- `test_run_ct2.py`
- `make_manifest.py`

### 현재 공통 진입점

- `runner_cli.py`
  - 웹 UI가 직접 호출하는 공통 진입점
  - 서브커맨드: `manifest`, `train`, `eval`, `infer`, `merge`, `ct2-export`

### 이 단계에서 정할 CLI 규약

- `manifest`
  - 입력: `--root`, `--wav-dir`, `--label-dir`, `--output`
- `train`
  - 입력: `--model-name`, `--manifest`, `--eval-manifest`, `--eval-ratio`, `--output-dir`
  - 입력: `--batch-size`, `--grad-accum`, `--lr`, `--max-steps`, `--fp16`
  - 입력: `--language`, `--task`, `--max-audio-sec`
  - 입력: `--lora-r`, `--lora-alpha`, `--lora-dropout`, `--target-modules`
- `eval`
  - 입력: `--manifest`, `--base-model`, `--lora-dir`, `--output-csv`, `--max-samples`
- `infer`
  - 입력: `--wav`, `--base-model`, `--lora-dir`, `--language`, `--task`
- `merge`
  - 입력: `--base-model`, `--lora-dir`, `--merged-dir`
- `ct2-export`
  - 입력: `--model-dir`, `--output-dir`, `--quantization`

## 2단계. run 산출물 규약 고정

### 학습 run 디렉토리 기준

- `output_dir/adapter_config.json` 등 PEFT 산출물은 그대로 유지
- `output_dir/runs/`
  - TensorBoard 로그
- `output_dir/run_config.json`
  - 실행 당시 CLI 인자 snapshot
- `output_dir/run_status.json`
  - 현재 상태
  - 필드 후보: `state`, `started_at`, `finished_at`, `pid`, `command`, `error`
- `output_dir/train_summary.json`
  - 마지막 step, best checkpoint, 저장된 adapter 위치, 실행 시간 요약

### 평가 산출물 기준

- `output_dir/eval_results.csv` 또는 사용자가 지정한 CSV
- `output_dir/eval_summary.json`
  - 필드 후보: `manifest`, `base_model`, `lora_dir`, `sample_count`, `avg_cer_base`, `avg_cer_lora`

### 추론/변환 산출물 기준

- `output_dir/infer_result.json`
  - 입력 wav, 모델, adapter, 추론 결과 텍스트
- `merged_dir/merge_summary.json`
- `ct2_output_dir/ct2_export_summary.json`

## 3단계. 기존 스크립트 정리 포인트

### 꼭 손볼 파일

- `merge_peft.py`
  - 현재 하드코딩된 `BASE_MODEL`, `LORA_DIR`, `MERGED_DIR` 제거
- `test_run_ct2.py`
  - 현재 하드코딩된 `model_path`, `audio_path` 제거
- `eval_dataset_lora.py`
  - 요약 JSON 출력 추가
- `infer_lora.py`
  - 결과 JSON 출력 옵션 추가
- `train_whisper_lora.py`
  - 시작/완료 시 메타 파일 작성
  - 예외 발생 시 실패 상태 기록

### 가능하면 같이 손볼 파일

- `make_manifest.py`
  - dry-run 또는 validate 모드 추가 검토
- `compare_infer.py`
  - 운영 UI 직접 연동보다는 개발용 도구로 위치를 분리
- `main.py`
  - placeholder 유지 여부 결정

## 4단계. `dalus_server`가 믿을 계약

### 호출 계약

- 모든 작업은 기본적으로 `uv run python runner_cli.py <subcommand> ...` 형식으로 실행한다.
- 성공 시 exit code `0`
- 실패 시 exit code `!= 0`
- 실패 시 stderr 요약 + summary/status JSON 동시 기록

### 상태 계약

- 장기 실행 작업은 상태 JSON을 갱신한다.
- 학습은 `run_status.json`, 평가는 `eval_status.json`을 사용한다.
- 최소 필드
  - `state`: `idle | running | completed | failed | stopped`
  - `message`
  - `updated_at`
  - `output_dir`
- 가능하면 추가
  - `current_step`
  - `total_steps`
  - `latest_checkpoint`
  - `tensorboard_log_dir`

### 경로 계약

- 상대 경로는 repo root 기준이 아니라, 호출자가 넘긴 경로를 그대로 해석한다.
- summary JSON 안에는 가능하면 절대 경로와 사용자 입력 원본을 모두 남긴다.
- README 예시는 개발 참고용이고, UI 연동은 README 예시 문자열에 의존하지 않는다.

## 5단계. 검증 순서

### 코드 수준

- argparse smoke test
- 메타 파일 생성 테스트
- 실패 시 non-zero 종료 테스트

### 샘플 데이터 기준

- `datasets/Sample`로 manifest 생성
- 짧은 소규모 train
- 생성된 adapter로 eval
- single-file infer
- merge
- CT2 export

### UI 연동 직전 체크

- 같은 인자를 수동 CLI로 실행했을 때와 `dalus_server`에서 실행했을 때 산출물 경로가 동일해야 한다.
- `dalus_server`는 stdout 파싱에 의존하지 않고 JSON 메타 파일만으로 상태를 읽을 수 있어야 한다.

## 지금 브랜치에서 바로 할 일

1. `README.md`와 `_forAI` 문서를 실제 소스 기본값과 `uv` 실행 기준에 맞춰 유지
2. `dalus_server`가 읽는 상태/요약 파일 계약이 바뀌면 문서를 같은 턴에 함께 갱신
3. 샘플 데이터 기준 최소 smoke run 절차를 계속 검증 가능한 형태로 정리

## 리스크

- 현재 스크립트는 사람 실행 기준이라, 장기 작업 상태 갱신과 실패 복구 정보가 부족하다.
- GPU/CUDA/torchcodec 환경차 때문에 CLI 계약보다 환경 문제에서 먼저 깨질 수 있다.
- 산출물 규약을 먼저 고정하지 않으면 `dalus_server` 쪽 구현을 해도 다시 뜯어고칠 가능성이 높다.
