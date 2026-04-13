# 데이터셋 준비 원칙

본 프로젝트(`lora_trainer_whisper`)에 데이터를 추가하거나 정리할 때 따를 기본 규칙입니다. 특정 디스크/경로 대신 데이터의 형태 처리에 집중합니다.

## 1. 표준 디렉터리 레이아웃
가장 권장하는 1:1 파싱 구조입니다.
```text
<root>/
├── wav/**/*.wav
└── lb/**/*.json  (필수키: {"script": {"text": "..."}})
```
이 틀에 맞추면 `runner_cli.py manifest`를 바로 쓸 수 있습니다.
- 폴더명이 다르면 `--wav-dir`, `--label-dir`로 맵핑합니다.
- Training/Validation 분리가 안 되어 있다면, 학습 단계에서 `--eval-ratio 0.05` 옵션으로 런타임에 동적 분리합니다.

## 2. 비표준 데이터 핸들링
- **통합 라벨 / 메타데이터 파일 (`.txt`, `.csv` 등)**
  개별 JSON 파일이 아니라도, 텍스트 정규화(예: `(A)/(B)` 기호 처리 등)를 거친 뒤 1:1 표준 JSON 포맷(`lb/`)으로 직접 변환하는 사전 스크립트를 작성하여 해결합니다.
- **초대용량 압축 데이터**
  작업 전 타겟 디스크의 여유 공간(TB급)을 반드시 확인한 후, 여유 공간이 충분한 스토리지에 압축을 풀어 `wav/`, `lb/` 하위로 병합합니다.

## 3. 오디오 경로 참조 (의사-절대경로)
생성된 `manifest.jsonl`의 `audio` 필드는 완전한 절대경로가 아닌 **의사-절대경로**(예: `/Training/wav/...`) 형태를 띱니다.
- 목적: 디스크 마운트 위치나 폴더 깊이가 변경되어도 manifest 파일을 재사용하기 위함.
- 방식: `manifest_utils.resolve_audio_path()`가 `manifest.jsonl` 파일 위치 기준으로 상위 디렉터리 구조를 역추적하여 실제 절대경로를 동적 해석합니다.

## 4. 필수 검증 (Validate)
매니페스트 생성이 끝난 후, 학습 투입 직전에 아래 검증 단계를 거쳐야 합니다. (혹은 독립 도구인 `whisper-dataset validate` 사용)

```bash
# 기본 파싱 라벨 검사
uv run python runner_cli.py validate <root>

# 정밀 검사 (WAV 헤더/오류 읽기)
uv run python runner_cli.py validate <root> --check-audio

# 오류 자동 제거 및 manifest 덮어쓰기
uv run python runner_cli.py validate <root> --check-audio --clean
```
