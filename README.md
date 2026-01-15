
# Lora Trainer for Whisper (for MSI DGX Sparks)




## prepare environment

```bash
sudo apt update
sudo apt install -y pkg-config cmake build-essential \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
    libavfilter-dev libavdevice-dev

```


## prepare virtual environment

please 
```bash
sh setup.sh
```


or..  

```bash
uv venv
uv pip install cmake ninja pybind11 setuptools wheel

# 3. 환경변수 자동 설정 (여기가 제일 중요합니다!)
# 3-1. CUDA 컴파일러 위치 지정
export CUDACXX=/usr/local/cuda/bin/nvcc
export PATH=/usr/local/cuda/bin:$PATH

# 3-2. pybind11 설치 경로를 파이썬에게 물어봐서 등록
# (uv run을 통해 가상환경 내부의 python을 호출)
PYBIND_PATH=$(uv run python -c "import pybind11; print(pybind11.get_cmake_dir())")
export CMAKE_PREFIX_PATH=$PYBIND_PATH:$CMAKE_PREFIX_PATH

echo "Found pybind11 at: $PYBIND_PATH"
echo "Found nvcc at: $CUDACXX"

# 4. 최종 동기화 (설치 시작)
# --no-build-isolation: 위에서 설치한 도구와 환경변수를 그대로 쓰라는 옵션
# UV_HTTP_TIMEOUT=600: 타임아웃 방지 (10분)
echo "Installing project dependencies..."
UV_HTTP_TIMEOUT=600 \
I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 \
uv sync --no-build-isolation

echo "✅ 모든 설치가 완료되었습니다!"
```

