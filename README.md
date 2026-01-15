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



## make manifest

이 과정은 학습하기위한 데이터셋의 메타정보를 담은 manifest 파일을 생성합니다.


```bash
python make_manifest.py --root ./datasets/Sample --wav_dir wav --label_dir lb
```

## train
```bash

torchrun --nproc_per_node=2 train_whisper_lora.py   --model_name "openai/whisper-large-v3"   --manifest "/home/agent01/works/dataset/71557/data/Training/manifest.jsonl"   --eval_manifest "/home/agent01/works/dataset/71557/data/Validation/manifest.jsonl"   --output_dir "outputs/large_v3_ddp"   --batch_size 32 --grad_accum 4 --fp16 --lr 1e-4   --use_gradient_checkpointing --max_audio_sec 30.0   --eval_steps 300  --max_steps 3000

torchrun --nproc_per_node=2 train_whisper_lora.py   --model_name "openai/whisper-large-v3"   --manifest "/home/agent01/works/dataset/71557/data/Training/manifest.jsonl"   --eval_manifest "/home/agent01/works/dataset/71557/data/Validation/manifest.jsonl"   --output_dir "outputs/large_v3_ddp_retry_2"   --batch_size 32   --grad_accum 2   --fp16   --lr 1e-5   --use_gradient_checkpointing   --max_audio_sec 30.0   --eval_steps 50   --max_steps 500

CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 train_whisper_lora.py   --model_name openai/whisper-large-v3   --manifest datasets/Sample/manifest.jsonl   --output_dir outputs/largev3_lora   --max_steps 300   --batch_size 16   --grad_accum 16   --fp16   --max_audio_sec 20   --use_gradient_checkpointing   --dataloader_workers 0


 # 단일 GPU
python train_whisper_lora.py --model_name "openai/whisper-small" --manifest "datasets/Sample/manifest.jsonl" --output_dir "outputs/small_lora" --batch_size 16 --grad_accum 2 --fp16 --lr 1e-4 --max_steps 300 --eval_steps 100
python train_whisper_lora.py   --model_name "openai/whisper-small"   --manifest "datasets/Sample/manifest.jsonl"   --output_dir "outputs/small_lora"   --batch_size 16 --grad_accum 2 --fp16 --lr 1e-4   --eval_ratio 0.01 --max_steps 10 

# GPU 전력 제한 (옵션)
sudo nvidia-smi -i 1 -pl 280

```


## evaluate
```bash

python eval_dataset_lora.py --manifest datasets/Sample/manifest.jsonl --base_model openai/whisper-small --lora_dir outputs/small_lora --output_csv comparison_results.csv
python eval_dataset_lora.py --manifest /home/agent01/works/dataset/71557/data/Validation/manifest.jsonl  --base_model openai/whisper-large-v3 --lora_dir outputs/large_v3_ddp --output_csv outputs/comparison_results.csv --max_samples 200

```

## merge lora weights to base model

```bash
 python merge_peft.py
```



## convert to ct2

추론 전용 모델로 변환 합니다.  

**--quantization int8_float16: 가중치는 8비트로 줄이고 연산은 16비트로 하여 속도와 정확도를 모두 잡습니다.**   

```bash
ct2-transformers-converter --model outputs/merged_small --output_dir outputs/ct2_small --quantization int8_float16 --force

```

