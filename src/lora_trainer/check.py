"""
date : 2026-01-15
filename: check.py
author: gbox3d
description: check workspace setup status and system environment ,cuda version etc.
please dont's edit this comment block
"""

import os
import subprocess
import sys
import torch
import torchcodec
import psutil


# system info
print("python:", sys.version)
print(f'Setup Complete! Torch: {torch.__version__}')
print("torchcodec:", torchcodec.__version__)

#cuda and gpu info
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
    

def bytes_to_gb(bytes_value):
    return f"{bytes_value / (1024 ** 3):.2f} GB"

print(f"--- System & GPU Memory Status (Python {sys.version.split()[0]}) ---")

# 1. System Memory (CPU RAM)
# Grace CPU는 LPDDR5X를 사용하여 매우 방대한 시스템 메모리를 가질 수 있습니다.
vm = psutil.virtual_memory()
print(f"\n[System Memory (RAM)]")
print(f"Total: {bytes_to_gb(vm.total)}")
print(f"Available: {bytes_to_gb(vm.available)}")
print(f"Used: {bytes_to_gb(vm.used)} ({vm.percent}%)")

# 2. GPU Memory (VRAM)
# nvidia-smi에서 안 보이더라도 PyTorch가 인식하는 메모리가 '진짜'입니다.
if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_id)
    
    # mem_get_info() returns (free, total)
    free_mem, total_mem = torch.cuda.mem_get_info(device_id)
    
    # 현재 PyTorch 텐서가 실제로 점유 중인 메모리
    allocated = torch.cuda.memory_allocated(device_id)
    reserved = torch.cuda.memory_reserved(device_id)

    print(f"\n[GPU Memory ({device_name})]")
    print(f"Total Capacity:   {bytes_to_gb(total_mem)}")
    print(f"Free (Available): {bytes_to_gb(free_mem)}")
    print(f"Allocated (Used): {bytes_to_gb(allocated)}")
    print(f"Reserved (Cache): {bytes_to_gb(reserved)}")
    
    # GB10 Unified Memory 특성 코멘트
    if total_mem > 100 * (1024**3): # 100GB 이상이면
        print("\n* Note: 100GB 이상의 VRAM이 잡힌다면 Grace-Blackwell의")
        print("        Unified Memory(CPU/GPU 메모리 공유)가 정상 작동 중인 것입니다.")
else:
    print("\n[GPU] CUDA device not found.")
    

# check.py 하단 추가 권장
print("\n--- Tensor Computation Test ---")
try:
    # 텐서를 만들어서 GPU로 보냄
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    y = torch.tensor([4.0, 5.0, 6.0]).cuda()
    
    # 연산 수행 (곱셈)
    z = x * y
    
    print(f"Success! Calculation result: {z}")
    print(f"Device used: {z.device}")
except Exception as e:
    print(f"Error during computation: {e}")