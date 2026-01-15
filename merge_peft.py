# merge_peft.py
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

# 경로 설정
BASE_MODEL = "openai/whisper-small"
LORA_DIR = "outputs/small_lora"
MERGED_DIR = "outputs/merged_small"

print(f"⏳ Loading Base Model: {BASE_MODEL}")
base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
processor = WhisperProcessor.from_pretrained(BASE_MODEL)

print(f"⏳ Loading LoRA Adapter: {LORA_DIR}")
model = PeftModel.from_pretrained(base_model, LORA_DIR)

print("⚡ Merging LoRA into Base Model...")
# 핵심: LoRA 가중치를 베이스 모델에 영구적으로 합침
model = model.merge_and_unload()

print(f"💾 Saving merged model to: {MERGED_DIR}")
model.save_pretrained(MERGED_DIR)
processor.save_pretrained(MERGED_DIR)

print("✅ Merge Complete!")