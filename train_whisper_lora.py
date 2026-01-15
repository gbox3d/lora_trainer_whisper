# train_whisper_lora.py
import os
import argparse
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType


# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_name", type=str, default="openai/whisper-small")
    p.add_argument("--manifest", type=str, default="datasets/Sample/manifest.jsonl")
    p.add_argument("--output_dir", type=str, default="outputs/lora")

    p.add_argument("--language", type=str, default="ko")
    p.add_argument("--task", type=str, default="transcribe")

    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)

    # 메모리/안정성 옵션
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--max_audio_sec", type=float, default=20.0)
    p.add_argument("--use_gradient_checkpointing", action="store_true")
    p.add_argument("--dataloader_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")

    # LoRA
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str, default="q_proj,v_proj")  # 콤마 구분

    # (옵션) 8bit 로딩 (bitsandbytes 필요)
    p.add_argument("--load_in_8bit", action="store_true")
    
    p.add_argument("--eval_manifest", type=str, default="", help="검증용 manifest.jsonl (없으면 train에서 자동 분리)")
    p.add_argument("--eval_steps", type=int, default=300, help="몇 step마다 eval 할지")
    p.add_argument("--eval_ratio", type=float, default=0.01, help="eval_manifest 없을 때 train에서 분리할 비율")


    return p.parse_args()


# -------------------------
# 핵심 해결: Whisper forward 호환 패치
# -------------------------
def patch_whisper_forward_for_peft(whisper_model: WhisperForConditionalGeneration):
    """
    PEFT가 base_model 호출 시 input_ids / inputs_embeds 등을 키워드로 넣어도
    Whisper가 죽지 않도록 forward를 패치합니다.

    - input_ids / inputs_embeds: 제거 (Whisper는 input_features를 사용)
    - signature 기반으로 Whisper forward가 실제로 받는 키만 전달
    - 모듈 구조/이름은 건드리지 않음 (LoRA 저장/로드 정상)
    """
    orig_forward = whisper_model.forward  # bound method
    sig = inspect.signature(orig_forward)
    allowed = set(sig.parameters.keys())
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    def patched_forward(*args, **kwargs):
        # PEFT가 항상 넘기는 "Whisper가 싫어하는 키" 제거
        kwargs.pop("input_ids", None)
        kwargs.pop("inputs_embeds", None)

        # 혹시라도 누군가 input_ids에 멜특징을 넣는 이상한 경우 대비(안전장치)
        if "input_features" not in kwargs and "input_ids" in kwargs:
            kwargs["input_features"] = kwargs.pop("input_ids")

        # Whisper forward가 실제로 받는 키만 남기기
        if not has_varkw:
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}

        return orig_forward(*args, **kwargs)

    whisper_model.forward = patched_forward


# -------------------------
# Data Collator
# -------------------------
@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: WhisperProcessor
    max_audio_sec: float = 20.0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_list = []
        for f in features:
            a = f["audio"]["array"]  # datasets.Audio decode 결과
            if isinstance(a, np.ndarray) and a.ndim == 2:  # stereo -> mono
                a = a.mean(axis=1)
            a = np.asarray(a, dtype=np.float32)

            # 길이 제한 (OOM 방지)
            sr = 16000
            max_len = int(self.max_audio_sec * sr)
            if len(a) > max_len:
                a = a[:max_len]

            audio_list.append(a)

        feats = self.processor.feature_extractor(
            audio_list, sampling_rate=16000, return_tensors="pt"
        )

        labels = self.processor.tokenizer(
            [f["text"] for f in features],
            return_tensors="pt",
            padding=True,
        ).input_ids

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Whisper가 받는 키만 구성
        batch = {
            "input_features": feats["input_features"],
            "labels": labels,
        }
        if "attention_mask" in feats:
            batch["attention_mask"] = feats["attention_mask"]

        return batch


# -------------------------
# Trainer (기본 Trainer로 충분)
# -------------------------
class WhisperTrainer(Trainer):
    # 혹시 Trainer 쪽에서 이상한 키를 섞어도 방어
    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs = super()._prepare_inputs(inputs)
        # 혹시 생기면 제거 (하지만 이제 forward 패치로도 안전)
        inputs.pop("input_ids", None)
        inputs.pop("inputs_embeds", None)
        return inputs


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset
    # dataset = load_dataset("json", data_files=args.manifest, split="train")
    # dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Dataset (Train / Eval 분리 로드)
    train_ds = load_dataset("json", data_files=args.manifest, split="train")
    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))

    if not args.eval_manifest:
        raise ValueError("--eval_manifest 를 Validation manifest.jsonl 로 지정해주세요.")

    eval_ds = load_dataset("json", data_files=args.eval_manifest, split="train")
    eval_ds = eval_ds.cast_column("audio", Audio(sampling_rate=16000))

    processor = WhisperProcessor.from_pretrained(
        args.model_name,
        language=args.language,
        task=args.task,
    )

    # Model load
    model_kwargs = {}
    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = "auto"

    model = WhisperForConditionalGeneration.from_pretrained(args.model_name, **model_kwargs)

    # Whisper 학습 안정화 옵션
    model.config.use_cache = False
    # if args.use_gradient_checkpointing:
    #     model.gradient_checkpointing_enable()
    
    if args.use_gradient_checkpointing:
        model.config.use_cache = False  # 이미 하시지만, 여기서 확실히
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # (안전장치) PEFT + checkpointing에서 입력 grad 경고/이슈 예방
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # 한국어/전사 강제(원하면 유지)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language, task=args.task
    )

    # ✅ 핵심: PEFT 호환 forward 패치
    patch_whisper_forward_for_peft(model)

    # LoRA
    target_modules = [s.strip() for s in args.target_modules.split(",") if s.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        fp16=bool(args.fp16),
        ddp_find_unused_parameters=False,

        logging_steps=10,
        save_steps=args.eval_steps,  # ✅ 입력받은 평가 주기(500)와 동일하게 자동 설정

        # -------------------------------------------------------
        # [수정] Tensorboard 사용 및 안전장치
        # -------------------------------------------------------
        report_to=["tensorboard"],                         # ✅ 텐서보드 활성화
        logging_dir=os.path.join(args.output_dir, "runs"), # ✅ 로그 경로

        save_total_limit=5,             # ✅ 최대 5개 모델 보관
        load_best_model_at_end=True,    # ✅ 학습 끝날 때 최고 모델 자동 로드
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # -------------------------------------------------------

        eval_strategy="steps",
        eval_steps=args.eval_steps,
        
        # report_to="none",  # <--- 🗑️ 이 줄은 반드시 삭제하거나 주석 처리하세요!

        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=bool(args.pin_memory),
    )
    trainer = WhisperTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,     # ✅ 변경
        eval_dataset=eval_ds,       # ✅ 추가
        data_collator=DataCollatorSpeechSeq2Seq(processor, max_audio_sec=args.max_audio_sec),
    )

    trainer.train()

    # LoRA adapter 저장
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print(f"\n✅ Done. Saved LoRA adapter to: {args.output_dir}\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
