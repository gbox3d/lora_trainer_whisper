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

    p.add_argument("--fp16", action="store_true")
    p.add_argument("--max_audio_sec", type=float, default=20.0)
    p.add_argument("--use_gradient_checkpointing", action="store_true")
    p.add_argument("--dataloader_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")

    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str, default="q_proj,v_proj")

    p.add_argument("--load_in_8bit", action="store_true")

    p.add_argument(
        "--eval_manifest",
        type=str,
        default="",
        help="검증용 manifest.jsonl (없으면 train에서 자동 분리)",
    )
    p.add_argument("--eval_steps", type=int, default=300, help="몇 step마다 eval 할지")
    p.add_argument(
        "--eval_ratio",
        type=float,
        default=0.01,
        help="eval_manifest 없을 때 train에서 분리할 비율",
    )

    return p.parse_args()


def patch_whisper_forward_for_peft(whisper_model: WhisperForConditionalGeneration):
    orig_forward = whisper_model.forward
    sig = inspect.signature(orig_forward)
    allowed = set(sig.parameters.keys())
    has_varkw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    def patched_forward(*args, **kwargs):
        kwargs.pop("input_ids", None)
        kwargs.pop("inputs_embeds", None)

        if "input_features" not in kwargs and "input_ids" in kwargs:
            kwargs["input_features"] = kwargs.pop("input_ids")

        if not has_varkw:
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}

        return orig_forward(*args, **kwargs)

    whisper_model.forward = patched_forward


@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: WhisperProcessor
    max_audio_sec: float = 20.0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_list = []
        for f in features:
            a = f["audio"]["array"]
            if isinstance(a, np.ndarray) and a.ndim == 2:
                a = a.mean(axis=1)
            a = np.asarray(a, dtype=np.float32)

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

        batch = {
            "input_features": feats["input_features"],
            "labels": labels,
        }
        if "attention_mask" in feats:
            batch["attention_mask"] = feats["attention_mask"]

        return batch


class WhisperTrainer(Trainer):
    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs = super()._prepare_inputs(inputs)
        inputs.pop("input_ids", None)
        inputs.pop("inputs_embeds", None)
        return inputs


def _load_and_cast_manifest(path: str):
    ds = load_dataset("json", data_files=path, split="train")
    return ds.cast_column("audio", Audio(sampling_rate=16000))


def _split_train_eval(train_ds, eval_ratio: float, seed: int):
    if eval_ratio <= 0.0 or eval_ratio >= 1.0:
        raise ValueError("--eval_ratio 는 0과 1 사이여야 합니다 (예: 0.01).")

    n = len(train_ds)
    if n < 2:
        raise ValueError("train dataset 이 너무 작아서 eval 분리가 불가능합니다.")

    test_size = max(1, int(round(n * eval_ratio)))
    if n - test_size < 1:
        test_size = n - 1

    split = train_ds.train_test_split(test_size=test_size, seed=seed, shuffle=True)
    return split["train"], split["test"]


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds = _load_and_cast_manifest(args.manifest)

    if args.eval_manifest:
        eval_ds = _load_and_cast_manifest(args.eval_manifest)
    else:
        train_ds, eval_ds = _split_train_eval(train_ds, args.eval_ratio, args.seed)

    processor = WhisperProcessor.from_pretrained(
        args.model_name,
        language=args.language,
        task=args.task,
    )

    model_kwargs = {}
    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = "auto"

    model = WhisperForConditionalGeneration.from_pretrained(args.model_name, **model_kwargs)

    model.config.use_cache = False

    if args.use_gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language, task=args.task
    )

    patch_whisper_forward_for_peft(model)

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
        save_steps=args.eval_steps,

        report_to=["tensorboard"],
        logging_dir=os.path.join(args.output_dir, "runs"),

        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        eval_strategy="steps",
        eval_steps=args.eval_steps,

        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=bool(args.pin_memory),
    )

    trainer = WhisperTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorSpeechSeq2Seq(
            processor, max_audio_sec=args.max_audio_sec
        ),
    )

    trainer.train()

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
