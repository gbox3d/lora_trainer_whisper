import argparse
from pathlib import Path

from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="openai/whisper-large-v3")
    p.add_argument("--lora-dir", default="outputs/small_lora")
    p.add_argument("--merged-dir", default="outputs/merged_small")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print(f"⏳ Loading Base Model: {args.base_model}")
    base_model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    processor = WhisperProcessor.from_pretrained(args.base_model)

    print(f"⏳ Loading LoRA Adapter: {args.lora_dir}")
    model = PeftModel.from_pretrained(base_model, args.lora_dir)

    print("⚡ Merging LoRA into Base Model...")
    model = model.merge_and_unload()

    merged_dir = Path(args.merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving merged model to: {merged_dir}")
    model.save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)

    print("✅ Merge Complete!")
    return {
        "base_model": args.base_model,
        "lora_dir": str(Path(args.lora_dir).resolve()),
        "merged_dir": str(merged_dir.resolve()),
    }


if __name__ == "__main__":
    main()
