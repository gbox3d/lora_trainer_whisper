"""Thin wrapper — delegates to src/lora_trainer/train_whisper_lora.py."""
from lora_trainer.train_whisper_lora import main

if __name__ == "__main__":
    try:
        main()
    finally:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
