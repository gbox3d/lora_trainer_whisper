from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from workflow_utils import checkpoint_names, utc_now_iso, write_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Whisper LoRA workflow runner")
    sub = p.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest")
    manifest.add_argument("--root", required=True)
    manifest.add_argument("--wav-dir", default="wav")
    manifest.add_argument("--label-dir", default="lb")
    manifest.add_argument("--output", default="manifest.jsonl")
    manifest.add_argument("--audio-path-mode", choices=("relative", "absolute"), default="relative")
    manifest.add_argument("--workers", type=int, default=16)

    train = sub.add_parser("train")
    train.add_argument("--model-name", default="openai/whisper-large-v3")
    train.add_argument("--manifest", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--language", default="ko")
    train.add_argument("--task", default="transcribe")
    train.add_argument("--max-steps", type=int, default=300)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--grad-accum", type=int, default=16)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--fp16", action="store_true")
    train.add_argument("--max-audio-sec", type=float, default=20.0)
    train.add_argument("--use-gradient-checkpointing", action="store_true")
    train.add_argument("--dataloader-workers", type=int, default=0)
    train.add_argument("--pin-memory", action="store_true")
    train.add_argument("--lora-r", type=int, default=8)
    train.add_argument("--lora-alpha", type=int, default=16)
    train.add_argument("--lora-dropout", type=float, default=0.05)
    train.add_argument("--target-modules", default="q_proj,v_proj")
    train.add_argument("--load-in-8bit", action="store_true")
    train.add_argument("--eval-manifest", default="")
    train.add_argument("--eval-steps", type=int, default=300)
    train.add_argument("--eval-ratio", type=float, default=0.01)

    eval_cmd = sub.add_parser("eval")
    eval_cmd.add_argument("--manifest", required=True)
    eval_cmd.add_argument("--base-model", default="openai/whisper-large-v3")
    eval_cmd.add_argument("--lora-dir", required=True)
    eval_cmd.add_argument("--output-csv", required=True)
    eval_cmd.add_argument("--language", default="ko")
    eval_cmd.add_argument("--task", default="transcribe")
    eval_cmd.add_argument("--max-samples", type=int, default=200)
    eval_cmd.add_argument("--seed", type=int, default=42)
    eval_cmd.add_argument("--max-new-tokens", type=int, default=128)
    eval_cmd.add_argument("--debug-first-n", type=int, default=0)
    eval_cmd.add_argument("--disable-adapter", action="store_true")

    infer = sub.add_parser("infer")
    infer.add_argument("--base-model", default="openai/whisper-large-v3")
    infer.add_argument("--lora-dir", required=True)
    infer.add_argument("--wav", required=True)
    infer.add_argument("--language", default="ko")
    infer.add_argument("--task", default="transcribe")
    infer.add_argument("--max-new-tokens", type=int, default=128)
    infer.add_argument("--compare-base", action="store_true")
    infer.add_argument("--result-json")

    merge = sub.add_parser("merge")
    merge.add_argument("--base-model", default="openai/whisper-large-v3")
    merge.add_argument("--lora-dir", required=True)
    merge.add_argument("--merged-dir", required=True)

    export = sub.add_parser("ct2-export")
    export.add_argument("--model-dir", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--quantization", default="int8_float16")

    return p.parse_args(argv)


def command_snapshot(namespace: argparse.Namespace) -> dict[str, Any]:
    data = vars(namespace).copy()
    return {
        "command": namespace.command,
        "args": data,
        "started_at": utc_now_iso(),
        "cwd": str(Path.cwd()),
        "python": sys.executable,
    }


def register_termination_handlers(status_path: Path, base_status: dict[str, Any]) -> None:
    def _handle_term(signum, _frame):
        payload = {
            **base_status,
            "state": "stopped",
            "updated_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
            "message": f"Interrupted by signal {signum}",
        }
        write_json(status_path, payload)
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)


def build_train_status(output_dir: Path, state: str, **extra: Any) -> dict[str, Any]:
    checkpoints = checkpoint_names(output_dir)
    return {
        "state": state,
        "output_dir": str(output_dir.resolve()),
        "latest_checkpoint": checkpoints[-1] if checkpoints else None,
        "checkpoint_count": len(checkpoints),
        "tensorboard_log_dir": str((output_dir / "runs").resolve()),
        **extra,
    }


def handle_manifest(args: argparse.Namespace) -> dict[str, Any]:
    import make_manifest

    return make_manifest.main(
        [
            "--root",
            args.root,
            "--wav_dir",
            args.wav_dir,
            "--label_dir",
            args.label_dir,
            "--output",
            args.output,
            "--audio_path_mode",
            args.audio_path_mode,
            "--workers",
            str(args.workers),
        ]
    )


def handle_train(args: argparse.Namespace) -> dict[str, Any]:
    import train_whisper_lora

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    status_path = output_dir / "run_status.json"
    summary_path = output_dir / "train_summary.json"

    snapshot = command_snapshot(args)
    write_json(config_path, snapshot)
    base_status = build_train_status(
        output_dir,
        "running",
        started_at=utc_now_iso(),
        updated_at=utc_now_iso(),
        pid=os_getpid(),
        message="Training process started",
    )
    write_json(status_path, base_status)
    register_termination_handlers(status_path, base_status)

    summary = train_whisper_lora.main(
        [
            "--model_name",
            args.model_name,
            "--manifest",
            args.manifest,
            "--output_dir",
            args.output_dir,
            "--language",
            args.language,
            "--task",
            args.task,
            "--max_steps",
            str(args.max_steps),
            "--batch_size",
            str(args.batch_size),
            "--grad_accum",
            str(args.grad_accum),
            "--lr",
            str(args.lr),
            "--seed",
            str(args.seed),
            "--max_audio_sec",
            str(args.max_audio_sec),
            "--dataloader_workers",
            str(args.dataloader_workers),
            "--lora_r",
            str(args.lora_r),
            "--lora_alpha",
            str(args.lora_alpha),
            "--lora_dropout",
            str(args.lora_dropout),
            "--target_modules",
            args.target_modules,
            "--eval_manifest",
            args.eval_manifest,
            "--eval_steps",
            str(args.eval_steps),
            "--eval_ratio",
            str(args.eval_ratio),
            *(["--fp16"] if args.fp16 else []),
            *(["--use_gradient_checkpointing"] if args.use_gradient_checkpointing else []),
            *(["--pin_memory"] if args.pin_memory else []),
            *(["--load_in_8bit"] if args.load_in_8bit else []),
        ]
    )

    final_status = build_train_status(
        output_dir,
        "completed",
        started_at=base_status["started_at"],
        finished_at=utc_now_iso(),
        updated_at=utc_now_iso(),
        pid=os_getpid(),
        message="Training completed",
    )
    write_json(status_path, final_status)
    payload = {
        **summary,
        "config_path": str(config_path.resolve()),
        "status_path": str(status_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "finished_at": utc_now_iso(),
    }
    write_json(summary_path, payload)
    return payload


def handle_eval(args: argparse.Namespace) -> dict[str, Any]:
    import eval_dataset_lora

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_csv.parent / "eval_summary.json"
    status_path = output_csv.parent / "eval_status.json"

    base_status = {
        "state": "running",
        "output_csv": str(output_csv.resolve()),
        "started_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "pid": os_getpid(),
        "message": "Evaluation started",
    }
    write_json(status_path, base_status)
    register_termination_handlers(status_path, base_status)

    result = eval_dataset_lora.main(
        [
            "--manifest",
            args.manifest,
            "--base_model",
            args.base_model,
            "--lora_dir",
            args.lora_dir,
            "--output_csv",
            args.output_csv,
            "--language",
            args.language,
            "--task",
            args.task,
            "--max_samples",
            str(args.max_samples),
            "--seed",
            str(args.seed),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--debug_first_n",
            str(args.debug_first_n),
            *(["--disable_adapter"] if args.disable_adapter else []),
        ]
    )

    write_json(status_path, {
        **base_status,
        "state": "completed",
        "updated_at": utc_now_iso(),
        "finished_at": utc_now_iso(),
        "message": "Evaluation completed",
    })
    payload = {
        **result,
        "summary_path": str(summary_path.resolve()),
        "status_path": str(status_path.resolve()),
        "completed_at": utc_now_iso(),
    }
    write_json(summary_path, payload)
    return payload


def handle_infer(args: argparse.Namespace) -> dict[str, Any]:
    import infer_lora

    result = infer_lora.main(
        [
            "--base_model",
            args.base_model,
            "--lora_dir",
            args.lora_dir,
            "--wav",
            args.wav,
            "--language",
            args.language,
            "--task",
            args.task,
            "--max_new_tokens",
            str(args.max_new_tokens),
            *(["--compare_base"] if args.compare_base else []),
        ]
    )
    if args.result_json:
        write_json(args.result_json, result)
    return result


def handle_merge(args: argparse.Namespace) -> dict[str, Any]:
    import merge_peft

    merged_dir = Path(args.merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    summary_path = merged_dir / "merge_summary.json"
    result = merge_peft.main(
        [
            "--base-model",
            args.base_model,
            "--lora-dir",
            args.lora_dir,
            "--merged-dir",
            args.merged_dir,
        ]
    )
    payload = {
        **result,
        "completed_at": utc_now_iso(),
        "summary_path": str(summary_path.resolve()),
    }
    write_json(summary_path, payload)
    return payload


def handle_ct2_export(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "ct2_export_summary.json"
    cmd = [
        "ct2-transformers-converter",
        "--model",
        args.model_dir,
        "--output_dir",
        args.output_dir,
        "--quantization",
        args.quantization,
        "--force",
    ]
    subprocess.run(cmd, check=True)
    payload = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "quantization": args.quantization,
        "converter_command": cmd,
        "completed_at": utc_now_iso(),
        "summary_path": str(summary_path.resolve()),
    }
    write_json(summary_path, payload)
    return payload


def os_getpid() -> int:
    import os

    return os.getpid()


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.command == "manifest":
            result = handle_manifest(args)
        elif args.command == "train":
            result = handle_train(args)
        elif args.command == "eval":
            result = handle_eval(args)
        elif args.command == "infer":
            result = handle_infer(args)
        elif args.command == "merge":
            result = handle_merge(args)
        elif args.command == "ct2-export":
            result = handle_ct2_export(args)
        else:
            raise ValueError(f"Unsupported command: {args.command}")
    except Exception as error:
        if args.command == "train":
            output_dir = Path(args.output_dir)
            status_path = output_dir / "run_status.json"
            summary_path = output_dir / "train_summary.json"
            write_json(
                status_path,
                build_train_status(
                    output_dir,
                    "failed",
                    updated_at=utc_now_iso(),
                    finished_at=utc_now_iso(),
                    pid=os_getpid(),
                    error=str(error),
                    message="Training failed",
                ),
            )
            write_json(
                summary_path,
                {
                    "output_dir": str(output_dir.resolve()),
                    "error": str(error),
                    "failed_at": utc_now_iso(),
                },
            )
        elif args.command == "eval":
            output_csv = Path(args.output_csv)
            status_path = output_csv.parent / "eval_status.json"
            write_json(
                status_path,
                {
                    "state": "failed",
                    "output_csv": str(output_csv.resolve()),
                    "updated_at": utc_now_iso(),
                    "finished_at": utc_now_iso(),
                    "pid": os_getpid(),
                    "error": str(error),
                    "message": "Evaluation failed",
                },
            )
        raise

    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
