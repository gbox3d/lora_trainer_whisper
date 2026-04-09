"""Thin wrapper — delegates to src/lora_trainer/runner_cli.py.

dalus_server calls: uv run python runner_cli.py <subcommand> ...
This file preserves that contract after the src/ migration.
"""
from lora_trainer.runner_cli import main

if __name__ == "__main__":
    main()
