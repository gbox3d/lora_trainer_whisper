"""Thin wrapper — delegates to src/lora_trainer/check.py."""
# check.py is a script-style module (no main function), so exec it directly.
if __name__ == "__main__":
    from pathlib import Path
    exec(Path(__file__).parent.joinpath("src/lora_trainer/check.py").read_text())
