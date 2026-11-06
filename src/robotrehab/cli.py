"""Entry point: python -m robotrehab.cli --config configs/full.yaml"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from .config import load_config
from .pipeline import PROJECT_ROOT, StageError, run_pipeline
from .stages import STAGES


def log(message: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Corrida completa de robot-rehab")
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    root = PROJECT_ROOT / "runs" / cfg["name"]
    try:
        run_pipeline(STAGES, cfg, root, args.workers, log)
    except StageError as exc:
        log(f"DETENIDO: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
