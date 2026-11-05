"""Run configuration."""
from pathlib import Path

import yaml


def load_config(path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
