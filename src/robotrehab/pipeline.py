"""Stages with fingerprints: a stage repeats only if its config, code or upstream changed."""
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PACKAGE_DIR = Path(__file__).parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]


class StageError(RuntimeError):
    """A failure the run must stop on, with a message meant for a person."""


@dataclass(frozen=True)
class Stage:
    name: str
    run: Callable
    config_keys: tuple[str, ...]
    sources: tuple[str, ...]  # paths relative to the package directory
    always: bool = False      # runs every time and does not feed later fingerprints


@dataclass
class Context:
    cfg: dict
    root: Path
    out: Path      # this stage's working directory
    workers: int
    log: Callable

    def stage_dir(self, name: str) -> Path:
        return self.root / name


def write_json(path: Path, data) -> None:
    """Atomic write, so a cut never leaves half a file."""
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(stage: Stage, cfg: dict, upstream: str, package_dir: Path = PACKAGE_DIR) -> str:
    digest = hashlib.sha256(upstream.encode())
    digest.update(json.dumps({key: cfg[key] for key in stage.config_keys}, sort_keys=True).encode())
    for source in stage.sources:
        digest.update((package_dir / source).read_bytes())
    return digest.hexdigest()


def run_pipeline(stages: list[Stage], cfg: dict, root: Path, workers: int, log: Callable = print,
                 package_dir: Path = PACKAGE_DIR) -> None:
    root.mkdir(parents=True, exist_ok=True)
    upstream = ""
    for stage in stages:
        key = fingerprint(stage, cfg, upstream, package_dir)
        if not stage.always:
            upstream = key
        final, work = root / stage.name, root / f"{stage.name}.tmp"
        done = final / "done.json"
        if not stage.always and done.exists() and read_json(done)["fingerprint"] == key:
            log(f"[{stage.name}] ya estaba hecha")
            continue
        shutil.rmtree(final, ignore_errors=True)
        marker = work / ".fingerprint"
        if work.exists() and (not marker.exists() or marker.read_text() != key):
            shutil.rmtree(work)
        work.mkdir(exist_ok=True)
        marker.write_text(key)
        log(f"[{stage.name}] empieza")
        stage.run(Context(cfg, root, work, workers, log))
        write_json(work / "done.json", {"fingerprint": key})
        work.replace(final)
        log(f"[{stage.name}] terminada")
