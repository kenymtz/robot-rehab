"""Fail in the first minute, not after hours of training."""
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import SAC

from ..pipeline import PROJECT_ROOT, Context, StageError, write_json
from ..policy import HIDDEN


def check_disk(path: Path, min_gb: float) -> float:
    free_gb = shutil.disk_usage(path).free / 2**30
    if free_gb < min_gb:
        raise StageError(f"Hay {free_gb:.1f} GB libres y se necesitan {min_gb} GB.")
    return free_gb


def check_simulator(env_id: str) -> None:
    try:
        env = gym.make(env_id)
        env.reset(seed=0)
        env.step(env.action_space.sample())
        env.close()
    except Exception as exc:
        raise StageError(f"No se pudo abrir el simulador {env_id}: {exc!r}") from exc


def check_tests() -> None:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise StageError("Fallan los tests:\n" + result.stdout[-2000:])


def measure_speed(env_id: str, steps: int) -> float:
    """Seconds per training step, from a short real SAC run."""
    torch.set_num_threads(1)
    model = SAC("MlpPolicy", gym.make(env_id), learning_starts=0, device="cpu",
                policy_kwargs={"net_arch": [HIDDEN, HIDDEN]})
    start = time.perf_counter()
    model.learn(steps)
    return (time.perf_counter() - start) / steps


def estimate_hours(seconds_per_step: float, steps: int, robots: int, workers: int) -> float:
    return seconds_per_step * steps * math.ceil(robots / workers) / 3600


def run(ctx: Context) -> None:
    section = ctx.cfg["preflight"]
    free_gb = check_disk(ctx.root, section["min_disk_gb"])
    check_simulator(ctx.cfg["env_id"])
    if section["run_tests"]:
        check_tests()
    seconds = measure_speed(ctx.cfg["env_id"], section["speed_steps"])
    hours = estimate_hours(seconds, ctx.cfg["train"]["initial_steps"], len(ctx.cfg["robots"]), ctx.workers)
    ctx.log(f"[preflight] {ctx.workers} procesos, {free_gb:.0f} GB libres, "
            f"entrenamiento estimado: {hours:.1f} h")
    write_json(ctx.out / "preflight.json",
               {"free_gb": free_gb, "seconds_per_step": seconds, "estimated_training_hours": hours})
