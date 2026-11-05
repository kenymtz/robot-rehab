"""Train the healthy robots with SAC, saving often enough to survive a power cut."""
import os
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC

from ..parallel import run_tasks
from ..pipeline import Context
from ..policy import HIDDEN


def robot_dir(base: Path, seed: int) -> Path:
    return base / f"robot_{seed}"


def _checkpoint(model: SAC, directory: Path) -> None:
    model.save(directory / "model.part.zip")
    os.replace(directory / "model.part.zip", directory / "model.zip")
    model.save_replay_buffer(directory / "buffer.part.pkl")
    os.replace(directory / "buffer.part.pkl", directory / "buffer.pkl")


def _load_or_create(directory: Path, env, seed: int, learning_starts: int) -> SAC:
    if (directory / "model.zip").exists():
        model = SAC.load(directory / "model.zip", env=env, device="cpu")
        if (directory / "buffer.pkl").exists():
            model.load_replay_buffer(directory / "buffer.pkl")
        return model
    return SAC("MlpPolicy", env, seed=seed, learning_starts=learning_starts, device="cpu",
               policy_kwargs={"net_arch": [HIDDEN, HIDDEN]})


def train_to(directory: Path, env_id: str, seed: int, target_steps: int, section: dict) -> int:
    """Train until `target_steps`, resuming from `directory`. Returns the steps reached."""
    directory.mkdir(parents=True, exist_ok=True)
    env = gym.make(env_id)
    model = _load_or_create(directory, env, seed, section["learning_starts"])
    while model.num_timesteps < target_steps:
        chunk = min(section["checkpoint_every"], target_steps - model.num_timesteps)
        model.learn(chunk, reset_num_timesteps=False)
        _checkpoint(model, directory)
    env.close()
    return model.num_timesteps


def _train_robot(task: dict) -> int:
    return train_to(Path(task["directory"]), task["env_id"], task["seed"], task["steps"], task["section"])


def run(ctx: Context) -> None:
    section = ctx.cfg["train"]
    tasks = [{"directory": str(robot_dir(ctx.out, seed)), "env_id": ctx.cfg["env_id"], "seed": seed,
              "steps": section["initial_steps"], "section": section} for seed in ctx.cfg["robots"]]
    run_tasks(_train_robot, tasks, ctx.workers)
