"""Extract each healthy network and record its memory."""
from pathlib import Path

from stable_baselines3 import SAC

from ..memory import build_memory, save_memory, subsample
from ..parallel import run_tasks
from ..pipeline import Context
from ..policy import from_sac, save_policy
from ..rollout import collect_pairs
from .train import robot_dir


def _record_robot(task: dict) -> None:
    section, seed, out = task["section"], task["seed"], Path(task["out"])
    policy = from_sac(SAC.load(Path(task["source"]) / "model.zip", device="cpu"))
    obs, actions = collect_pairs(policy, task["env_id"], section["pairs"], section["env_seed_start"])
    memory = build_memory(policy, obs, actions, section["repair_pairs"], seed=seed)
    small = subsample(memory, section["small_pairs"], section["small_repair_pairs"], seed=seed)
    save_policy(policy, out / f"policy_{seed}.pt")
    save_memory(memory, out / f"memory_large_{seed}.pt")
    save_memory(small, out / f"memory_small_{seed}.pt")


def run(ctx: Context) -> None:
    tasks = [{"out": str(ctx.out), "seed": seed, "env_id": ctx.cfg["env_id"],
              "section": ctx.cfg["memory"], "source": str(robot_dir(ctx.stage_dir("gate"), seed))}
             for seed in ctx.cfg["robots"]]
    run_tasks(_record_robot, tasks, ctx.workers)
