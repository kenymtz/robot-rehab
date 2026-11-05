"""The 15 cases. Rewards appear here for the first time, only to measure."""
from pathlib import Path

import numpy as np

from ..damage import apply_damage
from ..memory import load_memory
from ..parallel import run_tasks
from ..pipeline import Context, read_json, write_json
from ..policy import load_policy
from ..repair import action_mse, repair
from ..rollout import evaluate_policy

CONDITIONS = [("none", "large"), ("mutation", "large"), ("ridge", "large"), ("asr", "large"),
              ("bc", "large"), ("mutation", "small"), ("ridge", "small"), ("bc", "small")]
KEEP_TRAJECTORY = {("healthy", "large"), ("none", "large"), ("mutation", "large")}


def case_name(robot: int, pattern: int, condition: str, memory: str) -> str:
    return f"robot{robot}_pattern{pattern}_{condition}_{memory}"


def _save_trajectory(path: Path, stats: dict) -> None:
    temporary = path.with_name(path.name + ".part")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, qpos=stats["qpos"], qvel=stats["qvel"])
    temporary.replace(path)


def _run_case(task: dict) -> dict:
    out = Path(task["out"])
    robot, pattern, condition, memory_size = task["robot"], task["pattern"], task["condition"], task["memory"]
    name = case_name(robot, pattern, condition, memory_size)
    result_path = out / "cases" / f"{name}.json"
    if result_path.exists():
        return read_json(result_path)
    directory = Path(task["memory_dir"])
    policy = load_policy(directory / f"policy_{robot}.pt")
    memory = load_memory(directory / f"memory_{memory_size}_{robot}.pt")
    if condition != "healthy":
        apply_damage(policy, task["fraction"], pattern)
        policy = repair(condition, policy, memory, task["tuned"].get(condition, {}), pattern)
    stats = evaluate_policy(policy, task["env_id"], task["seeds"])
    if (condition, memory_size) in KEEP_TRAJECTORY:
        _save_trajectory(out / "traj" / f"{name}.npz", stats)
    result = {"robot": robot, "pattern": pattern, "condition": condition, "memory": memory_size,
              "return": stats["return"], "survivors": stats["survivors"], "distance": stats["distance"],
              "val_mse": action_mse(policy, memory.val_obs, memory.val_actions)}
    write_json(result_path, result)
    return result


def build_tasks(cfg: dict, base: dict) -> list[dict]:
    tasks = [dict(base, robot=r, pattern=-1, condition="healthy", memory="large") for r in cfg["robots"]]
    tasks += [dict(base, robot=r, pattern=p, condition=c, memory=m) for r in cfg["robots"]
              for p in cfg["final"]["patterns"] for c, m in CONDITIONS]
    return tasks


def run(ctx: Context) -> None:
    (ctx.out / "cases").mkdir(exist_ok=True)
    (ctx.out / "traj").mkdir(exist_ok=True)
    section = ctx.cfg["final"]
    start = section["env_seed_start"]
    base = {"out": str(ctx.out), "memory_dir": str(ctx.stage_dir("memory")), "env_id": ctx.cfg["env_id"],
            "seeds": list(range(start, start + section["episodes"])),
            "fraction": read_json(ctx.stage_dir("damage_curve") / "damage_level.json")["fraction"],
            "tuned": read_json(ctx.stage_dir("tune") / "tuned.json")}
    write_json(ctx.out / "final.json", run_tasks(_run_case, build_tasks(ctx.cfg, base), ctx.workers))
