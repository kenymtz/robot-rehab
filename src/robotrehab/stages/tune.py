"""Choose hyperparameters with the reserved memory. No simulator and no reward here."""
from itertools import product
from pathlib import Path

from ..damage import apply_damage
from ..memory import load_memory
from ..parallel import run_tasks
from ..pipeline import Context, read_json, write_json
from ..policy import load_policy
from ..repair import action_mse, repair


def param_grid(section: dict) -> list[dict]:
    """Every combination of the list values. Scalar values go into all of them."""
    grids = {key: value for key, value in section.items() if isinstance(value, list)}
    fixed = {key: value for key, value in section.items() if not isinstance(value, list)}
    return [dict(zip(grids, values), **fixed) for values in product(*grids.values())]


def _score(task: dict) -> float:
    directory = Path(task["memory_dir"])
    policy = load_policy(directory / f"policy_{task['robot']}.pt")
    memory = load_memory(directory / f"memory_large_{task['robot']}.pt")
    apply_damage(policy, task["fraction"], task["pattern"])
    repaired = repair(task["method"], policy, memory, task["params"], task["pattern"])
    return action_mse(repaired, memory.val_obs, memory.val_actions)


def run(ctx: Context) -> None:
    fraction = read_json(ctx.stage_dir("damage_curve") / "damage_level.json")["fraction"]
    patterns = ctx.cfg["damage_curve"]["patterns"]
    tasks = [{"memory_dir": str(ctx.stage_dir("memory")), "fraction": fraction, "method": method,
              "params": params, "robot": robot, "pattern": pattern}
             for method, section in ctx.cfg["tune"].items() for params in param_grid(section)
             for robot in ctx.cfg["robots"] for pattern in patterns]
    scores = run_tasks(_score, tasks, ctx.workers)
    grouped: dict[str, list] = {}
    for task, score in zip(tasks, scores):
        options = grouped.setdefault(task["method"], [])
        match = next((o for o in options if o["params"] == task["params"]), None)
        if match is None:
            match = {"params": task["params"], "scores": []}
            options.append(match)
        match["scores"].append(score)
    tuned = {method: min(options, key=lambda o: sum(o["scores"]) / len(o["scores"]))["params"]
             for method, options in grouped.items()}
    write_json(ctx.out / "tuned.json", tuned)
