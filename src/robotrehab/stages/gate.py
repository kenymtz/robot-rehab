"""A robot only counts if it really walks. Extend training until it does, up to a limit."""
import shutil
from pathlib import Path

from stable_baselines3 import SAC

from ..parallel import run_tasks
from ..pipeline import Context, StageError, read_json, write_json
from ..policy import from_sac
from ..rollout import evaluate_policy
from .train import robot_dir, train_to


def passes(stats: dict, section: dict) -> bool:
    return stats["return"] >= section["min_return"] and stats["survivors"] >= section["min_survivors"]


def _gate_robot(task: dict) -> dict:
    directory, section, train = Path(task["directory"]), task["gate"], task["train"]
    result_path = directory / "gate.json"
    if result_path.exists():
        return read_json(result_path)
    if not directory.exists():
        shutil.copytree(task["source"], directory)
    start = section["env_seed_start"]
    seeds = list(range(start, start + section["episodes"]))
    steps = train["initial_steps"]
    while True:
        model = SAC.load(directory / "model.zip", device="cpu")
        steps = model.num_timesteps
        stats = evaluate_policy(from_sac(model), task["env_id"], seeds)
        valid = passes(stats, section)
        if valid or not section["enforce"] or steps >= train["max_steps"]:
            break
        target = min(steps + train["extend_steps"], train["max_steps"])
        train_to(directory, task["env_id"], task["seed"], target, train)
    result = {"robot": task["seed"], "steps": steps, "return": stats["return"],
              "survivors": stats["survivors"], "valid": valid}
    write_json(result_path, result)
    return result


def run(ctx: Context) -> None:
    source = ctx.stage_dir("train")
    tasks = [{"directory": str(robot_dir(ctx.out, seed)), "source": str(robot_dir(source, seed)),
              "seed": seed, "env_id": ctx.cfg["env_id"], "gate": ctx.cfg["gate"], "train": ctx.cfg["train"]}
             for seed in ctx.cfg["robots"]]
    results = run_tasks(_gate_robot, tasks, ctx.workers)
    write_json(ctx.out / "gate.json", results)
    failed = [r for r in results if not r["valid"]]
    if failed and ctx.cfg["gate"]["enforce"]:
        detail = "; ".join(f"robot {r['robot']}: retorno {r['return']:.0f}, "
                           f"{r['survivors']} sin caerse, {r['steps']} pasos" for r in failed)
        raise StageError(f"Robots que no cumplen el criterio de marcha: {detail}")
