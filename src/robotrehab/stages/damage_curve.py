"""How much damage makes the robot fall? Uses tuning patterns only, never the final ones."""
from pathlib import Path

from ..damage import apply_damage, choose_damage_level
from ..parallel import run_tasks
from ..pipeline import Context, write_json
from ..policy import load_policy
from ..rollout import evaluate_policy


def _measure(task: dict) -> float:
    policy = load_policy(Path(task["memory_dir"]) / f"policy_{task['robot']}.pt")
    if task["fraction"] > 0:
        apply_damage(policy, task["fraction"], task["pattern"])
    return evaluate_policy(policy, task["env_id"], task["seeds"])["return"]


def run(ctx: Context) -> None:
    section = ctx.cfg["damage_curve"]
    start = section["env_seed_start"]
    base = {"memory_dir": str(ctx.stage_dir("memory")), "env_id": ctx.cfg["env_id"],
            "seeds": list(range(start, start + section["episodes"]))}
    healthy_tasks = [dict(base, robot=r, fraction=0.0, pattern=0) for r in ctx.cfg["robots"]]
    damaged_tasks = [dict(base, robot=r, fraction=f, pattern=p) for f in section["fractions"]
                     for r in ctx.cfg["robots"] for p in section["patterns"]]
    returns = run_tasks(_measure, healthy_tasks + damaged_tasks, ctx.workers)
    healthy = {t["robot"]: value for t, value in zip(healthy_tasks, returns)}
    ratios: dict[float, list[float]] = {f: [] for f in section["fractions"]}
    for task, value in zip(damaged_tasks, returns[len(healthy_tasks):]):
        ratios[task["fraction"]].append(value / healthy[task["robot"]])
    fraction, used_fallback = choose_damage_level(ratios, section["max_ratio"], section["fallback"])
    write_json(ctx.out / "curve.json", {str(f): values for f, values in ratios.items()})
    write_json(ctx.out / "damage_level.json", {"fraction": fraction, "used_fallback": used_fallback})
    ctx.log(f"[damage_curve] nivel elegido: {fraction}" + (" (por defecto)" if used_fallback else ""))
