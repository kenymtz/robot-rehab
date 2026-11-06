"""Record every stage of the demo: learning, damage and repair, with real measurements.

Run after scripts/demo_train.py. Writes clips and data.json to runs/demo/site.
Usage: uv run --frozen python scripts/demo_render.py [--quick]
"""
import json
import statistics
import sys
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch

from robotrehab.damage import apply_damage, choose_damage_level
from robotrehab.evaluate import recovery
from robotrehab.memory import Memory, build_memory
from robotrehab.parallel import run_tasks
from robotrehab.pipeline import PROJECT_ROOT
from robotrehab.policy import MaskedPolicy, load_policy
from robotrehab.repair import action_mse, repair
from robotrehab.rollout import collect_pairs, evaluate_policy

ENV = "Walker2d-v5"
ROOT = PROJECT_ROOT / "runs" / "demo"
SITE = ROOT / "site"
CLIP_SEED = 40_000
EVAL_SEEDS = list(range(30_000, 30_010))
DAMAGE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5]
TUNE_PATTERNS = [0, 1, 2]
TEST_PATTERNS = [100, 101, 102, 103, 104]
METHODS = ["none", "asr", "mutation", "ridge", "bc"]
GRIDS = {
    "none": [{}],
    "asr": [{}],
    "ridge": [{"lam": lam} for lam in (1e-4, 1e-2, 1.0)],
    "bc": [{"lr": lr, "steps": 2000, "batch_size": 256} for lr in (1e-4, 1e-3)],
    "mutation": [{"budget": 30000, "sigma": 0.002, "mutate_fraction": 0.02}],
}


def best_robot() -> tuple[int, list[dict]]:
    rows = [json.loads(line) for line in (ROOT / "progress.jsonl").read_text().splitlines()]
    last = max(r["steps"] for r in rows)
    finished = [r for r in rows if r["steps"] == last]
    best = max(finished, key=lambda r: (r["survivors"], r["return"]))
    return best["robot"], rows


def write_clip(qpos: np.ndarray, qvel: np.ndarray, name: str) -> None:
    env = gym.make(ENV, render_mode="rgb_array", width=480, height=368, camera_name="track")
    env.reset(seed=0)
    frames = []
    for index in range(0, len(qpos), 4):
        env.unwrapped.set_state(qpos[index], qvel[index])
        frames.append(env.render())
    env.close()
    frames += [frames[-1]] * 30  # hold the ending so a fall can be read
    imageio.mimsave(SITE / f"{name}.mp4", frames, fps=30, macro_block_size=16)


def measure(policy: MaskedPolicy, seeds: list[int] = EVAL_SEEDS) -> dict:
    stats = evaluate_policy(policy, ENV, seeds)
    return {"return": stats["return"], "survivors": stats["survivors"], "episodes": len(seeds),
            "distance": stats["distance"]}


def clip_of(policy: MaskedPolicy, name: str) -> None:
    stats = evaluate_policy(policy, ENV, [CLIP_SEED])
    write_clip(stats["qpos"], stats["qvel"], name)


def _damaged_return(task: dict) -> dict:
    policy = load_policy(task["path"])
    apply_damage(policy, task["fraction"], task["pattern"])
    return {"fraction": task["fraction"], "pattern": task["pattern"], **measure(policy, task["seeds"])}


def _tune(task: dict) -> float:
    policy = load_policy(task["path"])
    memory = Memory(**torch.load(task["memory"]))
    apply_damage(policy, task["fraction"], task["pattern"])
    repaired = repair(task["method"], policy, memory, task["params"], task["pattern"])
    return action_mse(repaired, memory.val_obs, memory.val_actions)


def _final(task: dict) -> dict:
    policy = load_policy(task["path"])
    memory = Memory(**torch.load(task["memory"]))
    apply_damage(policy, task["fraction"], task["pattern"])
    repaired = repair(task["method"], policy, memory, task["params"], task["pattern"])
    result = {"method": task["method"], "pattern": task["pattern"], **measure(repaired),
              "val_mse": action_mse(repaired, memory.val_obs, memory.val_actions)}
    if task["pattern"] == TEST_PATTERNS[0]:
        clip_of(repaired, f"repair_{task['method']}")
    return result


def main() -> None:
    quick = "--quick" in sys.argv
    SITE.mkdir(parents=True, exist_ok=True)
    robot, progress = best_robot()
    directory = ROOT / f"robot_{robot}"
    milestones = sorted({r["steps"] for r in progress})
    final_path = directory / f"policy_{milestones[-1]}.pt"
    healthy = load_policy(final_path)
    workers = 6
    print("robot", robot, "pasos", milestones[-1], flush=True)

    torch.manual_seed(0)
    learning = []
    for steps in [0] + milestones:
        policy = MaskedPolicy(17, 6).eval() if steps == 0 else load_policy(directory / f"policy_{steps}.pt")
        learning.append({"steps": steps, **measure(policy)})
        clip_of(policy, f"learn_{steps}")
        print("aprendizaje", learning[-1], flush=True)
    healthy_stats = learning[-1]

    pairs = 500 if quick else 10_000
    obs, actions = collect_pairs(healthy, ENV, pairs, 1000)
    memory = build_memory(healthy, obs, actions, int(pairs * 0.8), seed=robot)
    memory_path = ROOT / "demo_memory.pt"
    torch.save(vars(memory), memory_path)

    path = str(final_path)
    tasks = [{"path": path, "fraction": f, "pattern": p, "seeds": EVAL_SEEDS[:5]}
             for f in DAMAGE_LEVELS for p in TUNE_PATTERNS]
    curve = run_tasks(_damaged_return, tasks, workers)
    ratios = {f: [c["return"] / healthy_stats["return"] for c in curve if c["fraction"] == f] for f in DAMAGE_LEVELS}
    level, used_fallback = choose_damage_level(ratios, 0.4, 0.5)
    print("nivel", level, used_fallback, {f: round(statistics.median(v), 2) for f, v in ratios.items()}, flush=True)

    damage = []
    for fraction in DAMAGE_LEVELS:
        policy = load_policy(final_path)
        killed = apply_damage(policy, fraction, TEST_PATTERNS[0])
        clip_of(policy, f"damage_{int(fraction * 100)}")
        rows = [c for c in curve if c["fraction"] == fraction]
        damage.append({"fraction": fraction, "ratio": statistics.median(ratios[fraction]),
                       "return": statistics.median(c["return"] for c in rows),
                       "survivors": sum(c["survivors"] for c in rows),
                       "episodes": sum(c["episodes"] for c in rows), "killed": killed})

    grids = dict(GRIDS)
    if quick:
        grids["mutation"] = [{"budget": 50, "sigma": 0.01, "mutate_fraction": 0.2}]
        grids["bc"] = [{"lr": 1e-3, "steps": 50, "batch_size": 64}]
    chosen = {}
    for method, options in grids.items():
        if len(options) == 1:
            chosen[method] = options[0]
            continue
        tasks = [{"path": path, "memory": str(memory_path), "fraction": level, "pattern": p, "method": method,
                  "params": o} for o in options for p in TUNE_PATTERNS]
        scores = run_tasks(_tune, tasks, workers)
        n = len(TUNE_PATTERNS)
        means = [sum(scores[i * n:(i + 1) * n]) / n for i in range(len(options))]
        chosen[method] = options[means.index(min(means))]
    print("ajustes", chosen, flush=True)

    tasks = [{"path": path, "memory": str(memory_path), "fraction": level, "pattern": p, "method": m,
              "params": chosen[m]} for m in METHODS for p in TEST_PATTERNS]
    results = run_tasks(_final, tasks, workers)
    damaged = {r["pattern"]: r["return"] for r in results if r["method"] == "none"}
    methods = []
    for method in METHODS:
        rows = [r for r in results if r["method"] == method]
        values = [recovery(r["return"], damaged[r["pattern"]], healthy_stats["return"]) for r in rows]
        usable = [v for v in values if v is not None]
        methods.append({"method": method, "return": statistics.median(r["return"] for r in rows),
                        "survivors": sum(r["survivors"] for r in rows), "episodes": sum(r["episodes"] for r in rows),
                        "recovery": statistics.median(usable) if usable else None, "cases": len(usable),
                        "values": usable, "val_mse": statistics.median(r["val_mse"] for r in rows)})
        print(methods[-1], flush=True)

    policy = load_policy(final_path)
    killed = apply_damage(policy, level, TEST_PATTERNS[0])
    data = {"robot": robot, "steps": milestones[-1], "healthy": healthy_stats, "learning": learning,
            "all_robots": progress, "damage": damage, "level": level, "used_fallback": used_fallback,
            "killed": killed, "methods": methods, "params": chosen, "memory_pairs": pairs, "units": 256}
    (SITE / "data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("listo", flush=True)


if __name__ == "__main__":
    main()
