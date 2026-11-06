"""Short training for a visual demo, keeping a snapshot of the network at each milestone.

Not the official run: robots train far less than configs/full.yaml asks for.
Usage: uv run --frozen python scripts/demo_train.py
"""
import json
from pathlib import Path

from stable_baselines3 import SAC

from robotrehab.parallel import run_tasks
from robotrehab.pipeline import PROJECT_ROOT
from robotrehab.policy import from_sac, save_policy
from robotrehab.rollout import evaluate_policy
from robotrehab.stages.train import train_to

ENV = "Walker2d-v5"
ROOT = PROJECT_ROOT / "runs" / "demo"
SEEDS = [0, 1, 2]
MILESTONES = [20_000, 50_000, 100_000, 200_000, 300_000, 400_000]
SECTION = {"learning_starts": 10_000, "checkpoint_every": 50_000}


def train_robot(seed: int) -> None:
    directory = ROOT / f"robot_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    for steps in MILESTONES:
        path = directory / f"policy_{steps}.pt"
        if path.exists():
            continue
        train_to(directory, ENV, seed, steps, SECTION)
        policy = from_sac(SAC.load(directory / "model.zip", device="cpu"))
        stats = evaluate_policy(policy, ENV, list(range(20_000, 20_005)))
        save_policy(policy, path)
        row = {"robot": seed, "steps": steps, "return": stats["return"], "survivors": stats["survivors"]}
        with open(ROOT / "progress.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(row, flush=True)


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    run_tasks(train_robot, SEEDS, workers=len(SEEDS))
    print("listo", flush=True)
