"""Running episodes in the simulator. Repairs never import this module."""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from .policy import MaskedPolicy


@dataclass
class Episode:
    total_return: float
    length: int
    survived: bool
    distance: float
    qpos: np.ndarray
    qvel: np.ndarray


def run_episode(env, policy: MaskedPolicy, seed: int) -> Episode:
    obs, _ = env.reset(seed=seed)
    data = env.unwrapped.data
    start_x = float(data.qpos[0])
    qpos, qvel = [data.qpos.copy()], [data.qvel.copy()]
    total, length, terminated, truncated = 0.0, 0, False, False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, _ = env.step(policy.act(obs))
        total += float(reward)
        length += 1
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
    return Episode(total, length, not terminated, float(data.qpos[0]) - start_x,
                   np.array(qpos), np.array(qvel))


def evaluate_policy(policy: MaskedPolicy, env_id: str, seeds: list[int]) -> dict:
    """Mean return, episodes without a fall, mean distance, and the first episode's trajectory."""
    env = gym.make(env_id)
    episodes = [run_episode(env, policy, seed) for seed in seeds]
    env.close()
    return {
        "return": float(np.mean([e.total_return for e in episodes])),
        "survivors": sum(e.survived for e in episodes),
        "distance": float(np.mean([e.distance for e in episodes])),
        "qpos": episodes[0].qpos,
        "qvel": episodes[0].qvel,
    }


def collect_pairs(policy: MaskedPolicy, env_id: str, n_pairs: int,
                  seed_start: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Record what the policy saw and did until `n_pairs` pairs exist."""
    env = gym.make(env_id)
    observations, actions, seed = [], [], seed_start
    while len(observations) < n_pairs:
        obs, _ = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated) and len(observations) < n_pairs:
            action = policy.act(obs)
            observations.append(obs)
            actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
        seed += 1
    env.close()
    return (torch.as_tensor(np.array(observations), dtype=torch.float32),
            torch.as_tensor(np.array(actions), dtype=torch.float32))
