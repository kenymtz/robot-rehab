import torch

from robotrehab.policy import MaskedPolicy
from robotrehab.rollout import collect_pairs, evaluate_policy

ENV = "Walker2d-v5"


def walker_policy():
    torch.manual_seed(0)
    return MaskedPolicy(obs_dim=17, act_dim=6, hidden=16).eval()


def test_same_seed_gives_the_same_return():
    policy = walker_policy()
    first = evaluate_policy(policy, ENV, [3, 4])
    second = evaluate_policy(policy, ENV, [3, 4])
    assert first["return"] == second["return"]


def test_an_untrained_robot_falls_and_trajectory_is_recorded():
    stats = evaluate_policy(walker_policy(), ENV, [0])
    assert stats["survivors"] == 0
    assert stats["qpos"].shape[1] == 9
    assert len(stats["qpos"]) == len(stats["qvel"])


def test_collect_pairs_spans_episodes_and_stops_exactly():
    obs, actions = collect_pairs(walker_policy(), ENV, n_pairs=300, seed_start=1000)
    assert obs.shape == (300, 17)
    assert actions.shape == (300, 6)
