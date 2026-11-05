import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC

from robotrehab.policy import MaskedPolicy, clone, from_sac, load_policy, save_policy


def test_actions_are_bounded_and_have_the_right_shape(healthy):
    action = healthy.act(np.zeros(8, dtype=np.float32))
    assert action.shape == (3,)
    assert np.all(np.abs(action) <= 1)


def test_a_dead_unit_outputs_zero(healthy):
    healthy.m1.mask[3] = 0
    h1, _ = healthy.hidden(torch.randn(10, 8))
    assert torch.all(h1[:, 3] == 0)


def test_clone_does_not_share_weights(healthy):
    copy = clone(healthy)
    copy.head.weight.data.add_(1.0)
    assert not torch.equal(copy.head.weight, healthy.head.weight)


def test_save_and_load_keep_masks_and_weights(healthy, tmp_path):
    healthy.m2.mask[0] = 0
    save_policy(healthy, tmp_path / "p.pt")
    loaded = load_policy(tmp_path / "p.pt")
    obs = torch.randn(5, 8)
    assert torch.equal(healthy(obs), loaded(obs))
    assert loaded.m2.mask[0] == 0


def test_extraction_acts_like_the_sac_actor():
    env = gym.make("Walker2d-v5")
    model = SAC("MlpPolicy", env, seed=0, device="cpu", policy_kwargs={"net_arch": [32, 32]})
    policy = from_sac(model)
    obs, _ = env.reset(seed=0)
    for _ in range(50):
        expected, _ = model.predict(obs, deterministic=True)
        np.testing.assert_allclose(policy.act(obs), expected, atol=1e-5)
        obs, _, terminated, truncated, _ = env.step(expected)
        if terminated or truncated:
            obs, _ = env.reset(seed=1)
    assert isinstance(policy, MaskedPolicy)
