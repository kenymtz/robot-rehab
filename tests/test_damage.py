import pytest
import torch

from robotrehab.damage import apply_damage, choose_damage_level


def dead(policy, layer):
    return (dict((n, m) for n, _, m in policy.lesionable())[layer].mask == 0).nonzero().flatten().tolist()


def test_kills_the_same_fraction_in_each_layer(healthy):
    killed = apply_damage(healthy, 0.25, seed=0)
    assert set(killed) == {"h1", "h2"}
    assert len(killed["h1"]) == 8
    assert len(killed["h2"]) == 8


def test_changes_the_masks_in_place(healthy):
    killed = apply_damage(healthy, 0.25, seed=0)
    assert sorted(killed["h1"]) == dead(healthy, "h1")
    assert sorted(killed["h2"]) == dead(healthy, "h2")


def test_same_seed_kills_the_same_units(healthy):
    other = type(healthy)(**healthy.config)
    assert apply_damage(healthy, 0.5, seed=3) == apply_damage(other, 0.5, seed=3)


def test_different_seeds_kill_different_units(healthy):
    other = type(healthy)(**healthy.config)
    assert apply_damage(healthy, 0.5, seed=1) != apply_damage(other, 0.5, seed=2)


def test_layers_do_not_share_the_same_pattern(healthy):
    killed = apply_damage(healthy, 0.5, seed=0)
    assert sorted(killed["h1"]) != sorted(killed["h2"])


def test_zero_damage_changes_nothing(healthy):
    obs = torch.randn(4, 8)
    before = healthy(obs)
    killed = apply_damage(healthy, 0.0, seed=0)
    assert killed == {"h1": [], "h2": []}
    assert torch.equal(healthy(obs), before)


def test_does_not_touch_the_weights(healthy):
    before = healthy.fc1.weight.clone()
    apply_damage(healthy, 0.5, seed=0)
    assert torch.equal(healthy.fc1.weight, before)


def test_chooses_the_smallest_fraction_that_qualifies():
    ratios = {0.3: [0.5, 0.3, 0.2], 0.1: [0.9, 0.8, 0.7], 0.2: [0.6, 0.5, 0.5]}
    assert choose_damage_level(ratios, max_ratio=0.4, fallback=0.5) == (0.3, False)


def test_uses_the_median_not_the_mean():
    ratios = {0.1: [0.0, 0.3, 0.3, 0.3, 3.0]}
    assert choose_damage_level(ratios, max_ratio=0.4, fallback=0.5) == (0.1, False)


def test_a_median_equal_to_the_limit_qualifies():
    assert choose_damage_level({0.2: [0.4]}, max_ratio=0.4, fallback=0.5) == (0.2, False)


def test_falls_back_when_nothing_qualifies():
    ratios = {0.1: [0.9], 0.5: [0.8]}
    assert choose_damage_level(ratios, max_ratio=0.4, fallback=0.5) == (0.5, True)
