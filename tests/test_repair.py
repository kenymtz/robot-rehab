import ast
import inspect

import pytest
import torch

import robotrehab.repair as repair_module
from robotrehab.damage import apply_damage
from robotrehab.policy import clone
from robotrehab.repair import (action_mse, repair, repair_asr, repair_bc, repair_mutation,
                               repair_ridge)

METHODS = {
    "none": {},
    "mutation": {"budget": 150, "sigma": 0.05, "mutate_fraction": 0.2},
    "ridge": {"lam": 0.01},
    "asr": {},
    "bc": {"lr": 0.01, "steps": 150, "batch_size": 64},
}


@pytest.fixture
def damaged(healthy):
    policy = clone(healthy)
    apply_damage(policy, 0.3, seed=0)
    return policy


def val_error(policy, memory):
    return action_mse(policy, memory.val_obs, memory.val_actions)


def test_action_mse_is_zero_for_the_healthy_network(healthy, memory):
    assert action_mse(healthy, memory.obs, memory.actions) == pytest.approx(0.0, abs=1e-10)


def test_action_mse_known_value(healthy):
    obs = torch.zeros(2, 8)
    with torch.no_grad():
        actions = healthy(obs) + 0.5
    assert action_mse(healthy, obs, actions) == pytest.approx(0.25)


def test_action_mse_returns_a_float(healthy, memory):
    assert isinstance(action_mse(healthy, memory.obs, memory.actions), float)


def test_mutation_lowers_the_error(damaged, memory):
    repaired = repair_mutation(damaged, memory, budget=300, sigma=0.05, mutate_fraction=0.2, seed=0)
    assert val_error(repaired, memory) < val_error(damaged, memory)


def test_mutation_never_worsens_its_own_objective(damaged, memory):
    repaired = repair_mutation(damaged, memory, budget=50, sigma=5.0, mutate_fraction=1.0, seed=0)
    assert action_mse(repaired, memory.obs, memory.actions) <= action_mse(damaged, memory.obs, memory.actions)


def test_mutation_is_reproducible(damaged, memory):
    first = repair_mutation(damaged, memory, budget=50, sigma=0.05, mutate_fraction=0.2, seed=4)
    second = repair_mutation(damaged, memory, budget=50, sigma=0.05, mutate_fraction=0.2, seed=4)
    assert torch.equal(first.fc1.weight, second.fc1.weight)


def test_mutation_can_change_the_head(damaged, memory):
    repaired = repair_mutation(damaged, memory, budget=300, sigma=0.05, mutate_fraction=0.2, seed=0)
    assert not torch.equal(repaired.head.weight, damaged.head.weight)


def test_mutation_does_not_touch_dead_units_incoming_weights(damaged, memory):
    repaired = repair_mutation(damaged, memory, budget=300, sigma=0.05, mutate_fraction=1.0, seed=0)
    dead = damaged.m1.mask == 0
    assert torch.equal(repaired.fc1.weight[dead], damaged.fc1.weight[dead])


def test_ridge_lowers_the_error(damaged, memory):
    assert val_error(repair_ridge(damaged, memory, lam=0.01), memory) < val_error(damaged, memory)


def test_ridge_only_changes_the_head(damaged, memory):
    repaired = repair_ridge(damaged, memory, lam=0.01)
    assert torch.equal(repaired.fc1.weight, damaged.fc1.weight)
    assert torch.equal(repaired.fc2.weight, damaged.fc2.weight)
    assert not torch.equal(repaired.head.weight, damaged.head.weight)


def test_ridge_recovers_an_undamaged_network(healthy, memory):
    repaired = repair_ridge(healthy, memory, lam=1e-6)
    assert val_error(repaired, memory) < 1e-3


def test_asr_restores_the_activation_spread(damaged, memory):
    repaired = repair_asr(damaged, memory, clip=(0.01, 100.0))
    with torch.no_grad():
        h1, h2 = repaired.hidden(memory.obs)
    for name, activations, mask in (("h1", h1, repaired.m1.mask), ("h2", h2, repaired.m2.mask)):
        std, target = activations.std(0), memory.unit_std[name]
        usable = (mask > 0) & (target > 1e-3) & (std > 1e-6)
        assert usable.any()
        assert torch.allclose(std[usable], target[usable], rtol=0.2)


def test_asr_respects_the_clip(damaged, memory):
    repaired = repair_asr(damaged, memory, clip=(1.0, 1.0))
    assert torch.allclose(repaired.fc1.weight, damaged.fc1.weight)


def test_bc_lowers_the_error(damaged, memory):
    repaired = repair_bc(damaged, memory, lr=0.01, steps=150, batch_size=64, seed=0)
    assert val_error(repaired, memory) < val_error(damaged, memory)


def test_bc_keeps_the_best_weights_on_the_reserved_pairs(damaged, memory):
    repaired = repair_bc(damaged, memory, lr=10.0, steps=30, batch_size=64, seed=0)
    assert val_error(repaired, memory) <= val_error(damaged, memory) + 1e-9


@pytest.mark.parametrize("method", METHODS)
def test_dead_units_stay_dead(method, damaged, memory):
    repaired = repair(method, damaged, memory, METHODS[method], seed=0)
    assert torch.equal(repaired.m1.mask, damaged.m1.mask)
    assert torch.equal(repaired.m2.mask, damaged.m2.mask)
    with torch.no_grad():
        h1, h2 = repaired.hidden(memory.obs)
    assert torch.all(h1[:, damaged.m1.mask == 0] == 0)
    assert torch.all(h2[:, damaged.m2.mask == 0] == 0)


@pytest.mark.parametrize("method", METHODS)
def test_the_input_policy_is_left_untouched(method, damaged, memory):
    before = {key: value.clone() for key, value in damaged.state_dict().items()}
    repaired = repair(method, damaged, memory, METHODS[method], seed=0)
    assert repaired is not damaged
    for key, value in damaged.state_dict().items():
        assert torch.equal(value, before[key])


def test_none_returns_the_same_behavior(damaged, memory):
    repaired = repair("none", damaged, memory, {}, seed=0)
    assert val_error(repaired, memory) == val_error(damaged, memory)


def test_unknown_method_raises(damaged, memory):
    with pytest.raises(ValueError):
        repair("magic", damaged, memory, {}, seed=0)


def test_repair_module_cannot_reach_the_simulator():
    tree = ast.parse(inspect.getsource(repair_module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    forbidden = ("gymnasium", "mujoco", "stable_baselines3", "rollout")
    assert not any(word in name for name in imported for word in forbidden)


def test_repair_functions_do_not_receive_an_environment():
    for name, function in inspect.getmembers(repair_module, inspect.isfunction):
        if function.__module__ == repair_module.__name__:
            parameters = set(inspect.signature(function).parameters)
            assert not parameters & {"env", "environment", "reward", "rewards"}
