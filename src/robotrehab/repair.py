"""Repairs. They see only the damaged network and its memory: no simulator, no reward.

Every method returns a new policy and leaves its input untouched. Dead units stay dead.
"""
import torch

from .memory import Memory
from .policy import MaskedPolicy, clone


@torch.no_grad()
def action_mse(policy: MaskedPolicy, obs, actions) -> float:
    """Mean squared error between the policy's actions and the recorded ones."""
    return torch.mean((policy(obs) - actions) ** 2).item()


def _mutable_rows(policy: MaskedPolicy) -> list[tuple[torch.nn.Linear, torch.Tensor]]:
    """(layer, rows that may change): living units of each hidden layer, and every head output."""
    targets = [(layer, (mask.mask > 0).nonzero().flatten()) for _, layer, mask in policy.lesionable()]
    targets.append((policy.head, torch.arange(policy.head.out_features)))
    return targets


def _mutate(targets, sigma: float, mutate_fraction: float, generator: torch.Generator) -> list:
    """Add noise to a random subset of rows. Returns what is needed to undo it."""
    undo = []
    for layer, rows in targets:
        count = max(1, int(mutate_fraction * len(rows)))
        chosen = rows[torch.randperm(len(rows), generator=generator)[:count]]
        for param in (layer.weight, layer.bias):
            undo.append((param, chosen, param[chosen].clone()))
            param[chosen] += sigma * torch.randn(param[chosen].shape, generator=generator)
    return undo


@torch.no_grad()
def repair_mutation(policy: MaskedPolicy, memory: Memory, budget: int, sigma: float,
                    mutate_fraction: float, seed: int) -> MaskedPolicy:
    """(1+1)-ES hill climbing on `action_mse` over the repair pairs.

    Each step adds gaussian noise of size `sigma` to the incoming weights and biases of a random
    `mutate_fraction` of the living units of each lesionable layer, and of the head's outputs.
    The change stays only if the error goes down. `budget` is the number of steps.
    """
    policy = clone(policy)
    generator = torch.Generator().manual_seed(seed)
    targets = _mutable_rows(policy)
    best = action_mse(policy, memory.obs, memory.actions)
    for _ in range(budget):
        undo = _mutate(targets, sigma, mutate_fraction, generator)
        error = action_mse(policy, memory.obs, memory.actions)
        if error < best:
            best = error
        else:
            for param, rows, old in undo:
                param[rows] = old
    return policy


@torch.no_grad()
def repair_ridge(policy: MaskedPolicy, memory: Memory, lam: float) -> MaskedPolicy:
    """Recompute only the head by ridge regression, from surviving h2 activations.

    The target is atanh(clip(action, -0.999, 0.999)). The bias is not penalized.
    """
    policy = clone(policy)
    alive = policy.m2.mask > 0
    features = policy.hidden(memory.obs)[1][:, alive].double()
    design = torch.cat([features, torch.ones(len(features), 1, dtype=torch.double)], dim=1)
    target = torch.atanh(memory.actions.clamp(-0.999, 0.999)).double()
    penalty = lam * torch.eye(design.shape[1], dtype=torch.double)
    penalty[-1, -1] = 0.0  # the last column is the bias
    solution = torch.linalg.solve(design.T @ design + penalty, design.T @ target)
    policy.head.weight[:, alive] = solution[:-1].T.float()
    policy.head.bias.copy_(solution[-1].float())
    return policy


@torch.no_grad()
def repair_asr(policy: MaskedPolicy, memory: Memory, clip: tuple[float, float] = (0.25, 4.0)) -> MaskedPolicy:
    """Layer by layer, rescale each living unit so its activation std matches `memory.unit_std`."""
    policy = clone(policy)
    for index, (name, layer, mask) in enumerate(policy.lesionable()):
        current = policy.hidden(memory.obs)[index].std(0)  # measured after the previous layer changed
        scale = (memory.unit_std[name] / (current + 1e-6)).clamp(*clip)
        scale = torch.where(mask.mask > 0, scale, torch.ones_like(scale))
        layer.weight.mul_(scale.view(-1, 1))
        layer.bias.mul_(scale)
    return policy


def repair_bc(policy: MaskedPolicy, memory: Memory, lr: float, steps: int, batch_size: int,
              seed: int) -> MaskedPolicy:
    """Adam on the repair pairs. Return the weights with the lowest error on the reserved pairs."""
    policy = clone(policy)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    best = action_mse(policy, memory.val_obs, memory.val_actions)
    best_state = {key: value.clone() for key, value in policy.state_dict().items()}
    for _ in range(steps):
        batch = torch.randint(len(memory.obs), (batch_size,), generator=generator)
        optimizer.zero_grad()
        torch.mean((policy(memory.obs[batch]) - memory.actions[batch]) ** 2).backward()
        optimizer.step()
        error = action_mse(policy, memory.val_obs, memory.val_actions)
        if error < best:
            best = error
            best_state = {key: value.clone() for key, value in policy.state_dict().items()}
    policy.load_state_dict(best_state)
    return policy


def repair(method: str, policy: MaskedPolicy, memory: Memory, params: dict, seed: int) -> MaskedPolicy:
    """Dispatch on "none", "mutation", "ridge", "asr" or "bc". Unknown names raise ValueError."""
    if method == "none":
        return clone(policy)
    if method == "mutation":
        return repair_mutation(policy, memory, params["budget"], params["sigma"],
                               params["mutate_fraction"], seed)
    if method == "ridge":
        return repair_ridge(policy, memory, params["lam"])
    if method == "asr":
        return repair_asr(policy, memory)
    if method == "bc":
        return repair_bc(policy, memory, params["lr"], params["steps"], params["batch_size"], seed)
    raise ValueError(f"unknown repair method: {method}")
