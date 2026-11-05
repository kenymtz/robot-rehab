"""Deterministic walking policy whose hidden units can be killed through per-unit masks."""
import copy

import numpy as np
import torch
import torch.nn as nn

HIDDEN = 256


class UnitMask(nn.Module):
    """Multiplies each hidden unit by 0 (dead) or 1 (alive)."""

    def __init__(self, n_units: int):
        super().__init__()
        self.register_buffer("mask", torch.ones(n_units))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mask


class MaskedPolicy(nn.Module):
    """obs -> Linear -> ReLU -> mask h1 -> Linear -> ReLU -> mask h2 -> Linear (head) -> tanh -> action."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.config = {"obs_dim": obs_dim, "act_dim": act_dim, "hidden": hidden}
        self.fc1, self.m1 = nn.Linear(obs_dim, hidden), UnitMask(hidden)
        self.fc2, self.m2 = nn.Linear(hidden, hidden), UnitMask(hidden)
        self.head = nn.Linear(hidden, act_dim)

    def hidden(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Post-mask activations of both hidden layers."""
        h1 = self.m1(torch.relu(self.fc1(obs)))
        h2 = self.m2(torch.relu(self.fc2(h1)))
        return h1, h2

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.hidden(obs)[1]))

    def lesionable(self) -> list[tuple[str, nn.Linear, UnitMask]]:
        """(name, incoming layer, mask) for each layer whose units can die."""
        return [("h1", self.fc1, self.m1), ("h2", self.fc2, self.m2)]

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        return self(torch.as_tensor(obs, dtype=torch.float32)).numpy()


def clone(policy: MaskedPolicy) -> MaskedPolicy:
    return copy.deepcopy(policy)


def from_sac(model) -> MaskedPolicy:
    """Copy the deterministic actor of a Stable-Baselines3 SAC model."""
    actor = model.policy.actor
    first, second = (m for m in actor.latent_pi if isinstance(m, nn.Linear))
    policy = MaskedPolicy(first.in_features, actor.mu.out_features, first.out_features)
    policy.fc1.load_state_dict(first.state_dict())
    policy.fc2.load_state_dict(second.state_dict())
    policy.head.load_state_dict(actor.mu.state_dict())
    return policy.eval()


def save_policy(policy: MaskedPolicy, path) -> None:
    torch.save({"config": policy.config, "state_dict": policy.state_dict()}, path)


def load_policy(path) -> MaskedPolicy:
    data = torch.load(path, map_location="cpu")
    policy = MaskedPolicy(**data["config"])
    policy.load_state_dict(data["state_dict"])
    return policy.eval()
