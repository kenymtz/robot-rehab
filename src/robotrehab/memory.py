"""The only record a repair may use: what the healthy robot saw and did. It never holds rewards."""
from dataclasses import asdict, dataclass

import torch

from .policy import MaskedPolicy


@dataclass
class Memory:
    obs: torch.Tensor          # pairs available to the repair
    actions: torch.Tensor
    val_obs: torch.Tensor      # pairs reserved for choosing hyperparameters
    val_actions: torch.Tensor
    unit_mean: dict[str, torch.Tensor]  # healthy activation statistics per layer
    unit_std: dict[str, torch.Tensor]


@torch.no_grad()
def build_memory(policy: MaskedPolicy, obs: torch.Tensor, actions: torch.Tensor,
                 n_repair: int, seed: int = 0) -> Memory:
    """Split recorded pairs and store healthy statistics. `policy` must be the healthy network."""
    order = torch.randperm(len(obs), generator=torch.Generator().manual_seed(seed))
    keep, reserved = order[:n_repair], order[n_repair:]
    h1, h2 = policy.hidden(obs)
    return Memory(
        obs=obs[keep], actions=actions[keep], val_obs=obs[reserved], val_actions=actions[reserved],
        unit_mean={"h1": h1.mean(0), "h2": h2.mean(0)},
        unit_std={"h1": h1.std(0), "h2": h2.std(0)},
    )


def subsample(memory: Memory, n_total: int, n_repair: int, seed: int = 0) -> Memory:
    """Smaller memory with the same statistics and proportion of reserved pairs."""
    gen = torch.Generator().manual_seed(seed)
    keep = torch.randperm(len(memory.obs), generator=gen)[:n_repair]
    reserved = torch.randperm(len(memory.val_obs), generator=gen)[:n_total - n_repair]
    return Memory(memory.obs[keep], memory.actions[keep], memory.val_obs[reserved],
                  memory.val_actions[reserved], memory.unit_mean, memory.unit_std)


def save_memory(memory: Memory, path) -> None:
    torch.save(asdict(memory), path)


def load_memory(path) -> Memory:
    return Memory(**torch.load(path, map_location="cpu"))
