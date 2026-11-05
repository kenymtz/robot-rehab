"""Repairs. They see only the damaged network and its memory: no simulator, no reward.

Every method returns a new policy and leaves its input untouched. Dead units stay dead.
"""
from .memory import Memory
from .policy import MaskedPolicy


def action_mse(policy: MaskedPolicy, obs, actions) -> float:
    """Mean squared error between the policy's actions and the recorded ones."""
    raise NotImplementedError


def repair_mutation(policy: MaskedPolicy, memory: Memory, budget: int, sigma: float,
                    mutate_fraction: float, seed: int) -> MaskedPolicy:
    """(1+1)-ES hill climbing on `action_mse` over the repair pairs.

    Each step adds gaussian noise of size `sigma` to the incoming weights and biases of a random
    `mutate_fraction` of the living units of each lesionable layer, and of the head's outputs.
    The change stays only if the error goes down. `budget` is the number of steps.
    """
    raise NotImplementedError


def repair_ridge(policy: MaskedPolicy, memory: Memory, lam: float) -> MaskedPolicy:
    """Recompute only the head by ridge regression, from surviving h2 activations.

    The target is atanh(clip(action, -0.999, 0.999)). The bias is not penalized.
    """
    raise NotImplementedError


def repair_asr(policy: MaskedPolicy, memory: Memory, clip: tuple[float, float] = (0.25, 4.0)) -> MaskedPolicy:
    """Layer by layer, rescale each living unit so its activation std matches `memory.unit_std`."""
    raise NotImplementedError


def repair_bc(policy: MaskedPolicy, memory: Memory, lr: float, steps: int, batch_size: int,
              seed: int) -> MaskedPolicy:
    """Adam on the repair pairs. Return the weights with the lowest error on the reserved pairs."""
    raise NotImplementedError


def repair(method: str, policy: MaskedPolicy, memory: Memory, params: dict, seed: int) -> MaskedPolicy:
    """Dispatch on "none", "mutation", "ridge", "asr" or "bc". Unknown names raise ValueError."""
    raise NotImplementedError
