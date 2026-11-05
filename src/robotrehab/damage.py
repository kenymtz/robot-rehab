"""Killing hidden units, and choosing how much damage to study."""
import statistics

import torch

from .policy import MaskedPolicy


def apply_damage(policy: MaskedPolicy, fraction: float, seed: int) -> dict[str, list[int]]:
    """Kill `round(fraction * n_units)` random units in each lesionable layer, in place.

    Returns {layer name: indices of the units killed}. The same seed kills the same units.
    """
    generator = torch.Generator().manual_seed(seed)  # shared by all layers, so their patterns differ
    killed = {}
    for name, _, mask in policy.lesionable():
        n_units = mask.mask.numel()
        n_killed = int(round(fraction * n_units))
        indices = torch.randperm(n_units, generator=generator)[:n_killed]
        mask.mask[indices] = 0.0
        killed[name] = indices.tolist()
    return killed


def choose_damage_level(ratios: dict[float, list[float]], max_ratio: float,
                        fallback: float) -> tuple[float, bool]:
    """Pick the smallest fraction whose median ratio is at most `max_ratio`.

    `ratios` maps each damage fraction to its damaged/healthy return ratios.
    Returns (fraction, used_fallback). If no fraction qualifies, returns (fallback, True).
    """
    for fraction in sorted(ratios):
        if statistics.median(ratios[fraction]) <= max_ratio:
            return fraction, False
    return fallback, True
