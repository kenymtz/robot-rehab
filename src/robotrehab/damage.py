"""Killing hidden units, and choosing how much damage to study."""
from .policy import MaskedPolicy


def apply_damage(policy: MaskedPolicy, fraction: float, seed: int) -> dict[str, list[int]]:
    """Kill `round(fraction * n_units)` random units in each lesionable layer, in place.

    Returns {layer name: indices of the units killed}. The same seed kills the same units.
    """
    raise NotImplementedError


def choose_damage_level(ratios: dict[float, list[float]], max_ratio: float,
                        fallback: float) -> tuple[float, bool]:
    """Pick the smallest fraction whose median ratio is at most `max_ratio`.

    `ratios` maps each damage fraction to its damaged/healthy return ratios.
    Returns (fraction, used_fallback). If no fraction qualifies, returns (fallback, True).
    """
    raise NotImplementedError
