"""Recovery metric and the decision rule fixed before seeing results."""
import numpy as np


def recovery(repaired: float, damaged: float, healthy: float) -> float | None:
    """(repaired - damaged) / (healthy - damaged), not clipped.

    Returns None when the case was not degraded: healthy - damaged < 10% of healthy.
    """
    lost = healthy - damaged
    if lost < 0.1 * healthy:
        return None
    return (repaired - damaged) / lost


def bootstrap_median_ci(values: list[float], n_resamples: int, seed: int) -> tuple[float, float, float]:
    """(median, low, high): percentile 95% bootstrap interval of the median, resampling values."""
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    resamples = rng.choice(data, size=(n_resamples, len(data)), replace=True)
    low, high = np.percentile(np.median(resamples, axis=1), [2.5, 97.5])
    return float(np.median(data)), float(low), float(high)


def verdict(median: float, ci_low: float) -> str:
    """"seguir" if median >= 0.5 and ci_low > 0; "investigar" if median >= 0.1; else "explicar"."""
    if median >= 0.5 and ci_low > 0:
        return "seguir"
    if median >= 0.1:
        return "investigar"
    return "explicar"


def diagnosis(medians: dict[str, float]) -> list[str]:
    """Codes from the "mutation", "ridge" and "bc" medians, in this order when they apply.

    "busqueda": bc >= 0.5 and mutation < 0.5.
    "memoria_o_bucle": bc < 0.5.
    "ridge_principal": ridge > mutation.
    """
    codes = []
    if medians["bc"] >= 0.5 and medians["mutation"] < 0.5:
        codes.append("busqueda")
    if medians["bc"] < 0.5:
        codes.append("memoria_o_bucle")
    if medians["ridge"] > medians["mutation"]:
        codes.append("ridge_principal")
    return codes
