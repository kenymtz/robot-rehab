"""Recovery metric and the decision rule fixed before seeing results."""


def recovery(repaired: float, damaged: float, healthy: float) -> float | None:
    """(repaired - damaged) / (healthy - damaged), not clipped.

    Returns None when the case was not degraded: healthy - damaged < 10% of healthy.
    """
    raise NotImplementedError


def bootstrap_median_ci(values: list[float], n_resamples: int, seed: int) -> tuple[float, float, float]:
    """(median, low, high): percentile 95% bootstrap interval of the median, resampling values."""
    raise NotImplementedError


def verdict(median: float, ci_low: float) -> str:
    """"seguir" if median >= 0.5 and ci_low > 0; "investigar" if median >= 0.1; else "explicar"."""
    raise NotImplementedError


def diagnosis(medians: dict[str, float]) -> list[str]:
    """Codes from the "mutation", "ridge" and "bc" medians, in this order when they apply.

    "busqueda": bc >= 0.5 and mutation < 0.5.
    "memoria_o_bucle": bc < 0.5.
    "ridge_principal": ridge > mutation.
    """
    raise NotImplementedError
