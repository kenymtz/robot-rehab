"""Measure how well the offline action error predicts walking in the official run.

Reads runs/full/final/final.json, writes runs/full/offline_error.json and prints a summary.
Usage: uv run --frozen python scripts/offline_error.py
"""
import json
from itertools import permutations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "full"
REPAIRS = ("mutation", "ridge", "asr", "bc")
EDGES = [0.0, 0.005, 0.01, 0.03, 0.06, 1.0]
CLEAR_RATIO = 2.0  # one error must be at least this many times the other to count as different


def ranks(values) -> np.ndarray:
    """Rank from 1 upward. Ties share their average rank."""
    values = np.asarray(values, dtype=float)
    return np.array([(values < v).sum() + ((values == v).sum() + 1) / 2 for v in values])


def spearman(a, b) -> float:
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def bands(rows: list[dict], edges: list[float], attempts: int) -> list[dict]:
    """Share of attempts that stayed standing in each error range. Empty ranges are left out."""
    result = []
    for low, high in zip(edges, edges[1:]):
        inside = [r for r in rows if low <= r["val_mse"] < high]
        if inside:
            result.append({"low": low, "high": high, "cases": len(inside),
                           "standing": sum(r["survivors"] for r in inside) / (attempts * len(inside))})
    return result


def contrast(rows: list[dict]) -> tuple[dict, dict] | None:
    """The pair where clearly more error came with the largest gain in attempts without falling."""
    pairs = [(a, b) for a, b in permutations(rows, 2)
             if a["val_mse"] >= CLEAR_RATIO * b["val_mse"] and a["survivors"] > b["survivors"]]
    return max(pairs, key=lambda p: p[0]["survivors"] - p[1]["survivors"], default=None)


def correlation(rows: list[dict]) -> dict:
    return {"cases": len(rows), "return": spearman([r["val_mse"] for r in rows], [r["return"] for r in rows])}


def compact(row: dict) -> dict:
    return {key: row[key] for key in ("robot", "pattern", "condition", "val_mse", "survivors")}


def main() -> None:
    records = json.loads((RUN / "final" / "final.json").read_text(encoding="utf-8"))
    attempts = json.loads((RUN / "report" / "manifest.json").read_text(encoding="utf-8"))["config"]["final"]["episodes"]
    repaired = [r for r in records if r["condition"] in REPAIRS]
    large = [r for r in repaired if r["memory"] == "large"]
    mutation = [r for r in large if r["condition"] == "mutation"]
    higher, lower = contrast(mutation)
    result = {"attempts": attempts, "all_methods": correlation(large), "mutation": correlation(mutation),
              "bands": bands(repaired, EDGES, attempts), "band_cases": len(repaired),
              "contrast": {"higher": compact(higher), "lower": compact(lower)},
              "cases": [compact(r) for r in large]}
    (RUN / "offline_error.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("todos los métodos:", result["all_methods"])
    print("solo mutación:", result["mutation"])
    for band in result["bands"]:
        print(band)
    print("contraste:", result["contrast"])


if __name__ == "__main__":
    main()
