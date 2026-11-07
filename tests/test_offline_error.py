import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "offline_error.py"
spec = importlib.util.spec_from_file_location("offline_error", SCRIPT)
offline_error = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline_error)


def case(robot, error, survivors, condition="mutation", memory="large", pattern=100):
    return {"robot": robot, "pattern": pattern, "condition": condition, "memory": memory,
            "val_mse": error, "survivors": survivors, "return": survivors * 100.0}


def test_spearman_reads_order_only():
    assert offline_error.spearman([1, 2, 3, 4], [10, 200, 3000, 40000]) == pytest.approx(1.0)
    assert offline_error.spearman([1, 2, 3, 4], [9, 7, 5, 1]) == pytest.approx(-1.0)


def test_spearman_gives_ties_the_same_rank():
    assert offline_error.spearman([1, 1, 2], [1, 1, 2]) == pytest.approx(1.0)


def test_bands_count_attempts_and_skip_empty_ranges():
    rows = [case(0, 0.001, 20), case(0, 0.002, 10), case(1, 0.5, 0)]
    bands = offline_error.bands(rows, [0.0, 0.005, 0.01, 1.0], attempts=20)
    assert bands == [{"low": 0.0, "high": 0.005, "cases": 2, "standing": 0.75},
                     {"low": 0.01, "high": 1.0, "cases": 1, "standing": 0.0}]


def test_contrast_finds_more_error_with_more_walking():
    rows = [case(0, 0.02, 18), case(2, 0.008, 2), case(1, 0.03, 1), case(1, 0.002, 19)]
    higher, lower = offline_error.contrast(rows)
    assert (higher["robot"], lower["robot"]) == (0, 2)


def test_contrast_needs_the_error_to_differ_clearly():
    assert offline_error.contrast([case(0, 0.011, 18), case(2, 0.010, 2)]) is None
