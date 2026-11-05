import pytest

from robotrehab.pipeline import StageError
from robotrehab.stages.gate import passes
from robotrehab.stages.preflight import check_disk, check_simulator, estimate_hours
from robotrehab.stages.train import train_to


def test_disk_check_stops_when_space_is_short(tmp_path):
    with pytest.raises(StageError, match="GB libres"):
        check_disk(tmp_path, 10**9)


def test_simulator_check_passes_for_walker():
    check_simulator("Walker2d-v5")


def test_simulator_check_explains_a_bad_environment():
    with pytest.raises(StageError, match="simulador"):
        check_simulator("NoSuchEnv-v0")


def test_estimate_counts_waves_of_robots():
    assert estimate_hours(0.01, 360000, robots=3, workers=2) == pytest.approx(2.0)
    assert estimate_hours(0.01, 360000, robots=3, workers=8) == pytest.approx(1.0)


def test_gate_requires_both_return_and_survivors():
    section = {"min_return": 3000, "min_survivors": 18}
    assert passes({"return": 3000, "survivors": 18}, section)
    assert not passes({"return": 2999, "survivors": 20}, section)
    assert not passes({"return": 4000, "survivors": 17}, section)


def test_training_resumes_where_it_stopped(tmp_path):
    section = {"learning_starts": 10, "checkpoint_every": 50}
    assert train_to(tmp_path, "Walker2d-v5", 0, 100, section) == 100
    assert (tmp_path / "buffer.pkl").exists()
    assert train_to(tmp_path, "Walker2d-v5", 0, 150, section) == 150


