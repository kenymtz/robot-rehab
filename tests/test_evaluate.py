import pytest

from robotrehab.evaluate import bootstrap_median_ci, diagnosis, recovery, verdict


def test_full_recovery_is_one():
    assert recovery(3000, 1000, 3000) == pytest.approx(1.0)


def test_no_recovery_is_zero():
    assert recovery(1000, 1000, 3000) == pytest.approx(0.0)


def test_half_recovery():
    assert recovery(2000, 1000, 3000) == pytest.approx(0.5)


def test_recovery_is_not_clipped():
    assert recovery(500, 1000, 3000) == pytest.approx(-0.25)
    assert recovery(3500, 1000, 3000) == pytest.approx(1.25)


def test_a_case_without_degradation_is_excluded():
    assert recovery(3000, 2800, 3000) is None


def test_exactly_ten_percent_degradation_still_counts():
    assert recovery(2950, 2700, 3000) == pytest.approx(250 / 300)


def test_bootstrap_median_is_the_sample_median():
    median, low, high = bootstrap_median_ci([1.0, 2.0, 3.0, 4.0, 100.0], 2000, seed=0)
    assert median == pytest.approx(3.0)
    assert low <= median <= high


def test_bootstrap_interval_is_narrow_for_identical_values():
    assert bootstrap_median_ci([0.5] * 15, 500, seed=0) == pytest.approx((0.5, 0.5, 0.5))


def test_bootstrap_interval_contains_the_spread_of_the_data():
    values = [float(v) for v in range(15)]
    median, low, high = bootstrap_median_ci(values, 2000, seed=0)
    assert 0 < low < median < high < 14


def test_bootstrap_is_reproducible():
    values = [0.1, 0.4, 0.2, 0.9, 0.7]
    assert bootstrap_median_ci(values, 500, seed=3) == bootstrap_median_ci(values, 500, seed=3)


@pytest.mark.parametrize("median, low, expected", [
    (0.6, 0.2, "seguir"),
    (0.5, 0.01, "seguir"),
    (0.6, 0.0, "investigar"),
    (0.6, -0.1, "investigar"),
    (0.49, 0.2, "investigar"),
    (0.1, -0.5, "investigar"),
    (0.09, 0.05, "explicar"),
    (-0.2, -0.5, "explicar"),
])
def test_verdict(median, low, expected):
    assert verdict(median, low) == expected


def test_diagnosis_search_is_the_problem():
    assert diagnosis({"mutation": 0.2, "ridge": 0.1, "bc": 0.7}) == ["busqueda"]


def test_diagnosis_memory_or_loop_is_the_limit():
    assert diagnosis({"mutation": 0.2, "ridge": 0.1, "bc": 0.3}) == ["memoria_o_bucle"]


def test_diagnosis_ridge_wins():
    assert diagnosis({"mutation": 0.6, "ridge": 0.8, "bc": 0.7}) == ["ridge_principal"]


def test_diagnosis_combines_codes_in_order():
    assert diagnosis({"mutation": 0.2, "ridge": 0.3, "bc": 0.7}) == ["busqueda", "ridge_principal"]


def test_diagnosis_empty_when_all_is_well():
    assert diagnosis({"mutation": 0.7, "ridge": 0.5, "bc": 0.8}) == []
