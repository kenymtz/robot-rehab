from robotrehab.config import load_config
from robotrehab.pipeline import PROJECT_ROOT
from robotrehab.stages.final import CONDITIONS, build_tasks, case_name
from robotrehab.stages.tune import param_grid


def test_param_grid_combines_lists_and_keeps_scalars():
    grid = param_grid({"sigma": [1, 2], "fraction": [3, 4], "budget": 9})
    assert len(grid) == 4
    assert {"sigma": 2, "fraction": 3, "budget": 9} in grid


def test_final_has_15_cases_per_condition_with_the_full_config():
    cfg = load_config(PROJECT_ROOT / "configs" / "full.yaml")
    tasks = build_tasks(cfg, {})
    mutation = [t for t in tasks if t["condition"] == "mutation" and t["memory"] == "large"]
    assert len(mutation) == 15
    assert len([t for t in tasks if t["condition"] == "healthy"]) == 3
    assert len(tasks) == 3 + 15 * len(CONDITIONS)


def test_case_names_are_unique():
    assert case_name(0, 100, "bc", "large") != case_name(0, 100, "bc", "small")
