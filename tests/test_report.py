import numpy as np

from robotrehab.report import build_summary, plot_recovery, render_markdown
from robotrehab.video import compose, label


def records(repaired=2500.0, damaged=1000.0):
    rows = [{"robot": r, "pattern": -1, "condition": "healthy", "memory": "large", "return": 3000.0}
            for r in range(3)]
    for robot in range(3):
        for pattern in range(100, 105):
            rows.append({"robot": robot, "pattern": pattern, "condition": "none", "memory": "large",
                         "return": damaged})
            for condition, memory in (("mutation", "large"), ("ridge", "large"), ("asr", "large"),
                                      ("bc", "large"), ("mutation", "small")):
                rows.append({"robot": robot, "pattern": pattern, "condition": condition,
                             "memory": memory, "return": repaired})
    return rows


META = {"smoke": False, "level": {"fraction": 0.3, "used_fallback": False}, "video": True}


def test_summary_applies_the_rule():
    summary = build_summary(records(), 200)
    assert summary["conditions"]["mutation/large"]["median"] == 0.75
    assert summary["conditions"]["mutation/large"]["n_cases"] == 15
    assert summary["verdict"] == "seguir"


def test_median_case_is_chosen_from_the_main_method():
    case = build_summary(records(), 200)["median_case"]
    assert case["robot"] in range(3)
    assert case["pattern"] in range(100, 105)


def test_more_than_five_excluded_cases_warn():
    summary = build_summary(records(damaged=2900.0), 200)
    assert len(summary["excluded"]) == 15
    assert summary["warnings"]
    assert summary["verdict"] == "sin_datos"
    assert summary["median_case"] is None


def test_report_shows_verdict_and_table():
    text = render_markdown(build_summary(records(), 200), META)
    assert "Seguir" in text
    assert "Mutación con memoria" in text
    assert "SIN VALIDEZ" not in text


def test_smoke_report_is_marked():
    text = render_markdown(build_summary(records(), 200), dict(META, smoke=True))
    assert "SIN VALIDEZ CIENTÍFICA" in text


def test_report_flags_fallback_and_missing_video():
    meta = {"smoke": False, "level": {"fraction": 0.5, "used_fallback": True}, "video": False}
    text = render_markdown(build_summary(records(), 200), meta)
    assert "valor por defecto" in text
    assert "traj/" in text


def test_figure_is_written_even_without_data(tmp_path):
    plot_recovery(build_summary(records(), 200), tmp_path / "a.png")
    plot_recovery(build_summary(records(damaged=2900.0), 200), tmp_path / "b.png")
    assert (tmp_path / "a.png").stat().st_size > 0
    assert (tmp_path / "b.png").stat().st_size > 0


def test_compose_holds_the_last_frame_of_short_panels():
    long = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(3)]
    short = [np.full((4, 4, 3), 9, dtype=np.uint8)]
    frames = compose([long, short])
    assert len(frames) == 3
    assert frames[2].shape == (4, 8, 3)
    assert frames[2][0, 7, 0] == 9


def test_label_keeps_the_frame_size():
    frame = np.full((64, 64, 3), 255, dtype=np.uint8)
    assert label(frame, "Sano").shape == frame.shape
