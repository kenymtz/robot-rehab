import pytest

from robotrehab.pipeline import Stage, StageError, fingerprint, read_json, run_pipeline


@pytest.fixture
def package(tmp_path):
    directory = tmp_path / "pkg"
    directory.mkdir()
    (directory / "a.py").write_text("a = 1")
    (directory / "b.py").write_text("b = 1")
    return directory


def recorder(calls, name):
    def run(ctx):
        calls.append(name)
        (ctx.out / "result.txt").write_text(name)
    return run


def stages(calls):
    return [Stage("first", recorder(calls, "first"), ("x",), ("a.py",)),
            Stage("second", recorder(calls, "second"), ("y",), ("b.py",))]


def run(stage_list, package, root, cfg=None):
    run_pipeline(stage_list, cfg or {"x": 1, "y": 1}, root, 1, lambda _: None, package)


def test_stages_run_in_order_and_leave_results(package, tmp_path):
    calls = []
    run(stages(calls), package, tmp_path / "r")
    assert calls == ["first", "second"]
    assert (tmp_path / "r" / "second" / "result.txt").read_text() == "second"
    assert not (tmp_path / "r" / "first.tmp").exists()


def test_second_run_repeats_nothing(package, tmp_path):
    calls = []
    run(stages(calls), package, tmp_path / "r")
    calls.clear()
    run(stages(calls), package, tmp_path / "r")
    assert calls == []


def test_changing_later_code_keeps_earlier_stages(package, tmp_path):
    calls = []
    run(stages(calls), package, tmp_path / "r")
    calls.clear()
    (package / "b.py").write_text("b = 2")
    run(stages(calls), package, tmp_path / "r")
    assert calls == ["second"]


def test_changing_earlier_code_repeats_what_follows(package, tmp_path):
    calls = []
    run(stages(calls), package, tmp_path / "r")
    calls.clear()
    (package / "a.py").write_text("a = 2")
    run(stages(calls), package, tmp_path / "r")
    assert calls == ["first", "second"]


def test_changing_config_section_repeats_that_stage(package, tmp_path):
    calls = []
    run(stages(calls), package, tmp_path / "r")
    calls.clear()
    run(stages(calls), package, tmp_path / "r", {"x": 1, "y": 2})
    assert calls == ["second"]


def test_interrupted_stage_resumes_from_its_partial_work(package, tmp_path):
    attempts = []

    def flaky(ctx):
        attempts.append(1)
        progress = ctx.out / "progress.txt"
        if not progress.exists():
            progress.write_text("half")
            raise KeyboardInterrupt
        (ctx.out / "result.txt").write_text(progress.read_text() + "+rest")

    stage_list = [Stage("flaky", flaky, ("x",), ("a.py",))]
    with pytest.raises(KeyboardInterrupt):
        run(stage_list, package, tmp_path / "r")
    assert not (tmp_path / "r" / "flaky").exists()
    run(stage_list, package, tmp_path / "r")
    assert len(attempts) == 2
    assert (tmp_path / "r" / "flaky" / "result.txt").read_text() == "half+rest"


def test_partial_work_from_old_code_is_discarded(package, tmp_path):
    def fail(ctx):
        (ctx.out / "old.txt").write_text("old")
        raise StageError("stop")

    with pytest.raises(StageError):
        run([Stage("s", fail, ("x",), ("a.py",))], package, tmp_path / "r")
    (package / "a.py").write_text("a = 2")
    seen = []
    run([Stage("s", lambda ctx: seen.append((ctx.out / "old.txt").exists()), ("x",), ("a.py",))],
        package, tmp_path / "r")
    assert seen == [False]


def test_always_stage_runs_every_time_and_does_not_invalidate_the_rest(package, tmp_path):
    calls = []
    stage_list = [Stage("check", recorder(calls, "check"), ("x",), ("a.py",), always=True),
                  Stage("work", recorder(calls, "work"), ("y",), ("b.py",))]
    run(stage_list, package, tmp_path / "r")
    calls.clear()
    (package / "a.py").write_text("a = 2")
    run(stage_list, package, tmp_path / "r")
    assert calls == ["check"]


def test_done_file_stores_the_fingerprint(package, tmp_path):
    stage = Stage("first", recorder([], "first"), ("x",), ("a.py",))
    run([stage], package, tmp_path / "r")
    expected = fingerprint(stage, {"x": 1, "y": 1}, "", package)
    assert read_json(tmp_path / "r" / "first" / "done.json")["fingerprint"] == expected
