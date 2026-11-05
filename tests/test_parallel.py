import math

import pytest

from robotrehab.parallel import run_tasks
from robotrehab.pipeline import StageError


def test_results_keep_task_order_across_processes():
    assert run_tasks(math.sqrt, [16, 4, 1], workers=2) == [4, 2, 1]


def test_a_task_that_fails_once_is_retried():
    calls = []

    def flaky(value):
        calls.append(value)
        if len(calls) == 1:
            raise ValueError("first time")
        return value

    assert run_tasks(flaky, [7], workers=1) == [7]
    assert len(calls) == 2


def test_a_task_that_keeps_failing_stops_the_run():
    def broken(_):
        raise ValueError("always")

    with pytest.raises(StageError, match="fallaron dos veces"):
        run_tasks(broken, [1], workers=1)
