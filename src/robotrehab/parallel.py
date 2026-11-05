"""Run independent tasks across processes, retrying failures once."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import torch

from .pipeline import StageError


def _init_worker() -> None:
    torch.set_num_threads(1)


def _attempt_inline(fn, tasks, pending, results):
    failed, error = [], None
    for index in pending:
        try:
            results[index] = fn(tasks[index])
        except Exception as exc:
            failed.append(index)
            error = exc
    return failed, error


def _attempt_pool(fn, tasks, pending, results, workers):
    failed, error = [], None
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                             initializer=_init_worker) as pool:
        futures = {pool.submit(fn, tasks[index]): index for index in pending}
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                failed.append(index)
                error = exc
    return failed, error


def run_tasks(fn, tasks: list, workers: int, retries: int = 1) -> list:
    """Return results in task order. Raise StageError if a task still fails after `retries`."""
    results, pending, error = {}, list(range(len(tasks))), None
    for _ in range(retries + 1):
        if not pending:
            break
        if workers <= 1:
            pending, error = _attempt_inline(fn, tasks, pending, results)
        else:
            pending, error = _attempt_pool(fn, tasks, pending, results, workers)
    if pending:
        raise StageError(f"{len(pending)} tarea(s) fallaron dos veces. Último error: {error!r}")
    return [results[index] for index in range(len(tasks))]
