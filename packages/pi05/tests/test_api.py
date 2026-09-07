from __future__ import annotations

import time

import pytest

from vla_subtask_phase_probe.api import TaskApiDeliveryError, TaskSignalDispatcher
from vla_subtask_phase_probe.models import TaskSignal


def test_dispatcher_is_non_blocking_and_preserves_order() -> None:
    class SlowClient:
        def __init__(self) -> None:
            self.signals: list[TaskSignal] = []

        def send(self, signal: TaskSignal) -> None:
            time.sleep(0.02)
            self.signals.append(signal)

    client = SlowClient()
    dispatcher = TaskSignalDispatcher(client)
    signals = [
        TaskSignal("task", "pick", "picking"),
        TaskSignal("task", "place", "placing"),
        TaskSignal("task", "task", "complete"),
    ]
    started = time.perf_counter()
    for signal in signals:
        dispatcher.emit(signal)
    assert time.perf_counter() - started < 0.01
    dispatcher.wait_until_delivered(signals[-1], timeout_seconds=1)
    dispatcher.close(timeout_seconds=1)
    assert client.signals == signals


def test_dispatcher_stops_after_first_failure_without_retry() -> None:
    class FailingClient:
        calls = 0

        def send(self, signal: TaskSignal) -> None:
            self.calls += 1
            raise TaskApiDeliveryError("unavailable")

    client = FailingClient()
    dispatcher = TaskSignalDispatcher(client)
    dispatcher.emit(TaskSignal("task", "pick", "picking"))
    dispatcher.emit(TaskSignal("task", "place", "placing"))
    with pytest.raises(TaskApiDeliveryError, match="unavailable"):
        dispatcher.close(timeout_seconds=1)
    assert client.calls == 1
