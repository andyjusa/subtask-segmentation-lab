from __future__ import annotations

import json
import queue
import threading
import time
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import TaskSignal


class TaskApiDeliveryError(RuntimeError):
    pass


class SignalClient(Protocol):
    def send(self, signal: TaskSignal) -> None: ...


class TaskApiClient:
    """POST a signal once; never retry an ambiguously accepted request."""

    def __init__(self, endpoint: str, timeout_seconds: float = 180.0) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def send(self, signal: TaskSignal) -> None:
        request = Request(
            self.endpoint,
            data=json.dumps(signal.payload()).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                status = response.status
        except HTTPError as exc:
            raise TaskApiDeliveryError(f"Task API rejected signal: HTTP {exc.code}") from exc
        except URLError as exc:
            raise TaskApiDeliveryError(f"Task API request failed: {exc.reason}") from exc
        if not 200 <= status < 300:
            raise TaskApiDeliveryError(f"Task API rejected signal: HTTP {status}")


class TaskSignalDispatcher:
    """Deliver signals in order without blocking the robot control loop."""

    def __init__(self, client: SignalClient) -> None:
        self._client = client
        self._queue: queue.Queue[TaskSignal | None] = queue.Queue()
        self._errors: list[Exception] = []
        self._sent: list[TaskSignal] = []
        self._closed = False
        self._condition = threading.Condition()
        self._worker = threading.Thread(target=self._run, name="task-api", daemon=True)
        self._worker.start()

    @property
    def sent(self) -> tuple[TaskSignal, ...]:
        with self._condition:
            return tuple(self._sent)

    def emit(self, signal: TaskSignal) -> None:
        if self._closed:
            raise RuntimeError("dispatcher is closed")
        self._queue.put_nowait(signal)

    def wait_until_delivered(
        self, signal: TaskSignal, timeout_seconds: float | None = None
    ) -> None:
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        with self._condition:
            while signal not in self._sent and not self._errors:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("signal was not delivered before timeout")
                self._condition.wait(remaining)
            if self._errors:
                raise TaskApiDeliveryError(str(self._errors[0])) from self._errors[0]

    def close(self, timeout_seconds: float | None = None) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(None)
        self._worker.join(timeout_seconds)
        if self._worker.is_alive():
            raise TimeoutError("delivery did not finish before timeout")
        if self._errors:
            raise TaskApiDeliveryError(str(self._errors[0])) from self._errors[0]

    def _run(self) -> None:
        while True:
            signal = self._queue.get()
            try:
                if signal is None:
                    return
                if self._errors:
                    continue
                self._client.send(signal)
                with self._condition:
                    self._sent.append(signal)
                    self._condition.notify_all()
            except Exception as exc:
                with self._condition:
                    self._errors.append(exc)
                    self._condition.notify_all()
            finally:
                self._queue.task_done()
