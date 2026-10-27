"""One monotonic budget for a message, including queueing and answer retries."""
from __future__ import annotations

import contextvars
import os
import threading
import time
from contextlib import contextmanager

from fastapi import HTTPException


class RequestStopped(BaseException):
    # Deliberately not Exception: fallback/rewrite loops must not retry a stopped turn.
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class Deadline:
    def __init__(self, seconds: float, started: float | None = None):
        if not 0 < seconds <= 3600:
            raise ValueError("MESSAGE_TIMEOUT_SECONDS must be in (0, 3600]")
        self.ends = (time.monotonic() if started is None else started) + seconds
        self.stopped = threading.Event()
        self.reason: str | None = None
        self._lock = threading.Lock()

    def stop(self, reason: str):
        with self._lock:
            if not self.stopped.is_set():
                self.reason = reason
                self.stopped.set()

    def remaining(self) -> float:
        if self.stopped.is_set():
            raise RequestStopped(self.reason or "cancelled")
        left = self.ends - time.monotonic()
        if left <= 0:
            self.stop("timeout")
            raise RequestStopped("timeout")
        return left


current: contextvars.ContextVar[Deadline | None] = contextvars.ContextVar("rag_deadline", default=None)


@contextmanager
def using(deadline: Deadline | None):
    token = current.set(deadline)
    try:
        yield
    finally:
        current.reset(token)


def checkpoint():
    deadline = current.get()
    if deadline is not None:
        deadline.remaining()


def client_for_request(client):
    deadline = current.get()
    if deadline is None:
        return client
    # SDK retries and later rewrites all share the original deadline.
    from .observability import request_id
    options = {"timeout": deadline.remaining(), "max_retries": 0}
    if request_id():
        options["default_headers"] = {"X-Request-ID": request_id()}
    return client.with_options(**options)


def for_message(request) -> Deadline:
    return Deadline(float(os.getenv("MESSAGE_TIMEOUT_SECONDS", "180")),
                    getattr(request.state, "rag_started", None))


def remaining_http(deadline: Deadline) -> float:
    try:
        return deadline.remaining()
    except RequestStopped:
        raise HTTPException(504, "回答超過時間上限，請縮小問題後重試") from None


def install_gateway_budget():
    """Adapt the pinned gateway at runtime; never mutate the shared vendor files."""
    from agentkit import gateway
    original = gateway.client
    if getattr(original, "_rag_budget", False):
        return

    def scoped_client():
        from .gateway_compat import guard_client
        checkpoint()
        return client_for_request(guard_client(original()))

    scoped_client._rag_budget = True
    gateway.client = scoped_client
