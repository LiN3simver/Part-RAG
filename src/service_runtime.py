"""Bounded request admission and blocking work for the API process."""
from __future__ import annotations

import asyncio
import contextvars
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from fastapi import HTTPException
from starlette.responses import JSONResponse


def busy() -> HTTPException:
    return HTTPException(503, "查詢容量已滿，請稍後重試", headers={"Retry-After": "2"})


class WorkPool:
    """A timeout never frees capacity while the underlying thread still runs."""

    def __init__(self, workers: int = 4, waiting: int = 4, timeout: float = 30):
        if workers < 1 or waiting < 0 or timeout <= 0:
            raise ValueError("invalid query capacity or timeout")
        self._slots = threading.BoundedSemaphore(workers + waiting)
        self._executor = ThreadPoolExecutor(workers, thread_name_prefix="rag-query")
        self.timeout = timeout
        self.workers, self.waiting = workers, waiting
        self._lock = threading.Lock()
        self._occupied = 0

    async def run(self, fn, *args, timeout: float | None = None):
        wait = self.timeout if timeout is None else min(self.timeout, timeout)
        if wait <= 0:
            raise HTTPException(504, "查詢超過時間上限")
        if not self._slots.acquire(blocking=False):
            raise busy()
        with self._lock:
            self._occupied += 1
        def release(_=None):
            with self._lock:
                self._occupied -= 1
            self._slots.release()
        try:
            future = self._executor.submit(contextvars.copy_context().run, partial(fn, *args))
        except BaseException:
            release()
            raise
        future.add_done_callback(release)
        wrapped = asyncio.wrap_future(future)
        # Consume a late failure after the HTTP caller has timed out/cancelled.
        wrapped.add_done_callback(lambda f: None if f.cancelled() else f.exception())
        try:
            return await asyncio.wait_for(asyncio.shield(wrapped), wait)
        except TimeoutError:
            future.cancel()  # only cancels work which has not started
            raise HTTPException(504, "檢索超過時間上限，請縮小查詢條件") from None
        except asyncio.CancelledError:
            future.cancel()
            raise

    def close(self):
        self._executor.shutdown(wait=False, cancel_futures=True)

    def stats(self):
        with self._lock:
            return {"workers": self.workers, "waiting_limit": self.waiting,
                    "occupied": self._occupied, "timeout_seconds": self.timeout}


class AdmissionMiddleware:
    """Pure ASGI: retain SSE disconnect semantics and reserve health capacity."""

    def __init__(self, app, limit: int = 8, max_body: int = 262144):
        if limit < 1:
            raise ValueError("QUERY_MAX_REQUESTS must be positive")
        self.app = app
        self._slots = threading.BoundedSemaphore(limit)
        self.max_body = max_body

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] == "http" and path.startswith(("/api/", "/a2a/")):
            headers = dict(scope.get("headers", []))
            try:
                content_length = int(headers.get(b"content-length", b"0"))
            except ValueError:
                content_length = self.max_body + 1
            if content_length > self.max_body:
                response = JSONResponse({"detail": "請求內容超過大小上限"}, 413)
                return await response(scope, receive, send)
            original_receive = receive
            received = 0
            body_complete = False
            started = scope.get("state", {}).get("rag_started", time.monotonic())

            async def bounded_receive():
                nonlocal received, body_complete
                # A slow body must not occupy all message slots indefinitely.
                # Stop applying this timeout after the body ends: subsequent
                # receives belong to the SSE disconnect listener.
                if not body_complete and path in {"/api/chat", "/a2a/message"}:
                    left = started + float(os.getenv("MESSAGE_TIMEOUT_SECONDS", "180")) - time.monotonic()
                    if left <= 0:
                        raise HTTPException(408, "請求內容接收超過時間上限")
                    try:
                        message = await asyncio.wait_for(original_receive(), left)
                    except TimeoutError:
                        raise HTTPException(408, "請求內容接收超過時間上限") from None
                else:
                    message = await original_receive()
                if message["type"] == "http.request":
                    received += len(message.get("body", b""))
                    if received > self.max_body:
                        raise HTTPException(413, "請求內容超過大小上限")
                    body_complete = not message.get("more_body", False)
                return message

            receive = bounded_receive
        limited = scope["type"] == "http" and (
            path in {"/api/search", "/api/chat", "/api/export", "/a2a/message", "/a2a/search"}
            or path.startswith(("/api/part/", "/api/similar/", "/api/values/", "/a2a/tool/"))
        )
        if not limited:
            return await self.app(scope, receive, send)
        if not self._slots.acquire(blocking=False):
            response = JSONResponse({"detail": "查詢容量已滿，請稍後重試"}, 503,
                                    headers={"Retry-After": "2"})
            return await response(scope, receive, send)
        try:
            await self.app(scope, receive, send)
        finally:
            self._slots.release()


def query_pool() -> WorkPool:
    return WorkPool(int(os.getenv("QUERY_WORKERS", "4")),
                    int(os.getenv("QUERY_MAX_WAITING", "4")),
                    float(os.getenv("QUERY_TIMEOUT_SECONDS", "30")))
