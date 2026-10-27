"""Bounded synchronous SSE producers with async deadlines and disconnects."""
from __future__ import annotations

import asyncio
import contextvars
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import anyio
from fastapi.responses import StreamingResponse

from . import observability
from .request_budget import Deadline, RequestStopped, using
from .service_runtime import busy

_END = object()


class PersistenceError(RuntimeError):
    pass


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


class StreamPool:
    def __init__(self, workers: int = 4, buffer: int = 8):
        if workers < 1 or buffer < 1:
            raise ValueError("invalid stream capacity/buffer")
        self.workers, self.buffer = workers, buffer
        self._executor = ThreadPoolExecutor(workers, thread_name_prefix="rag-stream")
        self._lock = threading.Lock()
        self._jobs: set[StreamJob] = set()
        self._closed = False

    def reserve(self, deadline: Deadline):
        with self._lock:
            if self._closed or len(self._jobs) >= self.workers:
                raise busy()
            job = StreamJob(self, deadline)
            self._jobs.add(job)
            return job

    def _release(self, job):
        with self._lock:
            self._jobs.discard(job)

    def stats(self):
        with self._lock:
            return {"limit": self.workers, "occupied": len(self._jobs),
                    "detached": sum(j.deadline.reason in {"timeout", "disconnected", "shutdown"}
                                    for j in self._jobs),
                    "buffer_events": self.buffer}

    def close(self):
        with self._lock:
            self._closed = True
            jobs = list(self._jobs)
        for job in jobs:
            job.stop("shutdown")
        self._executor.shutdown(wait=False, cancel_futures=False)


class StreamJob:
    def __init__(self, pool: StreamPool, deadline: Deadline):
        self.pool, self.deadline = pool, deadline
        self.queue = asyncio.Queue(maxsize=pool.buffer)
        self.loop = asyncio.get_running_loop()
        self.future = None
        self._reported = False
        self.outcome = None

    def start(self, factory, cleanup=None):
        context = contextvars.copy_context()
        try:
            self.future = self.pool._executor.submit(context.run, self._produce, factory, cleanup)
        except BaseException:
            self.pool._release(self)
            raise
        self.future.add_done_callback(lambda _: self.pool._release(self))
        return self

    def _put(self, item):
        self.deadline.remaining()
        if self.loop.is_closed():
            raise RequestStopped("disconnected")
        pending = asyncio.run_coroutine_threadsafe(self.queue.put(item), self.loop)
        try:
            while True:
                try:
                    pending.result(timeout=min(.1, self.deadline.remaining()))
                    return
                except FutureTimeout:
                    continue
        finally:
            if not pending.done():
                pending.cancel()

    def _produce(self, factory, cleanup):
        iterator = None
        with using(self.deadline):
            try:
                self.deadline.remaining()
                iterator = iter(factory())
                while True:
                    self.deadline.remaining()
                    try:
                        item = next(iterator)
                    except StopIteration:
                        break
                    self._put(item)
            except RequestStopped:
                pass
            except BaseException as exc:
                self.outcome = "persistence" if isinstance(exc, PersistenceError) else "internal"
                observability.metrics.stopped(self.outcome)
                observability.event("stream_producer_error", error_type=type(exc).__name__)
                try:
                    self._put(sse("error", {"error": "回答處理失敗，請提供 request_id 給維護者",
                                            "error_type": self.outcome, "request_id": observability.request_id()}))
                    self._put(sse("done", {"llm_available": False, "completed": False}))
                except RequestStopped:
                    pass
            finally:
                # A generator may still be blocked in I/O. Only its own thread closes it.
                if iterator is not None:
                    try:
                        close = getattr(iterator, "close", None)
                        if close is not None:
                            close()
                    except BaseException as exc:
                        observability.event("stream_cleanup_error", error_type=type(exc).__name__)
                if cleanup is not None:
                    with using(None):
                        try:
                            cleanup()
                        except BaseException as exc:
                            self.outcome = "persistence"
                            observability.metrics.stopped("persistence")
                            observability.event("stream_persist_error", error_type=type(exc).__name__)
                            try:
                                self._put(sse("error", {"error": "對話保存失敗，請提供 request_id 給維護者",
                                                        "error_type": "persistence"}))
                                self._put(sse("done", {"llm_available": False, "completed": False}))
                            except RequestStopped:
                                pass
                try:
                    self._put(_END)
                except RequestStopped:
                    pass

    def stop(self, reason: str):
        self.deadline.stop(reason)
        if self.deadline.reason != reason:
            return
        if not self._reported and reason in {"timeout", "disconnected", "shutdown"}:
            self._reported = True
            observability.stream_stopped(reason)
        if self.future is None:
            self.pool._release(self)
        # Do not cancel a submitted producer before it starts: its cleanup must
        # pair the saved user turn even if the browser disconnected immediately.

    async def items(self):
        finished = False
        try:
            while True:
                try:
                    remaining = self.deadline.remaining()
                    item = await asyncio.wait_for(self.queue.get(), remaining)
                except (TimeoutError, RequestStopped):
                    self.stop("timeout")
                    yield sse("error", {"error": "回答超過時間上限，檢索結果仍可使用",
                                        "error_type": "timeout", "request_id": observability.request_id()})
                    yield sse("done", {"llm_available": False, "completed": False, "reason": "timeout"})
                    return
                if item is _END:
                    finished = True
                    return
                yield item
        finally:
            if self.deadline.reason is None:
                self.stop("completed" if finished else "disconnected")


class ManagedStreamingResponse(StreamingResponse):
    def __init__(self, job: StreamJob):
        self.job = job
        super().__init__(job.items(), media_type="text/event-stream",
                         headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def __call__(self, scope, receive, send):
        # ASGI >=2.4 alone relies on the next send to detect disconnects. We also
        # listen while the model is silent, so a stopped browser releases HTTP capacity.
        completed = False
        finishing = False

        async def response_send(message):
            nonlocal finishing
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                finishing = True
            await send(message)

        async def emit(group):
            nonlocal completed
            try:
                await self.stream_response(response_send)
                completed = True
            except OSError:
                self.job.stop("disconnected")
            finally:
                group.cancel_scope.cancel()

        async def listen(group):
            await self.listen_for_disconnect(receive)
            if finishing:
                return
            self.job.stop("disconnected")
            group.cancel_scope.cancel()

        async def expire(group):
            await asyncio.sleep(max(0, self.job.deadline.ends - time.monotonic()))
            self.job.stop("timeout")
            # Give a readable socket time to receive error+done. A stalled send
            # must also stop; it cannot bypass the total deadline by backpressure.
            await asyncio.sleep(.5)
            group.cancel_scope.cancel()

        try:
            async with anyio.create_task_group() as group:
                group.start_soon(emit, group)
                group.start_soon(listen, group)
                group.start_soon(expire, group)
        finally:
            if not completed:
                self.job.stop("disconnected")
            await self.body_iterator.aclose()
            scope.setdefault("state", {})["rag_stream_reason"] = self.job.outcome or self.job.deadline.reason


def stream_pool(app) -> StreamPool:
    pool = getattr(app.state, "answer_stream_pool", None)
    if pool is None:
        pool = StreamPool(int(os.getenv("STREAM_WORKERS", "4")),
                          int(os.getenv("STREAM_BUFFER_EVENTS", "8")))
        app.state.answer_stream_pool = pool
    return pool
