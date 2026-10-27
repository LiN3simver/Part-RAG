"""Safe request correlation and bounded process-local operational metrics."""
from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone

from fastapi.responses import JSONResponse, PlainTextResponse

_request_id = contextvars.ContextVar("rag_request_id", default=None)
_job_id = contextvars.ContextVar("rag_job_id", default=None)
_logger = logging.getLogger("partrag.events")
_logger.setLevel(logging.INFO)
_logger.propagate = False
if not _logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)

_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_BUCKETS = (.05, .1, .25, .5, 1., 2.5, 5., 10., 30., 60., 180., float("inf"))


def request_id():
    return _request_id.get()


def event(name: str, **fields):
    # Call sites supply operational fields only. Never pass request bodies, SQL,
    # exception messages, cookies, Authorization, environment values or URLs.
    _logger.info(json.dumps({"time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                             "event": name, "request_id": request_id(), "job_id": _job_id.get(), **fields},
                            ensure_ascii=False, separators=(",", ":")))


@contextmanager
def job_context():
    token = _job_id.set(uuid.uuid4().hex)
    started = time.monotonic()
    error_type = None
    event("sync_job_started")
    try:
        yield
    except BaseException as exc:
        error_type = type(exc).__name__
        raise
    finally:
        event("sync_job_finished", error_type=error_type,
              duration_ms=round((time.monotonic() - started) * 1000))
        _job_id.reset(token)


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.started = time.monotonic()
        self.active = 0
        self.counts = Counter()
        self.reasons = Counter()
        self.latencies = {}

    def begin(self):
        with self._lock:
            self.active += 1

    def finish(self, route, method, status, reason, seconds):
        with self._lock:
            self.active -= 1
            self.counts[(route, method, status)] += 1
            self.reasons[reason] += 1
            total, count, buckets = self.latencies.get((route, method), (0., 0, [0] * len(_BUCKETS)))
            for index, boundary in enumerate(_BUCKETS):
                buckets[index] += seconds <= boundary
            self.latencies[(route, method)] = (total + seconds, count + 1, buckets)

    def stopped(self, reason):
        with self._lock:
            self.reasons["stream_" + reason] += 1

    def snapshot(self):
        with self._lock:
            return {"scope": "process", "uptime_seconds": round(time.monotonic() - self.started, 1),
                    "active_http": self.active, "completed_http": sum(self.counts.values()),
                    "outcomes": dict(self.reasons)}

    def render(self):
        with self._lock:
            lines = ["# HELP rag_http_active HTTP requests currently in this process.",
                     "# TYPE rag_http_active gauge", f"rag_http_active {self.active}",
                     "# TYPE rag_http_requests_total counter"]
            for (route, method, status), count in sorted(self.counts.items()):
                labels = f'route={json.dumps(route)},method={json.dumps(method)},status="{status}"'
                lines.append(f"rag_http_requests_total{{{labels}}} {count}")
            lines.append("# TYPE rag_outcomes_total counter")
            for reason, count in sorted(self.reasons.items()):
                lines.append(f'rag_outcomes_total{{reason={json.dumps(reason)}}} {count}')
            lines.append("# TYPE rag_http_duration_seconds histogram")
            for (route, method), (total, count, buckets) in sorted(self.latencies.items()):
                labels = f'route={json.dumps(route)},method={json.dumps(method)}'
                for boundary, value in zip(_BUCKETS, buckets):
                    le = "+Inf" if boundary == float("inf") else str(boundary)
                    lines.append(f'rag_http_duration_seconds_bucket{{{labels},le="{le}"}} {value}')
                lines.extend([f"rag_http_duration_seconds_sum{{{labels}}} {total:.6f}",
                              f"rag_http_duration_seconds_count{{{labels}}} {count}"])
        return "\n".join(lines) + "\n"


metrics = Metrics()


def stream_stopped(reason):
    metrics.stopped(reason)
    event("stream_stopped", reason=reason)


def _outcome(status):
    if status == 503:
        return "unavailable"
    if status in {408, 504}:
        return "timeout"
    if status >= 500:
        return "server_error"
    if status in {413, 422}:
        return "validation"
    return "http_error" if status >= 400 else "ok"


class RequestMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        supplied = headers.get(b"x-request-id", b"").decode("latin-1")
        ident = supplied if _ID.fullmatch(supplied) else uuid.uuid4().hex
        token = _request_id.set(ident)
        started = time.monotonic()
        scope.setdefault("state", {}).update(request_id=ident, rag_started=started)
        status, reason = 500, "server_error"
        finished_body = False
        response_started = False
        metrics.begin()

        async def traced_send(message):
            nonlocal status, reason, finished_body, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status = message["status"]
                reason = _outcome(status)
                if status == 503 and any(k.lower() == b"retry-after" for k, _ in message.get("headers", [])):
                    reason = "busy"
                message = {**message, "headers": [(k, v) for k, v in message.get("headers", [])
                                                    if k.lower() != b"x-request-id"] +
                           [(b"x-request-id", ident.encode("ascii"))]}
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                finished_body = True
            await send(message)

        async def traced_receive():
            nonlocal reason
            message = await receive()
            if message["type"] == "http.disconnect" and not finished_body:
                reason = "disconnected"
            return message

        try:
            await self.app(scope, traced_receive, traced_send)
        except Exception as exc:
            reason = "server_error"
            event("http_exception", error_type=type(exc).__name__)
            if response_started:
                raise
            await JSONResponse({"detail": "伺服器處理失敗，請提供 request_id 給維護者",
                                "request_id": ident}, 500)(scope, receive, traced_send)
        except BaseException:
            reason = "cancelled"
            raise
        finally:
            stream_reason = scope.get("state", {}).get("rag_stream_reason")
            if stream_reason in {"timeout", "disconnected", "shutdown", "internal", "persistence"}:
                reason = stream_reason
            # Route templates prevent unbounded labels and hide part numbers/file IDs.
            route = getattr(scope.get("route"), "path", "<unmatched>")
            method = scope.get("method", "OTHER")
            if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
                method = "OTHER"
            seconds = time.monotonic() - started
            metrics.finish(route, method, status, reason, seconds)
            if route not in {"/livez", "/readyz", "/metrics", "/status"} or status >= 400:
                event("http_request", route=route, method=method, status=status,
                      outcome=reason, duration_ms=round(seconds * 1000, 2))
            _request_id.reset(token)


def mount(app):
    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint():
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/status", include_in_schema=False)
    async def status_endpoint():
        from . import db, retrieval
        query = getattr(app.state, "a2a_query_pool", None)
        streams = getattr(app.state, "answer_stream_pool", None)
        return JSONResponse({"runtime": metrics.snapshot(), "database_pool": db.stats(),
                             "query_pool": query.stats() if query is not None else None,
                             "stream_pool": streams.stats() if streams is not None else None,
                             "vector_ready": retrieval.embedding_ready(),
                             "message_timeout_seconds": float(os.getenv("MESSAGE_TIMEOUT_SECONDS", "180"))})
