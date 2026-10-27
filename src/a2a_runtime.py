"""Part-RAG's async transport; the shared agentkit card/tool/SSE contract is retained."""
from __future__ import annotations

import asyncio
import time

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import Field, field_validator

from agentkit import server
from agentkit.envelope import EV_DELTA, EV_DONE, EV_ERROR, EV_META, EV_RESULTS, sse, tool_err, tool_ok

from .probes import Probe
from .service_runtime import query_pool
from . import observability, chat
from .request_budget import for_message, install_gateway_budget, remaining_http
from .streaming import ManagedStreamingResponse, stream_pool


class MessageReq(server.MessageReq):
    query: str = Field(min_length=1, max_length=2000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=24)

    @field_validator("history")
    @classmethod
    def bounded_history(cls, history):
        if any(m.get("role") not in {"user", "assistant"} or len(m.get("content", "")) > 8000
               for m in history) or sum(len(m.get("content", "")) for m in history) > 32000:
            raise ValueError("history role/content exceeds the supported budget")
        return history


class SearchReq(server.SearchReq):
    query: str = Field(min_length=1, max_length=2000)


def build_router(spec, app):
    pool = getattr(app.state, "a2a_query_pool", None) or query_pool()
    probe = Probe(spec.status_fn, ttl=server.STATUS_TTL, wait=server.STATUS_DEADLINE)
    spec.status = probe.get
    app.state.a2a_query_pool = pool
    app.state.a2a_status_probe = probe
    streams = stream_pool(app)
    install_gateway_budget()
    router = server.build_router(spec)
    router.routes = [r for r in router.routes if r.path not in
                     {"/a2a/message", "/a2a/search", "/a2a/tool/{name}"}]
    ref = spec.ref()

    def retrieve(req):
        query = req.query
        if isinstance(req, MessageReq):
            query, _ = chat.retrieval_query(req.history, query)
        found = server._run_search(spec, query, req.top_k, req.filters)
        return found, server._to_hits(spec, found)

    @router.post("/search")
    async def search(req: SearchReq):
        found, hits = await pool.run(retrieve, req)
        return {"agent_id": ref.agent_id, "query": req.query, "hits": hits, "raw": found}

    @router.post("/tool/{name}")
    async def call_tool(name: str, body: dict | None = None):
        tool = spec.tool(name)
        if tool is None:
            return tool_err(f"未知的工具：{name}", available=[t.name for t in spec.tools])
        if not tool.ready:
            return tool_err(f"工具 {name} 尚未可用", reason=tool.not_ready_reason,
                            blocked_by=spec.blocked_by, status="not_ready")

        def execute():
            start = time.perf_counter()
            try:
                out = tool.handler(body or {})
            except Exception as exc:
                return tool_err(f"{type(exc).__name__}: {exc}")
            result = out if isinstance(out, dict) and "ok" in out else tool_ok(out)
            return {**result, "ms": round((time.perf_counter() - start) * 1000)}

        return await pool.run(execute)

    @router.post("/message")
    async def message(req: MessageReq, request: Request):
        deadline = for_message(request)
        job = None if req.no_llm else streams.reserve(deadline)
        try:
            found, hits = await pool.run(retrieve, req, timeout=remaining_http(deadline))
            status, _ = await asyncio.wait_for(probe.aget(), remaining_http(deadline))
            remaining_http(deadline)
        except BaseException as exc:
            if job is not None:
                job.stop("timeout" if isinstance(exc, TimeoutError) or
                         isinstance(exc, HTTPException) and exc.status_code == 504 else "cancelled")
            if isinstance(exc, TimeoutError):
                raise HTTPException(504, "回答超過時間上限") from None
            raise

        delegating = bool(req.delegate and spec.can_delegate)
        tool_loop = delegating or bool(req.use_tools and any(t.ready for t in spec.tools))
        meta = {"agent_id": ref.agent_id, "agent_name": ref.name, "status": status,
                "retrieval_query": found.get("query", req.query), "delegating": delegating,
                "tool_mode": tool_loop and not delegating, "request_id": observability.request_id()}
        if req.no_llm:
            async def retrieval_only():
                yield sse(EV_META, meta)
                yield sse(EV_RESULTS, {"hits": hits, "raw": found})
                yield sse(EV_DONE, {"llm_available": False, "skipped": "no_llm"})
            return StreamingResponse(retrieval_only(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        def generate():
            stream = None
            try:
                yield sse(EV_META, meta)
                yield sse(EV_RESULTS, {"hits": hits, "raw": found})
                got = False
                failed = False
                try:
                    if tool_loop:
                        from agentkit.delegate import stream_delegated
                        stream = stream_delegated(spec, req, found)
                        for event, payload in stream:
                            deadline.remaining()
                            got |= event == EV_DELTA and bool(payload.get("t"))
                            failed |= event == EV_ERROR
                            yield sse(event, payload)
                    elif spec.answer_stream is not None:
                        stream = spec.answer_stream(req.query, found, req.history)
                        for chunk in stream:
                            deadline.remaining()
                            if chunk:
                                got = True
                                yield sse(EV_DELTA, {"t": chunk})
                    else:
                        yield sse(EV_DELTA, {"t": server._no_llm_note(spec, status)})
                except Exception as exc:
                    failed = True
                    deadline.remaining()
                    payload = {"error": getattr(exc, "user_message", None) or f"{type(exc).__name__}: {exc}"}
                    for key in ("diag", "hint"):
                        value = getattr(exc, key, None)
                        if value is not None:
                            payload[key] = value
                    yield sse(EV_ERROR, payload)
                success = got and not failed
                done = {"llm_available": success, "completed": success}
                if failed:
                    done["reason"] = "llm_error"
                    observability.metrics.stopped("llm_error")
                yield sse(EV_DONE, done)
            finally:
                if stream is not None:
                    stream.close()

        return ManagedStreamingResponse(job.start(generate))

    return router
