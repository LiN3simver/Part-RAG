"""把任何一個既有 RAG/Agent 專案包成 A2A 端點，**不動它原本的程式**。

用法（各專案 src/a2a.py 裡）：

    spec = AgentSpec(agent_id="rfq", tools=[...], search_fn=..., hits_fn=...)
    router = build_router(spec)
    # main.py 裡加一行：app.include_router(router)

★ 為什麼是「加一個 /a2a 命名空間」而不是改寫既有 /api/*
  Part-RAG 與 RFQ-RAG 的前端與端點都在用了。改 /api/search 的請求形狀
  （partrag 用 {query, filters, top_k}、rfq 用 {q, limit}）就會同時弄壞兩個
  已經能用的頁面。/a2a/* 是純新增，舊路徑一個字都不動，兩者共存。

★ 為什麼 /a2a/message 是無狀態的（history 由呼叫端傳進來）
  partrag 有 DB 撐的對話表（partrag.conversation），rfq 完全沒有，
  material 與 bom 的 DB 還沒建。要求四個專案都先做對話持久化才能共用前端，
  等於把最慢的那個變成瓶頸。所以對話狀態一律由 Agent Hub 保管，
  agent 只負責「給我歷史 + 這一題，回我答案」。
  partrag 自己的 /api/chat 仍然照舊寫進它的對話表，行為未變。
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from . import registry
from .envelope import (
    EV_DELTA,
    EV_DONE,
    EV_ERROR,
    EV_META,
    EV_RESULTS,
    PROTOCOL,
    Hit,
    ToolSpec,
    sse,
    tool_err,
    tool_ok,
)

Status = str  # "ready" | "degraded" | "not_ready"

# 依賴檢查的硬期限與快取秒數，見 AgentSpec.status() 的說明。
# 2.5s 是刻意小於 client.CARD_TIMEOUT（6s）的：card 端點自己就不該是逾時的來源。
STATUS_DEADLINE = float(os.environ.get("A2A_STATUS_DEADLINE", "2.5"))
STATUS_TTL = float(os.environ.get("A2A_STATUS_TTL", "15"))


@dataclass
class AgentSpec:
    """一個 agent 要提供給 A2A 的全部東西。除了 agent_id 之外都可以省略。"""

    agent_id: str

    # 這個 agent 自己的工具。handler 收 dict、回 dict，不要拋例外。
    tools: list[ToolSpec] = field(default_factory=list)

    # 給 LLM 的系統提示。裡面應該寫清楚這份資料的限制（哪些欄位不可信、
    # 覆蓋率多少），漏掉會讓模型講出資料撐不住的話。
    system: str = ""

    # 空對話時前端顯示的範例問句：[{"label":..., "prompt":...}]
    examples: list[dict[str, str]] = field(default_factory=list)

    # 檢索：(query, top_k, filters) -> 該專案原生的 found dict
    search_fn: Callable[[str, int, dict], dict[str, Any]] | None = None
    # 正規化：found dict -> list[Hit]
    hits_fn: Callable[[dict[str, Any]], list[Hit]] | None = None
    # 摘要：(query, found, history) -> 文字串流
    answer_stream: Callable[[str, dict[str, Any], list[dict]], Iterator[str]] | None = None

    # 工具迴圈要怎麼把「自己這次檢索到的東西」講給模型聽：(query, found) -> 一段文字。
    #
    # ★ 沒有這個鉤子時，delegate.py 只會把 found["results"] 丟進去 —— 但每個
    #   專案的 found 都不只有 results：partrag 的工程圖在 found["drawings"]、
    #   清單統計在 found["listing"]、查無料號在 found["not_found"]。
    #   例如問「哪張工程圖有 <HSG 料號>」：
    #   同一句話不勾「用自己的工具」答得出來（走 answer_stream，看得到 drawings），
    #   勾了反而回「語料內無此料號」—— 卡片上那張材料表就畫在旁邊，模型卻沒收到。
    #   使用者看到的是「勾了工具反而變笨」，而不是少了一段 context。
    #   給了這個鉤子，兩條路徑就是同一份資料的兩種讀法。
    context_fn: Callable[[str, dict[str, Any]], str] | None = None

    # 依賴檢查：-> (status, detail dict)。裡面不要拋例外，掛掉要回 not_ready。
    status_fn: Callable[[], tuple[Status, dict[str, Any]]] | None = None
    # status != ready 時，卡片上要說明卡在哪（給人看，也給上游 agent 看）
    blocked_by: list[str] = field(default_factory=list)

    # 是否允許這個 agent 在回答時呼叫下游 agent 的工具（委派）
    can_delegate: bool = False

    # (量到的時間, (status, detail))。見 status() 的期限與快取說明。
    _status_cache: tuple[float, tuple[Status, dict[str, Any]]] | None = None

    def ref(self) -> registry.AgentRef:
        return registry.get(self.agent_id)

    def tool(self, name: str) -> ToolSpec | None:
        return next((t for t in self.tools if t.name == name), None)

    def status(self) -> tuple[Status, dict[str, Any]]:
        """依賴檢查，**保證 STATUS_DEADLINE 秒內回來**。

        ★ 為什麼需要期限（實測踩到的）
          partrag / rfqrag 的 pgvector 容器停掉時，它們的 status_fn 會卡在
          psycopg 的連線池上整整 **30 秒**（連線池預設逾時）才拋 PoolTimeout
          （兩個專案原有的 /api/health 也是同樣症狀）。
          card 是「快速回報我還活著嗎」的端點，被拖 30 秒等於整個統一前端
          在資料庫掛掉時要轉半分鐘，而且上游 agent 會直接判它離線。
          所以這裡給硬期限：逾時就回 degraded 並說明還在檢查，
          背景那條 thread 讓它自己跑完，結果進快取給下一次用。

        快取存在的另一個理由：統一前端每次載入要問四張 card，
        每張都重跑一次 GROUP BY（partrag 的 facets() 會掃整張主表）很浪費。
        """
        if self.status_fn is None:
            return "ready", {}

        now = time.monotonic()
        if self._status_cache and now - self._status_cache[0] < STATUS_TTL:
            return self._status_cache[1]

        box: dict[str, tuple[Status, dict[str, Any]]] = {}

        def run() -> None:
            try:
                box["v"] = self.status_fn()  # type: ignore[misc]
            except Exception as e:
                box["v"] = ("not_ready", {"error": f"{type(e).__name__}: {e}"})
            self._status_cache = (time.monotonic(), box["v"])

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(STATUS_DEADLINE)
        if "v" in box:
            return box["v"]
        # 還沒回來。有舊結果就先用舊的（並標明是舊的），沒有就說還在檢查。
        if self._status_cache:
            prev_st, prev_detail = self._status_cache[1]
            return prev_st, {**prev_detail, "stale": True,
                             "note": f"依賴檢查超過 {STATUS_DEADLINE}s 未回應，這是上一次的結果"}
        return "degraded", {
            "checking": True,
            "note": f"依賴檢查超過 {STATUS_DEADLINE}s 未回應（資料庫沒開時會這樣），"
                    "背景仍在檢查，稍後重新整理即可看到結果",
        }


class MessageReq(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict[str, Any] = Field(default_factory=dict)
    # 對話歷史由呼叫端（Agent Hub）保管，見模組 docstring
    history: list[dict[str, str]] = Field(default_factory=list)
    # 允許這一題向下游 agent 委派。預設 False —— 委派會多花好幾秒與好幾倍
    # token，必須是呼叫端明確要求的行為，不能偷偷發生。
    delegate: bool = False
    # 允許這一題呼叫**自己的**工具，但不准碰任何下游（「獨立但可以用自己的工具」）。
    #
    # ★ 為什麼要跟 delegate 分開
    #   這兩件事以前綁在同一個開關上：不開 delegate 就走純摘要，連自己的工具
    #   都不會被呼叫 —— rfq 的 estimate_cost / sanity_check_cost 想用就非得
    #   同時開放下游不可。但「要不要用自己的工具」與「要不要問別人」是兩個
    #   不同的授權：後者會把問題與資料送到另一個服務，前者不會。
    #   delegate=True 已經隱含要跑工具迴圈，所以不必兩個都勾。
    #
    # ★ 預設 True，跟 delegate 的預設不一樣，這是刻意的。
    #   資料不出這個 agent 的事不需要使用者先授權；預設關掉的效果只是讓
    #   agent 少掉自己本來就有的能力，而且要使用者先知道有這個開關、
    #   再知道這一題需要它，才拿得回來。真的不要工具的呼叫端（回測腳本、
    #   要比較「純檢索摘要」的實驗）明確傳 false 即可 —— 而那是知情的選擇。
    use_tools: bool = True
    # 限制這一題只能委派給哪些 agent（空 = registry 裡的全部下游）
    peers: list[str] = Field(default_factory=list)
    # 只要資料不要 LLM 摘要。前端「只檢索」模式與回測腳本用得到。
    no_llm: bool = False


class SearchReq(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict[str, Any] = Field(default_factory=dict)


def _run_search(spec: AgentSpec, query: str, top_k: int, filters: dict) -> dict[str, Any]:
    """檢索一次，永不拋例外。

    DB 掛掉時回 {"results": [], "error": ...} 而不是 500：呼叫端是另一個
    agent 的迴圈，它拿到 500 只能整條鏈中斷，拿到 error 還能繼續講「查不到」。
    """
    if spec.search_fn is None:
        st, detail = spec.status()
        return {
            "query": query,
            "results": [],
            "unavailable": True,
            "status": st,
            "note": "此 agent 尚未提供檢索能力",
            "blocked_by": spec.blocked_by,
            "detail": detail,
        }
    try:
        return spec.search_fn(query, top_k, filters or {})
    except Exception as e:
        return {"query": query, "results": [], "error": f"{type(e).__name__}: {e}"}


def _to_hits(spec: AgentSpec, found: dict[str, Any]) -> list[dict[str, Any]]:
    if spec.hits_fn is None:
        # 沒給正規化函式時退成「每筆一張只有 raw 的卡」，前端仍然畫得出來
        return [
            Hit(id=str(r.get("id", i)), title=str(r.get("id", i)), raw=r).to_dict()
            for i, r in enumerate(found.get("results") or [])
        ]
    try:
        return [h.to_dict() for h in spec.hits_fn(found)]
    except Exception as e:
        # 退回原始列而不是回 500：資料本身沒問題，是渲染層的 bug，
        # 使用者仍然應該看到查到的東西。錯誤訊息要**放進卡片**而不只是印在
        # 伺服器 log —— 只印 log 的話畫面上是一排無意義的空卡，
        # 沒有人會去翻 log（實測就這樣掉過一次：length_mm 是 array 不是純量）。
        msg = f"{type(e).__name__}: {e}"
        print(f"[a2a] hits_fn 失敗，退回原始列：{msg}")
        return [
            Hit(
                id=str(r.get("item_number") or r.get("rfq_no") or i),
                title="（此 agent 的卡片渲染失敗，以下為原始欄位）",
                subtitle=msg,
                raw=r,
            ).to_dict()
            for i, r in enumerate(found.get("results") or [])
        ]


def build_router(spec: AgentSpec) -> APIRouter:
    router = APIRouter(prefix="/a2a", tags=["a2a"])
    ref = spec.ref()

    @router.get("/card")
    def card() -> dict[str, Any]:
        """Agent 名片：我是誰、能做什麼、現在好不好、可以呼叫誰。

        上游 agent 與統一前端都只讀這一支就能把介面畫出來，
        不需要知道任何一個專案的內部端點。
        """
        st, detail = spec.status()
        return {
            "protocol": PROTOCOL,
            "agent_id": ref.agent_id,
            "name": ref.name,
            "role": ref.role,
            "project": ref.project,
            "base_url": ref.base_url,
            "native_ui": ref.native_ui,
            "tags": list(ref.tags),
            "status": st,
            "detail": detail,
            "blocked_by": spec.blocked_by if st != "ready" else [],
            "capabilities": {
                "search": spec.search_fn is not None,
                "chat": spec.answer_stream is not None,
                "tools": any(t.ready for t in spec.tools),
                "delegate": spec.can_delegate,
            },
            "upstream": list(ref.upstream),
            "downstream": list(ref.downstream),
            "tools": [t.public() for t in spec.tools],
            "examples": spec.examples,
            "system": spec.system,
        }

    @router.get("/health")
    def health() -> dict[str, Any]:
        st, detail = spec.status()
        return {"agent_id": ref.agent_id, "status": st, "detail": detail}

    @router.get("/tools")
    def tools() -> dict[str, Any]:
        """給上游掛工具用。system 一併給出去 —— 裡面寫的是這份資料的限制，
        漏掉會讓上游 agent 拿正確的數字講出錯誤的結論。"""
        return {
            "agent_id": ref.agent_id,
            "tools": [t.public() for t in spec.tools],
            "system": spec.system,
        }

    @router.post("/tool/{name}")
    def call_tool(name: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        t = spec.tool(name)
        if t is None:
            return tool_err(
                f"未知的工具：{name}",
                available=[x.name for x in spec.tools],
            )
        if not t.ready:
            return tool_err(
                f"工具 {name} 尚未可用",
                reason=t.not_ready_reason,
                blocked_by=spec.blocked_by,
                status="not_ready",
            )
        t0 = time.perf_counter()
        try:
            out = t.handler(body or {})
        except Exception as e:
            return tool_err(f"{type(e).__name__}: {e}")
        ms = round((time.perf_counter() - t0) * 1000)
        # handler 已經回 {"ok":...} 形狀時不要再包一層
        if isinstance(out, dict) and "ok" in out:
            return {**out, "ms": ms}
        return {**tool_ok(out), "ms": ms}

    @router.post("/search")
    def search(req: SearchReq) -> dict[str, Any]:
        """統一檢索：形狀固定為 {hits, raw}，前端不必認識各專案的欄位名。"""
        found = _run_search(spec, req.query, req.top_k, req.filters)
        return {
            "agent_id": ref.agent_id,
            "query": req.query,
            "hits": _to_hits(spec, found),
            "raw": found,
        }

    @router.get("/peers")
    def peers() -> dict[str, Any]:
        from .client import peer_cards

        return {"agent_id": ref.agent_id, "peers": peer_cards(ref.downstream)}

    @router.post("/message")
    async def message(req: MessageReq, request: Request):
        """統一對話端點（SSE）。事件順序見 envelope.py 的說明。

        ★ results 一定在聯絡 LLM 之前送出。gateway 掛掉、thinking 把
          max_tokens 吃光（實測會回空 text + stop_reason=max_tokens）時，
          使用者至少已經看到檢索結果了。
        """
        from fastapi.responses import StreamingResponse

        found = _run_search(spec, req.query, req.top_k, req.filters)
        hits = _to_hits(spec, found)

        # ★ 上游（Hub，或直接呼叫的另一個 agent）斷線時要真的停下來，不是
        #   繼續把 token 燒給一個沒人在聽的連線。cancel_event 由輪詢
        #   request.is_disconnected() 的背景 task 設起來，gen() 在每個
        #   事件之間檢查它，讓 spec.answer_stream() 的 generator 被明確
        #   close() —— 這會跳出 llm.stream_answer() 裡 `with ...stream() as s`
        #   的 with 區塊，真正關掉打去 Anthropic 的那條串流連線。
        cancel_event = threading.Event()

        async def watch_disconnect():
            try:
                while not cancel_event.is_set():
                    if await request.is_disconnected():
                        cancel_event.set()
                        return
                    await asyncio.sleep(0.25)
            except Exception:
                pass

        watcher = asyncio.ensure_future(watch_disconnect())

        def gen() -> Iterator[str]:
            st, _ = spec.status()
            # 三種模式，由呼叫端的兩個布林決定，預設是最省也最封閉的那個：
            #   都沒開            -> 純檢索 + 摘要，一個工具都不呼叫
            #   use_tools         -> 跑工具迴圈，但只掛自己的工具，不碰下游
            #   delegate          -> 跑工具迴圈，自己的工具 + 下游的工具
            delegating = bool(req.delegate and spec.can_delegate)
            has_tools = any(t.ready for t in spec.tools)
            tool_loop = delegating or bool(req.use_tools and has_tools)
            yield sse(EV_META, {
                "agent_id": ref.agent_id,
                "agent_name": ref.name,
                "status": st,
                "retrieval_query": found.get("query", req.query),
                "delegating": delegating,
                # 有跑工具迴圈但沒開放下游 —— 前端要能分辨這兩件事，
                # 否則使用者看到 trace 會以為它偷偷問了別人
                "tool_mode": tool_loop and not delegating,
            })
            yield sse(EV_RESULTS, {"hits": hits, "raw": found})

            if req.no_llm:
                yield sse(EV_DONE, {"llm_available": False, "skipped": "no_llm"})
                return

            got = False
            try:
                if tool_loop:
                    from .delegate import stream_delegated

                    # 同一個迴圈，掛不掛下游由 req.delegate 決定（見 delegate.py）
                    for ev, payload in stream_delegated(spec, req, found):
                        if cancel_event.is_set():
                            break
                        if ev == EV_DELTA and payload.get("t"):
                            got = True
                        yield sse(ev, payload)
                elif spec.answer_stream is not None:
                    stream = spec.answer_stream(req.query, found, req.history)
                    try:
                        for chunk in stream:
                            if cancel_event.is_set():
                                break
                            if chunk:
                                got = True
                                yield sse(EV_DELTA, {"t": chunk})
                    finally:
                        # 明確 close() 而不是讓它等 GC：這會在
                        # llm.stream_answer() 卡著的 yield 點丟 GeneratorExit，
                        # 跳出它 `with ...messages.stream() as s` 的區塊，
                        # 立刻關掉打去 Anthropic 的連線，真的停止繼續產生文字。
                        stream.close()
                else:
                    yield sse(EV_DELTA, {"t": _no_llm_note(spec, st)})
            except Exception as e:
                if not cancel_event.is_set():
                    # 例外可以帶三個給人看的屬性，帶了就用它們，沒帶就照舊
                    # 印類別名 + 訊息：
                    #   user_message  已經是人話的一句話（不要再加類別名）
                    #   hint          下一步該做什麼
                    #   diag          結構化診斷 dict，前端照欄位畫
                    # ★ 這是為了 LLM gateway 故障：原始訊息長成
                    #   "BadRequestError: Error code: 400 - {'error': {'message':
                    #   "400: {'error': 'anthropic_messages: Invalid model name…"
                    #   三層巢狀引號，貼在畫面上沒有人讀得出該做什麼。
                    payload: dict[str, Any] = {
                        "error": getattr(e, "user_message", None) or f"{type(e).__name__}: {e}"
                    }
                    diag = getattr(e, "diag", None)
                    if isinstance(diag, dict):
                        payload["diag"] = diag
                    hint = getattr(e, "hint", None)
                    if hint:
                        payload["hint"] = str(hint)
                    yield sse(EV_ERROR, payload)

            if not cancel_event.is_set():
                yield sse(EV_DONE, {"llm_available": got})

        def gen_and_cleanup() -> Iterator[str]:
            try:
                yield from gen()
            finally:
                # 正常跑完也要收掉 watcher，不然每一輪對話都留一個閒置的
                # polling task 在事件迴圈上，越聊越多。
                cancel_event.set()
                watcher.cancel()

        return StreamingResponse(
            gen_and_cleanup(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router


def _no_llm_note(spec: AgentSpec, status: Status) -> str:
    """沒有摘要能力時給的一句話。刻意寫成人看得懂的現況說明，
    而不是空白 —— 空白會讓人以為是壞掉而不是還沒做。"""
    ref = spec.ref()
    if status == "not_ready":
        why = "；".join(spec.blocked_by) or "尚未實作"
        return f"（{ref.name} 尚未就緒：{why}。已在上方回報現況，未捏造任何結果。）"
    return "（此 agent 未提供文字摘要，以上為檢索結果。）"
