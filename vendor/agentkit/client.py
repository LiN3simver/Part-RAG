"""呼叫別的 agent 的客戶端。Hub 與 agent 之間、agent 與 agent 之間都用這支。

★ 三條不可退讓的規則（每一條都對應一個實際會發生的故障）

  1. **對方離線不是例外，是常態。** 四個服務是四個各自手動啟動的 uvicorn，
     任何時刻都可能只開了兩個。所以連不上一律回
     {"ok": false, "error": ..., "offline": true}，絕不往上拋 —— 上游 agent
     拿到例外只能整條鏈中斷，拿到 offline 還能告訴使用者「材料建議 Agent
     沒開，以下只用歷史資料回答」。

  2. **一定要有逾時。** card 用 3 秒（只是讀狀態），工具呼叫用 90 秒
     （下游可能自己也在跑向量檢索 + LLM）。沒有逾時的話一個沒開的服務會
     讓整個前端轉圈到瀏覽器放棄。

  3. **card 要快取但 TTL 要短。** 統一前端每次切換 agent 都會問一輪四張
     card；10 秒快取足夠擋掉連點，又不會讓「剛剛才把服務開起來」要等很久
     才反映在畫面上。
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from typing import Any

import httpx

from . import registry

# card 端點自己有 2.5s 的依賴檢查期限（server.STATUS_DEADLINE），
# 6s 是留給程序啟動時第一次 import 的餘裕，不是給資料庫慢查詢的。
CARD_TIMEOUT = 6.0
TOOL_TIMEOUT = 90.0
MESSAGE_TIMEOUT = 300.0
FILE_TIMEOUT = 30.0
_CARD_TTL = 10.0

_card_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _url(agent_id: str, path: str) -> str:
    return registry.get(agent_id).base_url.rstrip("/") + path


def _offline(agent_id: str, e: Exception) -> dict[str, Any]:
    ref = registry.get(agent_id)
    return {
        "protocol": "a2a/1",
        "agent_id": agent_id,
        "name": ref.name,
        "role": ref.role,
        "project": ref.project,
        "base_url": ref.base_url,
        "status": "offline",
        "offline": True,
        "error": f"{type(e).__name__}: {e}",
        "capabilities": {"search": False, "chat": False, "tools": False, "delegate": False},
        "tools": [],
        "examples": [],
        "upstream": list(ref.upstream),
        "downstream": list(ref.downstream),
        "hint": f"服務未啟動？在 {ref.project} 目錄跑 run.bat（埠 {ref.port}）",
    }


def card(agent_id: str, *, fresh: bool = False) -> dict[str, Any]:
    now = time.monotonic()
    if not fresh:
        hit = _card_cache.get(agent_id)
        if hit and now - hit[0] < _CARD_TTL:
            return hit[1]
    try:
        r = httpx.get(_url(agent_id, "/a2a/card"), timeout=CARD_TIMEOUT)
        r.raise_for_status()
        out = r.json()
    except Exception as e:
        out = _offline(agent_id, e)
    _card_cache[agent_id] = (now, out)
    return out


def peer_cards(agent_ids: tuple[str, ...] | list[str]) -> list[dict[str, Any]]:
    return [card(a) for a in agent_ids if a in registry.BY_ID]


def all_cards(*, fresh: bool = False) -> list[dict[str, Any]]:
    return [card(a.agent_id, fresh=fresh) for a in registry.AGENTS]


def call_tool(
    agent_id: str, tool: str, args: dict[str, Any], *, timeout: float = TOOL_TIMEOUT
) -> dict[str, Any]:
    """呼叫下游 agent 的一個工具。回傳形狀固定 {"ok":bool, ...}。"""
    try:
        r = httpx.post(_url(agent_id, f"/a2a/tool/{tool}"), json=args, timeout=timeout)
        r.raise_for_status()
        out = r.json()
        if isinstance(out, dict) and "ok" in out:
            return out
        return {"ok": True, "result": out}
    except httpx.HTTPStatusError as e:
        return {"ok": False, "error": f"{agent_id} 回應 {e.response.status_code}"}
    except Exception as e:
        return {
            "ok": False,
            "offline": True,
            "error": f"連不上 {registry.get(agent_id).name}（{registry.get(agent_id).base_url}）：{e}",
        }


def file_bytes(agent_id: str, doc_id: str) -> tuple[bytes, str, str, int]:
    """代理下游 agent 的 /a2a/file/{doc_id}。回傳 (內容, content-type, content-disposition, 狀態碼)。

    ★ 為什麼要有這一支（見各專案 docker-compose.yml 的註解）
      agent 本身故意不對外開 host port，只有 Hub 有對外的固定位址；
      附件卡片上的 PDF 連結是**瀏覽器**直接點的，連得到的只能是 Hub。
      這支負責把請求轉進 ragnet 內部打到真正的 agent，再把位元組原樣吐回去。

    404/410 是下游自己的業務錯誤（查無附件、檔案不在原位置這台機器上），
    原樣把 status_code 轉出去，呼叫端不必認識下游用哪個碼表示什麼。
    連不上（agent 沒開）才是這裡自己的錯誤，用 RuntimeError 拋出去，
    跟 call_tool() 的 offline 語意一致。
    """
    try:
        r = httpx.get(_url(agent_id, f"/a2a/file/{doc_id}"), timeout=FILE_TIMEOUT)
    except Exception as e:
        ref = registry.get(agent_id)
        raise RuntimeError(f"連不上 {ref.name}（{ref.base_url}）：{type(e).__name__}: {e}") from e
    return (r.content, r.headers.get("content-type", "application/octet-stream"),
            r.headers.get("content-disposition", ""), r.status_code)


def search(agent_id: str, query: str, top_k: int = 10,
           filters: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        r = httpx.post(
            _url(agent_id, "/a2a/search"),
            json={"query": query, "top_k": top_k, "filters": filters or {}},
            timeout=TOOL_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"agent_id": agent_id, "query": query, "hits": [],
                "raw": {"error": str(e)}, "offline": True}


def stream_message(
    agent_id: str,
    payload: dict[str, Any],
    cancel_event: threading.Event | None = None,
) -> Iterator[tuple[str, dict]]:
    """轉發 /a2a/message 的 SSE，逐事件 yield (event, data)。

    自己解析而不是原樣 relay bytes，是為了讓 Hub 能在中途插入自己的事件
    （例如把 agent 名字塞進 meta、或在對方離線時補一個 error 事件）。

    cancel_event：使用者按了暫停時由呼叫端設起來。在兩行 SSE 之間檢查，
    設起來就直接 return —— 讓 `with httpx.stream(...)` 的 __exit__ 關掉
    這條到下游 agent 的連線，下游才會偵測到「client 斷了」進而真的停止
    LLM 呼叫，不是只有 Hub 這邊假裝停下來。
    """
    try:
        with httpx.stream(
            "POST", _url(agent_id, "/a2a/message"), json=payload, timeout=MESSAGE_TIMEOUT
        ) as r:
            r.raise_for_status()
            event = ""
            for line in r.iter_lines():
                if cancel_event is not None and cancel_event.is_set():
                    return
                if line.startswith("event: "):
                    event = line[7:].strip()
                elif line.startswith("data: "):
                    try:
                        yield event or "message", json.loads(line[6:])
                    except json.JSONDecodeError:
                        yield "error", {"error": f"無法解析 {agent_id} 的事件內容"}
    except Exception as e:
        ref = registry.get(agent_id)
        yield "error", {
            "error": f"連不上 {ref.name}（{ref.base_url}）：{type(e).__name__}: {e}",
            "offline": True,
            "hint": f"在 {ref.project} 目錄跑 run.bat（埠 {ref.port}）",
        }
        yield "done", {"llm_available": False}


# --------------------------------------------------------------------------
# 把下游 agent 的工具攤平成「本地工具」給模型用
# --------------------------------------------------------------------------
NS = "__"  # material__suggest_material。Anthropic 的工具名允許 [a-zA-Z0-9_-]{1,64}


ASK = "ask_"  # ask_bom / ask_material —— 把整句問題交給下游那個 agent


def ask_agent(agent_id: str, question: str, *, delegate: bool = True,
              timeout: float = MESSAGE_TIMEOUT) -> dict[str, Any]:
    """把一整句自然語言問題交給下游 agent，回它的答案與檢索結果。

    ★ 為什麼需要這個，而不是只有結構化工具（實測踩到的）
      bom agent 現階段自己的三個工具全是 not_ready（要等槽位樣板），
      所以 peer_tools() 給它時一個都不掛 —— 上游的 rfq 於是**完全叫不到它**，
      實測 rfq 被要求「請 Bom Agent 判斷…」時只能改叫自己的
      search_rfq，繞了兩圈回傳無關的 RFQ。
      「能對話」本身就是一種能力：bom 沒有語料但有兩個下游，
      問它一句話它會自己去問 material 與 partrag。
      這支就是 agent 之間的「講話」介面，結構化工具是「取資料」介面，兩者都要有。

    回傳裡 delegated_trace 會帶下游自己的委派軌跡，所以三層鏈
    （rfq -> bom -> material）在前端仍然看得到每一步是誰做的。
    """
    text, hits, trace, errors = "", [], [], []
    for ev, data in stream_message(agent_id, {
        "query": question, "top_k": 8, "history": [], "delegate": delegate,
    }):
        if ev == "delta":
            text += data.get("t") or ""
        elif ev == "results":
            hits = data.get("hits") or []
        elif ev == "trace":
            trace.append(data)
        elif ev == "error":
            errors.append(data.get("error"))
    ref = registry.get(agent_id)
    if not text and errors:
        return {"ok": False, "error": "；".join(str(e) for e in errors), "agent": ref.name}
    return {
        "ok": True,
        "agent": ref.name,
        "answer": text,
        # 只回 id 與標題：完整欄位讓上游需要時再用結構化工具去拿，
        # 整包塞回去會把 thinking 的 token 預算擠掉。
        "hits": [{"id": h.get("id"), "title": h.get("title")} for h in hits[:8]],
        "delegated_trace": [
            {"tool": t.get("tool"), "agent": t.get("agent_name"), "ok": t.get("ok")}
            for t in trace if t.get("phase") == "result"
        ],
        **({"warnings": errors} if errors else {}),
    }


def peer_tools(agent_ids: tuple[str, ...] | list[str]) -> list[dict[str, Any]]:
    """下游可以呼叫的東西：每個 ready 的工具，加上一個 `ask_<agent>` 對話入口。

    工具只掛 ready=True 的：把「未來會有」的工具掛給模型，它會呼叫、失敗、
    然後開始猜答案 —— 這比工具不存在更糟。
    離線的 agent 兩種都不掛，改由 system 提示裡說明它沒開。
    """
    out: list[dict[str, Any]] = []
    for aid in agent_ids:
        c = card(aid)
        if c.get("status") == "offline":
            continue
        name = c.get("name", aid)
        for t in c.get("tools") or []:
            if not t.get("ready", True):
                continue
            out.append({
                "name": f"{aid}{NS}{t['name']}"[:64],
                "description": f"[{name}] {t.get('description', '')}",
                "input_schema": t.get("input_schema") or {"type": "object", "properties": {}},
            })
        caps = c.get("capabilities") or {}
        if caps.get("chat") or caps.get("delegate") or caps.get("search"):
            out.append({
                "name": f"{ASK}{aid}"[:64],
                "description": (
                    f"用一句自然語言問「{name}」（{c.get('role', '')}）。"
                    "適用於：問題屬於它的專業範圍、但你不確定要用它的哪一支工具，"
                    "或它需要自己再去問更下游的 agent。"
                    "回傳它的答案與它引用的資料。"
                    + ("★ 這個 agent 目前尚未完全就緒（"
                       + "；".join(c.get("blocked_by") or []) + "），"
                       "它會如實說明哪部分還沒有，不要把它的說明改寫成肯定的答案。"
                       if c.get("blocked_by") else "")
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"question": {
                        "type": "string",
                        "description": "要問它的完整問題（帶上必要的料號與規格，它看不到你的對話）",
                    }},
                    "required": ["question"],
                },
            })
    return out


def split_ns(name: str) -> tuple[str | None, str]:
    """`material__suggest_material` -> ("material", "suggest_material")。"""
    if NS in name:
        aid, _, rest = name.partition(NS)
        if aid in registry.BY_ID:
            return aid, rest
    return None, name
