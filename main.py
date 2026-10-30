"""Part-rag API.

Run:  .venv/Scripts/python.exe -m uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import hmac
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from src import (
    aggregate, chat, llm, retrieval,
    sync_state, taxonomy, values, write,
)
from src import db, health as service_health, sessions
from src import observability
from src.service_runtime import AdmissionMiddleware, query_pool
from src.request_budget import RequestStopped, for_message, remaining_http, using
from src.streaming import ManagedStreamingResponse, PersistenceError, stream_pool

db.configure_api()

# --------------------------------------------------------------------------
# 暖機
#
# ★ 問題：容器說自己好了，但第一個真實查詢要付模型載入的錢
#   retrieval._model() 是惰性載入（lru_cache）。啟動流程原本長這樣：
#       容器起來（1 秒）→ HEALTHCHECK 打 /a2a/health，過了 → 標成 healthy
#       → Hub 開始導流量 → 第一個使用者送出問題 → 這時候才開始載模型 → 等 25 秒
#   關鍵在於 /a2a/health 根本不碰模型，所以它看不見這筆債。Dockerfile 的
#   start-period 涵蓋的是「行程起不起得來」，不是「模型載沒載完」。
#   uvicorn 開 N 個 worker 的話，每個 worker 各自付一次（各有一份 lru_cache）。
#
# ★ 為什麼是背景執行緒，不是在 startup 裡同步載
#   同步載只是把痛苦從第一個使用者搬到部署的人，而且 HEALTHCHECK 的
#   start-period 也得跟著調。背景載的話容器照樣 1 秒起來，模型在背後補上。
#
# ★ 萬一使用者比模型快怎麼辦：已經有保護了
#   retrieval.search() 的向量通道包在 try/except 裡，載不到就 ranked["vector"] = []
#   然後其餘九路照常回答。使用者拿到的是稍差一點的排名，不是一片空白 ——
#   而向量在 RRF 裡本來就是最低的 1.0 權重。
#
# ★ 換成 ONNX 之後模型載入與冷查詢都縮短到數秒等級。暖機仍然值得留（冷的
#   第一個查詢仍比熱的慢數倍），但它不再是「服務看起來好了其實要等半分鐘」那種等級的坑。
#
# ★ 順便暖 facets()：它有 300 秒 TTL，但冷的那一次仍要 ~0.5 秒，而它的呼叫端
#   之一是狀態查詢。沒理由讓第一個打 /api/health 的人去付這 0.5 秒。
# --------------------------------------------------------------------------
WARMUP = os.getenv("WARMUP", "1").strip() not in ("0", "false", "no", "")


def _warmup() -> None:
    # llm.diagnose() 一起暖：它的探測有 60 秒 TTL，但冷的那一次要等到 gateway
    # 回應為止 —— gateway 掛掉時就是整整 LLM_PROBE_TIMEOUT。
    # 不暖的話，重啟後第一個打 /api/health 的看板會等滿這段逾時，然後多半判定
    # 「這個服務有問題」，而實際上有問題的是 gateway、而且服務本身是好的。
    for name, fn in (("embedding model", lambda: retrieval.embed_query("暖機")),
                     ("facets", retrieval.facets),
                     ("llm probe", llm.diagnose)):
        t = time.monotonic()
        try:
            fn()
            observability.event("warmup_ready", component=name,
                                duration_ms=round((time.monotonic() - t) * 1000))
        except Exception as e:  # noqa: BLE001 -- 暖機失敗不該讓服務起不來
            observability.event("warmup_failed", component=name, error_type=type(e).__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if WARMUP:
        threading.Thread(target=_warmup, name="warmup", daemon=True).start()
    try:
        yield
    finally:
        for name in ("answer_stream_pool", "a2a_query_pool", "a2a_status_probe", "readiness_probe", "version_probe"):
            resource = getattr(app.state, name, None)
            if resource is not None:
                resource.close()
        db.close()


app = FastAPI(title="Part-rag", version="1.0", lifespan=lifespan)
app.add_middleware(AdmissionMiddleware, limit=int(os.getenv("QUERY_MAX_REQUESTS", "8")))
app.add_middleware(observability.RequestMiddleware)
app.state.a2a_query_pool = query_pool()
stream_pool(app)
service_health.mount(app)
observability.mount(app)


class Filters(BaseModel):
    category: list[str] | None = None
    # PLM 零組件 form fields (see src/taxonomy.PLM_FIELDS)
    prod_l1: list[str] | None = None      # 產品大類
    prod_l2: list[str] | None = None      # 產品中類
    prod_l3: list[str] | None = None      # 產品小類
    item_type: str | None = None          # 料件類型
    eco_std: str | None = None            # 環保標準
    plant: str | None = None              # 建立廠別 (needs the PLM import)
    announce_cat: str | None = None       # 公告大類 (needs the PLM import)
    # 建立人員/修改人員 are substring-matched: the stored value is
    # "<帳號> <中文名>" and nobody types the 中文名 half.
    created_by: str | None = None         # 建立人員 (needs the PLM import)
    modified_by: str | None = None        # 修改人員
    state: str | None = None              # 狀態
    created_after: datetime | None = None
    created_before: datetime | None = None
    pitch: str | None = None              # e.g. PH4.2
    conn_type: str | None = None          # e.g. HSG
    awg: int | None = None
    gen: int | None = None
    pins: int | None = None
    len_min: float | None = None
    len_max: float | None = None
    include_placeholder: bool = False
    # 作廢料號預設不進靠相似度找的通道；盤點時才會想看它們。
    # 打了完整料號的那條路不受這個旗標影響 —— 見 retrieval._filter_sql。
    include_obsolete: bool = False


class SearchReq(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    filters: Filters = Filters()
    top_k: int = Field(default=10, ge=1, le=100)


class ChatReq(SearchReq):
    # Omit to start a new conversation; the id comes back on the `meta` event.
    conv_id: int | None = None


class RenameReq(BaseModel):
    title: str = Field(min_length=1, max_length=80)


# Quick-ask chips shown by the UI. Served from the API rather than hard-coded in
# the page so they stay in step with what the retriever can actually do -- each
# one exercises a different channel.
# 展示版：範例問題清單取材自實際語料，已移除。
EXAMPLE_PROMPTS: list = []


def _find(query: str, filters: dict, top_k: int) -> dict[str, Any]:
    """One entry point for both retrieval modes.

    A question with a totality marker (列出/所有/共有幾筆) is answered by
    deterministic SQL in src/aggregate.py, not by ranked retrieval -- see that
    module's docstring for why a ranked window is the wrong shape for the
    question. Everything else goes down the hybrid pipeline unchanged.
    """
    spec = aggregate.plan(query, filters)
    if spec:
        return aggregate.run(spec)
    return retrieval.search(query, filters, top_k)


@app.get("/api/health")
def health() -> dict[str, Any]:
    db: dict[str, Any]
    try:
        f = retrieval.facets()
        db = {"ok": True, **f["stats"]}
    except Exception as e:
        db = {"ok": False, "error": str(e)}
    # `ok` / `detail` / `model` 的形狀一個字都沒動（有回歸測試逐鍵比對）；
    # diag 是純新增，讓看板不必自己認識 gateway 的錯誤格式就能顯示
    # 「該做什麼、誰能修、要不要換 API key」。與 a2a._status() 的 llm_diag 同一份。
    d = llm.diagnose()
    detail = "ok" if d["ok"] else f"{d['title']}：{d['detail']} → {d['fix']}"
    return {"db": db,
            "llm": {"ok": d["ok"], "detail": detail, "model": llm.MODEL, "diag": d},
            "sync": _sync_health()}


# 超過這麼久沒有任何 stage 成功就標 stale —— 不然「排程壞了三週」跟「昨天剛
# 跑完」在畫面上長得一樣（理由與 src/sync_state.py 的 last_run/last_success
# 分家相同）。門檻本身住在 src/sync_state.py，/a2a/health 讀的是同一個值。
SYNC_STALE_HOURS = sync_state.STALE_HOURS

# 展示版：公司系統對接已移除（原本取自來源設定模組的預設 profile）
DEFAULT_PROFILE = "default"


def _sync_health() -> dict[str, Any]:
    try:
        summary = sync_state.summarize(sync_state.report(DEFAULT_PROFILE), DEFAULT_PROFILE)
    except Exception as exc:
        return {"ok": False, "detail": f"無法讀取同步狀態：{type(exc).__name__}"}
    return {**summary, "detail": summary["note"],
            "stale_after_hours": sync_state.STALE_HOURS,
            "hours_since_success": summary["hours_since_last_success"],
            "stages": [{"stage": name, **row} for name, row in summary["stages"].items()]}


@app.get("/api/facets")
def get_facets() -> dict[str, Any]:
    return retrieval.facets()


@app.post("/api/search")
def api_search(req: SearchReq) -> dict[str, Any]:
    return _find(req.query, req.filters.model_dump(exclude_none=True), req.top_k)


@app.get("/api/export")
def api_export(query: str = Query(min_length=1, max_length=2000),
               limit: int = Query(default=200_000, ge=1, le=200_000)) -> StreamingResponse:
    """The complete result set of a 列出/所有 question, as CSV.

    Chat cannot hold tens of thousands of part numbers, so the listing block hands back a link
    to this instead of pretending 25 rows were the answer.
    """
    spec = aggregate.plan(query)
    if not spec:
        raise HTTPException(
            400, "此問題不是清單查詢；請用「列出所有…」的問法，或改用 /api/search"
        )
    name = quote(f"partrag_{spec.target_col or 'list'}.csv")
    return StreamingResponse(
        aggregate.csv_iter(spec, limit),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{name}"},
    )


@app.get("/api/values/{column}")
def api_values(column: str, limit: int = Query(default=500, ge=1, le=5000)) -> dict[str, Any]:
    """Distinct values of one PLM field, with counts.

    Feeds the 建立人員 filter dropdown, and answers "系統知道哪些人" directly --
    which is how a user checks whether a missing answer means "no such part" or
    "that person's parts were never imported".
    """
    if column not in aggregate.GROUPABLE:
        raise HTTPException(400, f"{column} 不是可列舉的欄位")
    label = taxonomy.field_label(column)
    if not values.column_counts().get(column):
        return {
            "column": column, "label": label, "loaded": 0, "values": [],
            "message": f"{retrieval.NOT_COVERED}：{label} 尚未匯入",
        }
    # Built directly rather than parsed out of a synthetic question -- the column
    # is already known here, so going through plan() would only add a way to fail.
    spec = aggregate.ListSpec(
        query=f"列出所有{label}",
        qnorm="",
        kind="values",
        count_only=False,
        matches=[],
        target_col=column,
        target_label=label,
    )
    out = aggregate.run(spec, sample=0)
    return {
        "column": column,
        "label": label,
        "loaded": values.column_counts()[column],
        "values": (out["listing"].get("values") or [])[:limit],
    }


@app.get("/api/part/{item_number}")
def api_part(item_number: str) -> dict[str, Any]:
    row = retrieval.get_part(item_number)
    if not row:
        # A bare 404 is a dead end for the caller. Hand back the block's real
        # neighbours so a number PLM has but the corpus does not still leads
        # somewhere useful.
        near = retrieval.search(item_number, {}, 8)
        raise HTTPException(
            404,
            {
                "detail": f"找不到料號 {item_number}",
                "hint": "此料號可能是 PLM 新開、尚未進到語料匯出檔",
                "nearby": [
                    {"item_number": r["item_number"], "name": r["name"]}
                    for r in near["results"]
                ],
            },
        )
    return row


# --------------------------------------------------------------------------
# 寫入端點
#
# ★ 為什麼要一把 token，而「沒有登入機制」卻是可接受的
#   讀取端點洩漏的是料號資料，寫入端點洩漏的是**改動語料的權力** —— 任何連得到
#   :8000 的人都能改品名、標作廢，而檢索結果會照著改動走，沒有任何痕跡。
#   「誰在問」不重要（那是登入解的事），「誰能改」不能不管。
#
# ★ 預設是關閉，不是預設開放
#   WRITE_TOKEN 沒設 → 這三支端點回 503。忘記設定的後果是「寫不進去」，
#   而不是「全世界都寫得進去」。
WRITE_TOKEN = os.getenv("WRITE_TOKEN", "").strip()


def require_write(x_write_token: str | None = Header(default=None)) -> None:
    if not WRITE_TOKEN:
        raise HTTPException(503, {
            "detail": "寫入端點未啟用",
            "hint": "在 .env 設 WRITE_TOKEN=<夠長的隨機字串> 後重啟服務",
        })
    # 常數時間比較：`!=` 遇到第一個不同的字元就返回，回應時間會洩漏猜對了幾個字。
    if not hmac.compare_digest((x_write_token or "").encode(), WRITE_TOKEN.encode()):
        raise HTTPException(401, {"detail": "X-Write-Token 不正確或未提供"})


class PartWriteReq(BaseModel):
    # 省略 name = 只改 PLM 欄位；新增一顆料號時必填。
    name: str | None = Field(default=None, min_length=1, max_length=500)
    # 欄位名可用英文 column 或表單原文標題（「建立廠別」「公告大類」…），
    # 與 CSV 匯入認得的完全一樣 —— 見 src/write.normalize_fields。
    fields: dict[str, Any] | None = None


class ObsoleteReq(BaseModel):
    reason: str | None = Field(default=None, max_length=200)


@app.put("/api/part/{item_number}", dependencies=[Depends(require_write)])
def api_part_upsert(item_number: str, req: PartWriteReq) -> dict[str, Any]:
    """新增或更新一顆料號。改品名會連 name_norm / search_text / 分類 / 屬性 /
    embedding 一起重算 —— 與 -m src.enrich --plm-csv 走同一套規則。"""
    try:
        return write.upsert(item_number, req.name, req.fields)
    except write.WriteError as e:
        raise HTTPException(400, {"detail": str(e)}) from e


@app.post("/api/part/{item_number}/obsolete", dependencies=[Depends(require_write)])
def api_part_obsolete(item_number: str, req: ObsoleteReq) -> dict[str, Any]:
    """標作廢（不刪除）。作廢後靠相似度找不到它，點名問仍查得到並標明已作廢。"""
    try:
        return write.set_obsolete(item_number, req.reason)
    except write.WriteError as e:
        raise HTTPException(404, {"detail": str(e)}) from e


@app.post("/api/part/{item_number}/restore", dependencies=[Depends(require_write)])
def api_part_restore(item_number: str) -> dict[str, Any]:
    try:
        return write.restore(item_number)
    except write.WriteError as e:
        raise HTTPException(404, {"detail": str(e)}) from e


@app.get("/api/plm-fields")
def api_plm_fields() -> dict[str, Any]:
    """The PLM 零組件 form's fields and where each value comes from.

    Lets the UI show why 公告大類 has no value (import-only) while 產品中類 does
    (derived), instead of both just looking empty.
    """
    return {
        "fields": [
            {
                "label": f.label,
                "column": f.column,
                "origin": f.origin,
                "required": f.required,
                "note": f.note,
            }
            for f in taxonomy.PLM_FIELDS
        ],
        "corpus_constants": {
            taxonomy.field_label(c): {"value": v, "why": w}
            for c, (v, w) in taxonomy.CORPUS_CONSTANTS.items()
        },
    }


# --------------------------------------------------------------------------
# 工程圖（見 src/drawing.py）
# --------------------------------------------------------------------------
def _pdf_dir(profile: str) -> Path:
    """這個來源的 PDF 目錄。

    不再寫死 data/pdf：排程支援多個來源設定，每個 profile
    的 PDF 落在各自的目錄，DB 的 local_file 只在該目錄底下唯一。

    解析規則必須與 `/a2a/file/{id}` 共用同一份（見 src/a2a.py）—— 兩支路由
    對「這份圖在哪」給不同答案的話，同一顆按鈕從原生介面點得開、從 Hub 點就 404。
    """
    # 展示版：公司系統對接已移除（原本由來源設定模組解析每個 profile 的目錄）
    return (Path("data/pdf") / profile).resolve()


@app.get("/api/drawing/{item_number}")
def api_drawing(item_number: str) -> dict[str, Any]:
    """這個料號的工程圖：標題欄 + 材料表 + PDF 連結。"""
    with retrieval.pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        rows = retrieval.drawings_for(cur, [item_number])
    if not rows:
        # 比照 api_part()：404 但不是死路。工程圖只覆蓋一小部分料號，
        # 「沒有圖」是最常見的答案，講清楚它跟「沒有這個料號」不是同一件事。
        raise HTTPException(
            404,
            {
                "detail": f"{retrieval.NOT_COVERED}：料號 {item_number} {retrieval.NO_DRAWING}",
                "hint": "工程圖目前只覆蓋一小部分料號；"
                        "這個料號本身可能是存在的，用 /api/part/ 查",
            },
        )
    return {"item_number": item_number, "drawings": rows}


@app.get("/api/drawing/file/{drawing_id}")
def api_drawing_file(drawing_id: int) -> FileResponse:
    """原始 PDF，瀏覽器直接開（不下載）。

    ★ 檔名只從 DB 的 local_file 取，**不接受呼叫端傳路徑**，而且取出來之後
      還要 resolve() 確認仍在 PDF 目錄底下。DB 裡的值是下載端
      清過的，但「上游清過了」不是拒絕路徑穿越的理由 —— 這條路由的參數是
      整數 id 正是為了讓檔名永遠不經過網路。
    """
    with retrieval.pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT local_file, source_profile FROM partrag.drawings WHERE id = %s",
            (drawing_id,),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(404, f"找不到工程圖 {drawing_id}")

    base = _pdf_dir(row[1])
    path = (base / row[0]).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise HTTPException(404, f"工程圖 {drawing_id} 的檔案不在 {base} 底下")
    return FileResponse(
        path,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename*=UTF-8\'\'{quote(path.name)}'},
    )


@app.get("/api/similar/{item_number}")
def api_similar(item_number: str, top_k: int = Query(default=10, ge=1, le=100)) -> dict[str, Any]:
    return {"item_number": item_number, "results": retrieval.find_similar(item_number, top_k)}


@app.get("/api/examples")
def api_examples() -> dict[str, Any]:
    return {"examples": EXAMPLE_PROMPTS}


# --------------------------------------------------------------------------
# conversations
# --------------------------------------------------------------------------
@app.get("/api/conversations")
def api_conversations(owner: str = Depends(sessions.owner_id)) -> dict[str, Any]:
    return {"conversations": chat.list_conversations(owner_id=owner)}


@app.post("/api/conversations")
def api_conversation_new(owner: str = Depends(sessions.owner_id)) -> dict[str, Any]:
    return chat.create_conversation(owner_id=owner)


@app.get("/api/conversations/{conv_id}")
def api_conversation(conv_id: int, owner: str = Depends(sessions.owner_id)) -> dict[str, Any]:
    conv = chat.conversation(conv_id, owner_id=owner)
    if not conv:
        raise HTTPException(404, f"找不到對話 {conv_id}")
    return {**conv, "messages": chat.get_messages(conv_id, owner_id=owner)}


@app.patch("/api/conversations/{conv_id}")
def api_conversation_rename(conv_id: int, req: RenameReq,
                            owner: str = Depends(sessions.owner_id)) -> dict[str, Any]:
    conv = chat.rename(conv_id, req.title, owner_id=owner)
    if not conv:
        raise HTTPException(404, f"找不到對話 {conv_id}")
    return conv


@app.delete("/api/conversations/{conv_id}")
def api_conversation_delete(conv_id: int, owner: str = Depends(sessions.owner_id)) -> dict[str, Any]:
    with chat.claim_turn(conv_id, owner_id=owner) as turn:
        if not turn.delete():
            raise HTTPException(404, f"找不到對話 {conv_id}")
    return {"deleted": conv_id}


@app.post("/api/chat")
async def api_chat(req: ChatReq, request: Request,
                   owner: str = Depends(sessions.owner_id)) -> StreamingResponse:
    """SSE: conversation metadata, retrieval results, then streamed LLM text.

    Event order is deliberate:
      meta    -> conv_id, so the client can attach to a conversation it did not
                 create explicitly, and the retrieval query actually used
      results -> emitted BEFORE the model is contacted, so the client still gets
                 the full hit list when the gateway is down
      delta*  -> answer text
      title   -> only on the first turn of a new conversation; the naming call
                 runs after the answer so it never delays the reply
      done
    """
    deadline = for_message(request)
    job = stream_pool(app).reserve(deadline)
    conv = turn = None
    conv_id = None
    history, rq, merged, first_turn, found = [], req.query, False, False, {}

    def sse(event: str, data: Any) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

    final_answer = {"ok": False, "text": "", "verified": None, "persist_attempted": False}

    def persist():
        if final_answer["persist_attempted"]:
            return
        final_answer["persist_attempted"] = True
        completed = final_answer["ok"] and deadline.reason not in {"timeout", "disconnected", "shutdown"}
        try:
            turn.add_message("assistant", final_answer["text"] if completed else "",
                             {**found, "generation": {
                                 "status": "completed" if completed else deadline.reason or "failed",
                                 "verified": final_answer["verified"],
                                 "request_id": observability.request_id()}})
        except Exception as exc:
            observability.event("chat_persist_error", error_type=type(exc).__name__)
            raise PersistenceError("conversation persistence failed") from None
        finally:
            turn.close()

    def gen():
        status: dict[str, Any] = {"ok": False, "error": "串流未開始"}
        text = ""
        answer_stream = None
        ok = False
        diag = None
        try:
            yield sse("meta", {"conv_id": conv_id, "title": conv["title"],
                               "retrieval_query": rq, "followup_merged": merged,
                               "request_id": observability.request_id()})
            yield sse("results", found)
            try:
                answer_stream = llm.stream_answer(req.query, found["results"], found, history, status=status)
                for chunk in answer_stream:
                    deadline.remaining()
                    if status.get("ok") is False and chunk.startswith("[LLM 無法使用："):
                        continue
                    if chunk:
                        text += chunk
                        yield sse("delta", {"t": chunk})
            except Exception as e:
                status = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            deadline.remaining()
            ok = bool(status.get("ok"))
            final_answer.update(ok=ok, text=text, verified=status.get("verified"))
            diag = None if ok else llm.diagnose(status.get("error"))
            if diag:
                yield sse("delta", {"t": f"[LLM 無法使用：{diag['title']}。{diag['fix']}]"})
                yield sse("error", {"error": f"{diag['title']}：{diag['detail']}", "diag": diag})
        finally:
            if answer_stream is not None:
                answer_stream.close()

        # Save before done: a client is allowed to close the socket on that event.
        persist()
        deadline.remaining()

        if first_turn:
            title = chat.ensure_title(conv_id, req.query, owner_id=owner)
            if title:
                yield sse("title", {"conv_id": conv_id, "title": title})

        # llm_error / llm_diag are additive; every pre-existing key of every
        # event is unchanged. The structured half is the same shape a2a and
        # /api/health hand out, so a client learns one format, not three.
        # verify 區塊只在核對真的跑過時出現（LLM_VERIFY=1 且這一題經過模型）。
        # first_pass 才是要看的數字：它是「模型第一次就沒有講出資料裡沒有的東西」
        # 的比例，掉下去代表該重跑一次 `python -m src.evalrun --layer answer`。
        # violations 是**最終仍未通過**的項目（已經重寫過一次），前端不必畫 ——
        # 回答已改成核對失敗提示，但看板與日誌仍需要原始違規種類。
        done: dict[str, Any] = {
            "llm_available": ok,
            "completed": ok and status.get("verified") is not False,
            "llm_error": None if ok else f"{diag['title']}：{diag['fix']}",
            "llm_diag": diag,
        }
        if "first_pass" in status:
            done["verify"] = {
                "ok": status.get("verified"),
                "first_pass": status.get("first_pass"),
                "attempts": status.get("attempts"),
                "violations": status.get("violations") or [],
            }
        yield sse("done", done)

    def prepare():
        nonlocal conv, turn, conv_id, history, rq, merged, first_turn, found
        user_saved = False
        with using(deadline):
            try:
                deadline.remaining()
                conv = chat.conversation(req.conv_id, owner_id=owner) if req.conv_id else None
                if req.conv_id is not None and conv is None:
                    raise HTTPException(404, f"找不到對話 {req.conv_id}")
                if conv is None:
                    conv = chat.create_conversation(owner_id=owner)
                conv_id = conv["id"]
                turn = chat.claim_turn(conv_id, owner_id=owner)
                turn.repair_pending()
                history = chat.history_for_llm(conv_id, owner_id=owner)
                rq, merged = chat.retrieval_query(history, req.query,
                    previous_query=chat.last_retrieval_query(conv_id, owner_id=owner))
                first_turn = not any(m["role"] == "user" for m in history)
                deadline.remaining()
                turn.add_message("user", req.query, {"retrieval_query": rq})
                user_saved = True
                found = _find(rq, req.filters.model_dump(exclude_none=True), req.top_k)
                deadline.remaining()
                # Hand ownership to a cleanup-capable producer BEFORE returning
                # the preparation result. A cancelled await cannot strand a
                # saved user message or a database lock in a discarded result.
                job.start(gen, cleanup=persist)
            except BaseException:
                if user_saved:
                    with using(None):
                        try:
                            turn.add_message("assistant", "", {"generation": {
                                "status": deadline.reason or "retrieval_failed",
                                "request_id": observability.request_id()}})
                        except Exception as exc:
                            observability.event("chat_persist_error", error_type=type(exc).__name__)
                if turn is not None:
                    turn.close()
                raise

    try:
        await app.state.a2a_query_pool.run(prepare, timeout=remaining_http(deadline))
    except BaseException as exc:
        job.stop("timeout" if isinstance(exc, RequestStopped) or
                 isinstance(exc, HTTPException) and exc.status_code == 504 else "cancelled")
        if isinstance(exc, RequestStopped):
            raise HTTPException(504, "回答超過時間上限") from None
        raise
    response = ManagedStreamingResponse(job)
    sessions.stamp(request, response)
    return response


# --------------------------------------------------------------------------
# A2A (/a2a/*) -- additive only; see src/a2a.py.
#
# Mounts the shared agent contract used by Agent Hub's unified frontend and by
# the upstream BOM agent. Everything above this line is untouched: the existing
# /api/* routes keep working byte-for-byte.
#
# Wrapped in try/except on purpose. agentkit lives in a sibling directory
# (../Agent_Hub_RAG_Frontend) and is deliberately NOT installed into this venv, so a
# missing or moved Agent_Hub_RAG_Frontend must degrade to "no /a2a routes", never to a
# service that will not start.
# --------------------------------------------------------------------------
try:
    from src import a2a

    app.state.a2a_mounted = a2a.mount(app)
except Exception as _e:  # pragma: no cover
    app.state.a2a_mounted = False
    observability.event("a2a_mount_failed", error_type=type(_e).__name__)
