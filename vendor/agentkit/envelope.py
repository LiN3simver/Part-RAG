"""A2A 統一資料信封（protocol `a2a/1`）。

四個專案的原生回傳形狀本來完全不同：

    partrag  /api/search  {query, filters, top_k} -> {results:[{item_number, name, ...}]}
    rfq      /api/search  {q, limit}              -> {results:[{rfq_no, spec_desc, ...}]}
    material 尚未有 API
    bom      尚未有 API

    SSE 事件名也不同：partrag 用 `delta`，rfq 用 `token`。

要共用同一個前端，就必須有一層「誰都能渲染」的中間形狀。這個檔就是那層。

★ 設計原則：原始列一定原封不動放在 `raw`。
  正規化只是為了讓前端能畫出通用卡片，不是取代原始資料。
  任何 agent 想給另一個 agent 完整欄位（例如 material 要拿 rfq 的 24 個成本欄），
  讀 `raw` 就好，不必為此擴充 Hit 的欄位。

★ 空值語意：kv 區塊裡 `None` 表示「未填」，`0` 表示「填了 0」。
  這兩者在 RFQ-RAG 是硬性需求（多數紀錄沒有成本明細，把「未填」畫成 0
  會讓人以為成本真的是零）。所以序列化一律留 JSON null，不要在後端轉成 ""
  或 0，前端才有辦法分開顯示。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

PROTOCOL = "a2a/1"

# --------------------------------------------------------------------------
# SSE 事件詞彙（統一版）
#
# 沿用 partrag 既有的 meta / results / delta / title / done，因為那份語意
# 已經在前端跑過一輪；rfq 的 `token` 只在它自己的 /api/chat 保留，
# /a2a/message 一律吐 `delta`。新增的只有 `trace`（委派軌跡）。
#
# 順序是刻意的，四個專案都必須照這個順序送：
#   meta    -> agent_id / conv_id / 實際送去檢索的字串
#   results -> ★ 在聯絡 LLM 之前就送出，gateway 掛掉時使用者仍拿到完整資料
#   trace*  -> 委派給下游 agent 的每一步（開始/結束各一次）
#   delta*  -> 答案文字
#   error   -> 可回復的錯誤（不中斷串流）
#   done    -> 收尾，帶 llm_available
# --------------------------------------------------------------------------
EV_META = "meta"
EV_RESULTS = "results"
EV_TRACE = "trace"
EV_DELTA = "delta"
EV_TITLE = "title"
EV_ERROR = "error"
EV_DONE = "done"

Tone = Literal["neutral", "green", "yellow", "warn", "good"]


def sse(event: str, data: Any) -> str:
    """一個 SSE 事件。default=str 是為了讓 Decimal / datetime 不炸掉串流。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@dataclass
class Block:
    """卡片裡的一個結構化區塊。

    kind:
      kv    rows = [[label, value], ...]      value 為 None 即「未填」
      text  text = "..."                      等寬呈現，用於已排版好的區塊
      table rows = [[...], ...] + head = [..]
    """
    kind: Literal["kv", "text", "table"]
    title: str = ""
    tone: Tone = "neutral"
    rows: list[list[Any]] = field(default_factory=list)
    head: list[str] = field(default_factory=list)
    text: str = ""


@dataclass
class Hit:
    """一筆檢索結果的通用形狀。

    `id` 一定要是「可以拿去別的 agent 再查一次」的鍵（料號 / RFQ 編號），
    前端的「帶去問其他 agent」按鈕就是靠它串接。
    """
    id: str
    title: str = ""
    subtitle: str = ""
    body: str = ""
    score: float | None = None
    why: list[str] = field(default_factory=list)
    badges: list[list[str]] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    # 資料來源的原始檔案：[[顯示文字, 絕對網址], ...]
    #
    # ★ 為什麼要有這個欄位
    #   卡片上的數字不全都是「從欄位讀出來的」—— rfq 的成本分析附件有一部分
    #   是掃描件，靠 OCR 讀的。使用者要能一鍵打開原檔自己核對，否則沒辦法
    #   判斷該不該相信那個價格。這是可稽核性，不是裝飾。
    #
    #   網址必須是**絕對**的：卡片在 Hub（:8600）上畫，相對路徑會打到 Hub 去，
    #   而檔案是各 agent 自己在服務。
    links: list[list[str]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolSpec:
    """Anthropic Messages API 的 tool schema + A2A 需要的兩個額外欄位。

    格式刻意對齊 Anthropic，因為公司的 LLM gateway
    走的就是這個介面，而且支援原生 tool use。
    這樣同一份 schema 可以直接餵給模型，不用再轉一層。

    ready=False 的工具仍然會出現在 card 裡（讓上游知道「未來會有」），
    但 tools_for_model() 不會把它掛給模型 —— 掛了只會讓模型呼叫失敗後亂猜。
    """
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Any = None          # Callable[[dict], dict]；不序列化
    ready: bool = True
    not_ready_reason: str = ""

    def public(self) -> dict[str, Any]:
        d = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "ready": self.ready,
        }
        if not self.ready and self.not_ready_reason:
            d["not_ready_reason"] = self.not_ready_reason
        return d

    def for_model(self) -> dict[str, Any]:
        """給 LLM 的乾淨 schema（不含 A2A 自己的欄位）。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def tool_ok(result: Any) -> dict[str, Any]:
    return {"ok": True, "result": result}


def tool_err(message: str, **extra: Any) -> dict[str, Any]:
    """工具失敗一律回這個形狀，不要拋例外。

    理由與 RFQ-RAG/src/agent_tools.py 的 dispatch() 相同：agent 迴圈拿到
    例外只能中斷，拿到 {"ok": false, "error": ...} 還能自己決定換工具或改問法。
    """
    return {"ok": False, "error": message, **extra}
