"""agentkit —— 四個 RAG/Agent 專案共用的 A2A 契約。

這個套件**不安裝**。各專案的 `src/a2a.py` 用一段 sys.path 引導把
`..\\Agent_Hub_RAG_Frontend` 加進路徑後 import 它，理由是：

  - 四個專案各有自己的 .venv（Bom-RAG 這次才建）。要「安裝」就得動四個
    venv，而 Part-RAG / RFQ-RAG 是已經在用的服務，動它們的環境是
    這次最不該冒的風險。零安裝的路徑引導，弄壞既有服務的機會是 0。
  - 只有一份原始碼，不會出現四份漂移的複本。

只依賴 fastapi / pydantic / httpx / anthropic / python-dotenv —— 四個 venv
都已經有（Bom-RAG 的 venv 照這份清單建）。

引導失敗時各專案的 a2a.py 會印警告並跳過掛載，主服務照舊啟動。
"""
from __future__ import annotations

from . import client, delegate, envelope, gateway, registry
from .envelope import PROTOCOL, Block, Hit, ToolSpec, sse, tool_err, tool_ok
from .server import AgentSpec, MessageReq, build_router

__all__ = [
    "AgentSpec",
    "Block",
    "Hit",
    "MessageReq",
    "PROTOCOL",
    "ToolSpec",
    "build_router",
    "client",
    "delegate",
    "envelope",
    "gateway",
    "registry",
    "sse",
    "tool_err",
    "tool_ok",
]
