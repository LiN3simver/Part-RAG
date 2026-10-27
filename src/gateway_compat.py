"""Carry generation policy in-band for the internal LLM gateway compatibility shim.

Changing `system` while retaining a similar user message did not reliably change
the answer. Repeat the same system policy in the message body and
identify the other generation settings. Preserve the original SDK system field,
tool results and request deadlines. No vendor edits.
"""
from __future__ import annotations

import hashlib
import json


_POLICY_FIELDS = ("model", "system", "max_tokens", "temperature", "top_p", "top_k",
                  "tools", "tool_choice", "thinking", "stop_sequences")


def policy_messages(kwargs: dict, mode: str) -> list[dict]:
    """Copy caller-owned history, including tool results, before adding identity."""
    policy = {key: kwargs[key] for key in _POLICY_FIELDS if key in kwargs}
    policy["mode"] = mode
    digest = hashlib.sha256(json.dumps(policy, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode("utf-8")).hexdigest()
    system = kwargs.get("system") or ""
    if not isinstance(system, str):
        system = "\n".join(block.get("text", "") for block in system if block.get("type") == "text")
    marker = (f"[request-policy:{digest}]（生成設定識別碼，回答請勿引用。）\n"
              f"【本次系統指令，與 system 欄位相同】\n{system}\n【本次訊息】\n")
    messages = [dict(message) for message in kwargs["messages"]]
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message["content"]
        if isinstance(content, str):
            message["content"] = marker + content
        else:
            # Anthropic requires tool_result blocks before any text in the
            # next user turn. Append policy without moving or editing results.
            message["content"] = [*content, {"type": "text", "text": marker}]
        break
    return messages


class _Messages:
    def __init__(self, original):
        self._original = original

    def create(self, **kwargs):
        return self._original.create(**{**kwargs, "messages": policy_messages(kwargs, "create")})

    def stream(self, **kwargs):
        return self._original.stream(**{**kwargs, "messages": policy_messages(kwargs, "stream")})

    def __getattr__(self, name):
        return getattr(self._original, name)


class _Client:
    def __init__(self, original):
        self._original = original
        self.messages = _Messages(original.messages)

    def with_options(self, **kwargs):
        return guard_client(self._original.with_options(**kwargs))

    def __getattr__(self, name):
        return getattr(self._original, name)


def guard_client(client):
    return client if isinstance(client, _Client) else _Client(client)
