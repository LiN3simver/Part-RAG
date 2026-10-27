"""Conversation persistence and follow-up handling.

Two concerns live here rather than in retrieval.py or main.py:

1.  CRUD over partrag.conversations / partrag.messages, including the one-shot
    LLM titling of a new conversation.

2.  Turning a follow-up into something retrievable. "那 300mm 的呢？" parses to
    a length and nothing else -- retrieval on that alone returns every 300mm
    cable in the corpus, in no useful order. The subject has to come from the
    previous turn.

Public routes always pass the signed session owner. owner_id=None is reserved
for trusted local CLI/acceptance operations, including legacy shared records.
"""
from __future__ import annotations

import json
import re

from psycopg.rows import dict_row

from . import llm
from .extract import normalize, parse_query
from .retrieval import pool
from .chat_turn import claim_turn

MAX_TITLE = 24
# How many prior turns to replay to the model. 6 covers the follow-up chains
# people actually type; beyond that the retrieval block for the current question
# matters far more than older chat.
HISTORY_TURNS = 6


# ---------------------------------------------------------------------------
# follow-up resolution
# ---------------------------------------------------------------------------
# Words that mean "the thing we were just talking about". A question containing
# one of these is almost certainly not self-contained.
_ANAPHORA_RE = re.compile(
    r"那|這|它|他們|上面|前面|剛才|剛剛|一樣|同樣|還有|其他|另外|"
    r"換成|改成|差別|差異|比較|呢[?？]?$|"
    r"这|它们|刚才|刚刚|一样|同样|其他|另外|换成|改成|差别|比较"
)
# Below this length a question rarely carries its own subject: "300mm 的",
# "有更短的嗎", "PH3.0 呢".
SHORT_QUESTION = 18


def is_followup(question: str) -> bool:
    n = normalize(question)
    return bool(_ANAPHORA_RE.search(n)) or len(n) <= SHORT_QUESTION


def retrieval_query(history: list[dict], question: str, *, previous_query: str | None = None) -> tuple[str, bool]:
    """-> (query to retrieve on, whether the previous turn was folded in).

    Carry user-derived context only. Native chat persists its resolved query so
    the subject survives beyond the six turns replayed to the language model.
    """
    if not history or not is_followup(question):
        return question, False
    if parse_query(question)["item_numbers"] and not _ANAPHORA_RE.search(normalize(question)):
        return question, False
    prev = previous_query
    if not prev:
        parts = []
        for message in reversed(history[-(HISTORY_TURNS * 2):]):
            if message["role"] != "user":
                continue
            text = message["content"]
            parts.append(text)
            if not is_followup(text) or parse_query(text)["item_numbers"]:
                break
        prev = " ".join(reversed(parts))
    if not prev:
        return question, False
    # A concrete part identifies the subject by itself. Replaying earlier
    # questions also replays their retrieval intents (AWG, connector, listing),
    # which can turn a drawing-existence follow-up into an unrelated drawing
    # search. Keep all user-named parts for comparisons, but only today's intent.
    subjects = parse_query(prev)["item_numbers"]
    if subjects:
        return f"{' '.join(subjects)} {question}", True
    return f"{prev[:2000]} {question}", True


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------
def _owner(owner_id: str | None, prefix: str = "") -> tuple[str, tuple]:
    return (f"{prefix}owner_id = %s", (owner_id,)) if owner_id is not None else ("true", ())


def list_conversations(limit: int = 200, *, owner_id: str | None = None) -> list[dict]:
    where, params = _owner(owner_id, "c.")
    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""SELECT c.id, c.title, c.titled, c.created_at, c.updated_at,
                      count(m.id) FILTER (WHERE m.role = 'user') AS turns
               FROM partrag.conversations c
               LEFT JOIN partrag.messages m ON m.conv_id = c.id
               WHERE {where}
               GROUP BY c.id
               ORDER BY c.updated_at DESC
               LIMIT %s""",
            (*params, limit),
        )
        return cur.fetchall()


def create_conversation(*, owner_id: str | None = None) -> dict:
    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        if owner_id is None:
            cur.execute("INSERT INTO partrag.conversations DEFAULT VALUES "
                        "RETURNING id, title, titled, created_at, updated_at")
        else:
            cur.execute("INSERT INTO partrag.conversations (owner_id) VALUES (%s) "
                        "RETURNING id, title, titled, created_at, updated_at", (owner_id,))
        return cur.fetchone()


def get_messages(conv_id: int, *, owner_id: str | None = None, limit: int | None = None) -> list[dict]:
    where, params = _owner(owner_id, "c.")
    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""SELECT m.id, m.role, m.content, m.payload, m.created_at
                FROM partrag.messages m JOIN partrag.conversations c ON c.id = m.conv_id
                WHERE m.conv_id = %s AND {where}
                ORDER BY m.id {('DESC LIMIT %s' if limit is not None else '')}""",
            (conv_id, *params, *((limit,) if limit is not None else ())),
        )
        rows = cur.fetchall()
        return list(reversed(rows)) if limit is not None else rows


def conversation(conv_id: int, *, owner_id: str | None = None) -> dict | None:
    where, params = _owner(owner_id)
    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, title, titled, created_at, updated_at "
            f"FROM partrag.conversations WHERE id = %s AND {where}",
            (conv_id, *params),
        )
        return cur.fetchone()


def rename(conv_id: int, title: str, *, owner_id: str | None = None) -> dict | None:
    where, params = _owner(owner_id)
    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""UPDATE partrag.conversations
               SET title = %s, titled = true, updated_at = now()
               WHERE id = %s AND {where}
               RETURNING id, title, titled, created_at, updated_at""",
            (title.strip()[:80] or "新對話", conv_id, *params),
        )
        return cur.fetchone()


def delete(conv_id: int, *, owner_id: str | None = None) -> bool:
    where, params = _owner(owner_id)
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(f"DELETE FROM partrag.conversations WHERE id = %s AND {where}", (conv_id, *params))
        return cur.rowcount > 0


def add_message(
    conv_id: int, role: str, content: str, payload: dict | None = None,
    *, owner_id: str | None = None
) -> int:
    with pool().connection() as conn:
        return _insert_message(conn, conv_id, role, content, payload, owner_id=owner_id)


def _insert_message(conn, conv_id, role, content, payload=None, *, owner_id=None):
    where, params = _owner(owner_id, "c.")
    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO partrag.messages (conv_id, role, content, payload)
               SELECT c.id, %s, %s, %s FROM partrag.conversations c
               WHERE c.id = %s AND {where} RETURNING id""",
            (role, content, json.dumps(payload, ensure_ascii=False, default=str)
             if payload is not None else None, conv_id, *params),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError("conversation is unavailable for this owner")
        mid = row[0]
        cur.execute(
            "UPDATE partrag.conversations SET updated_at = now() WHERE id = %s",
            (conv_id,),
        )
        return mid


def history_for_llm(conv_id: int, *, owner_id: str | None = None) -> list[dict[str, str]]:
    """Prior turns as role/content pairs, oldest first.

    Empty assistant turns are NOT filtered here: llm._messages needs to see the
    pairing to know which user turn went unanswered, and dropping the assistant
    row alone would leave two consecutive user messages.

    The stored retrieval payload is not replayed either -- 12 rendered rows per
    past turn would dominate the context and let the model answer a new question
    from a stale hit list.
    """
    msgs = get_messages(conv_id, owner_id=owner_id, limit=HISTORY_TURNS * 2)
    out = [{"role": m["role"], "content": m["content"]} for m in msgs]
    return out[-(HISTORY_TURNS * 2):]


def last_retrieval_query(conv_id: int, *, owner_id: str | None = None) -> str | None:
    where, params = _owner(owner_id, "c.")
    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"SELECT m.content, m.payload FROM partrag.messages m "
                    f"JOIN partrag.conversations c ON c.id=m.conv_id "
                    f"WHERE m.conv_id=%s AND m.role='user' AND {where} ORDER BY m.id DESC LIMIT 1",
                    (conv_id, *params))
        row = cur.fetchone()
    if not row:
        return None
    payload = row.get("payload") or {}
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload.get("retrieval_query") or None if isinstance(payload, dict) else None


# ---------------------------------------------------------------------------
# auto-titling
# ---------------------------------------------------------------------------
# The instruction goes in the USER turn, not the system prompt. With a system
# prompt of "你是標題產生器" and a bare question as the user turn, this gateway's
# `thinking` block spent its entire budget deliberating over whether the user
# wanted an answer or a title, and returned stop_reason=max_tokens with no text
# -- every conversation silently fell back to the trimmed question. Stated as a
# direct instruction it finishes in a few tokens with end_turn.
_TITLE_PROMPT = (
    f"把下面這個料號查詢問題濃縮成一個不超過 {MAX_TITLE} 字的繁體中文標題"
    "（英文型號、料號保留原文）。只輸出標題本身，不要說明、不要引號、不要句號。\n\n"
    "問題：{q}\n標題："
)


def _fallback_title(question: str) -> str:
    """Trim the question itself. Used when the gateway is unavailable -- a
    truncated question is a worse title than a summary but a much better one
    than '新對話'."""
    t = re.sub(r"\s+", " ", question).strip()
    return (t[:MAX_TITLE] + "…") if len(t) > MAX_TITLE else t


def make_title(question: str) -> str:
    try:
        from .request_budget import checkpoint, client_for_request
        checkpoint()
        msg = client_for_request(llm._client()).messages.create(
            model=llm.MODEL,
            # Generous despite the ~10-token output: the gateway emits a
            # `thinking` block first and charges it against max_tokens.
            max_tokens=min(llm.MAX_TOKENS, 1200),
            messages=[{"role": "user", "content": _TITLE_PROMPT.format(q=question)}],
        )
        checkpoint()
        text = "".join(
            b.text for b in msg.content if getattr(b, "type", None) == "text"
        ).strip()
        # Models like to wrap or explain; keep the first line and strip quotes.
        text = text.splitlines()[0].strip().strip("「」\"'。 ") if text else ""
        if text and len(text) <= MAX_TITLE * 2:
            return text[: MAX_TITLE * 2]
    except Exception as e:
        print(f"[chat] titling unavailable: {type(e).__name__}: {e}")
    return _fallback_title(question)


def ensure_title(conv_id: int, question: str, *, owner_id: str | None = None) -> str | None:
    """Name a conversation once, from its first question. -> new title or None.

    Called after the answer has streamed, so the extra round trip never delays
    the reply the user is waiting for.
    """
    conv = conversation(conv_id, owner_id=owner_id)
    if not conv or conv["titled"]:
        return None
    title = make_title(question)
    where, params = _owner(owner_id)
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE partrag.conversations SET title = %s, titled = true WHERE id = %s AND {where} AND NOT titled",
            (title, conv_id, *params),
        )
    return title
