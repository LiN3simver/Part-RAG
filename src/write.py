"""唯一一條「從外部寫語料」的路徑：新增／更新一顆料號、標作廢、還原。

在此之前 main.py 沒有任何寫語料的 API，更新必須進機器打指令（-m src.enrich）。
這個模組把 enrich 的規則收成三個函式，讓 /api/part 與 CLI 共用同一套行為 ——
兩邊各寫一次的話，API 遲早會少做 enrich 做的某一步（最可能少的就是重算 embedding）。

★ embedding 由呼叫方給模型，不在這裡 new
  enrich 是批次工具，每次跑 new 一個 SentenceTransformer 沒差；API 不行 ——
  一次請求載入一次 BGE-M3 是幾十秒，而服務裡早就有一份載好的
  （retrieval._model()，lru_cache）。所以這裡收一個 embed callable，
  預設就是去拿服務那一份。

★ 欄位白名單
  fields 直接來自 HTTP body。沒有白名單的話 "embedding" 或 "tsv" 這種欄位名
  會被組進 UPDATE 的 SET 裡 —— 前者會塞進一個非向量的值，後者是 GENERATED 欄位，
  寫它會直接讓整筆更新失敗。白名單重用 enrich.PLM_CSV_MAP 的目標欄位，
  所以 API 認得的欄位與 CSV 匯入認得的完全一樣，包括表單原文標題。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Iterable

from psycopg.rows import dict_row

from .db import pool
from .enrich import (
    DATE_COLS,
    INT_COLS,
    NUMERIC_COLS,
    PLM_CSV_MAP,
    _coerce,
    derive,
)
from .extract import is_placeholder, normalize, split_item, categorize

# 可以從外部寫的欄位 = CSV 匯入認得的那些，扣掉身分欄與自己算的欄位。
# item_number 走 URL、name 走專門的改名路徑、其餘衍生欄位一律由 derive() 產生。
_WRITABLE: frozenset[str] = frozenset(PLM_CSV_MAP.values()) - {"item_number", "name"}

# 作廢相關欄位不從這裡寫 —— 它們有自己的函式，帶時間與理由。
_FORBIDDEN = {"embedding", "tsv", "search_text", "plm_text", "is_obsolete",
              "obsoleted_on", "obsolete_reason", "is_placeholder"}


class WriteError(Exception):
    """呼叫端給的資料有問題（400 級），不是伺服器壞了。"""


def _embed_default(texts: list[str]) -> list[Any]:
    from .retrieval import _model

    return list(_model().encode(texts, normalize_embeddings=True, convert_to_numpy=True))


def _invalidate_caches() -> None:
    """寫完語料就讓 facets 的快取失效。

    retrieval.facets() 現在有 300 秒 TTL（見該處說明）。在此之前它每次都重算，
    所以「PUT 一顆新料號 → GET /api/facets 立刻看到 total+1」是成立的；不加這一
    行，同一個服務行程裡自己寫的東西最多要 5 分鐘才看得到自己，那是這層快取造成
    的退步，不是可以接受的取捨。

    只管 facets：values.py 的值索引本來就是純 TTL（那是它既有的設計，不是這次
    改動帶來的），維持原樣。

    延後 import 的理由同 _embed_default —— 這個模組也給 CLI 用，不該在 import
    的時候就把整條檢索路徑拉進來。
    """
    from .retrieval import invalidate_facets

    invalidate_facets()


def normalize_fields(fields: dict[str, Any] | None) -> dict[str, Any]:
    """把外部給的欄位（英文 column 或表單原文標題）轉成 column -> 值。

    走 enrich 的 _coerce，所以日期字串、"1,234" 這種數字、空字串代表 NULL
    的規則與 CSV 匯入完全一致。
    """
    out: dict[str, Any] = {}
    for k, v in (fields or {}).items():
        key = str(k).strip()
        col = PLM_CSV_MAP.get(key, key)
        if col in _FORBIDDEN:
            raise WriteError(f"欄位 {key} 不可從 API 寫入")
        if col not in _WRITABLE:
            raise WriteError(f"未知欄位 {key}")
        out[col] = _coerce(col, v) if v is not None else None
    return out


def get(item_number: str) -> dict | None:
    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT item_number, name, is_placeholder, is_obsolete, obsolete_reason, "
            "plm_synced, (embedding IS NOT NULL) AS has_embedding "
            "FROM partrag.parts WHERE item_number = %s",
            (item_number,),
        )
        return cur.fetchone()


def upsert(
    item_number: str,
    name: str | None = None,
    fields: dict[str, Any] | None = None,
    embed: Callable[[list[str]], Iterable[Any]] | None = None,
) -> dict[str, Any]:
    """新增或更新一顆料號。-> {action, item_number, renamed, reembedded}

    新增一定要給 name（沒有品名就沒有任何可檢索的內容，也算不出分類與屬性）。
    更新時 name 可以省略，代表「只改 PLM 欄位」。
    """
    item_number = (item_number or "").strip(" \t﻿​　").upper()
    if not item_number:
        raise WriteError("料號不可為空")
    cols = normalize_fields(fields)
    name = name.strip() if name else None

    embed = embed or _embed_default
    with pool().connection() as conn:
        # register_vector：這條路可能要寫 embedding，而連線池給的連線預設不認得
        # numpy array -> vector。db.pool() 已經在 configure 裡註冊過，這裡不重複。
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT name FROM partrag.parts WHERE item_number = %s", (item_number,)
            )
            existing = cur.fetchone()

        renamed = reembedded = False
        if existing is None:
            if not name:
                raise WriteError(f"新增 {item_number} 必須提供品名（name）")
            _insert(conn, cur_name=name, item_number=item_number, embed=embed)
            action, renamed, reembedded = "created", False, not is_placeholder(name)
        else:
            action = "updated"
            if name and name != (existing["name"] or "").strip():
                _rewrite_name(conn, item_number, name, embed)
                renamed = True
                reembedded = not is_placeholder(name)

        if cols:
            _set_columns(conn, item_number, cols)
        conn.commit()

    _invalidate_caches()
    return {
        "action": action,
        "item_number": item_number,
        "renamed": renamed,
        "reembedded": reembedded,
        "fields_written": sorted(cols),
    }


def _insert(conn, cur_name: str, item_number: str, embed) -> None:
    d = derive(item_number, cur_name)
    base, variant = split_item(item_number)
    p4, cat = categorize(item_number)
    ph = is_placeholder(cur_name)
    d.update({
        "name": cur_name,
        "name_norm": normalize(cur_name),
        "base_pn": base,
        "variant": variant,
        "prefix4": p4,
        "category": cat,
        "is_placeholder": ph,
        # 佔號列不算向量，與 ingest.embed() / enrich._insert_new 同一個規則。
        "embedding": None if ph else list(embed([normalize(cur_name)]))[0],
    })
    cols = list(d)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO partrag.parts ({','.join(cols)}) "
            f"VALUES ({','.join('%(' + c + ')s' for c in cols)}) "
            "ON CONFLICT (item_number) DO NOTHING",
            d,
        )


def _rewrite_name(conn, item_number: str, name: str, embed) -> None:
    """改名 = 改 name + 重算所有從 name 算出來的東西。理由見 enrich._apply_rename。"""
    d = derive(item_number, name)
    ph = is_placeholder(name)
    d.update({
        "name": name,
        "name_norm": normalize(name),
        "is_placeholder": ph,
        "embedding": None if ph else list(embed([normalize(name)]))[0],
    })
    cols = [c for c in d if c != "item_number"]
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE partrag.parts SET "
            + ", ".join(f"{c} = %({c})s" for c in cols)
            + " WHERE item_number = %(item_number)s",
            d,
        )


def _set_columns(conn, item_number: str, cols: dict[str, Any]) -> None:
    payload = dict(cols, item_number=item_number, plm_synced=True)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE partrag.parts SET "
            + ", ".join(f"{c} = %({c})s" for c in (*cols, "plm_synced"))
            + " WHERE item_number = %(item_number)s",
            payload,
        )


def set_obsolete(item_number: str, reason: str | None = None) -> dict[str, Any]:
    """標作廢。同 src.obsolete，但一次一顆、給 API 用。"""
    item_number = (item_number or "").strip().upper()
    reason = (reason or "").strip() or "API 標記"
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            """UPDATE partrag.parts
               SET is_obsolete = true, obsoleted_on = %s, obsolete_reason = %s
               WHERE item_number = %s AND NOT is_obsolete""",
            (datetime.now(), reason, item_number),
        )
        n = cur.rowcount
        if n == 0:
            # 分兩種：料號不存在（404 級）vs 已經是作廢（冪等，不算錯）
            cur.execute(
                "SELECT is_obsolete FROM partrag.parts WHERE item_number = %s",
                (item_number,),
            )
            row = cur.fetchone()
            if row is None:
                raise WriteError(f"查無此料號：{item_number}")
        conn.commit()
    _invalidate_caches()
    return {"item_number": item_number, "is_obsolete": True, "changed": n == 1,
            "reason": reason}


def restore(item_number: str) -> dict[str, Any]:
    item_number = (item_number or "").strip().upper()
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            """UPDATE partrag.parts
               SET is_obsolete = false, obsoleted_on = NULL, obsolete_reason = NULL
               WHERE item_number = %s AND is_obsolete""",
            (item_number,),
        )
        n = cur.rowcount
        if n == 0:
            cur.execute(
                "SELECT 1 FROM partrag.parts WHERE item_number = %s", (item_number,)
            )
            if cur.fetchone() is None:
                raise WriteError(f"查無此料號：{item_number}")
        conn.commit()
    _invalidate_caches()
    return {"item_number": item_number, "is_obsolete": False, "changed": n == 1}
