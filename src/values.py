"""Resolve a free-text query token to an exact PLM field value, from the DB.

src/taxonomy.py holds a HAND-WRITTEN table (VALUE_TERMS) of form values that
identify their field: <公告代碼> -> 公告大類, 100_示例廠 -> 建立廠別. That works
for a closed vocabulary, and stops working the moment the value set is open --
建立人員 is a list of people that grows every time someone joins, and nobody is
going to keep a dict of names in sync with PLM.

So this module builds the same mapping by asking the corpus what values it
actually holds:

    "ALICE 建立的线束"  ->  created_by = 'Alice_Wang 王小明'
    "100_示例厂"       ->  plant      = '100_示例廠'

Two consequences worth stating, because they are the whole point:

1.  A name that is NOT in the index cannot be resolved -- which is the honest
    answer ("目前系統尚未包含該資料"), not a fuzzy match onto the nearest
    colleague. `unresolved_person_like` reports that case explicitly.

2.  Nothing here is embedded. A person's name is an equality predicate; the
    dense channel has no opinion about who created a part, and asking it for one
    returns whichever row happens to mention a similar-looking token.
"""
from __future__ import annotations

import re
import time
from typing import NamedTuple

from psycopg.rows import dict_row

from . import taxonomy as T
from .db import pool
from .extract import CJK, normalize

# Columns whose values are worth matching a bare query token against. Ordered:
# an ambiguous token resolves to the earlier column, so 建立人員 wins over
# 修改人員 for a bare name -- "Alice 的料號" almost always means what Alice made.
INDEXED_COLUMNS: tuple[str, ...] = (
    "created_by", "modified_by", "locked_by",
    "plant", "announce_cat", "lot_control",
    "item_class", "item_subclass", "lifecycle", "state",
    "prod_l1", "prod_l2", "prod_l3", "item_type", "eco_std", "category",
)

# Columns whose values are also split into word-level keys, so "Alice" reaches
# "Alice_Wang 王小明" and "示例廠" reaches "100_示例廠".
#
# Deliberately NOT the taxonomy columns: prod_l2's words ("Cable", "Wire") are
# generic, and taxonomy.L2_ALIASES already resolves them properly. Splitting
# them here would make every query naming "cable" carry a prod_l2 filter.
_SPLIT_COLUMNS = frozenset({"created_by", "modified_by", "locked_by", "plant"})

# Columns whose non-null count is reported, which is a WIDER set than the ones
# worth value-indexing above.
#
# These two lists were one list, and that conflation made the pipeline lie: a
# numeric column like std_seconds is deliberately not value-indexed (indexing
# "700" as a searchable field value would make every query mentioning 700 a
# 標準工時 query), so column_counts() returned nothing for it, and
# retrieval._plm_notes read the missing entry as 0 and announced
# "標準工時(秒) 尚未從 PLM 匯入語料庫" over a fully imported column. Whether a value
# is worth indexing and whether it was imported are different questions.
COUNTED_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *INDEXED_COLUMNS,
            *(f.column for f in T.PLM_FIELDS if f.origin == "import" and f.column),
        )
    )
)

# A column with more distinct values than this is not a classification, it is
# free text; indexing it would make every query match something.
MAX_DISTINCT = 3000

# Rebuilt on demand. The corpus only changes when src/enrich.py runs, but that
# does not restart the server, so a short TTL beats a permanent cache.
TTL_SECONDS = 300.0

# Word-level keys that are too generic to be evidence of a field value.
_KEY_STOP = frozenset({
    "CABLE", "WIRE", "HARNESS", "CONN", "ASSY", "TYPE", "NEW", "OLD",
    "ADMIN", "USER", "TEST", "PLM", "SYSTEM",
    # ★ 自家公司名（展示版以 OURCO 代替）同時是示意廠別 '200_OURCO 分廠' 的
    #   一個詞級鍵。使用者講「OURCO 料號」指的是**自家料號**，不是那個分廠。
    #   不擋掉的話，任何一句提到公司名的問話都會被加上那個廠別的欄位加權，
    #   把真正的答案擠出前 10 名（客戶料號反查整組失準）。
    #   真的要查那個廠仍然查得到 —— 完整的廠別值本身還在索引裡。
    "OURCO",
    "成品", "半成品", "零件", "料号", "料號", "产品", "產品",
})


class FieldMatch(NamedTuple):
    """One PLM field pinned to one or more exact values by the query."""
    column: str
    values: list[str]   # ORed: a shared surname legitimately matches two colleagues
    label: str          # the form's printed label, e.g. 建立人員
    via: str            # the query token that matched, for the UI's "why"


class _Index(NamedTuple):
    # (match_key, column, canonical_value); longest key first
    keys: list[tuple[str, str, str]]
    counts: dict[str, int]           # column -> non-null row count
    built_at: float


_cache: _Index | None = None


def _split_keys(value_norm: str) -> list[str]:
    """Word-level keys for a value. Pure digits are dropped: plant
    '100_示例廠' would otherwise make every query mentioning 100mm a plant
    query."""
    out: list[str] = []
    for tok in re.split(r"[\s_/,()（）\-]+", value_norm):
        tok = tok.strip()
        if not tok or tok in _KEY_STOP or tok.isdigit():
            continue
        cjk = bool(re.search(f"[{CJK}]", tok))
        if len(tok) >= (2 if cjk else 3):
            out.append(tok)
    return out


def _indexable(key: str) -> bool:
    """這個字串夠不夠格當成「使用者點名了這個欄位的值」的證據。

    ★ 這道閘門是「換個問法就開始亂答」最貴的那一個案例的解法
      批號管控（lot_control）的值是 N / Y / 0 / Yes，單字元的 N、Y、0 全部
      進了索引。而 boundary_hit 的邊界是 (?<![A-Z0-9])，「PH3.0」裡小數點後面
      那個 0 兩側剛好一個是 '.'、一個是字串結尾 —— 於是：

        「我要查詢PH3.0 HSG相關料號」-> field_matches = [lot_control='0']

      lot_control='0' 的那幾筆料號因此各拿一份 _attr_bonus，把毫不相干的
      線材頂到第一名，命中理由還寫著「批號管控 0」——
      一個使用者從來沒有問過的欄位。這比查不到更糟：它看起來像答案。

      規則是「純數字不算證據、單一字元不算證據」，跟 _split_keys 對詞級鍵的
      判準同一套（那邊早就丟掉純數字了，只是全值鍵沒有走同一道關）。
      GC（2 個字母、環保標準）要留著，所以拉丁下限是 2 不是 3；
      單字元的 N/Y/0 與純數字的廠別代碼一律不進索引 —— 真的要查「批號管控是 Y 的
      料號」，走的是 vocab.master_fields 那條路（問句裡有「批號管控」四個字），
      不是靠句子裡剛好出現一個 Y。
    """
    k = key.strip()
    if not k or k in _KEY_STOP:
        return False
    # '0' / '100' / '3.0' —— 數字在問句裡是長度、PIN 數、pitch，不是欄位值。
    if re.fullmatch(r"[\d.]+", k):
        return False
    return len(k) >= 2


_PRIORITY = {col: i for i, col in enumerate(INDEXED_COLUMNS)}


def _build() -> _Index:
    # match_key -> column -> canonical values
    keys: dict[str, dict[str, list[str]]] = {}
    counts: dict[str, int] = {}

    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        # Which of the indexed columns actually exist on this database? The
        # ownership block arrives via a later migration, and this module must
        # keep working on a corpus that has not run it yet.
        cur.execute(
            """SELECT column_name FROM information_schema.columns
               WHERE table_schema = 'partrag' AND table_name = 'parts'"""
        )
        present = {r["column_name"] for r in cur.fetchall()}
        counted = [c for c in COUNTED_COLUMNS if c in present]
        cols = [c for c in INDEXED_COLUMNS if c in present]
        if not counted:
            return _Index([], {}, time.time())

        cur.execute(
            "SELECT " + ", ".join(f"count({c}) AS {c}" for c in counted)
            + " FROM partrag.parts"
        )
        counts = dict(cur.fetchone() or {})

        for col in cols:
            if not counts.get(col):
                continue  # column exists but nothing imported yet
            cur.execute(
                f"""SELECT {col} AS v, count(*) AS n FROM partrag.parts
                    WHERE {col} IS NOT NULL AND {col} <> ''
                    GROUP BY 1 ORDER BY n DESC LIMIT %s""",
                (MAX_DISTINCT + 1,),
            )
            rows = cur.fetchall()
            if len(rows) > MAX_DISTINCT:
                continue  # free text, not a classification
            for r in rows:
                raw = str(r["v"])
                vnorm = normalize(raw)
                cand = [vnorm, *(_split_keys(vnorm) if col in _SPLIT_COLUMNS else [])]
                for k in cand:
                    if not _indexable(k):
                        continue
                    vals = keys.setdefault(k, {}).setdefault(col, [])
                    if raw not in vals:
                        vals.append(raw)

    # A key that hits several columns resolves to the highest-priority one.
    # This is not a corner case: 建立人員 and 修改人員 hold the SAME set of people
    # (most parts were last modified by whoever created them), so every single
    # name key is multi-column. Dropping those left the index with no people in
    # it at all, and "Alice 建立的" silently fell back to trigram matching.
    #
    # Within the winning column every value is kept and ORed -- a shared surname
    # naming two colleagues is a real answer, not an error.
    flat: list[tuple[str, str, str]] = []
    for k, by_col in keys.items():
        col = min(by_col, key=lambda c: _PRIORITY.get(c, 99))
        for raw in by_col[col]:
            flat.append((k, col, raw))

    flat.sort(key=lambda t: (-len(t[0]), t[0]))
    return _Index(flat, counts, time.time())


def index() -> _Index:
    global _cache
    if _cache is None or time.time() - _cache.built_at > TTL_SECONDS:
        _cache = _build()
    return _cache


def invalidate() -> None:
    global _cache
    _cache = None


def column_counts() -> dict[str, int]:
    """column -> how many rows hold a non-null value. 0 means "not imported",
    which is the difference between "no such part" and "系統尚未包含該資料"."""
    return dict(index().counts)


def resolve(qnorm: str) -> list[FieldMatch]:
    """Exact PLM field values named anywhere in a normalized query.

    Longest key first, and a key is skipped once its column already matched --
    so "Alice_Wang 王小明" resolves once on the full name rather than three
    times on the name, the surname and the login.
    """
    hits: dict[str, dict[str, object]] = {}
    for key, col, raw in index().keys:
        if col in hits and hits[col]["via"] != key:  # type: ignore[index]
            continue
        if not T.boundary_hit(qnorm, key):
            continue
        e = hits.setdefault(col, {"via": key, "values": []})
        vals: list[str] = e["values"]  # type: ignore[assignment]
        if raw not in vals:
            vals.append(raw)
    return [
        FieldMatch(col, sorted(e["values"]), T.field_label(col), str(e["via"]))  # type: ignore[arg-type]
        for col, e in hits.items()
    ]


# A token shaped like a PLM login: ALICE_WANG, BOB_SMITH. Latin only -- a bare
# CJK fragment is far more likely to be part of a product description.
_PERSON_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,}(?:_[A-Z0-9]+)+)\b")
# The question is about authorship even when no name is spelled out:
# 「這是誰建立的」、「建立人員是誰」.
_CREATOR_HINT = re.compile(
    r"建立人员|建立者|建立的|建立了|所建|开的|开立|创建|谁建|谁开|谁做|"
    r"修改人员|修改者|修改的|上锁人员|CREATED BY|CREATOR"
)
OWNER_COLUMNS = ("created_by", "modified_by", "locked_by")


class CreatorGap(NamedTuple):
    """Why an authorship question cannot be answered from this corpus."""
    reason: str          # "not_imported" | "unknown_person"
    names: list[str]     # the unresolved person-shaped tokens, if any
    message: str


def creator_gap(qnorm: str, matches: list[FieldMatch]) -> CreatorGap | None:
    """-> a gap when the query asks about 建立/修改人員 and the corpus cannot say.

    This exists to stop a specific failure. "Alice_Wang 建立的料號" with the
    ownership block unimported resolves to no field value, so the query falls
    through to lexical matching -- and the name tokens DO hit product
    descriptions by trigram. The pipeline then returns confident-looking rows
    that have nothing to do with Alice. Reporting the gap is the only correct
    answer; NOT_COVERED is built on top of it.
    """
    if any(m.column in OWNER_COLUMNS for m in matches):
        return None
    names = sorted({m.group(1) for m in _PERSON_RE.finditer(qnorm)})
    hinted = bool(_CREATOR_HINT.search(qnorm))
    if not (names or hinted):
        return None

    counts = column_counts()
    if not counts.get("created_by") and not counts.get("modified_by"):
        return CreatorGap(
            "not_imported",
            names,
            "建立人員／修改人員尚未從 PLM 匯入語料庫（"
            "匯入：python -m src.enrich --skip-backfill --plm-csv <匯出.csv>）",
        )
    if not names:
        # The field is loaded and the question named no one -- nothing is missing;
        # ordinary retrieval can answer "who created this part".
        return None
    return CreatorGap(
        "unknown_person",
        names,
        "語料庫的建立人員清單中沒有 " + "、".join(names)
        + f"（目前共 {len({v for k, c, v in index().keys if c == 'created_by'})} 人）",
    )
