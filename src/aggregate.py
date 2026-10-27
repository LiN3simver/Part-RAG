"""Enumeration and counting: "列出所有 X"、"X 共有幾筆"。

Hybrid retrieval answers "which part best matches this spec". It structurally
cannot answer "列出目前所有零組件料號", and the failure is silent: RRF returns a
ranked window, so the reply would be the 10 rows that happened to rank highest,
presented as if it were the complete list. A wrong answer that looks complete is
worse than no answer.

So a question with a totality marker (列出 / 所有 / 全部 / 共有幾筆) leaves the
similarity pipeline entirely and becomes deterministic SQL:

    列出目前所有零組件料號        -> COUNT(*) + 分佈 + 前 N 筆 + CSV 匯出
    列出所有 100_示例廠的資料    -> WHERE plant = '100_示例廠'
    列出所有建立人員              -> SELECT created_by, count(*) GROUP BY 1
    <人名> 建立的所有料號         -> WHERE created_by = ... ORDER BY created_on DESC

Two rules keep it honest:

1.  A totality question whose conditions match 0 rows answers NOT_COVERED
    verbatim. It never degrades to "here are some similar parts".

2.  A residual keyword that resolved to NOTHING structural (no class, no value,
    no attribute) becomes a full-text filter rather than being ignored. Ignoring
    it would turn "列出所有 foobar 的資料" into "列出所有資料" -- the whole
    corpus presented as the answer to a question about foobar.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterator
from urllib.parse import quote

from psycopg.rows import dict_row

from . import taxonomy as T
from . import timerange
from . import values as V
from .db import pool
from .extract import parse_query
from .retrieval import (
    NOT_COVERED,
    SELECT_COLS,
    _attr_filters,
    _filter_sql,
    _jsonable,
    _wants_drawing,
    merge_matches,
)

# Totality markers. qnorm has been through extract.normalize(), which folds
# traditional to simplified CJK and upper-cases latin -- so only the simplified
# spelling is written here, and 「列出所有」/「列出所有」both reach it.
_TOTALITY_RE = re.compile(
    # 「分布」是整體詞：問「公告大類的完整分佈」要的是全庫的值分佈與各自筆數，
    # 不是十筆最相關的料號。少了它，這句話掉回十一路檢索，回的是十個湊巧
    # 帶著那幾個字的料號。
    r"列出|列举|所有|全部|清单|列表|统计|汇总|总共|共有|一览|分布|"
    r"有多少|多少笔|多少个|几笔|几个|数量是|LIST ALL|SHOW ALL|ALL OF THE"
)
# Wanting a number rather than a list.
_COUNT_ONLY_RE = re.compile(r"有多少|多少笔|多少个|几笔|几个|数量|总数|统计|共有几")

# Columns the corpus can enumerate distinct values of. Anything not here is
# either free text (name) or a measurement (length_mm) -- a GROUP BY over those
# produces thousands of one-row groups, which is not an answer.
GROUPABLE: tuple[str, ...] = (
    "created_by", "modified_by", "locked_by", "plant", "announce_cat",
    "item_class", "item_subclass", "unit", "lifecycle", "state", "major_rev",
    "lot_control", "prod_l1", "prod_l2", "prod_l3", "item_type", "eco_std",
    "category",
)

# Words people use for a target field that are not the form's printed label.
# Keys are normalized (simplified CJK, upper case) because that is what
# extract.normalize() produces.
_EXTRA_TARGETS: dict[str, str] = {
    "料号": "item_number", "零件料号": "item_number", "品号": "item_number",
    "品名": "name", "规格": "name", "描述": "name",
    "厂别": "plant", "工厂": "plant",
    "建立人": "created_by", "建立者": "created_by", "开单人": "created_by",
    "创建人": "created_by", "创建者": "created_by",
    "修改人": "modified_by", "修改者": "modified_by",
    "状态": "state", "版本": "major_rev", "版次": "generation",
    "分类": "item_class", "中类": "prod_l2", "大类": "prod_l1", "小类": "prod_l3",
}


def _target_terms() -> list[tuple[str, str, str]]:
    """(match_key, column, label), longest key first."""
    out: list[tuple[str, str, str]] = []
    for key, label in T.FIELD_TERMS.items():
        col = next((f.column for f in T.PLM_FIELDS if f.label == label), None)
        if col and len(key) >= 3:
            out.append((key, col, label))
    for key, col in _EXTRA_TARGETS.items():
        out.append((key, col, T.field_label(col)))
    # 長的優先；**同長度時具體欄位優先於 item_number / name**。
    #
    # ★ 為什麼要這條 tie-break
    #   「所有廠別各有多少筆料號」同時含「厂别」與「料号」，兩個都是 2 字，
    #   誰先命中純粹看 dict 的插入順序 —— 結果 target 變成 item_number，
    #   問「有哪些廠別」回的是一份料號清單（kind=rows 而不是 values）。
    #   料號與品名是每一句清單題都會出現的主詞，當 target 幾乎永遠不是使用者
    #   要的那個投影軸，所以同分時讓它們排後面。
    out.sort(key=lambda t: (-len(t[0]), t[1] in ("item_number", "name")))
    return out


TARGET_TERMS = _target_terms()

# Conversational filler stripped before looking for a residual keyword. Without
# this every question keeps a "資料"/"目前" token and the full-text fallback
# fires on words that carry no information.
_FILLER_RE = re.compile(
    r"列出|列举|所有|全部|清单|列表|统计|汇总|总共|共有|一览|目前|现在|系统|"
    # ★ 這幾個是跑分補上的，每一個都造成過**一個信心十足的 0**：
    #   「語料裡總共有多少筆料號？」殘留關鍵字「语料里」，於是全文檢索 0 筆，
    #   回答變成「語料裡有 0 筆料號」——「查不到」與「真的是零」在畫面上長得
    #   一模一樣。「所有廠別各有多少筆料號」殘留「各有」，同樣歸零。
    #
    # ★ 這裡**不能**放「料号」。它看起來是最該當填充詞的一個，但
    #   _residual_terms() 之後還會照 TARGET_TERMS 由長到短剪一次，而 TARGET_TERMS
    #   裡有更長的「零组件料号」——「列出目前所有零組件料號」原本是靠那個長鍵
    #   整段剪掉、殘留為空的。先在這裡把「料号」挖掉，長鍵就再也對不上，
    #   殘留變成「零组件」，而 tsv 沒有這個三字詞（CJK 是 bigram 切的），
    #   總筆數於是歸零。
    #   料號這類主詞交給 TARGET_TERMS 那一步剪就好，不要兩個地方搶著剪。
    r"语料|各有|各自|笔数|全库|整库|整个|一共|总计|合计|"
    r"资料库|向量库|数据库|资料|数据|信息|相关|有关|关于|里面|其中|"
    r"请|帮我|给我|我想|我要|查一下|查询|一下|出来|整理|表格|条列|"
    r"哪些|什么|多少|几笔|几个|以及|还有|以下|如下|全都|通通|"
    r"LIST|ALL|SHOW|ME|PLEASE|THE|OF|AND|WITH|FOR|ITEMS?|PARTS?|DATA"
)
_WORD_RE = re.compile(r"[A-Z0-9][A-Z0-9.+/*-]*|[一-鿿]+")


@dataclass
class ListSpec:
    query: str
    qnorm: str
    kind: str                                   # "rows" | "values"
    count_only: bool
    matches: list[V.FieldMatch]
    target_col: str | None = None
    target_label: str | None = None
    attr_sql: list[str] = field(default_factory=list)
    attr_params: list[Any] = field(default_factory=list)
    attr_labels: list[str] = field(default_factory=list)
    text_terms: str | None = None
    filters: dict = field(default_factory=dict)
    creator_gap: V.CreatorGap | None = None
    # 時間條件篩的是哪一欄（created_on / modified_on）。order 要跟它一致，
    # 見下面 order 的說明。沒有時間條件時是 None。
    date_col: str | None = None

    def where(self) -> tuple[str, list[Any]]:
        sql = ["true"]
        params: list[Any] = []
        for m in self.matches:
            sql.append(f"{m.column} = ANY(%s)")
            params.append(m.values)
        sql.extend(self.attr_sql)
        params.extend(self.attr_params)
        if self.text_terms:
            sql.append("tsv @@ websearch_to_tsquery('simple', %s)")
            params.append(self.text_terms)
        # An enumeration of a column's values only makes sense over rows that
        # have one; without this "列出所有建立人員" reports a huge NULL group.
        if self.kind == "values" and self.target_col:
            sql.append(f"{self.target_col} IS NOT NULL")
        fsql, fp = _filter_sql(self.filters)
        return " AND ".join(sql) + fsql, params + fp

    @property
    def order(self) -> str:
        """排序要用**這一題在問的那個日期**，不是永遠用 created_on。

        ★ 為什麼
          「列出所有料號的修改日期」以前排的是 `created_on DESC` —— 篩選、投影、
          排序三件事對到兩個不同的欄位，畫面上是一份「照修改日期查」卻「照建立
          日期排」的清單，而且 listing.order 還標著「建立日期新到舊」。使用者
          看到最上面那筆的修改日期不是最新的，只能當成資料有問題。
          同一個道理：問「誰修改的」要照修改時間排，問「誰建立的」照建立時間排。
        """
        owner_created = {"created_by"}
        owner_modified = {"modified_by"}
        cols = {m.column for m in self.matches}
        col = None
        if self.target_col in ("created_on", "modified_on"):
            col = self.target_col
        elif self.date_col:                       # 時間條件篩哪一欄就照哪一欄排
            col = self.date_col
        elif cols & owner_modified:
            col = "modified_on"
        elif cols & owner_created:
            col = "created_on"
        return f"{col} DESC NULLS LAST, item_number" if col else "item_number"

    @property
    def pinned_columns(self) -> set[str]:
        """Columns already fixed to a value. A breakdown over one of these is a
        single bar -- "產品中類分佈：Wire Harness N 筆" restates the filter."""
        cols = {m.column for m in self.matches}
        cols |= {s.split()[0].split("(")[0] for s in self.attr_sql}
        if self.target_col:
            cols.add(self.target_col)
        return cols

    def condition_labels(self) -> list[dict[str, Any]]:
        out = [
            {"label": m.label, "values": m.values, "via": m.via} for m in self.matches
        ]
        out += [{"label": lbl, "values": [], "via": ""} for lbl in self.attr_labels]
        if self.text_terms:
            out.append({"label": "關鍵字", "values": self.text_terms.split(), "via": ""})
        return out


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------
def plan(query: str, filters: dict | None = None) -> ListSpec | None:
    """-> a spec when the question asks about a whole set, else None.

    Note what is NOT a totality marker: 「有哪些」. "600mm 的 MCIO 線，有哪些
    料號？" is a spec search and belongs in the ranked pipeline -- the user wants
    the best matches, not a 400-row dump. Only an explicit 列出/所有/全部/共有
    switches paths.
    """
    q = parse_query(query)
    qnorm = q["norm"]
    if not _TOTALITY_RE.search(qnorm):
        return None

    # ★ 點名了料號、又在問圖上的東西 —— 這不是清單題，不管句子裡有沒有整體詞。
    #   `_find()` 的順序是 listing 先於 drawing，所以
    #   「<料號> 材料表數量是 A/R 的有哪幾列」裡的「数量是」會把
    #   一個明確在問材料表的問題整個劫走，回一份 SQL 統計。
    #   判斷刻意同時要求兩件事：**具體料號** ＋ **工程圖概念**。
    #   只看工程圖概念的話，「列出所有有工程圖的料號」這種真的清單題會被誤擋；
    #   而一個問句既點名了單一料號、又在講材料表／標題欄，它要的一定是那份圖。
    if q["item_numbers"] and _wants_drawing(qnorm):
        return None

    vocab = T.read_query(qnorm)
    matches = merge_matches(vocab, V.resolve(qnorm))

    target_col = target_label = None
    for key, col, label in TARGET_TERMS:
        if key in qnorm:
            target_col, target_label = col, label
            break

    attr_sql, attr_params, attr_labels = _attr_filters(q, vocab, matches)

    # 時間條件與規格條件一樣是硬篩選，所以接在同一組 attr_* 上 ——
    # where() / condition_labels() / 總筆數 / CSV 匯出四個地方就自動都吃到了。
    # 「列出 2026 年 3 月建立的所有料號」以前是把「2026」「3月」丟進全文檢索，
    # 撈到的是品名裡剛好有這些數字的料號。
    tq = timerange.parse(qnorm)
    date_col = None
    if tq:
        date_col = timerange.parts_column(tq)
        tsql, tparams = timerange.sql(tq, date_col)
        attr_sql.append(tsql)
        attr_params.extend(tparams)
        attr_labels.append(tq.label)

    # A target that is itself pinned to a value is a filter, not a projection:
    # "建立人員 <人名> 建立的所有料號" lists rows, it does not list people.
    pinned = {m.column for m in matches}
    if (
        target_col in GROUPABLE
        and target_col not in pinned
        and target_col not in ("item_number", "name")
    ):
        kind = "values"
    else:
        kind = "rows"

    spec = ListSpec(
        query=query,
        qnorm=qnorm,
        kind=kind,
        count_only=bool(_COUNT_ONLY_RE.search(qnorm)),
        matches=matches,
        target_col=target_col,
        target_label=target_label,
        attr_sql=attr_sql,
        attr_params=attr_params,
        attr_labels=attr_labels,
        filters=filters or {},
        date_col=date_col,
        # "列出 <登入名> 建立的所有料號" before the ownership import must answer
        # NOT_COVERED. Left to the keyword fallback below, the name tokens would
        # trigram-match product descriptions and produce a plausible wrong list.
        creator_gap=V.creator_gap(qnorm, matches),
    )
    # Only when nothing structural was resolved. "列出所有 100_示例廠的資料"
    # already has a plant filter; adding a keyword filter for the leftover "100"
    # would narrow it for no reason.
    if kind == "rows" and not (matches or attr_sql or spec.creator_gap):
        spec.text_terms = _residual_terms(qnorm, target_col)
    return spec


def _residual_terms(qnorm: str, target_col: str | None) -> str | None:
    """Content words left after the filler. -> a websearch_to_tsquery string."""
    stripped = _FILLER_RE.sub(" ", qnorm)
    for key, col, _ in TARGET_TERMS:
        if col == target_col:
            stripped = stripped.replace(key, " ")
    toks = [t for t in _WORD_RE.findall(stripped) if len(t) >= 2 and not t.isdigit()]
    seen: list[str] = []
    for t in toks:
        if t not in seen:
            seen.append(t)
    return " OR ".join(seen[:12]) if seen else None


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
# Facets reported alongside a row listing. "列出所有 <人名> 建立的料號" is much more
# useful with "其中 N 筆是線束" than as a flat list.
_BREAKDOWN: tuple[tuple[str, str], ...] = (
    ("prod_l2", "產品中類"),
    ("created_by", "建立人員"),
    ("plant", "建立廠別"),
    ("item_type", "料件類型"),
    ("state", "狀態"),
)
SAMPLE_ROWS = 60          # rows returned to the UI
LLM_ROWS = 25             # rows rendered into the prompt
MAX_VALUES = 300          # distinct values returned for a kind="values" listing


def run(spec: ListSpec, sample: int = SAMPLE_ROWS) -> dict[str, Any]:
    """Execute the spec. Returns a payload shaped like retrieval.search()'s, plus
    a `listing` block -- so the UI and llm.build_prompt keep working unchanged
    and only need to render the extra block."""
    where, params = spec.where()
    counts = V.column_counts()

    if spec.creator_gap:
        return _payload(
            spec,
            total=0,
            rows=[],
            listing_extra={
                "empty": True,
                "message": f"{NOT_COVERED}：{spec.creator_gap.message}",
                "reason": spec.creator_gap.reason,
            },
        )

    # A target column that was never imported is not an empty result, it is an
    # absent capability, and the two must not be reported the same way.
    if spec.target_col in counts and not counts.get(spec.target_col):
        return _payload(
            spec,
            total=0,
            rows=[],
            listing_extra={
                "empty": True,
                "message": f"{NOT_COVERED}：{spec.target_label} 尚未從 PLM 匯入語料庫",
                "reason": "column_not_imported",
            },
        )

    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"SELECT count(*) AS n FROM partrag.parts WHERE {where}", params)
        total = (cur.fetchone() or {}).get("n", 0)

        # How many rows the default NOT is_placeholder cut. Worth stating: a
        # sizeable share of the corpus is 領號用 stubs, and "所有料號" legitimately
        # means either count depending on who is asking.
        excluded = 0
        if not spec.filters.get("include_placeholder"):
            probe = dict(spec.filters, include_placeholder=True)
            w2, p2 = ListSpec(**{**spec.__dict__, "filters": probe}).where()
            cur.execute(f"SELECT count(*) AS n FROM partrag.parts WHERE {w2}", p2)
            excluded = ((cur.fetchone() or {}).get("n", 0)) - total

        # 同理要交代作廢列切掉幾筆。一個「總共 N 筆」的答案，如果其中幾筆是
        # PLM 已經作廢的，講不講出來是兩種不同的答案 —— 而盤點的人問的正是這個。
        excluded_obsolete = 0
        if not spec.filters.get("include_obsolete"):
            probe = dict(spec.filters, include_obsolete=True)
            w2, p2 = ListSpec(**{**spec.__dict__, "filters": probe}).where()
            cur.execute(f"SELECT count(*) AS n FROM partrag.parts WHERE {w2}", p2)
            excluded_obsolete = ((cur.fetchone() or {}).get("n", 0)) - total

        if total == 0:
            return _payload(
                spec,
                total=0,
                rows=[],
                listing_extra={
                    "empty": True,
                    "message": NOT_COVERED,
                    "reason": "no_match",
                    "excluded_placeholder": excluded,
                    "excluded_obsolete": excluded_obsolete,
                },
            )

        distinct: list[dict] = []
        if spec.kind == "values":
            cur.execute(
                f"""SELECT {spec.target_col} AS value, count(*) AS n,
                           max(created_on) AS last_created
                    FROM partrag.parts WHERE {where}
                    GROUP BY 1 ORDER BY n DESC, 1 LIMIT %s""",
                (*params, MAX_VALUES),
            )
            distinct = cur.fetchall()

        rows: list[dict] = []
        if not spec.count_only:
            cur.execute(
                f"""SELECT {SELECT_COLS} FROM partrag.parts WHERE {where}
                    ORDER BY {spec.order} LIMIT %s""",
                (*params, sample),
            )
            rows = cur.fetchall()

        breakdown = []
        pinned = spec.pinned_columns
        for col, label in _BREAKDOWN:
            if col in pinned or not counts.get(col, 1):
                continue
            cur.execute(
                f"""SELECT {col} AS value, count(*) AS n FROM partrag.parts
                    WHERE {where} AND {col} IS NOT NULL
                    GROUP BY 1 ORDER BY n DESC, value LIMIT 12""",
                params,
            )
            items = cur.fetchall()
            if items:
                breakdown.append({"label": label, "column": col, "items": items})
            if len(breakdown) >= 4:
                break

    return _payload(
        spec,
        total=total,
        rows=rows,
        listing_extra={
            "empty": False,
            "message": None,
            "values": distinct,
            "breakdown": breakdown,
            "excluded_placeholder": excluded,
            "excluded_obsolete": excluded_obsolete,
        },
    )


def _payload(
    spec: ListSpec, total: int, rows: list[dict], listing_extra: dict
) -> dict[str, Any]:
    listing = {
        "kind": spec.kind,
        "count_only": spec.count_only,
        "total": total,
        "target": {"column": spec.target_col, "label": spec.target_label}
        if spec.target_col
        else None,
        "conditions": spec.condition_labels(),
        # 標示要跟真正的 ORDER BY 一致 —— 標「建立日期新到舊」卻照 modified_on
        # 排，使用者只會覺得資料壞了。欄名從 spec.order 直接取，兩邊不可能再分岔。
        "order": (f"{T.field_label(_order_col)}新到舊"
                  if (_order_col := spec.order.split()[0]) != "item_number" else "料號"),
        "shown": len(rows),
        "truncated": len(rows) < total,
        "values": [],
        "breakdown": [],
        "excluded_placeholder": 0,
        "excluded_obsolete": 0,
        # Chat cannot hold tens of thousands of part numbers. The complete list is a file.
        "export_url": (
            f"/api/export?query={quote(spec.query)}" if total > len(rows) else None
        ),
        **listing_extra,
    }
    return {
        "query": spec.query,
        "parsed": {},
        "plm_vocab": {},
        "plm_notes": [],
        "field_matches": [
            {"column": m.column, "label": m.label, "values": m.values, "via": m.via}
            for m in spec.matches
        ],
        "not_found": [],
        "listing": listing,
        "total_candidates": total,
        "results": [_jsonable(r) for r in rows],
    }


# ---------------------------------------------------------------------------
# CSV export -- the honest answer to "列出全部" when "全部" is tens of thousands of rows
# ---------------------------------------------------------------------------
# Timestamps are formatted in SQL rather than by str(): the columns are
# timestamptz and psycopg renders them with a "+00:00" suffix, which reads as a
# UTC claim about a wall-clock value PLM exported with no offset at all.
EXPORT_COLS = (
    "item_number", "name", "prod_l1", "prod_l2", "prod_l3", "item_type",
    "eco_std", "plant", "announce_cat", "created_by",
    "to_char(created_on, 'YYYY-MM-DD HH24:MI') AS created_on",
    "modified_by",
    "to_char(modified_on, 'YYYY-MM-DD HH24:MI') AS modified_on",
    "major_rev", "state", "lifecycle",
)
_EXPORT_KEYS = tuple(c.split(" AS ")[-1] for c in EXPORT_COLS)
EXPORT_HEADERS = (
    "零組件料號", "品名及規格", "產品大類", "產品中類", "產品小類", "料件類型",
    "環保標準", "建立廠別", "公告大類", "建立人員", "建立日期",
    "修改人員", "修改日期", "版本", "狀態", "Lifecycle Phase",
)


def csv_iter(spec: ListSpec, limit: int = 200_000) -> Iterator[str]:
    """Stream the full result set as CSV.

    A named (server-side) cursor, so a full-corpus export does not materialise in
    the API process. The BOM is for Excel on a zh-TW box -- without it the
    Chinese headers open as mojibake.
    """
    where, params = spec.where()
    yield "﻿" + ",".join(EXPORT_HEADERS) + "\r\n"
    with pool().connection() as conn:
        with conn.cursor(name="partrag_export", row_factory=dict_row) as cur:
            cur.itersize = 2000
            cur.execute(
                f"""SELECT {", ".join(EXPORT_COLS)} FROM partrag.parts
                    WHERE {where} ORDER BY {spec.order} LIMIT %s""",
                (*params, limit),
            )
            for row in cur:
                yield ",".join(_csv_cell(row[c]) for c in _EXPORT_KEYS) + "\r\n"


def _csv_cell(v: Any) -> str:
    if v is None:
        return ""
    s = str(v)
    if any(c in s for c in ',"\r\n'):
        return '"' + s.replace('"', '""') + '"'
    return s
