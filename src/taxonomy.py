"""PLM form vocabulary <-> corpus vocabulary bridge.

The corpus export has exactly two columns: item_number and name. The PLM 零組件
form that engineers actually look at has many more, and the red (required) ones
are the words they type into search. Nothing in the CSV carries them, so before
this module a query naming a form field or a form-level class ("Wire Harness
cable") fell through to fuzzy matching against free text and returned noise.

This module does three things:

1.  Declares the PLM form's fields and, for each, WHERE its value can come
    from -- derived from the item_number / name, corpus-wide constant, or only
    available by importing the PLM master export (see src/enrich.py).
    `PLM_FIELDS` is the single source of truth for that; retrieval reads it to
    answer "<值> belongs to <欄位>, and that field isn't loaded yet" instead of
    guessing.

2.  Reconstructs the form's 3-level product taxonomy (產品大類/中類/小類) for
    every row. Prefix rules carry the number families we can attribute
    confidently; name rules cover the rest. Without it a large share of rows
    were category='其他' -- so "find related products" had nothing to work with.

3.  Holds the synonym table. 產品小類 is written in English in the form
    ("Wire Harness cable") and in Chinese in the data ("線束"), in traditional
    script in some rows and simplified in others. All three must resolve to the
    same class or the taxonomy is decorative.

展示版：PLM 欄位清單、料號前綴對照、同義詞與表單值對照都只留少數示意條目；
實際的表是依語料分析整理的，不包含在展示版中。
"""
from __future__ import annotations

import re
from typing import Any, NamedTuple

from .extract import to_simplified

# ---------------------------------------------------------------------------
# 1. the PLM form's fields, and where each value can actually come from
# ---------------------------------------------------------------------------
# origin:
#   "derived"  -- computed from item_number + name (this module / extract.py)
#   "constant" -- uniform across the corpus; asserted from the form, not per-row
#   "import"   -- exists only in PLM; needs the master export (src/enrich.py)
class PlmField(NamedTuple):
    label: str          # the form's label, as printed
    column: str | None  # partrag.parts column, if we hold it
    origin: str
    required: bool      # red label on the form
    note: str = ""


# 展示版：只留幾個示意欄位（三種 origin 各有代表）；實際的欄位清單依 PLM
# 表單與語料分析整理，不包含在展示版中。
PLM_FIELDS: list[PlmField] = [
    PlmField("零組件料號", "item_number", "derived", True),
    PlmField("產品中類", "prod_l2", "derived", True, "料號前綴 + 品名關鍵字"),
    PlmField("單位", "unit", "constant", True, "表單值 PC"),
    PlmField("料件類型", "item_type", "import", True,
             "derived 備援：由品名推導；PLM 匯入值較準確"),
]

# Fields whose value is the same for the whole corpus. Asserted from the PLM
# form, NOT measured per row -- retrieval reports them as corpus-level facts
# rather than pretending they discriminate between rows. Override by importing
# the real master export.
CORPUS_CONSTANTS: dict[str, tuple[str, str]] = {
    # column: (value, 說明)
    "unit": ("PC", "線材類料號的計價單位"),
}

_BY_LABEL = {f.label: f for f in PLM_FIELDS}
_BY_COLUMN = {f.column: f for f in PLM_FIELDS if f.column}


# ---------------------------------------------------------------------------
# 2. three-level product taxonomy
# ---------------------------------------------------------------------------
# (產品大類, 產品中類, 產品小類), in the PLM form's own wording, e.g.
# Cable / Wire Harness / Wire Harness cable.
Tri = tuple[str, str, str]

CABLE = "Cable"
CONN = "Connector"
ENCL = "Enclosure"
MODULE = "Module"
OPTIC = "Optical"

# Longest-prefix wins, so a 5-char block beats its 4-char parent.
# 展示版：前綴皆為虛構示意；實際對照表由語料的前綴/詞彙相關性分析而來，
# 不包含在展示版中。
PREFIX_TAXONOMY: dict[str, Tri] = {
    "ZXWH1": (CABLE, "Wire Harness", "Coaxial harness cable"),
    "ZXWH": (CABLE, "Wire Harness", "Wire Harness cable"),
    "ZXHS": (CABLE, "MCIO", "MCIO cable"),
    "ZXCN": (CONN, "Connector", "Connector"),
}

# Family-level fallback on a 3- or 2-char prefix, tried after the specific maps
# above and before the name rules. Covers the long tail: rows that prefix4/
# prefix5 + name rules leave unclassified cluster by number family far more
# cleanly than by any keyword in their descriptions.
# 展示版：同上，虛構示意。
PREFIX_COARSE: dict[str, Tri] = {
    "ZXW": (CABLE, "Wire Harness", "Wire Harness cable"),
    "ZXD": (CONN, "Connector", "Connector / 端子"),
    # Only families checked to be homogeneous belong here -- a wrong family
    # label is worse than 其他.
}

# Content rules for rows whose prefix we cannot attribute. Ordered: first match
# wins, so put the specific patterns above the generic ones.
NAME_RULES: list[tuple[re.Pattern[str], Tri]] = [
    (re.compile(r"QSFP28|100G|CWDM4"), (CABLE, "QSFP28 / DAC", "QSFP28 DAC cable")),
    (re.compile(r"QSFP\+|QSFP"), (CABLE, "QSFP+ / DAC", "QSFP+ DAC cable")),
    (re.compile(r"SFP\+|\bSFP\b"), (CABLE, "SFP+ / DAC", "SFP+ DAC cable")),
    (re.compile(r"OSFP"), (CABLE, "OSFP / DAC", "OSFP DAC cable")),
    (re.compile(r"TRANSCEIVER|收发模块|收發模組"), (OPTIC, "Transceiver", "Optical transceiver")),
    (re.compile(r"\bMPO\b|\bLC UPC\b|SINGLEMODE|FIBER OPTIC|光纤|光纖"),
     (OPTIC, "Fiber", "Fiber optic patch cable")),
    (re.compile(r"\bMCIO\b"), (CABLE, "MCIO", "MCIO cable")),
    (re.compile(r"EDSFF"), (CABLE, "EDSFF", "EDSFF cable")),
    (re.compile(r"SLIM\s?SAS|SLIMSAS"), (CABLE, "SlimSAS", "SlimSAS cable")),
    (re.compile(r"THIN\s?LINE\s?SAS|THINLINE"), (CABLE, "ThinLine SAS", "ThinLine SAS cable")),
    (re.compile(r"MINI\s?SAS\s?HD|SAS\s?HD"), (CABLE, "Mini SAS HD", "Mini SAS HD cable")),
    (re.compile(r"MINI\s?SAS"), (CABLE, "Mini SAS", "Mini SAS cable")),
    (re.compile(r"\bSAS\b"), (CABLE, "SAS", "SAS cable")),
    (re.compile(r"PCIE|PCI-E|\bRISER\b"), (CABLE, "PCIe", "PCIe cable")),
    (re.compile(r"\bSATA\b"), (CABLE, "SATA / 外接線", "SATA cable")),
    (re.compile(r"\bFFC\b|软排线|軟排線"), (CABLE, "FFC", "FFC cable")),
    (re.compile(r"FLAT CABLE|排线|排線|RIBBON"), (CABLE, "Flat Ribbon", "Flat ribbon cable")),
    (re.compile(r"RJ45|8P8C|\bCAT6\b|\bCAT5"), (CABLE, "Network", "RJ45 patch cable")),
    (re.compile(r"\bHDMI\b"), (CABLE, "USB / 外接線", "HDMI cable")),
    (re.compile(r"DISPLAYPORT|\bDP\b.*CABLE"), (CABLE, "USB / 外接線", "DisplayPort cable")),
    (re.compile(r"\bVGA\b|D-?SUB"), (CABLE, "USB / 外接線", "VGA / D-SUB cable")),
    (re.compile(r"USB\s?[234C]|\bUSB\b"), (CABLE, "USB / 外接線", "USB cable")),
    (re.compile(r"POWER CORD|AC POWER|电源线|電源線|IEC\s?60309"),
     (CABLE, "Power Cord", "AC power cord")),
    (re.compile(r"线束|線束|WIRE\s?HARN"), (CABLE, "Wire Harness", "Wire Harness cable")),
    (re.compile(r"\bHSG\b|HOUSING|\bIDC\b|CABLE ASS|CABLE ASSY|CABLE ASSEMBLY"),
     (CABLE, "Wire Harness", "Wire Harness cable")),
    (re.compile(r"OCULINK|OCULINK"), (CABLE, "PCIe", "OCuLink cable")),
    (re.compile(r"BASE-?T"), (CABLE, "Network", "BASE-T cable")),
    (re.compile(r"PIN HEADER|排针|排針|\bWAFER\b|BOX HEADER|\bHEADER\b"),
     (CONN, "Pin Header", "Pin Header / Wafer")),
    (re.compile(r"ANTENNA|天线|天線|\bSMA\b|\bBT ASSY\b|\bNFC\b"),
     (MODULE, "Antenna", "天線 / 無線元件")),
    (re.compile(r"RECEIVER|接收器|接收模块"), (MODULE, "Receiver", "無線接收模組")),
    (re.compile(r"\bMOUSE\b|KEYBOARD|滑鼠|鼠标"), (MODULE, "Peripheral", "滑鼠 / 鍵盤")),
    (re.compile(r"CHARGER|CRADLE|DOCKING|充电座|充電座"),
     (MODULE, "Accessory", "Charger / Cradle")),
    (re.compile(r"CONNECTOR|\bCONN\b|连接器|連接器|CRIMPED|TERMINAL|端子|插头|插頭|插座"),
     (CONN, "Connector", "Connector")),
    (re.compile(r"\bCASE\b|ENCLOSURE|外接盒|硬盘盒|硬碟盒"),
     (ENCL, "HDD Enclosure", "HDD enclosure")),
    (re.compile(r"\bPCBA\b"), (MODULE, "PCBA", "PCB assembly")),
    (re.compile(r"REMOTE|遥控|遙控"), (MODULE, "Remote Control", "Remote control")),
    (re.compile(r"\bMODULE\b|模块|模組"), (MODULE, "Module", "Module")),
    (re.compile(r"\bCABLE\b|\bCBL\b|线缆|線纜|\bCABEL\b"), (CABLE, "Cable", "Cable")),
    # Bulk conductor specs with no connector named (strands/diameter * cores,
    # insulation material). These are wire material, not a finished assembly.
    (re.compile(r"\d{1,2}/0\.\d{2,3}\s*[A-Z]{0,2}\s*\*|MM²|MM2\b|SR-PVC|\bUL\d{4}\b"),
     (CABLE, "Bulk Wire", "電線 / 導體材料")),
]

UNKNOWN: Tri = ("其他", "其他", "其他")


class Classified(NamedTuple):
    l1: str
    l2: str
    l3: str
    source: str        # prefix5 | prefix4 | name | unknown
    alt: Tri | None    # name evidence that disagrees with the number block


def classify(item_number: str, name_norm: str) -> Classified:
    """Reconstruct 產品大類/中類/小類 for a row.

    The number block decides, because that is how PLM works -- if ZXWH is the
    線束 block, every ZXWH row is Cable / Wire Harness / Wire Harness cable
    even when "Wire Harness" appears nowhere in its description.

    The block is not unanimous, though: some rows in an FFC block say WIRE
    HARNESS outright. Rather than pick a winner, the disagreeing name evidence
    is returned as `alt` and gets written into the row's searchable text, so
    those rows answer to both words while the displayed classification stays
    faithful to the block.
    """
    up = item_number.upper()
    name_tri: Tri | None = None
    for rx, tri in NAME_RULES:
        if rx.search(name_norm):
            name_tri = tri
            break

    for n in (5, 4):
        tri = PREFIX_TAXONOMY.get(up[:n])
        if tri:
            alt = name_tri if (name_tri and name_tri[1] != tri[1]) else None
            return Classified(*tri, f"prefix{n}", alt)

    # A specific keyword beats a family-level prefix guess.
    if name_tri:
        return Classified(*name_tri, "name", None)

    for n in (3, 2):
        tri = PREFIX_COARSE.get(up[:n])
        if tri:
            return Classified(*tri, f"prefix{n}", None)

    return Classified(*UNKNOWN, "unknown", None)


# ---------------------------------------------------------------------------
# 3. synonyms: PLM form wording <-> corpus wording
# ---------------------------------------------------------------------------
# Every alias on the right resolves to the taxonomy value on the left. Written
# in normalized form (simplified CJK, upper case) because that is what
# extract.normalize() produces for both data and query.
# 展示版：只留示意條目；實際的同義詞表由語料用詞分析整理，不包含在展示版中。
L2_ALIASES: dict[str, list[str]] = {
    "Wire Harness": [
        "WIRE HARNESS", "WIREHARNESS", "HARNESS", "线束", "線束",
        "WIRE HARNESS CABLE", "HSG", "HOUSING", "CABLE ASSY", "CABLE ASSEMBLY",
    ],
    "MCIO": ["MCIO", "MINI COOL EDGE", "MINI COOLEDGE"],
    "Connector": ["CONNECTOR", "连接器", "連接器", "CONN", "端子"],
}

# Which aliases are safe to write INTO a row's searchable text.
#
# L2_ALIASES is deliberately broad because it reads queries -- "HSG" should
# steer a query toward the wire-harness class. But injecting "HSG" into every
# wire-harness row would make a search for HSG return rows that have no
# HSG connector. So only the class name's language variants get injected;
# connector-level evidence stays query-side.
L2_INJECT: dict[str, list[str]] = {
    "Wire Harness": ["WIRE HARNESS", "WIRE HARNESS CABLE", "线束"],
    "MCIO": ["MCIO", "MCIO CABLE"],
    "Connector": ["CONNECTOR", "连接器"],
    # 產品小類 也可以有自己的注入詞（build_plm_text 對 l2 與 l3 都查）。
    "Coaxial harness cable": ["COAXIAL", "同轴"],
    "其他": [],
}

L1_ALIASES: dict[str, list[str]] = {
    CABLE: ["CABLE", "线材", "線材", "线缆", "線纜", "CBL"],
    CONN: ["CONNECTOR", "连接器", "連接器"],
    ENCL: ["ENCLOSURE", "外接盒", "机壳", "機殼"],
    MODULE: ["MODULE", "模块", "模組", "装置", "裝置"],
    OPTIC: ["OPTICAL", "光通讯", "光通訊"],
}

# 環保標準 (紅標題). Derivable because the standard is usually written into the
# description; rows that omit it stay NULL rather than being guessed.
ECO_ALIASES: dict[str, list[str]] = {
    "GC": ["GC", "GREEN COMPLIANCE", "绿色制程", "綠色製程", "GC 制程", "GC 製程"],
    "HF": ["HF", "HF GREEN", "HALOGEN FREE", "无卤", "無鹵"],
    "HB": ["HB"],
    "ROHS": ["ROHS", "ROHS2.0"],
}

# 環保標準比對「含不含這個代碼」，不是等號。retrieval 與 aggregate 共用同一條
# 規則，同一個問題走排序檢索與走清單統計才會給同一組筆數。
#
# ★ PLM 的環保標準欄有不少筆是 `GC HF` —— 一筆料同時具備綠色製程與無鹵，
#   兩個代碼都成立。存的是表單原值，所以比對不能是等號：等號的話問「GC」
#   會漏掉那些列，問「無鹵／HF」則永遠回 0 筆 —— 而上面的 ECO_ALIASES 本來
#   就收了 无卤/無鹵，等於把一個已經做好的能力關掉。
# ★ 用 regexp 而不是 LIKE '%GC%'：要卡在詞邊界上，`GC` 不能命中某個以 GC 開頭
#   的長代碼，`HB` 也不能命中 `GC HB2` 這種還沒出現、但隨時可能出現的值。
# ★ `%s::text[]` 的轉型不能省。array_to_string() 是 polymorphic，psycopg 送過去的
#   參數沒有型別標註，Postgres 會直接拋
#   `could not determine polymorphic type because input has type unknown`。
ECO_MATCH_SQL = (
    "eco_std ~* ('(^|[^A-Z0-9])(' || array_to_string(%s::text[], '|') || ')([^A-Z0-9]|$)')"
)

_ECO_SPLIT = re.compile(r"[^A-Z0-9]+")


def eco_matches(value: str | None, wanted: list[str]) -> bool:
    """ECO_MATCH_SQL 的 Python 版（給 retrieval._attr_bonus 用）。兩者必須一致 ——
    SQL 撈進來的列與 Python 給不給加權若用不同規則，排序就會自相矛盾。"""
    if not value:
        return False
    toks = set(_ECO_SPLIT.split(value.upper()))
    return any(w.upper() in toks for w in wanted)

# 料件類型 (紅標題) is inferred by extract._ITEM_TYPE_RULES -- it is a regex over
# the description, so it lives with the other extractors.

# Reverse index: alias -> (kind, canonical). Longest alias first so
# "WIRE HARNESS CABLE" wins over "CABLE".
def _build_alias_index() -> list[tuple[str, str, str]]:
    idx: list[tuple[str, str, str]] = []
    for canon, aliases in L2_ALIASES.items():
        for a in aliases:
            idx.append((a, "prod_l2", canon))
    for canon, aliases in L1_ALIASES.items():
        for a in aliases:
            idx.append((a, "prod_l1", canon))
    for canon, aliases in ECO_ALIASES.items():
        for a in aliases:
            idx.append((a, "eco_std", canon))
    idx.sort(key=lambda t: -len(t[0]))
    return idx


ALIAS_INDEX = _build_alias_index()

# PLM field labels that users type as search terms. Maps the word to the field
# it belongs to, so an unpopulated field can be reported honestly.
#
# Keys are keyed on to_simplified(...).upper() because read_query() receives a
# query already put through extract.normalize(). Writing 料件類型 here and
# comparing it against a normalized query silently never matched -- the query
# had become 料件类型.
def _nk(s: str) -> str:
    return to_simplified(s).upper()


# Both the full label and its parenthetical-free stem, because people type
# "標準工時" and the form prints "標準工時(秒)".
FIELD_TERMS: dict[str, str] = {}
for _f in PLM_FIELDS:
    FIELD_TERMS[_nk(_f.label)] = _f.label
    _stem = re.sub(r"[(（].*", "", _f.label).strip()
    if _stem and _stem != _f.label:
        FIELD_TERMS.setdefault(_nk(_stem), _f.label)

# Wordings that name a field without being any field's printed label -- e.g. a
# group heading the form prints above several fields. A question typed straight
# off the form asks about the heading, nothing is labelled that, so the query
# used to resolve to no field at all. One key may name several fields; only
# some are imported, and saying which is which is the answer.
# 展示版：示意條目。
_FIELD_ALIASES_RAW: dict[str, list[str]] = {
    "料件屬性": ["產品中類", "料件類型"],
}
FIELD_ALIASES: dict[str, list[str]] = {
    _nk(k): v for k, v in _FIELD_ALIASES_RAW.items()
}

# Values printed on the form that identify their field unambiguously. Written in
# traditional script; _nk() folds them for matching.
# 展示版：只留通用的料件類型值；實際表含各表單欄位的代碼值，不包含在展示版中。
_VALUE_TERMS_RAW: dict[str, tuple[str, str]] = {
    "製造件": ("料件類型", "製造件"),
    "外包件": ("料件類型", "外包件"),
    "採購件": ("料件類型", "採購件"),
    "虛擬件": ("料件類型", "虛擬件"),
    # 單位 PC is deliberately absent: "PC" occurs in ordinary descriptions and
    # queries far too often to be treated as a field value.
}
VALUE_TERMS: dict[str, tuple[str, str]] = {
    _nk(k): v for k, v in _VALUE_TERMS_RAW.items()
}


def _boundary_hit(hay: str, needle: str) -> bool:
    """Alias match. Latin aliases need word boundaries so 'GC' does not fire
    inside 'GCXX'; CJK aliases have no boundaries so use substring."""
    if re.search(r"[一-鿿]", needle):
        return needle in hay
    return re.search(rf"(?<![A-Z0-9]){re.escape(needle)}(?![A-Z0-9])", hay) is not None


# Public alias: src/values.py resolves DB-sourced values (people, plants) with
# exactly the same boundary rule, and two copies of it would drift.
boundary_hit = _boundary_hit


class QueryVocab(NamedTuple):
    prod_l1: list[str]
    prod_l2: list[str]
    eco_std: list[str]
    item_type: list[str]
    # (column, value) for imported master fields the query named, e.g.
    # ("<column>", "<value>"). Searchable as soon as the export is loaded,
    # which is the whole point of naming the field's value in a query.
    master: list[tuple[str, str]]
    # Columns named WITHOUT a value: a field label alone means "show me the
    # parts that have one", so it becomes a col IS NOT NULL filter.
    master_fields: list[str]
    expand_terms: list[str]   # extra lexical terms to widen recall
    plm_terms: list[dict[str, Any]]  # PLM fields the query names


def read_query(qnorm: str) -> QueryVocab:
    """Resolve PLM vocabulary in a normalized query.

    `qnorm` must come from extract.normalize(): simplified CJK, upper case. The
    same normalization runs over the data, which is the only reason a
    traditional-script query can hit a simplified-script row.
    """
    l1: list[str] = []
    l2: list[str] = []
    eco: list[str] = []
    terms: set[str] = set()

    for alias, kind, canon in ALIAS_INDEX:
        if not _boundary_hit(qnorm, alias):
            continue
        bucket = {"prod_l1": l1, "prod_l2": l2, "eco_std": eco}[kind]
        if canon not in bucket:
            bucket.append(canon)
        # Every alias of a matched class becomes a lexical term: the form says
        # "Wire Harness cable" but the data says 線束, and vice versa.
        src = {"prod_l1": L1_ALIASES, "prod_l2": L2_ALIASES, "eco_std": ECO_ALIASES}[kind]
        terms.update(src[canon])

    # 產品中類 implies its 大類 -- searching "MCIO" is also searching Cable.
    for canon in l2:
        for tri in PREFIX_TAXONOMY.values():
            if tri[1] == canon and tri[0] not in l1:
                l1.append(tri[0])
                break

    itype: list[str] = []
    master: list[tuple[str, str]] = []
    plm: list[dict[str, Any]] = []
    for word, (field, value) in VALUE_TERMS.items():
        if not _boundary_hit(qnorm, word):
            continue
        f = _BY_LABEL.get(field)
        if field == "料件類型" and value not in itype:
            itype.append(value)
        # An imported field with a value becomes a real filter. Before this,
        # a query naming a form value could not reach the row that holds it.
        elif f and f.origin == "import" and f.column:
            pair = (f.column, value)
            if pair not in master:
                master.append(pair)
        entry = {
            "field": field,
            "value": value,
            "origin": f.origin if f else "import",
            "column": f.column if f else None,
            "note": f.note if f else "",
        }
        if entry not in plm:
            plm.append(entry)

    named_fields: list[str] = []
    lookup: list[tuple[str, str]] = list(FIELD_TERMS.items())
    lookup += [(k, lbl) for k, labels in FIELD_ALIASES.items() for lbl in labels]
    for key, label in lookup:
        if len(key) < 3 or key not in qnorm:
            continue
        f = _BY_LABEL[label]
        if f.origin == "import" and f.column and not any(
            c == f.column for c, _ in master
        ):
            named_fields.append(f.column)
        if not any(p["field"] == label for p in plm):
            plm.append(
                {
                    "field": label,
                    "value": None,
                    "origin": f.origin,
                    "column": f.column,
                    "note": f.note,
                }
            )

    # Drop aliases already present verbatim -- no point re-sending them.
    expand = sorted(t for t in terms if t not in qnorm)
    return QueryVocab(l1, l2, eco, itype, master, named_fields, expand[:40], plm)


def build_plm_text(c: Classified, attrs: dict[str, Any]) -> str:
    """PLM-vocabulary text appended to a row's search_text.

    This is what makes the form's wording searchable: the row says
    "PH4.2 3P HSG TO PH4.2 3P HSG Cable Ass'y L=420mm" and the engineer types
    "Wire Harness cable" or "6 pin", neither of which appears in the row.

    Corpus constants (如 單位 PC) are NOT written here on purpose -- they are
    identical on every row, so injecting them would add a term that matches
    everything and ranks nothing. Retrieval answers those from
    CORPUS_CONSTANTS instead.
    """
    parts: list[str] = [
        c.l1, c.l2, c.l3,
        *L2_INJECT.get(c.l2, []),
        *L2_INJECT.get(c.l3, []),
    ]
    if c.alt:  # name evidence that disagrees with the number block
        parts += [c.alt[1], c.alt[2], *L2_INJECT.get(c.alt[1], [])]

    if attrs.get("item_type"):
        parts.append(attrs["item_type"])
    if attrs.get("eco_std"):
        parts.append(attrs["eco_std"])
    parts.extend(attrs.get("pitch_series") or [])
    parts.extend(attrs.get("pin_layout") or [])
    # Total pin count as its own token: "2*3P" should be findable as "6P".
    parts.extend(f"{v}P" for v in (attrs.get("pins_total") or []))
    parts.extend(f"{v:g}MM" for v in (attrs.get("length_mm") or []))
    parts.extend(attrs.get("conn_types") or [])
    if attrs.get("is_coaxial"):
        parts.append("同轴")

    seen: list[str] = []
    for p in parts:
        p = (p or "").strip().upper()
        if p and p != "其他" and p not in seen:
            seen.append(p)
    return " ".join(seen)


def field_label(column: str) -> str:
    """partrag.parts column -> the PLM form's printed label."""
    f = _BY_COLUMN.get(column)
    return f.label if f else column


def constant_answer(field: str) -> str | None:
    """Corpus-level answer for a constant field, e.g. 單位 -> 'PC'."""
    for col, (value, why) in CORPUS_CONSTANTS.items():
        f = _BY_COLUMN.get(col)
        if f and f.label == field:
            return f"{field}={value}（{why}）"
    return None
