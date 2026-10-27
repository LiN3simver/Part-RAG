"""Normalization + attribute extraction.

Imported by BOTH ingest.py and retrieval.py. That symmetry is load-bearing:
a length written as "L=600mm" at ingest time and typed as "600mm" in a query
must land on the same normalized value, or attribute matching silently fails.
"""
from __future__ import annotations

import re
from typing import Any

from opencc import OpenCC

# Traditional -> Simplified. The source CSV mixes both scripts, often inside the
# same string (圖號 vs 图号, 製程 vs 制程), so without this a simplified-script
# query cannot match traditional-script data.
_T2S = OpenCC("t2s")

CJK = r"一-鿿"

# --- item_number structure -------------------------------------------------
# Most rows are a 12-char base + "-" + variant. The variant is NOT a revision:
# Z0000XX00000-3 and Z0000XX00000-6 can be unrelated products. Never dedupe or
# merge on base_pn -- it is a sibling hint only.
# The (?=[A-Z0-9]*\d) guard is load-bearing: without it an ordinary word like
# "ZEROHALOGEN" or "ZIPPERTUBING" is read as a part number, which then reports
# itself as a not-found part number. Every real item_number contains a digit.
# The boundaries are ASCII look-arounds, NOT \b: CJK characters are \w in
# Python's Unicode mode, so \b finds no boundary between a part number and the
# Chinese glued to it. "Z0000XX00000-1是誰建立的" made \b backtrack and yield the
# bare base "Z0000XX00000" (losing the variant), and "料號Z0000XX00000-1的..."
# matched nothing at all -- the item channel died on the most natural phrasing.
ITEM_RE = re.compile(
    r"(?<![0-9A-Za-z])(Z(?=[A-Z0-9]*\d)[A-Z0-9]{9,14}(?:-[0-9A-Z]{1,4})?)(?![0-9A-Za-z])",
    re.I,
)
BASE_RE = re.compile(r"^(.*?)-([0-9A-Z]{1,4})$", re.I)

# prefix4 -> product family. Derived from measured prefix/vocabulary
# correlation over the corpus.
# 展示版：前綴皆為虛構示意；實際對照表由語料分析而來，不包含在展示版中。
CATEGORY_MAP = {
    "ZXWH": "線束/Wire Harness",
    "ZXHS": "MCIO 高速線",
    "ZXCN": "連接器",
}


def to_simplified(text: str) -> str:
    return _T2S.convert(text)


def normalize(text: str) -> str:
    """Canonical comparison form: simplified CJK, upper-case latin, tidy spaces."""
    if not text:
        return ""
    t = to_simplified(text)
    t = t.replace("：", ":").replace("，", ",").replace("（", "(").replace("）", ")")
    t = re.sub(r"\s+", " ", t)
    return t.strip().upper()


# CJK 與 ASCII 英數黏在一起的地方補一個空格。
#
# ★ 為什麼需要這個，而不是逐條改屬性正則
#   Python 的 \w 含中日韓文字，所以「查詢PH3.0」的 詢 與 P 之間**沒有** \b。
#   ITEM_RE 早就為了同一個理由改成 ASCII look-around（見上面那段註解），但
#   下面二十幾條屬性正則（pitch / AWG / GEN / PIN / 連接器詞彙…）全都還在用 \b，
#   於是同一句話換個寫法就整組抽不到東西：
#       'PH3.0 HSG'          -> pitch_series=['PH3.0'], conn_types=['HSG']
#       '我要查詢PH3.0 HSG相關料號' -> {}          ← 一模一樣的規格，零屬性
#   屬性一旦是空的，retrieval._attr_bonus 就沒有東西可以加權，排序完全交給
#   字面與向量 —— 這正是「換個方式問就開始亂答」的機制。使用者換的是問法，
#   不是規格，系統卻換了一組完全不同的證據。
#
#   逐條把 \b 換成 (?<![A-Za-z0-9]) 要改二十幾處、每一處都有自己調過的前後界
#   （_PITCH_PREFIX_RE 的 (?![\d.]) 就是踩過坑才寫成那樣的），改錯一處就是
#   靜默的召回下降。在**進正則之前**把邊界補出來只有一個地方會錯，而且
#   「中文和英數之間插一個空格」不可能切斷任何一個英數 token。
_CJK_ASCII_RE = re.compile(f"(?<=[{CJK}])(?=[A-Za-z0-9])|(?<=[A-Za-z0-9])(?=[{CJK}])")


def space_cjk_ascii(text: str) -> str:
    """「料號Z0000XX00000-1的規格」→「料號 Z0000XX00000-1 的規格」。"""
    return _CJK_ASCII_RE.sub(" ", text or "")


def cjk_bigrams(text: str) -> list[str]:
    """Character bigrams for every CJK run.

    Postgres has no Chinese tokenizer (the pgvector image ships no zhparser) and
    the built-in parsers treat a whole CJK run as ONE token, which makes中文
    full-text search useless. Emitting bigrams gives tsvector real Chinese terms.
    """
    out: list[str] = []
    for run in re.findall(f"[{CJK}]+", text):
        if len(run) == 1:
            out.append(run)
        else:
            out.extend(run[i : i + 2] for i in range(len(run) - 1))
    return out


def build_search_text(name_norm: str, plm_text: str = "") -> str:
    """tsvector source: latin/alnum tokens as-is + CJK bigrams appended.

    `plm_text` carries the PLM form's vocabulary for the row (產品大類/中類/小類,
    料件類型, 環保標準, pitch series, pin layout) -- see taxonomy.build_plm_text.
    Bigrams are generated over both so 線束 / 线束 are both reachable.
    """
    blob = f"{name_norm} {plm_text}".strip()
    return " ".join([blob, *cjk_bigrams(blob)])


def split_item(item_number: str) -> tuple[str, str | None]:
    m = BASE_RE.match(item_number)
    return (m.group(1), m.group(2)) if m else (item_number, None)


def categorize(item_number: str) -> tuple[str, str]:
    p4 = item_number[:4].upper()
    return p4, CATEGORY_MAP.get(p4, "其他")


# --- placeholder detection -------------------------------------------------
# A large share of rows are number-reservation stubs, e.g.
#   領號用=>客戶名稱(<客戶>),建立者(<人員>),原狀態(新出圖)
# plus rows whose name is "/" or similarly degenerate. Kept in the table
# (still findable by exact item_number) but excluded from semantic search.
def is_placeholder(name: str) -> bool:
    n = name.strip()
    if "領號用" in n or "领号用" in n:
        return True
    if len(n) <= 3:
        return True
    if not re.search(r"[A-Za-z0-9一-鿿]", n):
        return True
    return False


# --- attribute regexes -----------------------------------------------------
_LEN_RE = re.compile(
    r"(?i)\bL\s*[=:]\s*([\d]+(?:\.\d+)?(?:\s*[/+]\s*[\d]+(?:\.\d+)?)*)\s*(MM|CM|M|INCH|IN|FT)?"
)
_LEN_BARE_RE = re.compile(r"(?i)(?<![\d.])(\d{2,5}(?:\.\d+)?)\s*MM\b")
_PIN_RE = re.compile(r"(?i)(?<![\d.])(\d{1,3})\s*(?:PIN|P)\b")
_MATRIX_RE = re.compile(r"(?i)(?<![\d.])(\d{1,2})\s*\*\s*(\d{1,3})\s*P\b")
_AWG_RE = re.compile(r"(?i)\b(\d{2})\s*AWG\b")
_GEN_RE = re.compile(r"(?i)\bGEN\s*(\d)\b")
_PITCH_RE = re.compile(r"(?i)\b(\d\.\d{1,2})\s*MM\b")
_SPEED_RE = re.compile(r"(?i)(?<![\d.])(\d{2,3})\s*G\b(?!HZ)")
# Bare "P/N" is the wire supplier's own spec code (see "P/N <供應商規格碼>" in cable
# descriptions), NOT the customer's part number for our finished part -- it must
# NOT be in this pattern. GPN ("customer Global Part Number") and explicit 客户料号/客户P/N
# wording are the only reliable customer-side signals.
_CUSTPN_RE = re.compile(
    r"(?i)(?:GPN|客户料号|客户p/?n)\s*[:：=]?\s*([A-Za-z0-9][A-Za-z0-9._\-/]{3,})"
)
# "工程图号为:ABC-0001" / "工程图号是ABC-0001" -- the linking word sits between the
# label and the colon/value, so it must be consumed here or the whole match fails.
_DRAW_RE = re.compile(
    r"(?i)(?:图号|工程图号|客户图号)\s*(?:为|是)?\s*[:：]?\s*([A-Za-z0-9][A-Za-z0-9._\-]{3,})"
)
_COLOR_RE = re.compile(
    r"(?i)\b(BLACK|WHITE|BLUE|RED|GREEN|YELLOW|GRAY|GREY)\b|(黑色|白色|蓝色|红色|绿色|黄色|灰色)"
)
_TO_SPLIT_RE = re.compile(r"(?i)\s+TO\s+|转|轉")

_UNIT_MM = {"MM": 1.0, "CM": 10.0, "M": 1000.0, "INCH": 25.4, "IN": 25.4, "FT": 304.8}

# --- connector pitch series ------------------------------------------------
# The single strongest discriminator in the 線束 family: PH4.2 and PH3.0 rows
# are otherwise word-for-word identical. Notations seen in the corpus:
#   PH4.2 / PH 3.0      PH=2.54mm      0.4PH / 0.5PH      PITCH 1.25
# All fold to one canonical token, "PH4.2", so any of them can find any other.
# The trailing guard is (?![\d.]) and not \b: "PH=2.54mm" has no word boundary
# between "4" and "m", so \b forces a backtrack to "2" and reports PH2.0.
_PITCH_PREFIX_RE = re.compile(r"(?i)\bPH\s*=?\s*(\d(?:\.\d{1,2})?)(?![\d.])")
_PITCH_SUFFIX_RE = re.compile(r"(?i)(?<![\d.])(\d\.\d{1,2})\s*PH\b")
_PITCH_WORD_RE = re.compile(r"(?i)\bPITCH\s*[:=]?\s*(\d\.\d{1,2})|(\d\.\d{1,2})\s*MM\s*PITCH")
# Bare sub-10mm "0.5MM": in this corpus a millimetre value that small is a
# pitch, never a cable length (lengths start at 10mm -- see _lengths_mm).
_PITCH_BARE_RE = re.compile(r"(?i)(?<![\d.])(\d\.\d{1,2})\s*MM\b")

# --- pin layout ------------------------------------------------------------
# "2*3P" is 2 rows x 3 = 6 pins, and is NOT the same product as "3P". The old
# extractor pushed both 6 and 3 into one array, so a query for 3P scored 2*3P
# rows exactly as high as single-row 3P rows. Layout and total are kept apart.
_LAYOUT_RE = re.compile(r"(?i)(?<![\d.])(\d{1,2})\s*\*\s*(\d{1,3})\s*(?:P|PIN|POS)\b")
_PIN_TOTAL_RE = re.compile(r"(?i)(?<![\d.*])(\d{1,3})\s*(?:PIN|POS|P)\b(?!\s*\*)")

# --- 環保標準 (紅標題) -----------------------------------------------------
# Only tagged when the description says so; most rows don't. The rest stay
# NULL rather than being guessed -- the authoritative value lives in PLM.
_ECO_RE = re.compile(r"(?i)\b(HF\s*GREEN|HALOGEN\s*FREE|ROHS2?\.?0?|ROHS|GC|HF|HB)\b|(无卤|無鹵)")
_ECO_CANON = {
    "HFGREEN": "HF", "HALOGENFREE": "HF", "HF": "HF", "HB": "HB",
    "GC": "GC", "ROHS": "ROHS", "ROHS2": "ROHS", "ROHS20": "ROHS",
}

# --- connector types -------------------------------------------------------
# Controlled vocabulary. Feeds 是否有接頭 (報關欄位) and gives the taxonomy
# channel something concrete to match on.
_CONN_VOCAB: list[tuple[str, re.Pattern[str]]] = [
    ("HSG", re.compile(r"(?i)\bHSG\b|\bHOUSING\b")),
    ("IDC", re.compile(r"(?i)\bIDC\b")),
    ("D-SUB", re.compile(r"(?i)\bD-?SUB\b|\bDB\s?\d{1,2}\s?P\b")),
    ("MCIO", re.compile(r"(?i)\bMCIO\b")),
    ("SLIMSAS", re.compile(r"(?i)SLIM\s?SAS")),
    ("THINLINE SAS", re.compile(r"(?i)THIN\s?LINE\s?SAS|\bTHINLINE\b")),
    ("MINI SAS HD", re.compile(r"(?i)MINI\s?SAS\s?HD|\bSAS\s?HD\b|\bHD\s?\d+X\b")),
    ("MINI SAS", re.compile(r"(?i)MINI\s?SAS")),
    ("SATA", re.compile(r"(?i)\bSATA\b")),
    ("QSFP28", re.compile(r"(?i)\bQSFP28\b")),
    ("QSFP+", re.compile(r"(?i)\bQSFP\+?\b")),
    ("SFP+", re.compile(r"(?i)\bSFP\+?\b")),
    ("OSFP", re.compile(r"(?i)\bOSFP\b")),
    ("EDSFF", re.compile(r"(?i)\bEDSFF\b|\bE[13]\.S\b")),
    ("PCIE", re.compile(r"(?i)\bPCI-?E(?:X\d+)?\b|\bCEM\b")),
    ("RJ45", re.compile(r"(?i)\bRJ-?45\b|\b8P8C\b")),
    ("FFC", re.compile(r"(?i)\bFFC\b|\bFPC\b")),
    ("USB-C", re.compile(r"(?i)\bUSB\s?-?\s?C\b|\bTYPE\s?-?C\b")),
    ("USB-A", re.compile(r"(?i)\bUSB\s?\d?\.?\d?\s?A\b|\bUSB\s?-?\s?A\b")),
    ("HDMI", re.compile(r"(?i)\bHDMI\b")),
    ("DISPLAYPORT", re.compile(r"(?i)\bDISPLAYPORT\b|\bMINI\s?DP\b")),
    ("MOLEX", re.compile(r"(?i)\bMOLEX\b")),
    ("PIN HEADER", re.compile(r"(?i)\bPIN\s?HEADER\b|排针")),
    ("RING TER", re.compile(r"(?i)\bRING\s?TER\b|\bRING\s?TERMINAL\b")),
    ("LC", re.compile(r"(?i)\bLC\s?(?:UPC|APC)\b")),
    ("MPO", re.compile(r"(?i)\bMPO\b")),
    ("OPEN", re.compile(r"(?i)\bOPEN\b|\bCUT\s?END\b")),
]

_COAX_RE = re.compile(r"(?i)\bCOAXIAL\b|\bCOAX\b|同轴")
_VOLT_RE = re.compile(r"(?i)(?<![\d.])(\d{1,4})\s*V\b(?!ER)")

# --- 料件類型 (紅標題): 製造/外包/虛擬/採購 --------------------------------
# Inferred from the description only. First match wins, so the 佔號 test runs
# before the 製造件 test -- a reservation stub is 虛擬, whatever else it says.
_ITEM_TYPE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)领号用|領號用|占用料号|佔用料號|\bDUMMY\b|\bVIRTUAL\b"), "虛擬件"),
    (re.compile(r"(?i)外包|OUTSOURC"), "外包件"),
    (re.compile(r"(?i)CABLE ASS|ASSEMBLY|组立|組立|成品|线束|線束|\bHSG\b|压接|壓接"), "製造件"),
    (re.compile(r"(?i)CONNECTOR|\bCONN\b|连接器|連接器|端子|TRANSCEIVER"), "採購件"),
]


def _lengths_mm(text: str) -> list[float]:
    """All lengths in mm. Returns a list because 'L=600/500mm' occurs in real names."""
    out: list[float] = []
    for m in _LEN_RE.finditer(text):
        factor = _UNIT_MM.get((m.group(2) or "MM").upper(), 1.0)
        for part in re.split(r"[/+]", m.group(1)):
            part = part.strip()
            if part:
                out.append(round(float(part) * factor, 2))
    if not out:
        for m in _LEN_BARE_RE.finditer(text):
            out.append(round(float(m.group(1)), 2))
    return sorted(set(out))


def _pins(text: str) -> list[int]:
    out: set[int] = set()
    for m in _MATRIX_RE.finditer(text):
        rows, cols = int(m.group(1)), int(m.group(2))
        out.add(rows * cols)
        out.add(cols)
    for m in _PIN_RE.finditer(text):
        v = int(m.group(1))
        if 1 <= v <= 300:
            out.add(v)
    return sorted(out)


def _pitch_series(text: str) -> list[str]:
    """Canonical pitch tokens, e.g. ['PH4.2', 'PH1.5'].

    A name can carry two: "(PH3.0 2*6P+PH1.5 12P)HSG" is one connector at 3.0mm
    pitch and one at 1.5mm, so this is a list, not a scalar.
    """
    vals: set[float] = set()
    for rx in (_PITCH_PREFIX_RE, _PITCH_SUFFIX_RE):
        for m in rx.finditer(text):
            vals.add(float(m.group(1)))
    for m in _PITCH_WORD_RE.finditer(text):
        vals.add(float(m.group(1) or m.group(2)))
    if not vals:
        # Fall back to a bare millimetre value only in the FFC pitch range --
        # anything larger is far more likely to be a length or a wire spec.
        for m in _PITCH_BARE_RE.finditer(text):
            v = float(m.group(1))
            if v <= 2.54:
                vals.add(v)
    out = []
    for v in sorted(vals):
        if not 0.3 <= v <= 10:
            continue
        # Match how the data writes it: PH1.0 / PH3.0 / PH4.2 / PH2.54.
        out.append(f"PH{int(v)}.0" if v == int(v) else f"PH{v:g}")
    return out


def _pin_layout(text: str) -> tuple[list[str], list[int], list[int]]:
    """-> (layouts, totals, per_row).

    layouts:  ['2*3P']        the notation as a canonical token
    totals:   [6]             rows*cols, or a bare pin count
    per_row:  [3]             the cols of a matrix; empty for bare counts

    Keeping totals and per_row apart is the fix for the 3P/2*3P collision: a
    query for "3P" matches a single-row 3P part on `totals`, and only brushes a
    2*3P part on `per_row`, which scores far lower.
    """
    layouts: list[str] = []
    totals: set[int] = set()
    per_row: set[int] = set()
    for m in _LAYOUT_RE.finditer(text):
        rows, cols = int(m.group(1)), int(m.group(2))
        if not (1 <= rows <= 20 and 1 <= cols <= 300):
            continue
        tok = f"{rows}*{cols}P"
        if tok not in layouts:
            layouts.append(tok)
        totals.add(rows * cols)
        per_row.add(cols)
    for m in _PIN_TOTAL_RE.finditer(text):
        v = int(m.group(1))
        if 1 <= v <= 300:
            totals.add(v)
    return layouts, sorted(totals), sorted(per_row)


def _eco_std(text: str) -> str | None:
    """環保標準. Strictest-wins: a row saying both HF Green and ROHS is HF."""
    found: set[str] = set()
    for m in _ECO_RE.finditer(text):
        if m.group(2):
            found.add("HF")
            continue
        key = re.sub(r"[\s.]", "", m.group(1)).upper()
        if key in _ECO_CANON:
            found.add(_ECO_CANON[key])
    for pref in ("GC", "HF", "HB", "ROHS"):
        if pref in found:
            return pref
    return None


def _conn_types(text: str) -> list[str]:
    """Controlled-vocabulary connector list. First match of an overlapping pair
    wins (MINI SAS HD before MINI SAS) because _CONN_VOCAB is ordered."""
    out: list[str] = []
    for label, rx in _CONN_VOCAB:
        if rx.search(text):
            # 'MINI SAS' is a substring of 'MINI SAS HD'; keep only the specific.
            if any(label in got and label != got for got in out):
                continue
            out.append(label)
    return out


# Trailing spec noise that is not part of the connector name itself.
_CONN_STOP_RE = re.compile(
    r"(?i)\s*(?:,|;|\(|\bCABLE\b|\bASS(?:'?Y|EMBLY)?\b|\bL\s*[=:]|\b\d+\s*AWG\b|\bGEN\s*\d)"
)


def _trim_connector(s: str) -> str | None:
    s = _CONN_STOP_RE.split(s.strip(), maxsplit=1)[0].strip(" -,")
    return s[:80] or None


def extract_attrs(name: str) -> dict[str, Any]:
    """Extract structured attributes. Input should be raw name; we normalize here."""
    # 補中英之間的邊界後才進正則 —— 理由見 space_cjk_ascii()。對語料裡的品名
    # 幾乎沒有影響（品名本來就是英數為主），影響的是使用者打的自然語言問句。
    n = space_cjk_ascii(normalize(name))
    pitch = _PITCH_RE.search(n)
    awg = _AWG_RE.search(n)
    gen = _GEN_RE.search(n)
    speed = _SPEED_RE.search(n)
    color = _COLOR_RE.search(n)

    lengths = _lengths_mm(n)
    # A pitch like "1.25mm" is not a cable length -- drop sub-10mm false hits.
    lengths = [v for v in lengths if v >= 10]

    conn_a = conn_b = None
    if _TO_SPLIT_RE.search(n):
        parts = _TO_SPLIT_RE.split(n, maxsplit=1)
        if len(parts) == 2:
            conn_a = _trim_connector(parts[0])
            conn_b = _trim_connector(parts[1])

    layouts, pins_total, pins_row = _pin_layout(n)
    conn_types = _conn_types(n)
    volt = _VOLT_RE.search(n)

    item_type = None
    for rx, label in _ITEM_TYPE_RULES:
        if rx.search(n):
            item_type = label
            break

    return {
        "length_mm": lengths,
        "pin_counts": _pins(n),
        "pin_layout": layouts,
        "pins_total": pins_total,
        "pins_row": pins_row,
        "pitch_series": _pitch_series(n),
        "awg": int(awg.group(1)) if awg else None,
        "gen": int(gen.group(1)) if gen else None,
        "pitch_mm": float(pitch.group(1)) if pitch else None,
        "speed_g": int(speed.group(1)) if speed else None,
        "color": (color.group(1) or color.group(2)) if color else None,
        "eco_std": _eco_std(n),
        "item_type": item_type,
        "conn_types": conn_types,
        "has_connector": bool(conn_types),
        "is_coaxial": bool(_COAX_RE.search(n)),
        "rated_voltage": int(volt.group(1)) if volt else None,
        "customer_pn": sorted({m.group(1).rstrip(".,") for m in _CUSTPN_RE.finditer(n)}),
        "drawing_no": sorted({m.group(1).rstrip(".,") for m in _DRAW_RE.finditer(n)}),
        "conn_a": conn_a,
        "conn_b": conn_b,
        "has_to_pattern": bool(conn_a and conn_b),
    }


def parse_query(q: str) -> dict[str, Any]:
    """Pull structured intent out of a user query using the SAME regexes."""
    n = normalize(q)
    attrs = extract_attrs(q)
    item_hits = [m.group(1).upper() for m in ITEM_RE.finditer(n)]
    # A bare code that looks like a customer P/N (no leading P + right shape)
    #
    # ★ 邊界跟 ITEM_RE 一樣是 ASCII look-around，不是 \b —— 理由完全相同：
    #   CJK 在 Python 是 \w，所以「客户料号<客戶料號>是我们的哪个料号」
    #   兩側都沒有 \b，codes 抽出來是空的，客戶料號那一路整條不會啟動，
    #   回來的前幾名跟該客戶料號毫無關係。
    code_hits = sorted(
        {
            m.group(0)
            for m in re.finditer(r"(?<![A-Z0-9])[A-Z0-9][A-Z0-9._\-/]{5,}(?![A-Z0-9])", n)
            if any(c.isdigit() for c in m.group(0))
        }
    )
    return {
        "norm": n,
        "item_numbers": item_hits,
        "codes": code_hits,
        "length_mm": attrs["length_mm"],
        "pin_counts": attrs["pin_counts"],
        "pin_layout": attrs["pin_layout"],
        "pins_total": attrs["pins_total"],
        "pitch_series": attrs["pitch_series"],
        "conn_types": attrs["conn_types"],
        "eco_std": attrs["eco_std"],
        "awg": attrs["awg"],
        "gen": attrs["gen"],
        "speed_g": attrs["speed_g"],
    }
