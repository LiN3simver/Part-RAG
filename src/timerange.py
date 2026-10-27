"""問句裡的「時間」——把它變成日期區間，而不是拿去做字串比對。

★ 這一支要解掉的案例
  「給我修改日期為三月跟五月的工程圖」回的是「查不到…標題欄與材料表中均無
  含『三月』或『五月』字串」。病灶就在那句話：系統把日期拿去比對**圖上的文字**，
  而日期其實在 `drawings.doc_modified_on` 這個欄位上，那兩個月都有圖。
  回答不但錯，還講得很有說服力，使用者沒有理由懷疑。

★ 為什麼要獨立一個模組
  時間條件同時是三個地方的篩選：`parts.created_on` / `parts.modified_on`
  （料號）與 `drawings.doc_modified_on`（工程圖）。三個呼叫端（retrieval、
  aggregate、以及未來的匯出）如果各寫一份解析，「2026 三月」在清單題與檢索題
  會得到不同的區間 —— 那種不一致比解析不到更難查。

★ 邊界一律是 [start, end)：左閉右開
  月底、閏年、時區都不必特別處理，而且 `< 下個月一號` 與 `<= 月底 23:59:59`
  的差別（那一秒）永遠不會咬人。

★ 沒講年份時用「今年」，而且**只在句子裡完全沒有年份**時
  「2026年3月到5月」的 5 月不能被補成今年 —— 它屬於前面那個 2026。
  所以解析是「先找出句子裡的年份，再解析月份」，不是逐段獨立猜。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

# 中文數字 -> 阿拉伯數字。只收 1~12（月份）與 1~31（日）會用到的寫法。
_CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "兩": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _cn_int(s: str) -> int | None:
    """「十二」-> 12、「三」-> 3、「二十五」-> 25。純阿拉伯數字直接轉。"""
    s = s.strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if s in _CN_DIGIT:
        return _CN_DIGIT[s]
    # 十X / X十 / X十X
    m = re.fullmatch(r"(.*)十(.*)", s)
    if m:
        head, tail = m.group(1), m.group(2)
        tens = _CN_DIGIT.get(head, 1) if head else 1
        ones = _CN_DIGIT.get(tail, 0) if tail else 0
        if tens is not None and ones is not None:
            return tens * 10 + ones
    return None


@dataclass(frozen=True)
class Span:
    """一段日期區間，左閉右開。"""
    start: date
    end: date
    label: str

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end


# 問的是哪一種日期。parts 有建立與修改兩個欄位，工程圖只有一個
# （檔案在 PLM 的更新時間），所以工程圖那條路不看這個值。
CREATED, MODIFIED = "created", "modified"

_CREATED_RE = re.compile(r"建立|新增|开立|開立|创建|創建|开的|開的|新开|新開|CREATED")
_MODIFIED_RE = re.compile(r"修改|更新|变更|變更|改过|改過|MODIFIED|UPDATED|REVISED")


@dataclass(frozen=True)
class TimeQuery:
    spans: tuple[Span, ...]      # 多段是 OR（「三月跟五月」）
    field: str                   # CREATED | MODIFIED
    explicit_field: bool         # 使用者有沒有明講是建立還是修改

    @property
    def label(self) -> str:
        who = "建立日期" if self.field == CREATED else "修改日期"
        return f"{who} {' 或 '.join(s.label for s in self.spans)}"


def _month_span(y: int, m: int) -> Span:
    start = date(y, m, 1)
    end = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    return Span(start, end, f"{y} 年 {m} 月")


def _year_span(y: int) -> Span:
    return Span(date(y, 1, 1), date(y + 1, 1, 1), f"{y} 年")


def _add_months(d: date, n: int) -> date:
    total = (d.year * 12 + d.month - 1) + n
    return date(total // 12, total % 12 + 1, 1)


# --- 各種寫法 ---------------------------------------------------------------
# ★ 年份一律寫成 (19|20)\d{2} 而且左邊不准是英數
#   料號裡到處都是四位數：`Z0000XX02074-9` 裡的 `2074-9` 剛好長得像「2074 年 9 月」，
#   而 2074 落在任何一個「合理年份」的範圍裡。左邊界擋掉黏在料號中間的那些，
#   19|20 開頭擋掉 2074/7321 這種本來就不可能是年份的數字。
_YEAR = r"(?<![0-9A-Za-z])((?:19|20)\d{2})"
# 完整日期：2026-06-29 / 2026/6/29 / 2026年6月29日
_YMD_RE = re.compile(_YEAR + r"\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})\s*日?")
# 年月：2026-03 / 2026/3 / 2026年3月 / 2026 三月（中文數字也收）
_YM_RE = re.compile(_YEAR + r"\s*[-/年]?\s*(\d{1,2}|十[一二]?|[一二三四五六七八九])\s*月?")
# 只有月：3月 / 三月（年份靠句子裡別處的年，或今年）
_M_RE = re.compile(r"(?<!\d)(\d{1,2}|十[一二]?|[一二三四五六七八九])\s*月")
# 年份：2026年 / 2026 年
_Y_RE = re.compile(_YEAR + r"\s*年")
# 相對：最近/近 N 天|週|個月|年
_REL_RE = re.compile(
    r"(?:最近|近|过去|過去|前)\s*(\d{1,3}|半|[一二三四五六七八九十]+)\s*"
    r"(天|日|周|週|个月|個月|月|年|季)")
# 上半年 / 下半年 / 第 N 季 / QN
_HALF_RE = re.compile(r"(上|下)半年")
_QUARTER_RE = re.compile(r"第\s*([1-4]|[一二三四])\s*季|(?<![A-Z0-9])Q([1-4])(?![0-9])")
# 之後 / 以後 / 之前 / 以前 —— 開放區間
_AFTER_RE = re.compile(r"(之后|之後|以后|以後|以来|以來|起|後|后)")
_BEFORE_RE = re.compile(r"(之前|以前|前|早于|早於)")

# 這幾個詞出現才算「在問時間」。少了這道閘門，任何帶四位數的料號
# （例如 `ABC-20251` 裡的 2025）都可能被讀成年份。
_TIME_WORDS_RE = re.compile(
    r"日期|时间|時間|年|月|日|季|最近|近期|今年|去年|前年|明年|本月|这个月|這個月|"
    r"上个月|上個月|上月|本周|本週|这周|這週|上周|上週|建立于|建立於|修改于|修改於|"
    r"DATE|MONTH|YEAR|RECENT|"
    # 「2026/03/15 之後修改的」「2026 Q2 的工程圖」整句話一個「年月日」都沒有，
    # 少了這兩條就整個被閘門擋在外面（都回 None）。
    r"(?<![0-9A-Za-z])(?:19|20)\d{2}\s*[-/]\s*\d{1,2}|"
    r"(?<![A-Z0-9])Q[1-4](?![0-9])")

# 一次最多取幾段。「三月跟五月跟七月跟…」再多就不是人在問問題了。
MAX_SPANS = 6


def _field_of(qnorm: str) -> tuple[str, bool]:
    """問的是建立日期還是修改日期。

    ★ 預設是**修改**而不是建立
      這個判斷只在使用者沒明講時才用得到，而沒明講的時候他要的幾乎都是
      「最近有動過的東西」。工程圖那條路更是只有一個日期欄位（檔案更新時間），
      預設成建立日期會讓「2026 三月的工程圖」對到一個根本不存在的欄位。
    """
    if _CREATED_RE.search(qnorm):
        return CREATED, True
    if _MODIFIED_RE.search(qnorm):
        return MODIFIED, True
    return MODIFIED, False


def _relative(qnorm: str, today: date) -> list[Span]:
    out: list[Span] = []
    for m in _REL_RE.finditer(qnorm):
        raw, unit = m.group(1), m.group(2)
        n = 0.5 if raw == "半" else (_cn_int(raw) or 0)
        if not n:
            continue
        if unit in ("天", "日"):
            days = int(n)
            start = date.fromordinal(today.toordinal() - days)
        elif unit in ("周", "週"):
            start = date.fromordinal(today.toordinal() - int(n * 7))
        elif unit in ("个月", "個月", "月"):
            start = _add_months(today.replace(day=1), -int(n * 2) // 2 if n == 0.5 else -int(n))
            if n == 0.5:                      # 「近半個月」當成 15 天
                start = date.fromordinal(today.toordinal() - 15)
        elif unit == "季":
            start = _add_months(today.replace(day=1), -int(n) * 3)
        else:                                  # 年
            months = int(n * 12)
            start = _add_months(today.replace(day=1), -months)
        end = date.fromordinal(today.toordinal() + 1)   # 含今天
        out.append(Span(start, end, f"最近{raw}{unit}"))
    return out


def _named(qnorm: str, today: date) -> list[Span]:
    """今年 / 去年 / 本月 / 上個月 這類固定詞。"""
    out: list[Span] = []
    if "今年" in qnorm:
        out.append(_year_span(today.year))
    if "去年" in qnorm:
        out.append(_year_span(today.year - 1))
    if "前年" in qnorm:
        out.append(_year_span(today.year - 2))
    if re.search(r"本月|这个月|這個月|当月|當月", qnorm):
        out.append(_month_span(today.year, today.month))
    if re.search(r"上个月|上個月|上月", qnorm):
        prev = _add_months(today.replace(day=1), -1)
        out.append(_month_span(prev.year, prev.month))
    return out


def _year_in(qnorm: str, today: date) -> int:
    """句子裡講到的年份；沒講就是今年。多個年份時取第一個 ——
    「2025 到 2026」這種跨年寫法由 _explicit 那一段自己處理。

    ★ 「去年第四季」的年份是**去年**，不是今年
      這裡不看 去年/前年 的話，`_explicit` 會拿 today.year 去組第四季，
      於是「去年第四季」解出來的是 2026 Q4（未來），而使用者要的是 2025 Q4。
      實測就是這樣錯的。
    """
    m = _Y_RE.search(qnorm)
    if m:
        return int(m.group(1))
    # 沒有「年」字但有 2026-03 這種寫法
    m = _YM_RE.search(qnorm)
    if m:
        return int(m.group(1))
    if "前年" in qnorm:
        return today.year - 2
    if "去年" in qnorm:
        return today.year - 1
    if "明年" in qnorm:
        return today.year + 1
    return today.year


def _explicit(qnorm: str, today: date) -> list[Span]:
    """年月日 / 年月 / 只有月 / 半年 / 季。"""
    out: list[Span] = []
    used: list[tuple[int, int]] = []           # 已經吃掉的字元區間

    def taken(a: int, b: int) -> bool:
        return any(not (b <= s or a >= e) for s, e in used)

    # 1) 完整日期 -> 當天
    for m in _YMD_RE.finditer(qnorm):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            start = date(y, mo, d)
        except ValueError:
            continue
        out.append(Span(start, date.fromordinal(start.toordinal() + 1),
                        f"{y}-{mo:02d}-{d:02d}"))
        used.append(m.span())

    # 2) 年 + 月
    for m in _YM_RE.finditer(qnorm):
        if taken(*m.span()):
            continue
        y = int(m.group(1))
        mo = _cn_int(m.group(2))
        if not mo or not 1 <= mo <= 12 or not 1900 < y < 2200:
            continue
        out.append(_month_span(y, mo))
        used.append(m.span())

    # 3) 只有月份（年份取句子裡的年，沒有就今年）。
    #    「2026年3月到5月」的 5 月走這一條，年份沿用 2026 —— 這就是為什麼
    #    年份要先從整句抓出來，而不是每段各自猜。
    year = _year_in(qnorm, today)
    for m in _M_RE.finditer(qnorm):
        if taken(*m.span()):
            continue
        mo = _cn_int(m.group(1))
        if not mo or not 1 <= mo <= 12:
            continue
        out.append(_month_span(year, mo))
        used.append(m.span())

    # 4) 上/下半年、第 N 季
    for m in _HALF_RE.finditer(qnorm):
        y = _year_in(qnorm, today)
        if m.group(1) == "上":
            out.append(Span(date(y, 1, 1), date(y, 7, 1), f"{y} 年上半年"))
        else:
            out.append(Span(date(y, 7, 1), date(y + 1, 1, 1), f"{y} 年下半年"))
    for m in _QUARTER_RE.finditer(qnorm):
        raw = m.group(1) or m.group(2)
        q = _cn_int(raw) if raw else None
        if not q or not 1 <= q <= 4:
            continue
        y = _year_in(qnorm, today)
        out.append(Span(date(y, (q - 1) * 3 + 1, 1), _add_months(date(y, (q - 1) * 3 + 1, 1), 3),
                        f"{y} 年第 {q} 季"))

    # 5) 只有年份、沒有月份 -> 整年
    if not out:
        for m in _Y_RE.finditer(qnorm):
            out.append(_year_span(int(m.group(1))))

    return out


def _drop_coarser(spans: list[Span]) -> list[Span]:
    """一段完全包住另一段時，丟掉外面那一段。

    ★ 「今年上半年」不是「今年」再加「上半年」，是**今年裡的上半年**
      `_named` 會為「今年」產生一整年，`_explicit` 會為「上半年」產生半年。
      兩段重疊，`_merge` 一合併就變回整年 —— 使用者講了「上半年」三個字，
      系統卻回他一整年的資料，而且沒有任何地方看得出來被放寬了。
      同樣的道理適用於「去年第四季」「2026 年 3 月」。
      更精確的那一段一定被包在較粗的那一段裡，所以規則是「有人包住我就留我」。
    """
    out: list[Span] = []
    for s in spans:
        if any(o is not s and s.start <= o.start and o.end <= s.end
               and (o.start, o.end) != (s.start, s.end) for o in spans):
            continue          # 我包住了別人 -> 我是比較粗的那個，丟掉
        out.append(s)
    # 完全一樣的區間去重（「2026年3月」寫兩次）
    seen: set[tuple[date, date]] = set()
    uniq: list[Span] = []
    for s in out:
        if (s.start, s.end) not in seen:
            seen.add((s.start, s.end))
            uniq.append(s)
    return uniq or spans


def _merge(spans: list[Span]) -> list[Span]:
    """把相鄰或重疊的區間併起來，並照時間排序。

    「2026年3月到5月」會解析出 3 月與 5 月兩段（中間的「到」不產生東西），
    這裡看到 `到/至/~` 時已經被 parse() 展開過了；這一支只負責去重與合併，
    讓 label 與 SQL 都乾淨。"""
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: (s.start, s.end))
    out = [spans[0]]
    for s in spans[1:]:
        last = out[-1]
        if s.start <= last.end and s.end > last.end:
            out[-1] = Span(last.start, s.end, f"{last.label}～{s.label}")
        elif s.start >= last.end:
            out.append(s)
        # 完全被包住的丟掉
    return out[:MAX_SPANS]


_RANGE_JOIN_RE = re.compile(r"到|至|~|～|-{1,2}\s*(?=\d{1,2}\s*月)|起.*?止")


def parse(qnorm: str, today: date | None = None) -> TimeQuery | None:
    """-> TimeQuery，或 None（這句話沒有在問時間）。

    `qnorm` 必須是 extract.normalize() 的輸出（簡體、大寫）。
    """
    if not qnorm or not _TIME_WORDS_RE.search(qnorm):
        return None
    today = today or datetime.now().date()

    spans = _named(qnorm, today) + _relative(qnorm, today) + _explicit(qnorm, today)
    if not spans:
        return None
    spans = _drop_coarser(spans)

    field, explicit = _field_of(qnorm)

    # 「3月到5月」：兩段之間有連接詞 -> 併成一整段，而不是兩個孤立的月份。
    # 「三月跟五月」沒有連接詞 -> 保持兩段（OR）。這個差別是使用者真的講出來的，
    # 不要幫他決定。
    if len(spans) >= 2 and _RANGE_JOIN_RE.search(qnorm):
        lo = min(s.start for s in spans)
        hi = max(s.end for s in spans)
        lab = f"{min(spans, key=lambda s: s.start).label}～{max(spans, key=lambda s: s.end).label}"
        spans = [Span(lo, hi, lab)]

    # 「X 之後」/「X 之前」：把區間打開。判斷詞要在日期後面才算。
    if len(spans) == 1 and _AFTER_RE.search(qnorm) and not _BEFORE_RE.search(qnorm):
        s = spans[0]
        spans = [Span(s.start, date(2200, 1, 1), f"{s.label} 之後")]
    elif len(spans) == 1 and _BEFORE_RE.search(qnorm):
        s = spans[0]
        spans = [Span(date(1900, 1, 1), s.start, f"{s.label} 之前")]

    return TimeQuery(tuple(_merge(spans)), field, explicit)


# 把時間講法整段拿掉時要剪的東西。給「這句話除了時間以外還講了什麼」用。
_STRIP_RES = (
    _YMD_RE, _YM_RE, _M_RE, _Y_RE, _REL_RE, _HALF_RE, _QUARTER_RE,
    re.compile(r"修改日期|建立日期|更新日期|日期|时间|時間|"
               r"今年|去年|前年|明年|本月|这个月|上个月|上月|本周|上周|"
               r"之后|之後|以后|以後|之前|以前|最近|近期"),
)


def strip(qnorm: str) -> str:
    """把時間相關的字整段剪掉，剩下的才是「他還講了什麼」。

    ★ 為什麼需要它
      `retrieval._cjk_phrases()` 是拿剩下的中文串去比對材料表的。不剪時間的話，
      「給我修改日期為2026三月跟五月的工程圖」會抽出「修改日期为」當成材料名，
      系統於是判定「這句話有內容條件」，跳過純日期那條分支，回 0 份圖 ——
      而使用者一個材料都沒提過。
    """
    s = qnorm or ""
    for rx in _STRIP_RES:
        s = rx.sub(" ", s)
    return s


def sql(tq: TimeQuery, column: str) -> tuple[str, list[Any]]:
    """(SQL 片段, 參數)。多段是 OR，整串包在一層括號裡，可以直接 AND 進 WHERE。"""
    parts, params = [], []
    for s in tq.spans:
        parts.append(f"({column} >= %s AND {column} < %s)")
        params += [s.start, s.end]
    return "(" + " OR ".join(parts) + ")", params


def parts_column(tq: TimeQuery) -> str:
    return "created_on" if tq.field == CREATED else "modified_on"


# 工程圖只有一個日期欄位：檔案在 PLM 的更新時間。
# 使用者說「建立日期」時也對到它 —— 這不是將就，是因為圖沒有別的日期可談，
# 而回答裡會照實說明這是「檔案更新時間」。
DRAWING_COLUMN = "doc_modified_on"


def describe(tq: TimeQuery, for_drawing: bool = False) -> str:
    """回答裡要講的那一句：我照哪個欄位、篩了哪一段。"""
    rng = " 或 ".join(s.label for s in tq.spans)
    if for_drawing:
        return f"工程圖的檔案更新時間（PLM doc_modified_on）落在 {rng}"
    who = "建立日期" if tq.field == CREATED else "修改日期"
    return f"料號的{who}落在 {rng}"
