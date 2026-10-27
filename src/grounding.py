"""答案核對：模型講的每一件可查證的事，都要在我們送給它的資料裡找得到。

★ 為什麼核對的基準是「prompt 的字面」而不是資料庫
  「這句話是不是幻覺」只有一個可操作的定義：**模型有沒有可能從我們給的東西讀到
  它**。拿資料庫當基準會放過一整類錯誤 —— 某個料號確實存在於庫裡，
  但這一題根本沒檢索到它，模型卻拿它當答案，這仍然是幻覺（而且是最像真的那種）。
  所以 check() 收的是**送出去的那段 prompt 原文**，不是 found 的結構。

★ 這一層刻意不呼叫 LLM
  三個理由，按重要性排：
  1. 用 LLM 判 LLM 有共同模式失效 —— 同一顆模型讀錯欄位標籤時，它當裁判也會
     讀錯同一個標籤，於是核對變成蓋章。
  2. 每題多一次 gateway 呼叫，而 gateway 現在是整條鏈上最慢也最容易掛的一環。
  3. 幻覺裡最貴的那一種（憑空的料號、憑空的工程圖檔名、憑空的長度）全部是
     **字串比對就抓得到的**，不需要語意理解。
  需要語意層核對的（「答非所問」「把兩個料號的規格互相張冠李戴」）由
  llm.judge() 另外處理，那一支才是 LLM 裁判，而且預設只在跑分時開。

★ severity 分兩級的理由
  block 級會觸發重新生成（多一次 gateway 呼叫、多好幾秒），所以只放**誤判成本
  極低**的檢查：一個料號要嘛在 prompt 裡，要嘛不在，沒有模糊空間。
  warn 級（例如「查無料號卻沒講」）判準是啟發式的，誤判會把正確答案擋掉重跑，
  所以只回饋、不重跑。
"""
from __future__ import annotations

import re
from typing import Any, NamedTuple

from .extract import ITEM_RE, normalize

BLOCK, WARN = "block", "warn"
LIST_VALUE_LIMIT = 40
UNVERIFIED_NOTICE = (
    "模型回答未通過資料核對，無法提供可靠的摘要。"
    "請查看本輪檢索結果或工程圖原文，也可以縮小問題後再試一次。"
)


class Violation(NamedTuple):
    kind: str        # unknown_item | unknown_file | unsupported_spec | missing_source |
    #                  refused_with_results | wrong_drawing_subject | denied_present |
    #                  raw_dump | not_found_ignored
    severity: str    # block | warn
    detail: str      # 給人看的一句話（寫進 eval 報表與 API payload）
    fix: str         # 回饋給模型的祈使句（重新生成時附上）


# 規格數值：這幾種在檢索結果裡一律是**照抄**的，模型沒有任何正當理由算出一個
# prompt 裡沒有的值。
#
# ★ 刻意不收 PIN 數。「2*3P」是 6 PIN 是 SYSTEM 明文要求模型做的換算，
#   收進來會把唯一一個「要求模型推理」的地方判成幻覺。寧可漏抓也不要擋掉對的答案。
_SPEC_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("長度", re.compile(r"(?i)(?<![\d.])(\d{2,5}(?:\.\d+)?)\s*MM(?![A-Z])")),
    ("AWG", re.compile(r"(?i)(?<![\d.])(\d{2})\s*AWG")),
    ("GEN", re.compile(r"(?i)GEN\s*(\d)(?![\d.])")),
    ("pitch", re.compile(r"(?i)PH\s*(\d(?:\.\d{1,2})?)(?![\d.])")),
)

# 檔名比對不能用「抓出來的字串是否等於真檔名」，兩邊都會對不上：
#   答案側 「來源：<文件編號>_<公司>_….pdf」 -> 前面黏著「來源：」
#   語料側 「<料號>  客戶 簽.pdf」 -> 檔名**含空白**，抓出來只剩「簽.pdf」
# 所以中日韓標點先排除掉（減少前一種），再用**雙向包含**判定（解決後一種）。
# 誤判的代價是把正確答案擋掉重跑，比漏抓一個編造的檔名貴 —— 而編造的檔名
# 與任何真檔名都不會互為子字串，仍然抓得到。
_PDF_RE = re.compile(
    r"[^\s\"'`（）()【】\[\]：，。、；！？「」『』《》〈〉]+\.pdf", re.I)

# 工程圖 PDF 的連結。只比對路徑那一段：同一個連結在 prompt 裡可能是相對路徑
# （`/api/drawing/file/35`，純摘要那條路）或絕對網址（`http://…/a2a/file/35`，
# 統一前端的卡片），而流水號才是會被編出來的部分。
_LINK_RE = re.compile(r"/(?:api/drawing|a2a)/file/\d+")

# 「答案有沒有寫出來源檔名」的比對，截短的檔名至少要對上前幾個字。
# 24 是因為工程圖檔名的前段幾乎都是料號（`Z0000XX00000-1…`、`ABC-0001(…`），
# 那一段就足以指認是哪一份圖。見下面 missing_source 那一條的說明。
_SOURCE_PREFIX = 24

# 幾份圖以上算「清單題」。與 llm.DRAWING_SUMMARY_MIN 同一個門檻（那邊決定要不要
# 改用總結型的任務指令）——兩邊不一致的話，會出現「提示叫它寫總結、核對卻按單份
# 圖的規則要求檔名」這種互相矛盾的要求。刻意不 import llm：這一層不呼叫任何
# 產生答案的東西，維持單向依賴。
_LISTING_DRAWINGS = 3

# 「我沒有這筆資料」的說法。收得寬一點：這是 warn 級判準，寧可放過也不要
# 把一個講得很委婉的正確答案標成違規。
_NEGATIVE_RE = re.compile(
    r"查無|查不到|沒有|沒收錄|未收錄|不在|找不到|無此|尚未|无此|没有|查无|找不到|未包含"
)


def _squash(s: str) -> str:
    """比對規格數值用的形式：大寫、去掉所有空白。

    模型寫「622 mm」、檢索結果寫「622mm」，這兩個是同一件事；不壓掉空白的話
    每一題都會誤報。"""
    return re.sub(r"\s+", "", normalize(s))


def _ocr_fold(s: str) -> str:
    """OCR 的 O/0 混淆折平。**只用在核對的那一側**，不改任何顯示。

    ★ 為什麼要專門處理這一個字元
      工程圖是 OCR 讀出來的，而 `O` 與 `0` 是這份語料裡已知且未解決的誤判
      （llm.SYSTEM 與 src/drawing.py 都寫了）。例如材料表上是
      `LABEL:W4Omm(REF)`（字母 O），模型答的是 `W40mm` —— 模型做的是**把已知的
      OCR 錯字讀對**，不是編造數值。不折平的話，核對器會把「模型讀對了」
      判成幻覺，然後叫它重寫成錯的。
      折平只發生在比對，違規訊息裡印的仍是原文。
    """
    return s.replace("O", "0")


# 模型在做**比較**而不是陳述規格。
#
# ★ 為什麼要放過這一種
#   「<料號>（650mm）比 622mm 長 28mm」裡的 28mm 是算出來的，
#   檢索結果裡當然沒有 —— 但它是正確且有用的。這是整個核對器裡唯一允許
#   「資料裡沒有」的情況，判準是句子裡先出現了比較詞。
#   刻意**不是**只看「長／短」兩個字：那樣連「這條線長 900mm」都會被放過，
#   而那正是最典型的幻覺句型。一定要有「比 / 相差 / 差」才算在比較。
_COMPARE_RE = re.compile(r"比|相差|差了|差異|差异|多出|少了|之差")
_COMPARE_WINDOW = 20


# 未解碼的 Unicode escape（`線束`）。只要出現就一定是把 JSON 原樣貼出來。
_ESCAPED_RE = re.compile(r"\\u[0-9a-fA-F]{4}")


def _looks_raw(ans: str) -> bool:
    """這段回答是不是「工具輸出原樣貼上」而不是寫給人看的答案。"""
    s = ans.strip()
    if _ESCAPED_RE.search(s):
        return True
    if not s.startswith(("{", "[")):
        return False
    # 開頭是大括號還不夠 —— 可能只是引了一小段。看它佔整段的比重。
    brace = s.count('":') + s.count('" :')
    return brace >= 3


def items_in(text: str) -> set[str]:
    """一段文字裡出現的所有料號（用與檢索同一條 ITEM_RE）。"""
    return {m.group(1).upper() for m in ITEM_RE.finditer(normalize(text))}


_CN_MONTH = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七",
             8: "八", 9: "九", 10: "十", 11: "十一", 12: "十二"}
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

# 判定是以**子句**為單位，不是字元窗口。
#
# ★ 為什麼不是「前後 N 個字裡有否定詞」（自測就打到自己了）
#   「3 月只有 2 份，沒有更多了；5 月 10 份」——「沒有」離「3 月」
#   只有十個字，字元窗口把這句完全正確的話判成否認。而句讀本來就是中文
#   切分語意的方式：逗號後面那半句講的是另一件事。
_CLAUSE_SPLIT_RE = re.compile(r"[，,。；;：:！!？?、\n]+")
# 這個子句裡有講出份數（「2 份」「10 筆」）-> 它在陳述數量，不是在否認。
_COUNT_RE = re.compile(r"\d+\s*(?:份|筆|张|張|个|個)")
# 被否定的東西要是「圖／資料」本身才算否認這一段時間有東西。
#
# ★ 少了這一條會打到「三月那幾份沒有更新的版本了」
#   那句話否定的是「更新的版本」，不是「三月有圖」——而它是完全正確的陳述。
#   要求同一個子句裡出現圖／資料／收錄這類詞，句子的主詞才對得上。
_SUBJECT_RE = re.compile(r"图|圖|资料|資料|收录|收錄|档案|檔案|记录|記錄|数据|數據")


def _period_variants(bucket: dict[str, Any]) -> list[str]:
    """一段時間在答案裡可能被寫成的樣子（都是 _squash 之後的形式）。

    ★ 只有「整整一個月」才展開成月份寫法
      區間也可能是單日（2026-06-29）或跨月合併（三月～五月）。那些情況拿
      「3 月」去比對會打到不該打的句子，所以只留 label 本身的字面 ——
      漏抓一句比擋掉一個正確答案便宜。
    """
    out = [_squash(bucket.get("label") or "")]
    a, b = _DATE_RE.match(str(bucket.get("start") or "")), \
        _DATE_RE.match(str(bucket.get("end") or ""))
    if a and b:
        y, mo = int(a.group(1)), int(a.group(2))
        nxt = (y + 1, 1) if mo == 12 else (y, mo + 1)
        whole_month = (a.group(3) == "01" and b.group(3) == "01"
                       and (int(b.group(1)), int(b.group(2))) == nxt)
        if whole_month:
            out += [f"{y}年{mo}月", f"{y}年{mo:02d}月", f"{y}-{mo:02d}",
                    f"{y}/{mo}", f"{y}/{mo:02d}", f"{y}.{mo}",
                    f"{_CN_MONTH[mo]}月", f"{mo}月"]
    return list(dict.fromkeys(v for v in out if v))


def _denied_periods(ans: str, buckets: list[dict[str, Any]]) -> list[str]:
    """答案否認掉、但實際上有資料的時間區間（給人看的說明字串）。"""
    if not buckets:
        return []
    squashed = _squash(ans)
    clauses = [c for c in _CLAUSE_SPLIT_RE.split(squashed) if c]
    out: list[str] = []
    for b in buckets:
        n = int(b.get("n") or 0)
        if n <= 0:                     # 真的沒有 -> 說沒有是正確答案
            continue
        # 答案引用了這一段裡的任何一份圖（料號或檔名）-> 它沒有否認這一段。
        refs = [*(b.get("items") or []), *(b.get("files") or [])]
        if any(r and _squash(str(r)) in squashed for r in refs):
            continue
        variants = _period_variants(b)
        denied = any(
            any(v in c for v in variants)
            and _NEGATIVE_RE.search(c)
            and _SUBJECT_RE.search(c)            # 否定的是「圖／資料」本身
            and not _COUNT_RE.search(c)          # 有講份數就是在陳述數量
            for c in clauses
        )
        if denied:
            out.append(f"{b.get('label') or variants[0]}（實際有 {n} 份）")
    return out


def check(answer: str, context: str, found: dict[str, Any] | None = None,
          question: str = "") -> list[Violation]:
    """`context` 是**實際送出去的那段 prompt**（llm.build_prompt / build_context
    的回傳值）。結構化欄位值也必須出現在實際 context 才能要求答案列出。"""
    found = found or {}
    ans = (answer or "").strip()
    if not ans:
        return []

    out: list[Violation] = []
    ctx_items = items_in(context) | items_in(question)
    ctx_squashed = _squash(context)

    # Enforce an explicit exhaustive request only for a small, fully supplied
    # value list. Counts, examples and clipped contexts must not force invention.
    listing = found.get("listing") or {}
    values = listing.get("values") or []
    if (listing.get("kind") == "values" and not listing.get("count_only")
            and 0 < len(values) <= LIST_VALUE_LIMIT
            and re.search(r"全部|所有|完整", normalize(question))
            and not re.search(r"舉例|举例|例如|例子|(?:前|最多|只要|只列)\s*\d+", normalize(question))):
        missing_values = []
        normalized_context, normalized_answer = normalize(context), normalize(ans)
        for value in values:
            literal = str(value.get("value") or "")
            if not literal:
                continue
            pattern = re.compile(r"(?<![A-Z0-9_])" + re.escape(normalize(literal)) + r"(?![A-Z0-9_])", re.I)
            if pattern.search(normalized_context) and not pattern.search(normalized_answer):
                missing_values.append(literal)
        if missing_values:
            label = (listing.get("target") or {}).get("label") or "欄位"
            out.append(Violation(
                "missing_listing_values", BLOCK,
                f"使用者要求完整列出{label}，回答漏列已提供的值：" + "、".join(missing_values),
                f"請逐一列出資料中的{label}值與對應料號筆數，不要只回答某一種，"
                "也不要把其他欄位的分佈當成答案。漏列：" + "、".join(missing_values)))

    # ---- 0. 把工具的原始輸出當成答案吐出來 -------------------------------
    # ★ 這一條是實際抓到的，而且長得很嚇人
    #   工具模式下模型會直接把 tool_result 的 JSON 原樣當回答送出，畫面上是：
    #     {"item_number": "Z0000XX00000-1", "category": "線束/Wire Harness", …}
    #   使用者看到的是一串 \u 亂碼。它不是幻覺（內容全是真的），但它完全沒有
    #   回答問題，而且會讓人以為系統壞了。擋下來重寫一次就好。
    if _looks_raw(ans):
        out.append(Violation(
            "raw_dump", BLOCK,
            "回答是工具輸出的原始 JSON，不是給人看的答案",
            "你剛才把工具回傳的原始 JSON 直接當成答案送出去了。"
            "請改用繁體中文寫成給人看的回答：講重點、附完整料號，"
            "不要貼 JSON、不要留 \\uXXXX 這種未解碼的字元。",
        ))

    # ---- 1. 憑空的料號 ---------------------------------------------------
    ghosts = sorted(items_in(ans) - ctx_items)
    if ghosts:
        out.append(Violation(
            "unknown_item", BLOCK,
            "回答出現檢索結果裡沒有的料號：" + "、".join(ghosts[:5]),
            "以下料號**不在**這一題的檢索結果裡，請把它們從回答中完全移除，"
            "不要改寫、不要當成參考料號保留："
            + "、".join(ghosts[:5])
            + "。只能引用「檢索結果」區塊實際列出的料號。",
        ))

    # ---- 2. 憑空的工程圖檔名 --------------------------------------------
    # 引用工程圖一定要附檔名，所以檔名是模型最常編造的東西之一，
    # 而且編出來的檔名長得跟真的一模一樣（同樣的 <文件編號>_<公司>_ 格式）。
    ctx_files = {f.lower() for f in _PDF_RE.findall(context)}
    ctx_files |= {f.lower() for f in (
        (d.get("plm_filename") or d.get("local_file") or "")
        for d in (found.get("drawings") or [])) if f}
    bad_files = sorted({
        f for f in _PDF_RE.findall(ans)
        if not any(c in f.lower() or f.lower() in c for c in ctx_files)
    })
    if bad_files:
        out.append(Violation(
            "unknown_file", BLOCK,
            "回答引用了不存在的工程圖檔名：" + "、".join(bad_files[:3]),
            "以下檔名不是這一題提供的工程圖："
            + "、".join(bad_files[:3])
            + "。請只使用「工程圖」區塊裡原封不動的檔名，一個字元都不要改。"
            # ★ 講出「為什麼你以為要改」，否則它會堅持同一個修正
            #   系列通用圖的檔名 `Z0000XX10XXX-0_XX_V01.pdf` 會被模型拿同一行的
            #   料號換成 `Z0000XX10200-0_XX_V01.pdf` —— 它以為自己在修 OCR。
            #   只被告知「這個檔名不對」時，它換了寫法但照樣把 X 換掉，
            #   重寫次數用完，整題的正確答案被丟掉。
            "**檔名裡如果有一串 X（`…10XXX-0…`），那是一份涵蓋整個料號系列的"
            "通用圖，圖上就是這樣印的，不是 OCR 沒讀好** —— 不要拿料號去替換它，"
            "換掉之後那個檔名並不存在。",
        ))

    # ---- 3. 憑空的規格數值 ----------------------------------------------
    bad_specs: list[str] = []
    ctx_folded = _ocr_fold(ctx_squashed)
    for label, rx in _SPEC_RES:
        for m in rx.finditer(ans):
            token = _squash(m.group(0))
            if not token or token in ctx_squashed or _ocr_fold(token) in ctx_folded:
                continue
            # 前面 20 個字內出現比較詞 -> 這是算出來的差值，不是規格陳述。
            if _COMPARE_RE.search(ans[max(0, m.start() - _COMPARE_WINDOW):m.start()]):
                continue
            bad_specs.append(f"{label} {m.group(0).strip()}")
    bad_specs = sorted(dict.fromkeys(bad_specs))[:5]
    if bad_specs:
        out.append(Violation(
            "unsupported_spec", BLOCK,
            "回答出現檢索結果裡沒有的規格數值：" + "、".join(bad_specs),
            "以下數值在提供的資料裡找不到，請刪除或改成資料中真正有的值："
            + "、".join(bad_specs)
            + "。規格數值一律照抄，不要換算、不要取整、不要推估。",
        ))

    # ---- 4. 有圖卻不附來源 ----------------------------------------------
    # SYSTEM 的硬性要求（同一張圖有好幾個版本，版本錯了材料就錯了）。
    dwg_files = [d.get("plm_filename") or d.get("local_file") or ""
                 for d in (found.get("drawings") or [])]
    dwg_files = [f for f in dwg_files if f]
    # ★ 比對前把兩邊的空白**整個拿掉**（不是壓成一個）
    #   檔名可能有兩個連續空白、模型會抄成一個；反方向也有：檔名
    #   `客图<料號>_02.pdf` 一個空白都沒有，模型在中英交界處**自己加了一個**
    #   —— 一個完全正確的引用被判成沒寫來源，然後被叫去重寫。
    #   空白在這裡從來不是資訊，全部拿掉兩個方向一次解決。
    _ws = lambda s: re.sub(r"\s+", "", s.lower())  # noqa: E731
    ans_ws = _ws(ans)
    # ★ 截短的檔名也算「有寫出來源」
    #   清單題（一次幾十份圖）的索引為了讓每一份都列得下，會把檔名截短
    #   （llm.INDEX_FILE_CLIP）。模型照抄的就是那個截短版，而舊寫法要求整個
    #   檔名一字不差地出現 —— 於是一個**確實指出了來源**的答案被判成沒寫，
    #   然後被叫去重寫。
    #   前綴比對的誤判方向是「放過」，代價遠低於擋掉正確答案。
    has_source = any(_ws(f) in ans_ws for f in dwg_files) or any(
        _ws(f[:_SOURCE_PREFIX]) in ans_ws
        for f in dwg_files if len(f) > _SOURCE_PREFIX)
    # ★ 清單題（一次撈到很多份圖）用**料號**指認來源就夠了
    #   「某個月的工程圖」撈到幾十份，正確的答案是份數加幾個例子，而模型舉的
    #   例子是 `<料號> (<日期>)` —— 料號而不是檔名。舊寫法只認檔名，
    #   於是一個完全正確的清單回答被判成「沒寫來源」擋下重寫。
    #   檔名之所以是硬性要求，是因為引用**圖上的值**（材料表、版本）時同一顆料號
    #   會有好幾個版本，指不明就會拿錯版本的材料。而清單題引用的是「這批圖」
    #   本身，沒有引用任何一份圖上的值 —— 料號足以讓使用者接著追問那一份，
    #   而那時候就是單份題，檔名的硬性要求照樣生效。
    if not has_source and len(found.get("drawings") or []) >= _LISTING_DRAWINGS:
        ids = [str(d.get(k) or "") for d in (found.get("drawings") or [])
               for k in ("item_number", "dwg_part_no")]
        has_source = any(i and _ws(i) in ans_ws for i in ids)
    if dwg_files and not has_source:
        out.append(Violation(
            "missing_source", BLOCK,
            "引用了工程圖但沒有寫出來源檔名",
            "回答的最後要獨立一行寫出工程圖來源：檔名 ＋ 檔案更新時間 ＋ 連結。"
            "檔名是：" + "、".join(dwg_files[:2]),
        ))

    # ---- 4b. 手上有結果卻說查不到 ---------------------------------------
    #
    # ★ 這是「過度拒答」，而它跟幻覺一樣是核對該擋的東西
    #   例如「<料號> 有哪些相似料號」：檢索給了 10 筆，模型回
    #   「資料中沒有符合您查詢條件的相似料號」，理由還寫著「由於您沒有明確要求
    #   列出相關/相似的替代方案」—— 而使用者問句裡就有「相似」兩個字。
    #   系統提示要求「沒問就不要給近似的」之後，模型把它過度套用成「少講一點
    #   比較安全」。空手而回的答案跟編出來的答案一樣不能用，而且它更難被發現：
    #   畫面上明明有十張卡片，文字說沒有。
    #
    #   判準刻意收得很緊：**有結果、答案講了否定詞、而且一個料號都沒提到**。
    #   只要它提到任何一筆檢索到的料號（哪怕是「以下這些都不完全符合」），
    #   就不算拒答。
    rows = found.get("results") or []
    dwgs = found.get("drawings") or []
    # ★ 有三種情況「說查不到」**就是正確答案**，不能拿這條規則去打它：
    #   1. 問了工程圖但那顆料號沒有圖（found["drawing_note"]）。這時 results 裡
    #      那一筆是**留給那句話當主詞用的**，不是答案 —— 見 retrieval.search 的
    #      收斂段。
    #   2. 清單題。它的證據是總筆數與值分佈，不是料號；一份正確的清單回答
    #      可以一個料號都不提。
    #   3. 問句點名的料號根本不在語料裡（not_found）—— 那句「查不到」是
    #      第 5 條規則**要求**它講的。
    gap = (bool(found.get("drawing_note")) and not dwgs) or bool(found.get("listing")) \
        or bool(found.get("not_found"))
    if (rows or dwgs) and not gap and _NEGATIVE_RE.search(ans):
        got = {str(r.get("item_number")) for r in rows}
        got |= {str(d.get("item_number")) for d in dwgs}
        got |= {str(d.get("local_file")) for d in dwgs if d.get("local_file")}
        if not any(g and g.lower() in ans.lower() for g in got):
            n = len(rows) or len(dwgs)
            out.append(Violation(
                "refused_with_results", BLOCK,
                f"檢索到 {n} 筆結果，回答卻說查不到，而且一筆都沒提到",
                f"這一題**檢索到 {n} 筆結果**，就在上面的資料區塊裡。"
                "請用它們回答，不要說「查不到」或「沒有符合的」。"
                "如果它們不完全符合使用者的條件，就逐筆講清楚差在哪裡 ——"
                "「有這些但不完全一樣」跟「什麼都沒有」是兩個不同的答案。",
            ))

    # ---- 4c. 拿別的圖回答點名的那份圖 ------------------------------------
    #
    # ★ 這一條抓的是整條工程圖線上最像真的一種幻覺
    #   檢索端的兩個成因已經修掉了（見 retrieval.drawing_code_hits 與
    #   _drawing_note），這一條是最後一道：使用者點名 `ABC-0002`，語料裡
    #   只有流水號相鄰的 `ABC-0001`，而模型寫出「這是 ABC-0002 的工程圖」，
    #   標題欄、版本、材料表全部照抄 ABC-0001。**每一個值都是真的**
    #   —— 真的存在於某一份圖上，只是不是這一份。所以前面幾條規則全部放它過去：
    #   料號在問句裡（不算憑空）、檔名是真的（不算編造）、規格值在 prompt 裡
    #   （不算沒有依據）。唯一能抓到它的判準是**主詞對不對**。
    #
    #   判準：這個編號在 `drawing_absent` 裡（檢索端確定沒有它的圖）、答案裡
    #   寫了這個編號、而且整段沒有任何一句否定。三個條件同時成立才算 ——
    #   模型只要講了「ABC-0002 沒有收錄」就不會被打到，那正是要它講的話。
    absent_dwg = [c for c in (found.get("drawing_absent") or []) if c]
    if absent_dwg and not _NEGATIVE_RE.search(ans):
        ans_up = normalize(ans).upper()
        claimed = [c for c in absent_dwg if c.upper() in ans_up]
        if claimed:
            out.append(Violation(
                "wrong_drawing_subject", BLOCK,
                "回答把別份工程圖的內容講成 " + "、".join(claimed[:3]) + " 的圖",
                "語料裡**沒有** " + "、".join(claimed[:3])
                + " 這幾個編號的工程圖。請第一句就明講這件事，"
                "不要把其他編號（例如流水號相鄰的那一份）的標題欄、版本或"
                "材料表當成它的內容 —— 那些值本身是真的，但屬於另一份圖，"
                "而使用者沒有辦法從答案裡看出拿錯了圖。",
            ))

    # ---- 3b. 憑空的工程圖連結 -------------------------------------------
    #
    # ★ 為什麼連結要單獨抓
    #   清單題的資料裡只有最前面幾份附連結。模型舉例時挑了第 4 份，檔名照抄
    #   正確、時間照抄正確，連結卻寫成 `/api/drawing/file/4` —— 它從第 1 份的
    #   `…/file/1` 推出「第 4 份就是 file/4」。那個 id 是資料庫流水號，跟清單
    #   順序無關，點進去會是**另一份圖**。
    #   這種錯誤沒有任何一條既有規則抓得到（檔名是真的、料號是真的），
    #   但它是純字面比對就能判的：網址要嘛在 prompt 裡，要嘛不在。
    #   ★ 比對要用「抽出來的集合」，不能用 `in context`
    #     `/api/drawing/file/4` 是 `/api/drawing/file/40` 的子字串 —— 用 in 的話
    #     只要語料裡有 file/40，編出來的 file/4 就永遠算過關（自測就這樣漏掉了）。
    ctx_links = set(_LINK_RE.findall(context))
    bad_links = sorted({u for u in _LINK_RE.findall(ans) if u not in ctx_links})
    if bad_links:
        out.append(Violation(
            "unknown_link", BLOCK,
            "回答出現資料裡沒有的工程圖連結：" + "、".join(bad_links[:3]),
            "以下連結不在提供的資料裡：" + "、".join(bad_links[:3])
            + "。連結裡的編號是資料庫流水號，**不是清單的第幾份**，不可以推算。"
            "只有附了連結的那幾份可以寫連結；其餘請只寫檔名與檔案更新時間。",
        ))

    # ---- 4d. 說某一段時間沒有資料，而那一段其實有 ------------------------
    #
    # ★ 這是「局部拒答」，4b 抓不到它
    #   「給我修改日期為三月跟五月的工程圖」兩個月都有圖，模型回對了總份數、
    #   列出幾筆五月的，然後補一句「語料中沒有修改日期為三月的工程圖」。
    #   前面每一條規則都放它過去：總數是對的、列出的料號與檔名都是真的、
    #   它也沒有整段拒答（4b 要求「一筆都沒提到」）。唯一錯的是那半句話，
    #   而那半句話就是使用者要的答案。
    #
    #   成因不在模型：答案層只展開得下前幾份圖，而排序是新到舊 —— 它看到的
    #   四份全是五月的。所以修正有兩半：把每段的份數寫進 prompt（見
    #   llm._drawing_scope 與 retrieval.drawing_buckets），以及這一條 ——
    #   份數是資料庫算出來的確定值，「有 N 份」與「沒有」之間沒有模糊空間，
    #   誤判成本低到可以放進 block 級。
    #
    #   判準要三個條件同時成立：那一段實際有圖、答案在那一段的寫法附近出現
    #   否定詞、而且答案**一份都沒引用**那一段裡的圖（料號或檔名）。
    #   第三個條件是為了放過「三月那 2 份是…，沒有更多了」這種正確的句子。
    denied = _denied_periods(ans, found.get("drawing_buckets") or [])
    if denied:
        out.append(Violation(
            "denied_present", BLOCK,
            "回答說這幾段時間沒有工程圖，但檢索結果裡有：" + "、".join(denied[:3]),
            "以下時間區間**確實有**工程圖，份數是資料庫直接統計的，就在上面的"
            "「這一題的全部工程圖」清單裡：" + "、".join(denied[:3])
            + "。請把「沒有／查無」那句話刪掉，改成講出每一段各有幾份。"
            "上面只展開了前幾份的標題欄與材料表，**沒有展開不等於不存在**。",
        ))

    # ---- 5. 查無料號卻當成查到了（warn） --------------------------------
    missing = [pn for pn in (found.get("not_found") or [])]
    if missing and not _NEGATIVE_RE.search(ans):
        out.append(Violation(
            "not_found_ignored", WARN,
            "語料中沒有 " + "、".join(missing[:3]) + "，但回答沒有講出來",
            "請先明確說出 " + "、".join(missing[:3])
            + " 不在語料中（通常是 PLM 剛開、尚未進到匯出檔），"
            "再把鄰近料號當成參考列出，不要讓使用者以為找到了。",
        ))

    return out


def blocking(violations: list[Violation]) -> list[Violation]:
    return [v for v in violations if v.severity == BLOCK]


def feedback(violations: list[Violation]) -> str:
    """重新生成時附在對話後面的那一段。只講「哪裡錯、要怎麼改」，
    不重述整題 —— 原本的 prompt 還在同一串訊息裡。"""
    if not violations:
        return ""
    lines = [
        "你剛才的回答沒有通過事實核對。以下每一項都是**根據上面提供的資料**"
        "逐字比對出來的，不是意見：",
    ]
    lines += [f"{i}. {v.fix}" for i, v in enumerate(violations, 1)]
    lines.append(
        "請重寫一次完整回答。寧可說「資料中沒有這一項」，也不要保留任何一項上述內容。"
    )
    return "\n".join(lines)


def note(violations: list[Violation]) -> str:
    """重寫過仍然沒通過時，附在答案末尾告訴使用者的那一句。

    ★ 這一段一定要寫出來。攔不下來的幻覺如果安靜地送出去，使用者拿到的是
      一個「看起來被核對過」的錯答案 —— 比沒有核對機制更危險。"""
    if not violations:
        return ""
    return (
        "\n\n---\n⚠️ 自動事實核對未通過（已重新生成一次仍有下列問題），"
        "以下內容請以左側檢索結果卡片為準：\n"
        + "\n".join(f"- {v.detail}" for v in violations)
    )


def as_payload(violations: list[Violation]) -> list[dict[str, str]]:
    """給 API / eval 報表的結構化形式。"""
    return [{"kind": v.kind, "severity": v.severity, "detail": v.detail}
            for v in violations]
