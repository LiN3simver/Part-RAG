"""一份工程圖 PDF -> 標題欄（右下）+ 材料表（左下）。純解析，不碰 DB。

圖框其他部分（尺寸標註、LABEL 說明、NOTE、簽核人 D.R./CH.K/APV.D）不抽。

版面前提（對實際圖檔跑出來的觀察，不是推論）
--------------------------------------------------
* **大多數圖沒有可用的文字層**：字是 CAD 畫成的向量線條，或整份是掃描檔
  —— 所以 **OCR 是主路線，文字層才是那條免費的捷徑**，不是反過來。
* 頁面尺寸有 A4 / Letter / A3、橫式直式好幾種，所以所有切點都必須從
  anchor 的實際座標算，寫死比例一定會壞。
* CAD 把文字也畫成向量線段（單頁上萬條 edge），所以 pdfplumber 的
  find_tables() 在這類圖上找不到材料表 —— 每個字母的筆畫都被當成框線。
  **用長度過濾**（>= 150pt 的水平線）之後，剩下的正好就是材料表的列分隔線，
  這是這類圖最乾淨的一個訊號。

★ 唯一的關鍵陷阱：兩個表格在同一個 y 帶上
  只按閱讀順序排會黏成一列，長這樣：
    `3 TER:PHOSPHOR BRONZ... 6PCS RoHS COMPLIANT 0.0=±.5 CH.K <簽核人> CBL,SERIAL...`
  所以一定要**先按 x 切欄再按 y 分行**。split_bottom() 做這件事。

★ OCR 尺度：不是 DPI，是**每一塊自己的渲染像素寬**
  直式 A4 表單量出來的 300 DPI 不適用橫式 CAD 圖（842pt 寬）。同一個 clip
  只改渲染寬度，讀得出來的列數是**非單調**的 —— 這是偵測模型內部 resize
  的結果，不是解析度不夠。

  但真正的關鍵不是挑哪一檔，而是**渲染哪一塊**：整條 842pt 寬的 band 壓到
  1450px，等於每個字只有 124 DPI，連字號會整批被吃掉
  （`ABC-0001` -> `ABC0001`、`GOLD-PLATED` -> `GOLDPLATED`）。
  同樣 1450px 只給 520pt 寬的材料表那一塊，就是 209 DPI，連字號全部保住。

  所以流程是三段：
      1. 定位：整條 band 跑一次，只為了拿 ITEM/Q'TY/TITLE/PART NO. 的 x 座標
      2. 材料表那一塊，用它自己的寬度算 DPI，再跑一次
      3. 標題欄那一塊，同樣算法再跑一次
  第 2、3 段的結果優先，第 1 段只補它們沒蓋到的位置。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from . import drawing_frames, pdf_ocr

# 下方要看的比例。有向量格線時會用格線把它收得更準（見 _band_top）。
BAND_FRAC = 0.32

# 目標渲染寬度（像素）。整條 band 的定位那一次，與切出來的每一塊，
# 都各自照自己的 pt 寬換算 DPI。見模組 docstring。
LOCATE_PX = 1450
BLOCK_PX = 1450

# 材料表那一「格」（只有表格本身，不含上方的 NOTE 區）單獨再跑一次時用的目標寬度。
# 比 BLOCK_PX 高，是因為這一塊只有 50pt 高，貴不到哪去；見 _bom_clip()。
BOM_PX = 2000

# 判斷「這一頁是不是圖框那一頁」用的字樣。
ANCHORS = ("TITLE", "PART NO", "REV", "SHEET", "SCALE", "SIZE",
           "PD./FILE", "ITEM", "Q'TY")

# 材料表列分隔線的最小長度（pt）。CAD 的字母筆畫都 < 10pt，
# 150 足以只留下真正的表格線。
MIN_RULE_LEN = 150.0

Word = pdf_ocr.Word


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def _cx(w: Word) -> float:
    return (w[0] + w[2]) / 2


def _cy(w: Word) -> float:
    return (w[1] + w[3]) / 2


def _up(s: str) -> str:
    return (s or "").upper()


def _txt(ws: list[Word]) -> str:
    """一組字框照閱讀順序（先 y 後 x）串起來。"""
    return " ".join(w[4] for w in sorted(ws, key=lambda w: (_cy(w), w[0]))).strip()


# --------------------------------------------------------------------------
# 頁面挑選 + 取字
# --------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _rules(path: str, page_no: int) -> tuple[list[float], list[float]]:
    """材料表的水平列線與垂直欄線（PDF point，y 由上而下）。

    只留長度 >= MIN_RULE_LEN 的線 —— 不過濾的話，CAD 畫成線段的每一個
    字母筆畫都會被當成框線（單頁上千條水平 edge，真正的表格線只有十幾條）。

    有 cache 是因為解析這種頁很貴：pdfplumber 要把上千條線與曲線全部建成
    物件，單頁要數秒。同一頁在 find_page 裡會被問到兩次以上。
    """
    try:
        import pdfplumber
    except ModuleNotFoundError:
        return [], []
    try:
        with pdfplumber.open(path) as pdf:
            pg = pdf.pages[page_no]
            h = sorted({round(e["top"], 1) for e in pg.edges
                        if e["orientation"] == "h"
                        and (e["x1"] - e["x0"]) >= MIN_RULE_LEN})
            v = sorted({round(e["x0"], 1) for e in pg.edges
                        if e["orientation"] == "v"
                        and (e["bottom"] - e["top"]) >= 40})
            return h, v
    except Exception:
        return [], []


def _merge_segments(segs: dict[float, list[tuple[float, float]]],
                    min_len: float) -> list[tuple[float, float, float]]:
    """同一條線的線段接起來，再按長度過濾。回傳 (座標, 起, 迄)。

    順序不能顛倒 —— 先過濾再合併的話，CAD 一格一格畫出來的長線會被當成
    一堆短線丟掉（見 _vspans）。
    """
    out: list[tuple[float, float, float]] = []
    for k, spans in segs.items():
        cur_lo = cur_hi = None
        for lo, hi in sorted(spans):
            # 容差 1pt：相接的兩段端點偶爾差個零點幾。
            if cur_hi is not None and lo <= cur_hi + 1.0:
                cur_hi = max(cur_hi, hi)
                continue
            if cur_hi is not None and cur_hi - cur_lo >= min_len:
                out.append((k, cur_lo, cur_hi))
            cur_lo, cur_hi = lo, hi
        if cur_hi is not None and cur_hi - cur_lo >= min_len:
            out.append((k, cur_lo, cur_hi))
    return sorted(out)


@lru_cache(maxsize=8)
def _hspans(path: str, page_no: int) -> tuple[tuple[float, float, float], ...]:
    """水平格線，連它橫跨哪一段 x 一起回傳：(y, x0, x1)。

    ★ 為什麼客戶圖框不能直接用 `_rules()` 的 hrules
      那一支只回 y 座標，於是「圖上別的框的橫線」與「這張表的列線」分不出來。
      材料表右邊緊鄰另一個框格時，那一格的橫線跟材料表在 x 上完全沒有交集，
      卻會把材料表的某一列切成兩列 —— 規格與顏色/數量被分到兩列去，項次
      也跟著錯開。

      列線的定義是「橫跨這張表」，所以要連 x 範圍一起看。
    """
    try:
        import pdfplumber
    except ModuleNotFoundError:
        return ()
    try:
        with pdfplumber.open(path) as pdf:
            pg = pdf.pages[page_no]
            segs: dict[float, list[tuple[float, float]]] = {}
            for e in pg.edges:
                if e["orientation"] != "h":
                    continue
                segs.setdefault(round(e["top"], 1), []).append(
                    (round(e["x0"], 1), round(e["x1"], 1)))
    except Exception:
        return ()
    return tuple(_merge_segments(segs, MIN_RULE_LEN))


@lru_cache(maxsize=8)
def _vspans(path: str, page_no: int) -> tuple[tuple[float, float, float], ...]:
    """垂直格線，連**它自己有多長**一起回傳：(x, y0, y1)。

    `_rules()` 只回 x 座標就夠給預設路徑用（那條路的欄界是表頭字框，格線只
    用來切列）。客戶圖框需要長度：有些客戶圖框的**標題欄列距與材料表列距一樣**，
    「間距突然變大就是表格結束」那條規則因此完全停不下來 —— 材料表會往下
    一路收到頁底，把標題欄與最下面那段保密聲明都當成材料列。

    能停下來的訊號是欄界那幾條線自己的長度：材料表的直線只畫在材料表那一段
    y 上，標題欄是另外一疊。長度過濾沿用 `_rules()` 的 40pt。

    ★ 過濾長度**之前**一定要先把共線相接的線段接起來
      CAD 是一格一格畫的：同一條欄界在檔案裡常常是好幾個首尾相接的線段。
      先過濾再合併的話，短的那幾段會被 40pt 門檻丟掉，那條欄界的長度就只剩
      最長的一段 —— 表格的上界因此往下縮，材料表**上面幾列整批消失**，
      而且不會有任何症狀。
    """
    try:
        import pdfplumber
    except ModuleNotFoundError:
        return ()
    try:
        with pdfplumber.open(path) as pdf:
            pg = pdf.pages[page_no]
            segs: dict[float, list[tuple[float, float]]] = {}
            for e in pg.edges:
                if e["orientation"] != "v":
                    continue
                segs.setdefault(round(e["x0"], 1), []).append(
                    (round(e["top"], 1), round(e["bottom"], 1)))
    except Exception:
        return ()
    return tuple(_merge_segments(segs, 40.0))


def _band_top(page_h: float, hrules: list[float]) -> float:
    """圖框那一條的上緣。

    有格線就用「落在下方 45% 裡最上面的那條長水平線」再往上留 10pt ——
    材料表有幾列是每張圖都不一樣的，固定比例會把長材料表
    的上面幾列切掉，而切掉的部分不會有任何症狀，只是那幾列就是不見了。
    """
    default = page_h * (1 - BAND_FRAC)
    inband = [y for y in hrules if y > page_h * 0.55]
    if not inband:
        return default
    return max(page_h * 0.45, min(default, min(inband) - 10))


@lru_cache(maxsize=8)
def _page_text_words(path: str, page_no: int) -> tuple[Word, ...]:
    """整頁文字層的字框，座標轉到 **/Rotate 之後的顯示空間**，score 一律 1.0。

    ★ 不轉的話座標會整批轉 90 度，而且完全不會報錯
      `get_text()` 回的是未旋轉空間的座標，OCR 是在 `get_pixmap()` 的點陣圖上
      做的、那張圖已經照 /Rotate 轉正 —— 兩邊混用的症狀是「座標全部合法、
      只是整批轉了 90 度」，切欄與歸格照樣「成功」，只是切在錯的地方。
      帶 /Rotate 90/270 的圖確實存在，而且這類圖的表格常常**大半只存在於
      文字層**，轉錯就等於那一半整批放錯格。

    有 cache 是因為同一頁會被問到好幾次（定位、材料表、標題欄各一次）。
    """
    import fitz

    doc = fitz.open(path)
    try:
        pg = doc[page_no]
        m = pg.rotation_matrix
        out: list[Word] = []
        for w in pg.get_text("words"):
            t = w[4].strip()
            if not t:
                continue
            r = fitz.Rect(w[:4]) * m
            out.append((r.x0, r.y0, r.x1, r.y1, t, 1.0))
        return tuple(out)
    finally:
        doc.close()


def _text_words(path: str, page_no: int, clip) -> list[Word]:
    """文字層的字框，只留與 clip 有交集的。形狀對齊 pdf_ocr.words()。"""
    ws = _page_text_words(path, page_no)
    if clip is None:
        return list(ws)
    return [w for w in ws
            if w[2] > clip.x0 and w[0] < clip.x1
            and w[3] > clip.y0 and w[1] < clip.y1]


def _anchor_count(ws: list[Word]) -> int:
    t = _up(" ".join(w[4] for w in ws))
    return sum(1 for a in ANCHORS if a in t)


def _covers(o: Word, w: Word, overlap: float) -> bool:
    """o 算不算已經蓋住 w。overlap=0 是「w 的中心落在 o 裡（±2pt）」，
    overlap>0 是「兩框的交集佔**較小那一框**面積這麼多以上」。

    ★ 分母要取較小那一框，不能取 w
      OCR 常常把文字層的好幾個字讀成一個寬框（文字層的四個短字 -> OCR 的
      一個寬框）。用 w（那個寬框）
      當分母的話，交集只佔它的四分之一，於是它被當成新內容補進來 ——
      同一句話在同一格裡出現兩次。取較小那一框當分母，「小框完全在大框裡」
      就會被正確判成重複，而「兩個框只是邊緣壓到」還是分得開。
    """
    if overlap <= 0:
        cx, cy = _cx(w), _cy(w)
        return o[0] - 2 <= cx <= o[2] + 2 and o[1] - 2 <= cy <= o[3] + 2
    ix = min(o[2], w[2]) - max(o[0], w[0])
    iy = min(o[3], w[3]) - max(o[1], w[1])
    if ix <= 0 or iy <= 0:
        return False
    area = max(min((w[2] - w[0]) * (w[3] - w[1]),
                   (o[2] - o[0]) * (o[3] - o[1])), 1e-6)
    return (ix * iy) / area >= overlap


def _merge(base: list[Word], extra: list[Word],
           self_check: bool = True, overlap: float = 0.0) -> list[Word]:
    """兩個 OCR 尺度的聯集：base 全留，extra 只補 base 沒蓋到的位置。

    ★ 刻意是「第一輪優先」，不是「信心高的優先」。試過照 rec 分數排，結果更差：
      第二輪常把同一行切成好幾個短框（`Z0000XX00000 1` 變成
      `Z0000XX` / `00000` / `1`），而短框的分數往往是 1.00，於是它們贏過
      完整的那一框（0.95），完整的再因為重疊被丟掉 —— 欄位值就碎了。
      同樣的機制也會把 `GOLD-PLATED` 換成 `GOLDPLATED`。

      聯集的用途是**補漏檢**，不是挑字。所以主尺度先寫死，第二尺度只補洞。

    ★ self_check=False：只跟 base 比，補進來的字彼此不比
      預設是 True（既有三個呼叫端的行為不變）。但「文字層 + OCR」那個聯集
      一定要關掉：同一輪 OCR 的相鄰兩框本來就會互相壓到 ±2pt，開著的話
      **後面那一框會被前面那一框擋掉**。例如簽核人的名字，中心剛好落在
      同一輪讀到的 `APPROVED BY` 框 +2 的範圍裡，於是整個值消失 —— 這樣
      被自己那一輪的鄰居擋掉的字不是少數。

    ★ overlap>0：改用「交集佔面積的比例」判斷重複，給文字層與 OCR 的聯集用
      中心點那個判準對**兩塊貼在一起的字**太粗：OCR 讀到的標籤
      `DESCRIPTION:` 的中心，可能剛好落在文字層下一行某個字框往上 2pt 的
      容差裡（只差零點幾 pt），標籤整個被丟掉。標籤沒了，那一格的內容就被
      隔壁兩格各吃一份。兩框的交集其實只佔標籤面積的一成不到，用面積比就分得開。
    """
    out = list(base)
    for w in extra:
        ref = out if self_check else base
        if any(_covers(o, w, overlap) for o in ref):
            continue
        out.append(w)
    return out


def _ocr_block(path: str, page_no: int, clip, target_px: int) -> list[Word]:
    """一塊區域，用它自己的寬度換算 DPI 之後 OCR 一次。

    DPI 由 target_px 反推而不是寫死，是因為決定認字品質的是**每個字有幾個
    像素**，而不同區塊的 pt 寬差三倍（整條 band 842pt vs 標題欄 200pt）。
    """
    # ★ 上限 300 不是保險，是必要條件：標題欄那一塊只有兩百 pt 上下寬，
    #   照 1450px 反推接近 600 DPI —— 偵測模型在那個尺度反而會把連字號
    #   吃掉，而且單這一塊的 OCR 時間就會變成整份的好幾倍。
    #   下限 120 則是避免長材料表被壓到認不出字。
    width_pt = max(clip.width, 1.0) / 72.0
    dpi = max(120, min(300, round(target_px / width_pt)))
    try:
        return pdf_ocr.words(path, page_no, dpi=dpi, clip=clip)
    except Exception as e:  # noqa: BLE001 -- 一塊掛掉，其他塊照跑
        print(f"    ⚠ OCR {dpi}dpi 失敗（{type(e).__name__}: {e}）")
        return []


@dataclass
class Page:
    page_no: int
    method: str                      # text | ocr
    words: list[Word]
    hrules: list[float]
    vrules: list[float]
    rect: Any
    clip: Any = None
    # OCR 那條路裡混進來的文字層字數（0 = 純 OCR）。只為了誠實回報
    # extract_method，下游會照它決定要不要叫使用者以原圖為準。
    text_n: int = 0


def find_page(path: str, max_ocr_pages: int = 3) -> Page | None:
    """找到有圖框的那一頁，並把該頁下方那條的字取回來（定位用的那一次）。

    先掃全部頁的**文字層**（免費）：anchor >= 4 就走文字層，只有少數圖會走到，
    但走到的那些免費而且比 OCR 準。

    其餘走 OCR。多頁圖的材料表**通常**在 sheet 1、但每頁都有標題欄，所以從第 1 頁
    開始試，讀到 >= 2 個 anchor 就收工 —— 不然一份 6 頁的圖要 OCR 6 次。

    ★ 「通常」不是「一定」，這一支要多繞一圈找材料表：有些多頁圖的 sheet 1
      只有標題欄跟一張有標尺寸的裝配示意圖，材料表印在 sheet 2 甚至更後面。
      舊版一讀到 sheet 1 的 TITLE/PART NO./REV 這幾個 anchor 就收工返回 ——
      這幾個 anchor 只代表「這頁有標題欄」，跟「材料表在這頁」是兩件事，
      混在同一個門檻裡誤判了。
      症狀是 `bom_rows()` 對著一頁沒有材料表的畫面找表頭，**回空 list**：
      不是壞在 OCR，是壞在「材料表一定在第一頁找到的那頁」這個版面假設。

      所以多留一個 `fallback`：先記住第一頁湊到 >= 2 個 anchor 的結果（舊版
      的答案），但只要湊到的 anchor 裡沒有 `ITEM`／`Q'TY`，就當「這頁只有
      標題欄」，繼續往下一頁找**材料表表頭**；找到就直接用那一頁蓋過
      fallback。掃到 `max_ocr_pages` 都沒有材料表表頭（本來就沒有材料表的圖，
      例如規格書）才退回 fallback ——
      這正是舊版原本的行為，不影響 sheet 1 就有材料表的圖：
      它們在第一次迴圈就同時湊到標題欄與材料表 anchor，跟舊版一樣立刻返回，
      不會多 OCR 任何一頁。
    """
    import fitz

    doc = fitz.open(path)
    try:
        n = len(doc)
        rects = [doc[i].rect for i in range(n)]
    finally:
        doc.close()
    if not n:
        return None

    def band(i):
        h, v = _rules(path, i)
        clip = fitz.Rect(rects[i].x0, _band_top(rects[i].height, h),
                         rects[i].x1, rects[i].y1)
        return h, v, clip

    # --- 捷徑：文字層
    # 這一圈刻意**不**碰 pdfplumber：解析一頁要數秒，而這裡只是在數
    # anchor，用固定比例的粗略 band 就夠。真的挑中某一頁之後才去要格線。
    for i in range(n):
        rough = fitz.Rect(rects[i].x0, rects[i].height * (1 - BAND_FRAC),
                          rects[i].x1, rects[i].y1)
        ws = _text_words(path, i, rough)
        if _anchor_count(ws) >= 4:
            h, v, clip = band(i)
            return Page(i, "text", _text_words(path, i, clip), h, v, rects[i], clip)

    if not pdf_ocr.available():
        return None

    # --- 主路線：OCR
    #
    # ★ 走 OCR 不代表文字層是空的，所以文字層**照樣要收**（_merge 的 base）。
    #   二選一是這批圖最貴的一個假設：anchor >= 4 那道門檻是對「整個標題欄
    #   都是文字」設計的，而客戶圖框常常是**一半文字一半向量** ——
    #   有的材料表整張在文字層裡、標題欄是 CAD 畫的，有的剛好相反。
    #   丟掉文字層就等於把已經是完美文字的那一半拿去重新猜：例如 `UNIT` 與
    #   `QT'Y` 被 OCR 讀成同一個框（`UNIT QTY`），兩個表頭都認不出來，
    #   於是整欄數量全空 —— 而文字層裡它們本來就是兩個字。
    #
    #   順序是 base=文字層、extra=OCR：文字層是照抄，OCR 是猜的。
    last = min(n, max_ocr_pages) - 1
    fallback: Page | None = None
    for i in range(last + 1):
        h, v, clip = band(i)
        ws = _ocr_block(path, i, clip, LOCATE_PX)
        tw = _text_words(path, i, clip)
        merged = _merge(tw, ws, self_check=False, overlap=0.5) if tw else ws
        if _anchor_count(merged) >= 2:
            page = Page(i, "ocr", merged, h, v, rects[i], clip, text_n=len(tw))
            if fallback is None:
                fallback = page
            # 這頁除了標題欄，材料表表頭也在嗎？用跟 split_bottom() 完全
            # 同一套比對（_label_hits + _BOM_HEADERS），這裡判斷「找對頁了」
            # 才會跟下游「這頁抽不抽得到材料表」是同一個答案。
            if _label_hits(merged, _BOM_HEADERS):
                return page
        elif fallback is None and i == last:
            return Page(i, "ocr", merged, h, v, rects[i], clip, text_n=len(tw))
        if i == last:
            return fallback
    return fallback


# --------------------------------------------------------------------------
# 按 x 切欄
# --------------------------------------------------------------------------
_TITLE_LABELS = (
    ("title", re.compile(r"^TITLE$")),
    ("part_no", re.compile(r"^PART\s*N[O0]\.?$")),
    ("rev", re.compile(r"^REV\.?$")),
    ("size", re.compile(r"^SIZE$")),
    ("scale", re.compile(r"^SCALE$")),
    # SHEET 也寫成 `SHEET NO.`，而且 OCR 會把中間吃掉成 `SH,T NO.`（實測）。
    # 認不出來的代價不只是少一欄：SHEET 同時是 REV. 那一格的下界，漏掉的話
    # REV. 會把整條 `01 SH,T NO. 1OF1` 吞進去。
    ("sheet", re.compile(r"^SH[.,;]?E{0,2}[.,;]?T\.?(N[O0]\.?)?$")),
    # `PD./FILE NO.` 與**光是 `FILE NO.`** 都要收：兩種圖框都存在。
    # 漏掉後者的代價不是少一個欄位 —— 那個標籤沒被認出來，就不會成為
    # TITLE 那一格的右界，於是 TITLE 會把「FILE NO.」和它的值一起吃進去
    # （症狀：TITLE 變成 `FILE NO. <標題> <文件編號> <標題> ...`，
    #  同時 pd_file_no 是空的）。
    ("pd_file_no", re.compile(r"^(P\.?D\.?\s*/?\s*)?FILE\s*N[O0]\.?$")),
)

# ★ ITEM 寫成 `[I1L|]?TEM`，不是 `ITEM`：表頭會被讀成 `TEM`
#   （而且信心 1.00 —— 開頭那個 I 緊貼著表格左框線，被當成框線的一部分吃掉）。
#   表頭認不出來的後果不是少一個欄位，是**整個 ITEM 欄的欄界消失**，
#   於是每一列的編號都被歸到描述欄裡去，而且完全不會報錯。
_BOM_HEADERS = (
    ("item", re.compile(r"^[I1L|]?TEM$")),
    ("description", re.compile(r"^DESCRIPTI[O0]N$")),
    ("qty", re.compile(r"^Q\s*['`’]?\s*TY$")),
)


def _label_hits(ws: list[Word], table) -> dict[str, Word]:
    """每個標籤取最右下角的那一個 —— 圖框在右下，同樣的字在圖裡別處
    （例如零件說明）出現時不該被選走。"""
    out: dict[str, Word] = {}
    for w in ws:
        t = re.sub(r"\s+", "", _up(w[4])).replace("，", ",")
        for key, rx in table:
            if rx.match(t):
                cur = out.get(key)
                if cur is None or (_cy(w), w[0]) > (_cy(cur), cur[0]):
                    out[key] = w
    return out


def _label_hits_all(ws: list[Word], table) -> dict[str, list[Word]]:
    """`_label_hits()` 的「全部候選」版，給 title_block_multi() 用。"""
    out: dict[str, list[Word]] = {}
    for w in ws:
        t = re.sub(r"\s+", "", _up(w[4])).replace("，", ",")
        for key, rx in table:
            if rx.match(t):
                out.setdefault(key, []).append(w)
    return out


def split_bottom(ws: list[Word]) -> tuple[list[Word], list[Word], dict, dict]:
    """(材料表的字, 標題欄的字, 材料表表頭, 標題欄標籤)

    切點用 ITEM/DESCRIPTION/Q'TY 表頭與 TITLE/PART NO. 等標籤的**實際 x 位置**
    決定，不寫死比例 —— 版面有 842x595、792x612、1191x842 三種以上。

    中間的公差區（`0.0=±.5`）與簽核人（D.R./CH.K/APV.D）落在兩個切點之間，
    沒有 anchor，自然被排除，不必另外寫規則去踢。
    """
    bom_h = _label_hits(ws, _BOM_HEADERS)
    title_h = _label_hits(ws, _TITLE_LABELS)

    # 材料表的右界：Q'TY 欄的右緣。沒有 Q'TY 就退到 DESCRIPTION 右邊一點。
    if "qty" in bom_h:
        bom_right = bom_h["qty"][2] + 6
    elif "description" in bom_h:
        bom_right = bom_h["description"][2] + 40
    else:
        bom_right = None

    # 標題欄的左界：所有標題欄標籤裡最左邊那個。簽核人在它左邊，被排除。
    title_left = min((w[0] for w in title_h.values()), default=None)

    left = [w for w in ws if bom_right is not None and _cx(w) < bom_right]
    right = [w for w in ws if title_left is not None and _cx(w) >= title_left - 2]
    return left, right, bom_h, title_h


# --------------------------------------------------------------------------
# 標題欄：anchor 相對取值
# --------------------------------------------------------------------------
def title_block(ws: list[Word], labels: dict[str, Word],
                band_bottom: float) -> dict[str, Any]:
    """每個標籤的值 = 它那一格裡、除了它自己以外的字。

    格子邊界不用幾何比例，用**別的標籤**推：
      右界 = 同一列（y 有重疊）、在它右邊的下一個標籤的 x0
      下界 = 在它下面、而且 x 範圍與**標籤本身**有重疊的下一個標籤的 y0

    這兩條規則是量出來的，不是設計出來的。一張典型圖框的七個欄位剛好涵蓋
    各種擺法，同一組規則全部命中：
      SCALE/SIZE/PART NO./PD./FILE NO. 的值在標籤**正下方**
      REV./SHEET 的值在標籤**右邊同一列**
      TITLE 的值是下方**兩行**（`CBL,SERIAL,...` + `<第二行>`）

    只用「正下方」會漏掉 REV. 與 SHEET；只用「右邊」會漏掉另外四個。
    而純幾何比例（例如「PART NO. 在頁寬 83% 處」）會被非 A4 的那幾張打壞。

    ★ 下界為什麼比的是「標籤本身」的 x 範圍，不是整格的寬度：
      PART NO. 的格子右界被 REV. 撐開，而 SHEET 的標籤剛好落在那個寬度裡
      —— 用整格寬度去比，SHEET 會被當成 PART NO. 正下方的下一個標籤，
      下界砍在料號值的上方，**整個欄位就變成空的**。改比標籤自身的 x 範圍後，SHEET 與
      PART NO. 不重疊，下界落到圖框底，值才取得到。
    """
    label_boxes = list(labels.values())
    out: dict[str, Any] = {}
    conf: dict[str, float] = {}

    for key, lb in labels.items():
        got = _cell_words(ws, lb, label_boxes, band_bottom)
        if got:
            out[key] = _txt(got)
            conf[key] = min(w[5] for w in got)

    out["_conf"] = conf
    return out


def _cell_words(ws: list[Word], lb: Word, label_boxes: list[Word],
                band_bottom: float,
                hspans: list[tuple[float, float, float]] | None = None) -> list[Word]:
    """一個標籤那一格裡的字（不含標籤本身）。規則見 title_block()。

    ★ hspans（客戶圖框才給）：格子的下界改用**這一格自己的橫線**
      「下一個標籤的上緣」在標籤稀疏的圖框會估得太寬：`SCALE:` 下面沒有
      任何標籤時，那一格一路開到圖框底，把兩列之外的字也吃進來，比例變成
      `NONE B`。
      橫線是 CAD 畫的格子邊界，比「猜下一個標籤在哪」準。
    """
    lx0, ly0, lx1, ly1 = lb[0], lb[1], lb[2], lb[3]

    right = min(
        (o[0] for o in label_boxes
         if o is not lb and o[0] > lx1 - 1 and o[3] > ly0 and o[1] < ly1),
        default=float("inf"),
    )
    bottom = min(
        (o[1] for o in label_boxes
         if o is not lb and o[1] > ly1 - 1 and o[2] > lx0 and o[0] < lx1),
        default=band_bottom,
    )
    if hspans:
        # 標籤下面第一條「橫跨這個標籤」的線。要橫跨標籤本身（不是整格），
        # 理由與 title_block() 的下界規則相同。
        line = min((y for y, a, b in hspans
                    if y > ly1 + 1 and a <= lx0 + 1 and b >= lx1 - 1),
                   default=None)
        if line is not None:
            bottom = min(bottom, line)
    return [
        w for w in ws
        if w not in label_boxes
        and lx0 - 2 <= _cx(w) <= right
        and ly0 - 2 <= _cy(w) <= bottom
    ]


def title_block_multi(ws: list[Word], cands: dict[str, list[Word]],
                      band_bottom: float,
                      hspans: list[tuple[float, float, float]] | None = None,
                      ) -> dict[str, Any]:
    """客戶圖框版：同一個標籤在圖上出現好幾次時，取**最下面那個有值的**。

    ★ `_label_hits()` 的「取最右下角那一個」在預設圖框成立，在客戶圖框會取到空格
      例如 `REV.` 印了兩次 —— 上面那格是本公司的版次（有值），下面那格是
      客戶版次（客戶沒填，空白）。只看最下面的，`dwg_rev` 就是空的，而圖上
      明明有寫。反過來一律取最上面也不對（預設圖框的標籤本來就會在
      註記區重複出現）。能用的判準是「那一格有沒有值」——空格不是候選。

      仍然是**由下往上**試，所以「兩格都有值」時的結果與既有規則一致。
    """
    boxes = [w for lst in cands.values() for w in lst]
    out: dict[str, Any] = {}
    conf: dict[str, float] = {}
    cells: dict[str, tuple[float, float, float, float]] = {}
    used: set[Word] = set()
    for key, lst in cands.items():
        for lb in sorted(lst, key=lambda w: (_cy(w), w[0]), reverse=True):
            got = _cell_words(ws, lb, boxes, band_bottom, hspans)
            if got:
                out[key] = _txt(got)
                conf[key] = min(w[5] for w in got)
                cells[key] = (min(w[0] for w in got), min(w[1] for w in got),
                             max(w[2] for w in got), max(w[3] for w in got))
                used.update(got)
                break
    out["_conf"] = conf
    # 每個欄位的值落在哪個矩形——只給信心低的那幾格重讀用（見 _extract_frame
    # 的 _rescue_low_conf_cells），不是每一格都要重讀：已經讀對的欄位
    # （信心已經很高）沒有理由拿去冒風險，重讀反而可能把對的字改錯。
    out["_cells"] = cells
    # 哪些字已經被某個欄位認領了。呼叫端要靠它找出「標題欄裡沒有欄位名稱的
    # 那幾格」（例如某些客戶圖框的保密聲明），見 _title_notes()。
    out["_used"] = used | set(boxes)
    return out


# --------------------------------------------------------------------------
# 材料表
# --------------------------------------------------------------------------
@dataclass
class BomRow:
    item_no: int | None = None
    description: str = ""
    qty_raw: str = ""
    conf: float = 1.0
    vendor_pn: list[str] = field(default_factory=list)
    kind: str | None = None
    # 以下是客戶圖框才有的欄（見 drawing_frames.py）。預設圖框一律留空，
    # 所以預設路徑的抽取結果與加這些欄之前完全相同。
    #
    # table_tag 分辨「這一列是哪一張表的」：有的客戶圖上有兩張表，
    # 另一張是各料號各自的長度（PN_LENGTH），混進 BOM 的話「這張圖
    # 用了幾種材料」會多算好幾列。
    table_tag: str = "BOM"
    unit: str = ""
    remark: str = ""
    color: str = ""
    length_raw: str = ""


def _grid_bands(hrules: list[float], header_top: float) -> list[tuple[float, float]]:
    """材料表的列範圍，直接從 CAD 的格線來。抓不到就回空 list。

    抽出來獨立一支，是因為 extract() 也要問同一個問題 —— `extract_method` 要
    誠實區分 `ocr`（自己按字高分群猜的列）與 `ocr_grid`（列是圖上畫的）。
    只看「這一頁有沒有格線」會把兩者混為一談：頁面有框線、但材料表那一段
    沒有列線的圖是存在的。

    ★ 「離上一條列線太近」的線要跳過，那不是列界。
      描述寫兩行的材料列（很常見：`CONN: ... PHOSPHOR BRONZE,` 換行接
      `G/F. WITH NOSE OR EQUIVALENT`）在 CAD 圖上常常中間還有一條短線，
      收下去就把一列切成兩列 —— 上半列有編號、下半列沒有，於是材料表憑空
      多一列 `? | ... |`，而且 _fill_item_no() 的位置多數決也會被這一列拖垮。
      典型情況：列高約 8pt，而那條短線距離上一條列線只有 3pt。
      門檻取典型列高的一半 —— 真正的列不會只有半列高，這是版面規則不是巧合。

    ★ 兩個門檻都從「上一條**收下來的**線」量，不是從原始序列的前一條量。
      不然一連兩條雜線就會各自跟彼此比、雙雙通過。
    """
    lines = sorted((y for y in hrules if y <= header_top + 2), reverse=True)
    if len(lines) < 3:
        return []
    gaps = [lines[i] - lines[i + 1] for i in range(len(lines) - 1)]
    typical = sorted(gaps)[len(gaps) // 2]
    keep = [lines[0]]
    for y in lines[1:]:
        gap = keep[-1] - y
        if gap > max(typical * 2.5, typical + 8):
            break
        if gap < typical * 0.5:
            continue
        keep.append(y)
    keep.reverse()
    if len(keep) < 3:
        return []
    return [(keep[i], keep[i + 1]) for i in range(len(keep) - 1)]


def _row_bands(ws: list[Word], hrules: list[float],
               header_top: float) -> list[tuple[float, float]]:
    """材料表的列範圍。

    有格線（多數圖）就直接用長水平線 —— 那是 CAD 畫的真值，一條線一列，
    連 OCR 整列漏讀都還看得出「這裡有一列」。

    沒格線（純掃描檔等）退回按字高分群：以描述欄的字為準，
    y 中心相差超過一個字高就換列。不用固定門檻，因為列高在不同圖差兩倍以上。

    ★ 由表頭往上一條一條收，遇到「明顯比列高大的空隙」就停。
      hrules 裡不只有材料表的線，還有整個圖框的框線。全收的話，材料表頂端
      到上一條框線之間會變成一個高兩百多 pt 的「列」，把圖上方的註記當成材料表的一列
      吃進來 —— 表格外的字混進材料表，比少一列更難發現。
    """
    grid = _grid_bands(hrules, header_top)
    if grid:
        return grid

    rows: list[tuple[float, float]] = []
    for w in sorted(ws, key=_cy):
        if _cy(w) >= header_top:
            continue
        h = max(w[3] - w[1], 4.0)
        if rows and abs(_cy(w) - (rows[-1][0] + rows[-1][1]) / 2) <= h * 0.7:
            lo, hi = rows[-1]
            rows[-1] = (min(lo, w[1]), max(hi, w[3]))
        else:
            rows.append((w[1], w[3]))
    return rows


def _assign(ws: list[Word], bands: list[tuple[float, float]]) -> list[list[Word]]:
    """每一個字**只能**歸給一列。回傳與 bands 等長的字群（可能是空的）。

    ★ 這個函式存在的唯一理由是「一個字被算進兩列」這個 bug：格線切出來的列是
      相鄰的（前一列的 hi 就是下一列的 lo），而原本的歸屬條件帶 ±1 的容差，
      於是**貼著格線的那一行字會同時落進上下兩列**。症狀是描述的第二行
      同時被算進相鄰兩個 band，材料表因此多出一列編號是 NULL 的
      `? | <描述第二行> |` —— 圖上根本沒有這一列。

      多出來的那一列不只是雜訊：它讓 _fill_item_no() 的「位置對得上」多數決
      算不過，於是 OCR 漏讀的編號也一起不補了。

      容差不能直接砍掉（列高只有 3~10pt，字框比格子高一點點是常態），
      所以改成「有容差、但只給最重疊的那一列」。

    ★ 比的是**字框與列的重疊高度**，不是中心點落在哪一格。中心點會在騎線的字
      上面翻船：同一行裡被 OCR 切成兩框的 `<廠商料號>` 與 `<線材規格>…`，
      前者的中心剛好落到隔壁列去，於是那一行被拆成兩列、其中一列沒有編號。
      重疊高度看的是整個字框，騎線也還是歸給它待得比較多的那一列。
    """
    out: list[list[Word]] = [[] for _ in bands]
    if not bands:
        return out
    for w in ws:
        i = _pick(w, bands)
        lo, hi = bands[i]
        if lo - 1 <= _cy(w) <= hi + 1:
            out[i].append(w)
    return out


def _pick(w: Word, bands: list[tuple[float, float]]) -> int:
    """這個字框該歸給哪一列。主鍵是重疊高度，完全沒重疊時退回中心點距離
    （取負值讓「越近越大」與主鍵同向）。"""
    cy = _cy(w)
    return max(
        range(len(bands)),
        key=lambda k: (min(bands[k][1], w[3]) - max(bands[k][0], w[1]),
                       -max(bands[k][0] - cy, cy - bands[k][1], 0.0)),
    )


# 編號框要壓過列線多少才算「騎線」。小於這個值只是字框緊貼著線，那是常態。
STRADDLE = 1.0


def _unsplit(bands: list[tuple[float, float]],
             nums: list[Word]) -> list[tuple[float, float]]:
    """ITEM 欄的編號騎在某條列線上 -> 那條線是合併儲存格的內線，不是列界。

    ★ 描述寫兩三行的材料列，CAD 常常是把 DESCRIPTION 那一格**跨列合併**，
      而框線照樣整條畫過去（長度過濾擋不掉，它就是一條 150pt 以上的實線）。
      照收的話一列會被切成兩列：有編號的那半留下，另一半變成
      `? | Cable:... 28AWG... |` —— 圖上沒有這一列。

      能分辨的是**編號自己**：合併儲存格的編號是照整格高度置中的，所以它會
      壓在內線上；而正常列的編號完整落在自己那一格裡，碰不到上下兩條線。
      三行描述的客戶圖框與兩行描述的預設圖框版式完全不同，同一條規則都命中。

    ★ 光看「有沒有編號騎線」會**合併掉真的兩列**：列高緊的表格裡，OCR 的數字框
      本來就常常壓到上下兩條線各一兩個 pt。所以還要加一條硬性條件 ——
      **合併後那一列不能有兩個編號**。兩邊各有自己的編號，就表示這條線是真的
      列界，不管數字框壓到哪裡。沒有這道保險時，相鄰的第 1、2 列會被併成一列
      （數量欄變成 `1PCS 1PCS`），而第 1 列的描述反而掉成 `?` 列。

    只處理相鄰且真的共用邊界的 band（也就是格線來的那種）；字高分群那條路徑
    的 band 本來就不連續，不會誤觸。
    """
    if len(bands) < 2 or not nums:
        return bands
    owner: dict[int, int] = {}
    for w in nums:
        i = _pick(w, bands)
        owner[i] = owner.get(i, 0) + 1

    out = [bands[0]]
    held = owner.get(0, 0)          # 目前這一列（可能已經併過）帶著幾個編號
    for i, (lo, hi) in enumerate(bands[1:], 1):
        edge = out[-1][1]
        mine = owner.get(i, 0)
        crossed = (
            abs(edge - lo) < 0.01
            and held + mine <= 1
            and any(w[1] + STRADDLE <= edge <= w[3] - STRADDLE for w in nums)
        )
        if crossed:
            out[-1] = (out[-1][0], hi)
            held += mine
        else:
            out.append((lo, hi))
            held = mine
    return out


# ITEM 欄的最大寬度（pt）。欄裡只有 1~3 位數，實測都在 10pt 上下；
# 25 是寬鬆的上限，用來擋掉「一整欄描述文字剛好都以數字開頭」那種假群。
ITEM_COL_W = 25.0
_BARE_INT = re.compile(r"^\d{1,3}$")


def _find_item_column(body: list[Word], qty_lo: float) -> tuple[float, float] | None:
    """ITEM 表頭沒被認出來時，用**編號自己**把那一欄找回來。

    ★ 這條路徑存在的理由：表頭認不出來的代價不是少一個欄位，是**整個 ITEM 欄
      沒有邊界**（item_lo/item_hi 是 ±inf），於是每一列的編號都被歸進描述欄，
      整張表的 item_no 全是 NULL —— 這是 `?` 列的主要來源之一。
      `_BOM_HEADERS` 的 regex 已經放寬成 `^[I1L|]?TEM$` 收「開頭的 I 被表格
      左框線吃掉」那種，但有些圖是連 `TEM` 都沒讀到 —— 放寬字樣救不了。

    ★ 找法不靠格線，靠**編號的幾何特徵**：材料表的編號是一疊 1~3 位數的短字框，
      x 對齊在一條很窄的帶上（10pt 上下），而且一列最多一個。
      垂直格線試過，不可靠：同一條線 CAD 常會重複畫好幾次（座標差零點幾 pt），
      而且欄界跟文字起點對不上（描述起點與最近的格線可以差到四十幾 pt）。
      編號自己反而是最乾淨的訊號。

    ★ 找到也要**驗過才用**：要求至少 3 個不同的 y 位置各出現一個，否則
      「規格裡剛好有一串對齊的數字」也會被當成 ITEM 欄。寧可回 None 留 NULL，
      也不要把描述欄的某一段切走 —— 切走的字不會有任何症狀，只是不見了。
    """
    cands = [w for w in body if _cx(w) < qty_lo and _BARE_INT.match(w[4].strip())]
    if len(cands) < 3:
        return None
    best: tuple[float, float] | None = None
    best_n = 0
    for seed in cands:
        lo = _cx(seed) - ITEM_COL_W / 2
        hi = lo + ITEM_COL_W
        rows = {round(_cy(w), 0) for w in cands if lo <= _cx(w) <= hi}
        if len(rows) > best_n:
            best_n, best = len(rows), (lo, hi)
    if best is None or best_n < 3:
        return None
    # 收斂到實際命中的那幾個字框上，不要留 25pt 的空殼 —— 太寬會把描述欄
    # 開頭的短數字（`30AWG` 前面那種）一起吃進來。
    inside = [w for w in cands if best[0] <= _cx(w) <= best[1]]
    return (min(w[0] for w in inside) - 3, max(w[2] for w in inside) + 3)


# 一列的左緣比材料表自己的左緣還右這麼多（pt），就當它是別張表的列。
# 實際的落差通常是兩百多 pt，60 是很寬鬆的門檻。
FOREIGN_X = 60.0


def _trim_foreign_bands(bands: list[tuple[float, float]],
                        body: list[Word]) -> list[tuple[float, float]]:
    """把「其實是另一張表」的列切掉。

    ★ `_grid_bands()` 從表頭往上沿著格線一條一條收，遇到明顯的大空隙才停。
      但圖上**還有別的表格，而且緊貼在材料表上面**時，兩張表的格線間距差不多，
      那個空隙判斷就攔不住 —— 典型版面：

          下方  材料表      x 從 ITEM 欄起      Q'TY 是 1PCS / A/R / 16PCS
          上方  netname 表  x 從頁面中段起      右欄是一串訊號名稱
          兩張表之間的空隙只比材料表自己的列高大一點 —— 差距不足以停下來。

      結果是一整串 `I2C_SDA... | I2C_(` 這種 netname 被當成材料進了資料庫，
      而且它們自己那一欄的 pin 編號還會被讀成 ITEM 編號，跟材料表的 1..11
      撞號。使用者問的是材料，圖上的 netname 表不在問題範圍內，直接不看。

    ★ 判準是**左緣**，不是內容。材料表一定從 ITEM 欄開始（整張表最左邊），
      別張表落在它右邊某處 —— 這是版面事實，不用去猜哪個字串像材料。
      基準取「最靠近表頭的三列」的左緣：那幾列鐵定是材料表自己的。

    ★ 一碰到別張表就停，不再往上找。表格是連續的，跳過一段再收下去
      只會把更上面某個不相干的區塊也收進來。
    """
    if len(bands) < 3:
        return bands
    lefts = [min((w[0] for w in cells), default=None) for cells in _assign(body, bands)]
    base = min((x for x in lefts[-3:] if x is not None), default=None)
    if base is None:
        return bands
    for i in range(len(bands) - 1, -1, -1):
        if lefts[i] is not None and lefts[i] > base + FOREIGN_X:
            return bands[i + 1:]
    return bands


# 一列的左緣比材料表自己的左緣還右這麼多（pt），就當它是別張表的列。
# 實際的落差通常是兩百多 pt，60 是很寬鬆的門檻。
FOREIGN_X = 60.0


def _trim_foreign_bands(bands: list[tuple[float, float]],
                        body: list[Word]) -> list[tuple[float, float]]:
    """把「其實是另一張表」的列切掉。要的只有左下的材料表，圖上其他表格不看。

    ★ `_grid_bands()` 從表頭往上沿著格線一條一條收，遇到明顯的大空隙才停。
      但圖上**還有別的表格、而且緊貼在材料表上面**時，兩張表的格線間距差不多，
      那個空隙判斷就攔不住 —— 典型版面：

          下方  材料表      x 從 ITEM 欄起      Q'TY 是 1PCS / A/R / 16PCS
          上方  netname 表  x 從頁面中段起      右欄是一串訊號名稱
          兩張表之間的空隙只比材料表自己的列高大一點 —— 差距不足以停下來。

      結果是一整串 `I2C_SDA... | I2C_(` 這種 netname 被當成材料入庫，而且
      netname 表自己那一欄的 pin 編號會被讀成 ITEM 編號，跟材料表真正的 1..11
      撞號（DB 裡這份圖因此有兩組 1、兩組 2 …）。這些字還會一起進
      `search_text`，「哪些料號用了 XXX」就會被它們撈到。

    ★ 判準是**左緣**，不是內容。材料表一定從 ITEM 欄開始（整張表最左邊），
      別張表落在它右邊某處 —— 這是版面事實，不必去猜哪個字串像材料。
      基準是「從表頭往上走到目前為止看過的最小左緣」，逐列更新。

    ★ **連續兩列都是外來的才停**，單獨一列不算。這一條是量出來的：
      材料表裡有一列漏讀編號時，它的左緣就從 ITEM 欄跳到描述欄
      （那一列的左緣跳到一百多 pt，其餘列都在 25 上下），只看一列就停的話，
      長材料表會被砍到只剩表頭附近那一兩列 —— 修一個問題製造一個更大的。
      別張表是一整塊，不會只有一列。

    ★ 一碰到別張表就停，不再往上找。表格是連續的，跳過一段再往上收，
      只會把更上面某個不相干的區塊也收進來。
    """
    if len(bands) < 3:
        return bands
    lefts = [min((w[0] for w in cells), default=None) for cells in _assign(body, bands)]

    def foreign(i: int, base: float) -> bool:
        # 空列不算外來的：它沒有內容可以判斷，也不該把一段表格切斷。
        return lefts[i] is not None and lefts[i] > base + FOREIGN_X

    base = None
    for i in range(len(bands) - 1, -1, -1):
        if lefts[i] is not None and (base is None or lefts[i] < base):
            base = lefts[i]
        if base is None:
            continue
        if foreign(i, base) and (i == 0 or foreign(i - 1, base)):
            return bands[i + 1:]
    return bands


# 補救那一列時試的 (左邊界內縮 pt, DPI) 組合。300dpi、緊貼欄界（pad=0）
# ——生產環境整表用的那個尺度與切法——已經證明讀不到，不重試。
#
# ★ 兩個維度都要掃，不能只掃 DPI
#   pad=0（clip 左邊緣貼著 ITEM 欄右界）在任何 DPI 都讀不到：偵測模型
#   需要欄界外一點點空白當「這是一行字的開頭」的線索，貼死了等於連起手勢
#   都沒有。但 pad 也不能太大——pad>=16 會把 ITEM 欄自己的殘影（`3`／`M`
#   這種被拉長的殘影）圈進來，讀到的不是 description，是編號欄的雜訊，
#   比留空更糟（假內容）。8~14 是量出來的安全帶。
#   DPI 這個維度本身非單調（見模組 docstring），不同圖要的 pad/DPI 組合
#   並不相同（有的要 600~750dpi，有的 450dpi 就夠），所以兩個維度都要掃過。
_RESCUE_PADS = (10, 12, 14)
_RESCUE_DPI = (450, 600, 750)

# 少於這個長度就當雜訊丟掉，不當成補救成功。真的 description 最短也有
# 十幾個字（`PULL TAB:COLOR:BLUE`）；pad 太大時撈到的 ITEM 欄殘影是
# 單字元（`M`／`3`）。門檻取在兩者之間，寧可漏補、不要把假內容寫回去——
# 這一列本來就是空的，錯的內容比留空更誤導使用者。
_RESCUE_MIN_LEN = 6


def _rescue_blank_description(path: str, page_no: int, rows: list[BomRow],
                              bands: list[tuple[float, float]],
                              item_hi: float, qty_lo: float) -> None:
    """材料表整條一起 OCR 時，少數列的 description 會整段讀不到——不是
    信心不足被 MIN_SCORE 濾掉，是那個解析度下 RapidOCR 對那一列的字型／
    字距完全沒偵測到任何框（少數列、分散在不同的圖，編號跟數量都在，
    唯獨 description 空）。

    ★ 為什麼編號／數量不受影響、只有 description 出事
      ITEM 跟 Q'TY 都是 1~3 個字的孤立短 token，任何堪用的解析度都認得出來；
      description 是整行擠在一起的長字串（`TER:PH2.54PHOSPHOR BRONZE...`），
      整條表格選的那個解析度剛好是它的偵測死角，換一個解析度、換一個切法
      重跑「這一列自己」就找得到——同一批像素，生產環境的切法＋300dpi
      讀到 0 個字，換成 pad=10、600dpi 讀到信心 0.96 的完整描述。這是模組
      docstring 已經記過的 DPI 非單調現象，這裡只是把它用在「事後補救單獨
      一列」而不是「事前選一個解析度」。

    ★ 只挑「有編號、有數量、description 空」的列，不是所有空列
      沒編號也沒數量的列本來就會在 bom_rows() 前面被整批丟掉（那是報價單
      版式，不是材料表）；有編號沒數量的材料表也是合法的存在（有些客戶圖框
      整欄都沒有 Q'TY）。只有「兩邊都有、就 description 沒有」是可疑的 ——
      那正是這一列的字真的存在、只是這一個解析度沒認出來的訊號。

    重試失敗（找不到、或找到的太短像雜訊）就照原樣留空，不編內容。
    """
    targets = [(r, b) for r, b in zip(rows, bands)
              if r.item_no is not None and r.qty_raw and not r.description]
    if not targets:
        return

    import fitz

    for row, (lo, hi) in targets:
        for pad in _RESCUE_PADS:
            clip = fitz.Rect(item_hi - pad, lo - 1, qty_lo, hi + 1)
            found = False
            for dpi in _RESCUE_DPI:
                try:
                    ws = pdf_ocr.words(path, page_no, dpi=dpi, clip=clip)
                except Exception as e:  # noqa: BLE001 -- 補救失敗就留空，不擋整份圖
                    print(f"    ⚠ description 補救失敗（ITEM {row.item_no}, "
                          f"pad={pad}, {dpi}dpi）：{type(e).__name__}: {e}")
                    continue
                text = _txt(ws)
                if len(text) >= _RESCUE_MIN_LEN:
                    row.description = text
                    row.vendor_pn = vendor_pn(text)
                    row.kind = kind(text)
                    row.conf = min(row.conf, min((w[5] for w in ws), default=row.conf))
                    found = True
                    break
            if found:
                break


def bom_rows(ws: list[Word], headers: dict[str, Word],
             hrules: list[float], path: str | None = None,
             page_no: int = 0) -> list[BomRow]:
    """材料表逐列。欄位歸屬用表頭的實際 x 位置，列用格線（或字高分群）。

    ★ 欄界不能取「ITEM 與 DESCRIPTION 兩個表頭的中點」—— DESCRIPTION 是置中
      標題，中點會落在 x=150，而描述文字是從 x=47 開始的，整欄會被切成兩半。
      要用的是各欄**表頭字框本身**的範圍：ITEM 欄只有窄窄的數字，Q'TY 欄同理。

    `path` / `page_no` 只給 _rescue_blank_description() 用（見該函式）。
    沒帶（測試或舊呼叫端）就跳過補救，行為與加這個功能之前一樣。
    """
    if "item" not in headers and "qty" not in headers:
        return []

    anchor = headers.get("item") or headers.get("qty")
    header_top = anchor[1] - 1

    qty_lo = headers["qty"][0] - 12 if "qty" in headers else float("inf")
    body = [w for w in ws if _cy(w) < header_top]

    if "item" in headers:
        item_lo = headers["item"][0] - 5
        item_hi = headers["item"][2] + 5
    else:
        # 表頭沒認出來就從編號自己找那一欄，找不到才退回「沒有 ITEM 欄」。
        item_lo, item_hi = _find_item_column(body, qty_lo) or (
            float("-inf"), float("-inf"))

    bands = _row_bands(body, hrules, header_top)
    bands = _trim_foreign_bands(bands, body)
    bands = _unsplit(bands, [w for w in body if item_lo <= _cx(w) <= item_hi])
    out: list[BomRow] = []
    out_bands: list[tuple[float, float]] = []
    for band, cells in zip(bands, _assign(body, bands)):
        if not cells:
            continue
        num = [w for w in cells if item_lo <= _cx(w) <= item_hi]
        qty = [w for w in cells if _cx(w) >= qty_lo]
        desc = [w for w in cells
                if not (item_lo <= _cx(w) <= item_hi) and _cx(w) < qty_lo]
        d = _txt(desc)
        q = _txt(qty)
        if not d and not q:
            continue
        row = BomRow(
            item_no=_int(_txt(num)),
            description=d,
            qty_raw=q,
            conf=min((w[5] for w in cells), default=1.0),
        )
        row.vendor_pn = vendor_pn(d)
        row.kind = kind(d)
        out.append(row)
        out_bands.append(band)

    # ★ 「一列編號都沒有、一列數量也沒有」的東西不是材料表，整批丟掉。
    #
    #   有些 PDF 其實是客戶的報價單版式，不是工程圖框。抽出來的「列」全是
    #   條款文字（`* NCNR (Non-cancelable & non-returnable)` 這類）
    #   —— 每一列都沒有編號也沒有數量。這比留 `?` 列更糟：卡片上會出現一張
    #   看起來像材料表、實際上是合約條款的表，而且 search_text 也會吃進去，
    #   「哪些料號用了 XXX」就會被這些字撈到。
    #
    #   條件是 AND 而不是 OR：數量欄整欄空白的真材料表是存在的
    #   （有些客戶圖框整欄都沒有 Q'TY），那種有編號，擋不到它。
    if not any(r.item_no is not None for r in out) and not any(r.qty_raw for r in out):
        return []

    # 材料表在工程圖上是由下往上編號（第 1 列在表頭正上方），out 目前是
    # 由上而下，所以反過來就是 1..N 的順序。
    out.reverse()
    out_bands.reverse()
    _rescue_glued_item_no(out)
    _fill_item_no(out)

    # description 補救排在 _fill_item_no() 之後，不是之前：那一列的編號
    # 常常本來就是 OCR 漏讀的（見 _fill_item_no），漏讀跟
    # description 空是同一種「這個解析度認不出來」的症狀，一起發生很常見。
    # 排在前面的話，這種列在補救那一刻 item_no 還是 None，會被
    # _rescue_blank_description() 的判準（要求 item_no 已知）跳過，
    # 明明救得回來卻沒救到。
    pairs = sorted(zip(out, out_bands),
                   key=lambda p: (p[0].item_no is None, p[0].item_no or 0))
    out = [p[0] for p in pairs]
    out_bands = [p[1] for p in pairs]

    if path is not None:
        _rescue_blank_description(path, page_no, out, out_bands, item_hi, qty_lo)

    return out


# 黏在描述頭尾的裸編號。要求前後有空白隔開、而且只有 1~3 位數 ——
# `30AWG(...)` 不會命中（那不是獨立的整數 token），`4U"` 也不會。
_GLUED_HEAD = re.compile(r"^(\d{1,3})\s+(?=\S)")
_GLUED_TAIL = re.compile(r"\s+(\d{1,3})$")


def _rescue_glued_item_no(rows: list[BomRow]) -> None:
    """編號被 OCR 跟描述讀成同一個字框時，把它從描述裡救回來。

    ★ 這是 `?` 列最大的一個來源。症狀長這樣 —— ITEM 表頭有認出來、欄界也在，但是：

          None | 1 WIRE:... 30AWG ...                            | A/R
          None | 3 TER:PHOSPHOR BRONZE, GOLD PLATED ,...         | 40PCS
          None | ACETATECLOTHTAPE:COLOR:BLACK,HF 4                | A/R

      偵測模型把 ITEM 格與 DESCRIPTION 格框成同一塊，而歸欄看的是字框中心的
      x —— 中心落在描述欄，所以 ITEM 欄一個字都沒有。編號**在頁面上讀到了**，
      只是被放在錯的地方；這跟「OCR 根本沒認出那個數字」是兩件事。
      注意編號有時在頭、有時在尾，兩邊都要看。

    ★ **只有整批對得上位置才算數**，而且驗算前不改任何一列。這一條非常重要：
      單看一列，「描述開頭的裸數字是不是編號」是猜的；但材料表是 1..N 連號的，
      所以「把候選值排成一列，看它們是不是剛好等於 1,2,3…」就是一個可以驗算的
      假設。對不上就整批放回去，描述一個字都不動 —— 亂剝數字會把
      `30AWG` 的規格值吃掉，而那種錯誤沒有任何症狀。
      門檻沿用 _fill_item_no() 的三分之二多數決，兩邊是同一個判準。
    """
    todo = [(i, r) for i, r in enumerate(rows) if r.item_no is None and r.description]
    if len(todo) < 2:
        return

    cand: list[tuple[int, BomRow, int, str]] = []
    for i, r in todo:
        for rx, cut in ((_GLUED_HEAD, "head"), (_GLUED_TAIL, "tail")):
            m = rx.search(r.description)
            if not m:
                continue
            rest = (r.description[m.end():] if cut == "head"
                    else r.description[:m.start()]).strip()
            if rest:
                cand.append((i, r, int(m.group(1)), rest))
                break

    # 位置對得上才動手。i 是由下而上的列序，圖上的編號就是 i+1。
    if len(cand) < 2:
        return
    agree = sum(1 for i, _, n, _ in cand if n == i + 1)
    if agree * 3 < len(cand) * 2:
        return
    for _, r, n, rest in cand:
        r.item_no = n
        r.description = rest
        r.vendor_pn = vendor_pn(rest)
        r.kind = kind(rest)


def _fill_item_no(rows: list[BomRow]) -> None:
    """OCR 漏讀的列編號，用「它在表上的第幾列」補回來。

    rows 已經照圖上的順序（由下而上 = 1..N）排好。常見的情況是某一列的編號
    整個沒被偵測到 —— 但那一列的描述與數量都在，格線也證明那裡有一列。

    ★ 門檻是「三分之二的列對得上位置」，不是「全部對得上」。例如某一列
      的 `9` 被讀成 `6`（跟第 6 列撞號），全等的話就整批不補，反而留下一個
      重複編號 + 一個 NULL。格線是 CAD 畫的真值、OCR 的數字是猜的，
      多數決對得上時就該信格線。

      對不上三分之二就整批不補：那代表這張圖的編號本來就不是 1..N 連號
      （或列被切錯了），這時候用位置去猜等於自己編一個編號出來 ——
      寧可留 NULL 讓 --report 標出來人工看。
    """
    known = [(i, r.item_no) for i, r in enumerate(rows) if r.item_no is not None]
    if len(known) < 2:
        return
    agree = sum(1 for i, n in known if n == i + 1)
    if agree * 3 < len(known) * 2:
        return
    for i, r in enumerate(rows):
        r.item_no = i + 1


def _int(s: str) -> int | None:
    m = re.search(r"\d+", s or "")
    return int(m.group()) if m else None


# --------------------------------------------------------------------------
# 數量 / 廠商料號 / 類別
# --------------------------------------------------------------------------
_QTY_NUM = re.compile(r"(\d+(?:\.\d+)?)\s*(?:PCS|PC|EA|SET|M|MM)?\b", re.I)
_AR = re.compile(r"A\s*/?\s*R", re.I)


def qty_num(raw: str) -> float | None:
    """`8PCS` -> 8.0；`A/R` -> None。

    ★ A/R（As Required）是**有效數量**，不是缺值 —— 膠帶、熱縮套管、電線這類
      常寫 A/R。原文一律存在 qty_raw，這裡只負責「數得出來的
      才給數字」。塞 0 會被讀成「不需要」，比留空更糟。
    """
    if not raw or _AR.search(raw):
        return None
    m = _QTY_NUM.search(raw)
    return float(m.group(1)) if m else None


# 廠商前綴詞表 + 通用 P/N 樣式。寫法很一致：`<廠商> P/N:ABC-0001`，
# 偶爾漏冒號寫成 `<廠商> PN:ABC-0002`。
# 展示版：前綴詞只列幾個公開的連接器品牌當示意；實際詞表是從語料整理出來的，未收錄。
_VENDOR_PN = re.compile(
    r"(?:^|[\s,;(])(?:HR|JST|MOLEX|AMP|TE|JAE)?\s*"
    r"P\s*/?\s*N\s*[:：]?\s*([A-Z0-9][A-Z0-9\-/_.]{3,})",
    re.I,
)


def vendor_pn(desc: str) -> list[str]:
    """描述裡的廠商料號。讓「哪些料號用了 <廠商料號> 這顆端子」反查得到。"""
    out: list[str] = []
    for m in _VENDOR_PN.finditer(desc or ""):
        v = _o_to_zero(m.group(1).upper().rstrip(".,;"))
        if v and v not in out:
            out.append(v)
    return out


# 類別詞取冒號前那一段。這類圖的寫法非常一致（WIRE: / HSG: / TER: / LABEL: /
# HS TUBE: ...），所以是「取冒號前」而不是關鍵字比對 —— 後者遇到沒見過的
# 類別就會漏，前者至少存得下來。
_KIND_OK = re.compile(r"^[A-Z][A-Z0-9 /._-]{1,24}$")
_KIND_FALLBACK = ("ACETATE CLOTH TAPE", "GLUE", "RD CBL", "LABEL", "HS TUBE",
                  "WIRE", "HSG", "TER", "TAPE", "SHIELD", "BRAID")


def kind(desc: str) -> str | None:
    d = (desc or "").strip()
    if not d:
        return None
    head = d.split(":", 1)[0].strip().upper() if ":" in d else ""
    if head and _KIND_OK.match(head):
        return head
    up = d.upper()
    for k in _KIND_FALLBACK:
        if up.startswith(k):
            return k
    return None


# --------------------------------------------------------------------------
# 料號正規化
# --------------------------------------------------------------------------
# 這類圖固定的兩種 OCR 錯誤：連字號被吃掉、字母 O 與數字 0 互換。
# `Z0000XX00000 1` 要還原成 `Z0000XX00000-1`，`ABC-1O01` 的 O 要還原成 0。
def _o_to_zero(s: str) -> str:
    """只在「數字之間或緊鄰數字」的位置把 O 還原成 0。

    整串無條件 O->0 會把 `NONE` 變成 `N0NE`、`OR EQU` 變成 `0R EQU`。
    """
    return re.sub(r"(?<=\d)[Oo](?=[\dA-Z-])|(?<=[A-Z0-9])[Oo](?=\d)", "0", s)


# PLM 自己加在料號前面的系統前綴。圖上不會印它 —— 那是 PLM 裡的分類碼，
# 不是圖框上的欄位。展示版：前綴值為示意。
#
# ★ 只收這一個前綴，而且寫死
#   這不是「差幾個字就算對上」的模糊比對，是一條公司規定：`PT-ABC-0001`
#   與圖上印的 `ABC-0001` 是**同一顆料**。放寬成「開頭差一段就算」的話，
#   `Z0000XX00000-1` 與 `Z0000XX00000-2` 這種真的是兩顆的也會被判成同一顆，
#   而 part_no_ok 這一欄的用途正是「這個料號敢不敢直接講」。
_PLM_PREFIXES = ("PT-",)


def normalize_pn(raw: str, authoritative: str | None) -> tuple[str | None, bool]:
    """(要存的值, 是否與 PLM 權威料號核對一致)

    ★ 對不上就存 OCR 原值 + False，**不硬改**。原則是：沒把握就留白，
      不拿「看起來像那個料號」的某一列充數。
      錯的值一旦被 LLM 引用，比沒有值更糟。

    ★ 完全對上時存**權威值**，只差 PLM 前綴時存**圖上印的值**
      兩種都是 True，但存的東西不一樣，理由是這一欄的定義是「圖上的 PART NO.」：
        完全對上 -> 兩邊字面相同，存權威值等於順手清掉 OCR 的雜訊
                    （被吃掉的連字號、O/0 混淆），存的還是圖上那個號。
        只差前綴 -> 存權威值就變成往圖上那一格填一個**圖上沒有印的字串**
                    （`PT-ABC-0001`），而卡片上那一格的標題是
                    「標題欄（圖框右下）」—— 使用者拿卡片對原圖會發現對不上。
      所以前綴這條路只改「核對過了」這個判斷，不改值。歸檔料號本來就另有
      `item_number` 一欄在存。
    """
    raw = (raw or "").strip()
    if not raw:
        return (None, False)

    # OCR 的候選寫法：原樣、空白換連字號、O->0、兩者都做
    cands = {raw}
    for base in (raw, _o_to_zero(raw)):
        cands.add(base)
        cands.add(re.sub(r"\s+", "-", base.strip()))
        cands.add(re.sub(r"\s+", "", base.strip()))
    cands = {re.sub(r"[^A-Z0-9-]", "", c.upper()) for c in cands}

    if authoritative:
        # 跟 cands 用同一個正規化（去掉非英數連字號的字元），否則帶標點的
        # 權威料號永遠比不上——例如 `ABC-1.5M`：candidates 那一輪把
        # `.` 濾掉變成 `ABC-15M`，這裡的 auth 卻原樣留著 `.`，於是
        # 完全相同的字串因為兩邊濾法不一致而判定「對不上」，明明圖上印的
        # 就是這個號碼。
        auth = re.sub(r"[^A-Z0-9-]", "", authoritative.strip().upper())
        if auth in cands:
            return (authoritative.strip(), True)
        # 連字號位置也可能被吃掉，去掉連字號再比一次
        flat = auth.replace("-", "")
        if any(c.replace("-", "") == flat for c in cands):
            return (authoritative.strip(), True)
        for pre in _PLM_PREFIXES:
            if not auth.startswith(pre):
                continue
            bare = auth[len(pre):].replace("-", "")
            if any(c.replace("-", "") == bare for c in cands):
                return (raw, True)

    return (raw, False)


# --------------------------------------------------------------------------
# 對外唯一入口
# --------------------------------------------------------------------------
_SHEET = re.compile(r"(\d+)\s*[O0]F\s*(\d+)?", re.I)


def parse_sheet(s: str) -> tuple[int | None, int | None]:
    """`1 OF 2` / `1OF2` / `1OF` 三種形狀都要吃 —— OCR 會把空白吃掉，
    而 `OF` 的 O 也常被讀成數字 0（`1OF` 與 `10F` 都出現過）。"""
    m = _SHEET.search(s or "")
    if not m:
        n = _int(s)
        return (n, None)
    return (int(m.group(1)), int(m.group(2)) if m.group(2) else None)


@dataclass
class Drawing:
    page_no: int
    method: str
    title: str | None = None
    part_no: str | None = None
    part_no_ok: bool = False
    rev: str | None = None
    size: str | None = None
    scale: str | None = None
    sheet_no: int | None = None
    sheet_of: int | None = None
    pd_file_no: str | None = None
    dwg_date: str | None = None
    ocr_conf: float | None = None
    title_raw: str = ""
    items_raw: str = ""
    items: list[BomRow] = field(default_factory=list)
    # 客戶圖框才有的標題欄欄位（MATERIAL / P.CODE / CUSTOMER PART NO. /
    # DRAWN BY …）。刻意是一包 key->value 而不是各給一個資料庫欄位：
    # 每一種客戶圖框的欄位都不一樣，加欄位等於每遇到一種新圖框就改一次
    # schema，而這一包直接進 jsonb 就查得到。
    title_fields: dict[str, str] = field(default_factory=dict)
    frame: str | None = None         # 命中的客戶圖框樣式（None = 公司預設圖框）


_DATE = re.compile(r"\b(\d{1,2}[/.]\d{1,2}[/.]\d{2,4}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b")


def _title_bottom(pg: Page, labels: dict[str, Word]) -> float:
    """標題欄格子的下界：從最上面那個標籤開始，沿著格線的「梯子」往下走到它斷掉為止。

    ★ 不能用頁面底緣。CAD 圖框沿著下緣印著欄位標記（`1 2 3 … 8`），它們在圖框
      線外面，但 x 上正落在標題欄那幾格的正下方。下界放到頁底，`PART NO.`
      那一格就會把最右邊那個 `8` 一起吃進來 —— 料號變成 `Z0000XX00000-1 8`：
      值本身是對的，只是尾巴多一個字，於是核對不過。這不是個案。

    ★ 也不能用「最底下那條長水平線」：那條通常是整張圖框的外框（跟標題欄
      底線之間還隔著一段），欄位標記就印在這兩條之間。

    ★ 更不能用「最低的標籤下面第一條線」。試過，會壞在 `SHEET` 被 OCR 連著值
      讀成一整塊（`SH,T NO. 1OF 1`）而認不出標籤的那些圖：最低標籤變成上一列的
      `SCALE`，第一條線剛好落在料號值的上方 —— **整欄被切掉**。

      能用的是格線本身的節奏：標題欄的橫線是均勻的一疊（間距幾 pt 到十 pt），
      而它與下面的圖框外框之間有一個明顯的大空隙（二十 pt 上下）。所以從標題欄頂端
      往下逐條走，走到間距突然變大就停 —— 停的那條就是標題欄的底。
    """
    if not (pg.hrules and labels):
        return pg.rect.y1
    top = min(w[1] for w in labels.values())
    ladder = sorted(y for y in pg.hrules if y > top)
    if not ladder:
        return pg.rect.y1
    # 標籤字高約 7pt；間距超過它的兩倍多就不是同一疊格線了。
    heights = sorted(w[3] - w[1] for w in labels.values())
    gap_max = max(14.0, 2.2 * heights[len(heights) // 2])
    bottom = ladder[0]
    for y in ladder[1:]:
        if y - bottom > gap_max:
            break
        bottom = y
    return bottom


def _refine(path: str, pg: Page, lo: float, hi: float) -> list[Word]:
    """把 band 的一塊（x 從 lo 到 hi）用它自己的寬度重跑一次 OCR。

    定位那一次是整條 842pt 壓成 1450px；這一次只有這一塊，同樣 1450px
    等於兩倍以上的每字像素數 —— 連字號就是在這一步救回來的。
    """
    import fitz

    if pg.method != "ocr" or pg.clip is None or hi - lo < 40:
        return []
    clip = fitz.Rect(max(pg.rect.x0, lo), pg.clip.y0,
                     min(pg.rect.x1, hi), pg.clip.y1)
    return _ocr_block(path, pg.page_no, clip, BLOCK_PX)


def _bom_clip(pg: Page, headers: dict[str, Word]):
    """材料表**表格本身**的矩形（不含上方的 NOTE 區），沒有格線就回 None。

    ★ 為什麼要再切一塊：_refine() 給材料表那一塊的 y 範圍是整條 band
      （將近 200pt 高），裡面除了表格還有 NOTE、測試條件、標籤說明那一整段
      自由文字。那些雜訊會讓偵測模型整行漏掉 —— 同一份 PDF、同一個 x 範圍，
      只把 y 收成表格那幾十 pt，能讀到那一行的 DPI 檔位就從極少數變成大多數。

      漏掉的往往是描述最長的那一行（例如 `CONN: ... IDC TYPE ... SOCKET, ...`）
      —— 整列最有價值的那一行。
      DPI 掃描是非單調的（模組 docstring 已記載這件事），所以解法不是挑一個
      幸運的 DPI，是**把不相干的東西移出畫面**。

    y 上界取最上面那條列線（而不是 band 頂），下界取表頭字框底 ——
    表頭要留著，_merge() 才不會把它當成新位置補進來一份重複的 ITEM/Q'TY。
    """
    import fitz

    anchor = headers.get("item") or headers.get("qty")
    if pg.method != "ocr" or pg.clip is None or anchor is None:
        return None
    grid = _grid_bands(pg.hrules, anchor[1] - 1)
    if not grid:
        return None
    x0 = min(w[0] for w in headers.values()) - 14
    x1 = max(w[2] for w in headers.values()) + 10
    return fitz.Rect(
        max(pg.rect.x0, x0), max(pg.rect.y0, min(lo for lo, _ in grid) - 3),
        min(pg.rect.x1, x1), min(pg.rect.y1, anchor[3] + 2),
    )


def _method(pg: Page, used_grid: bool) -> str:
    """誠實回報這份圖的字是怎麼來的。下游（llm.py / a2a.py）照它決定要不要
    叫使用者以原圖為準，所以「OCR 混了文字層」不能講成純 OCR ——
    那一半是照抄的，可信度跟猜出來的不一樣。"""
    if pg.method == "text":
        return "text"
    base = "ocr_grid" if used_grid else "ocr"
    return f"{base}+text" if pg.text_n else base


def extract(path: str, authoritative_pn: str | None = None) -> Drawing | None:
    """一份 PDF -> Drawing。讀不到圖框那一頁時回 None。"""
    pg = find_page(path)
    if pg is None:
        return None

    # 客戶自己的圖框（見 drawing_frames.py）走另一套版面詞彙。
    # 偵測看的是**整頁**的字，不只下方那條 —— 有的圖框的關鍵表格
    # （例如 `P/N AND LENGTH`）與它的標題都在圖的上半，band 裡看不到。
    frame = drawing_frames.detect(list(pg.words) + list(_page_text_words(path, pg.page_no)))
    if frame is not None:
        return _extract_frame(path, pg, frame, authoritative_pn)

    # 第 1 段：定位。只為了知道兩個表格各自在哪一段 x。
    left, right, bom_h, title_h = split_bottom(pg.words)

    # 第 2、3 段：兩塊各自重跑，結果優先，定位那一輪只補沒蓋到的位置。
    if left:
        lo = min(w[0] for w in left)
        hi = max(w[2] for w in left)
        fine = _refine(path, pg, lo - 12, hi + 12)
        if fine:
            merged = _merge(fine, left)
            left, _, bom_h2, _ = split_bottom(merged)
            left = left or merged
            bom_h = bom_h2 or bom_h

        # 第 2b 段：只有表格那一塊，再跑一次補漏（見 _bom_clip）。
        # ★ 順序是 _merge(left, tight)，不是反過來 —— tight 是**補漏檢**，
        #   只填 left 沒蓋到的位置。反過來會讓這一輪的切框覆蓋掉已經讀對的行，
        #   那正是 _merge() docstring 記著的那個坑（完整的一行被切成好幾個
        #   短框，欄位值就碎了）。
        clip = _bom_clip(pg, bom_h)
        if clip is not None:
            tight = _ocr_block(path, pg.page_no, clip, BOM_PX)
            if tight:
                left = _merge(left, tight)
    if right:
        lo = min(w[0] for w in right)
        hi = max(w[2] for w in right)
        fine = _refine(path, pg, lo - 12, hi + 12)
        if fine:
            merged = _merge(fine, right)
            _, right2, _, title_h2 = split_bottom(merged)
            right = right2 or merged
            title_h = title_h2 or title_h

    tb = (title_block(right, title_h, _title_bottom(pg, title_h))
          if title_h else {"_conf": {}})
    rows = bom_rows(left, bom_h, pg.hrules, path, pg.page_no)

    raw_pn = tb.get("part_no") or ""
    pn, ok = normalize_pn(raw_pn, authoritative_pn)
    sheet_no, sheet_of = parse_sheet(tb.get("sheet") or "")
    if sheet_no is None:
        # SHEET 的標籤與值被 OCR 讀成同一塊時（`SH,T NO. 1OF 1`）認不出標籤，
        # 這一格就是空的。整塊標題欄原文再找一次 `N OF M` —— 圖框裡不會有
        # 第二個這種寫法。
        sheet_no, sheet_of = parse_sheet(_txt(right))

    conf = tb.get("_conf") or {}
    all_conf = list(conf.values()) + [r.conf for r in rows]

    # 材料表的列到底是圖上畫的還是自己按字高猜的 —— 這一欄會被 llm.py 拿去
    # 決定要不要叫使用者以原圖為準，所以要照實回報，不能只看「這頁有框線」。
    anchor = bom_h.get("item") or bom_h.get("qty")
    used_grid = bool(anchor and _grid_bands(pg.hrules, anchor[1] - 1))

    d = Drawing(
        page_no=pg.page_no,
        method=_method(pg, used_grid),
        title=tb.get("title"),
        part_no=pn,
        part_no_ok=ok,
        rev=_clean_rev(tb.get("rev")),
        size=tb.get("size"),
        scale=tb.get("scale"),
        sheet_no=sheet_no,
        sheet_of=sheet_of,
        pd_file_no=_clean_pd_file(tb.get("pd_file_no")),
        ocr_conf=round(min(all_conf), 4) if all_conf else None,
        title_raw=_txt(right),
        items_raw=_txt(left),
        items=rows,
    )
    m = _DATE.search(d.title_raw)
    d.dwg_date = m.group(1) if m else None
    return d


# --------------------------------------------------------------------------
# 客戶圖框（drawing_frames.py）
# --------------------------------------------------------------------------
_SHEET_SLASH = re.compile(r"(\d+)\s*/\s*(\d+)")


def _parse_sheet_cell(s: str) -> tuple[int | None, int | None]:
    """`SHEET` 那一格。預設圖框寫 `1 OF 2`，部分客戶圖框寫 `Page 1/1` 或 `1/1`。

    斜線那種**只在這一格認**，不拿去掃整塊標題欄原文：原文裡還有日期
    （`9/30/20XX`）與比例（`1:1`），掃整塊會把 `9/30` 讀成第 9 張共 30 張。
    """
    no, of = parse_sheet(s)
    if of is None:
        m = _SHEET_SLASH.search(s or "")
        if m:
            return (int(m.group(1)), int(m.group(2)))
    return (no, of)


def _rect_of(words: list[Word]) -> Word:
    """一群字的外接框，形狀對齊 Word（給 row_bands / table_rect 當「表頭」用）。"""
    return (min(w[0] for w in words), min(w[1] for w in words),
            max(w[2] for w in words), max(w[3] for w in words), "", 1.0)


def _in_rect(w: Word, r) -> bool:
    return (_cx(w) >= r.x0 and _cx(w) <= r.x1
            and _cy(w) >= r.y0 and _cy(w) <= r.y1)


def _frame_table(path: str, pg: Page, tbl: Any,
                 pool: list[Word]) -> tuple[Any, list[BomRow]] | None:
    """客戶圖框的一張表 -> (表格矩形, 逐列)。找不到表頭或列就回 None。"""
    hits = drawing_frames.header_row(pool, tbl)
    if not hits:
        return None
    header = _rect_of(list(hits.values()))
    box = drawing_frames.grid_box(hits, list(_vspans(path, pg.page_no)))
    # 列線只認「橫跨這張表」的那幾條（見 _hspans）。抓不到格線資訊就退回
    # 整頁的 hrules，行為與加這道過濾之前相同。
    rules = drawing_frames.row_rules(list(_hspans(path, pg.page_no)), box) \
        or pg.hrules
    bands = drawing_frames.row_bands(
        rules, header, tbl.rows_below_header,
        limit=(box[2], box[3]) if box else None)
    if not bands:
        return None
    bounds = drawing_frames.column_bounds(tbl, hits, pg.vrules)
    if not bounds:
        return None
    rect = drawing_frames.table_rect(bounds, bands, header, box)

    words = [w for w in pool if _in_rect(w, rect)]
    if tbl.ocr_refine:
        # 只給表格那一塊，用它自己的寬度反推 DPI。整條 band 壓成 1450px 時
        # 某些客戶圖框只剩 120 DPI，序號與數量那些孤立的一位數會整批讀不到；
        # 同樣的像素預算只給三百多 pt 寬的表格就是 300 DPI。理由與 _bom_clip() 相同。
        tight = _ocr_block(path, pg.page_no, rect, BOM_PX)
        if tight:
            words = _merge(words, tight, self_check=False)

    # 合併儲存格的內線要併回去。判準與預設路徑的 _unsplit() 完全相同
    # （項次的字框騎在線上 = 那條線是合併儲存格的內線）。不併的話，三行描述
    # 的那一列會被切成兩列：上半列有項次，下半列變成一列沒有項次的描述殘段，
    # 圖上根本沒有這一列。
    item_col = next((c.key for c in tbl.columns if c.role == "item"), None)
    if item_col and item_col in bounds:
        lo_x, hi_x = bounds[item_col]
        bands = _unsplit(bands, [w for w in words if lo_x <= _cx(w) <= hi_x
                                 and _BARE_INT.match(w[4].strip())])

    role = {c.key: c.role for c in tbl.columns}
    dicts = drawing_frames.read_rows(words, tbl, hits, bounds, bands)
    _rescue_faint_numbers(path, pg.page_no, tbl, bounds, box, dicts)

    rows: list[BomRow] = []
    for cells in dicts:
        vals = {role.get(k, "extra"): v for k, v in cells.items()
                if not k.startswith("_")}
        desc = vals.get("description", "")
        row = BomRow(
            item_no=_int(drawing_frames.plain_digits(vals.get("item", ""))),
            description=desc,
            qty_raw=vals.get("qty", ""),
            conf=cells["_conf"],
            table_tag=tbl.name,
            unit=vals.get("unit", ""),
            remark=vals.get("remark", ""),
            color=vals.get("color", ""),
            length_raw=vals.get("length", ""),
        )
        row.kind = vals.get("kind") or kind(desc)
        row.vendor_pn = vendor_pn(desc)
        # P/N AND LENGTH 那張表的「描述」就是料號本身，所以它自己要進
        # vendor_pn —— 「<料號> 是哪一份圖上的」要靠這一欄反查，
        # 而那張表上的料號多半不在料號主檔（parts）裡。
        if tbl.name == "PN_LENGTH" and desc and desc not in row.vendor_pn:
            row.vendor_pn.insert(0, desc)
        rows.append(row)

    # 表頭在下的表格（預設圖框與部分客戶圖框的材料表）是由下往上編號，讀出來是
    # 由上而下 —— 反過來才是圖上的 1..N。
    if not tbl.rows_below_header:
        rows.reverse()
    if item_col:
        _carry_item_no(rows)
    else:
        # 這張表沒有項次欄（P/N AND LENGTH 就是），編號用它在圖上的第幾列 ——
        # 那是圖上看得到的順序，不是猜的，而且卡片與 CSV 都要有個排序依據。
        for i, r in enumerate(rows, 1):
            r.item_no = i
    return rect, rows


# 補救單格時用的尺度與二值化門檻。600dpi 是量出來的：同一格在 300dpi
# 被讀成 `T`、450dpi 讀不到、600dpi 讀成 `1`（信心 1.00）。
_FAINT_DPI = 600
_FAINT_THRESHOLD = 252
_NUMERIC_ROLES = ("item", "qty")
# 格子邊界要往內縮幾 pt。**這個維度一定要掃**，不能寫死：貼著格線切的話
# 邊框會跟數字黏成一塊被讀成別的字（實測同一格 pad=0 讀成 `TL`、
# pad=2 讀成 `1`、信心 0.99），縮太多又會把數字自己切掉。
_FAINT_PADS = (2, 1, 3)


def _rescue_faint_numbers(path: str, page_no: int, tbl: Any,
                          bounds: dict[str, tuple[float, float]],
                          box, dicts: list[dict[str, Any]]) -> None:
    """項次欄與數量欄裡讀不到數字的格子，一格一格重讀一次。

    ★ 為什麼只有這兩欄、而且只在「讀不到數字」的時候
      這兩欄的內容**照版面定義就是數字**，所以「這一格沒有數字」是一個可以
      驗算的失敗訊號，不是猜的。規格欄不做這件事 —— 那是自由文字，沒有可以
      驗算的期望值，重讀只會拿另一個版本的字去覆蓋已經讀對的字。

    ★ 為什麼一格一格：整張表一起加強對比沒有用（見 pdf_ocr.words 的 boost）。
      成本是每個空格一次小圖 OCR，實測一格 0.3 秒上下。

    讀不到就照原樣留空，不編數字。
    """
    if box is None:
        return
    numeric = [c for c in tbl.columns if c.role in _NUMERIC_ROLES
               and c.key in bounds]
    if not numeric:
        return

    import fitz

    for row in dicts:
        lo, hi = row["_band"]
        for col in numeric:
            cur = row.get(col.key, "")
            if re.search(r"\d", drawing_frames.plain_digits(cur)):
                continue
            # ±inf 的外界收回到格線外框：最外側那一欄照樣要有矩形可以 render。
            x0, x1 = bounds[col.key]
            x0, x1 = max(x0, box[0]), min(x1, box[1])
            best: tuple[float, str] | None = None
            for pad in _FAINT_PADS:
                clip = fitz.Rect(x0 + pad, lo + pad, x1 - pad, hi - pad)
                if clip.width < 4 or clip.height < 4:
                    continue
                try:
                    got = pdf_ocr.words(path, page_no, dpi=_FAINT_DPI,
                                        clip=clip, boost=_FAINT_THRESHOLD)
                except Exception as e:  # noqa: BLE001 -- 補救失敗就留空
                    print(f"    ⚠ 淡色數字補救失敗（{col.key}）："
                          f"{type(e).__name__}: {e}")
                    break
                text = _txt(got)
                if not (text and re.search(r"\d",
                                           drawing_frames.plain_digits(text))):
                    continue
                score = min(w[5] for w in got)
                if best is None or score > best[0]:
                    best = (score, text)
                # 信心夠高就不必再試別的內縮量。
                if score >= 0.9:
                    break
            if best is not None:
                row[col.key] = best[1]
                row["_conf"] = min(row["_conf"], best[0])


def _carry_item_no(rows: list[BomRow]) -> None:
    """序號欄是合併儲存格時，把編號帶到同一組的其他列。

    ★ 中文圖框的 `序号 1` 是**一格跨三列**（三條不同顏色的電子線同屬一個
      項次），而那一格的字是照整格高度置中的 —— 所以它只落在中間那一列的
      y 帶裡，上下兩列的 item_no 是 None。留 None 的代價是「序号 1 有哪幾條線」
      只查得到三分之一。

    ★ 判準是「這一段空白的**兩端是不是同一個編號**」，不是「往下帶就好」
      一段連續的空白，上下兩端的已知編號相同（或某一端就是表格邊界）時，
      那段空白只可能是合併儲存格 —— 圖上不存在「1、空、1」這種編號。
      而兩端不同（`1、空、3`）就是真的漏讀了一列的編號，這時候**不填** ——
      那一列可能是 2，也可能是圖上本來就沒編號，猜錯不會有任何症狀。
      寧可留 NULL 讓 `ingest_drawing --report` 標出來人工看。

    rows 進來時已經是圖上的 1..N 順序。這條規則對方向不敏感（看的是一段
    空白的兩端），所以不必分「由上往下」與「由下往上」兩種寫法。
    """
    seq = rows
    i = 0
    while i < len(seq):
        if seq[i].item_no is not None:
            i += 1
            continue
        j = i
        while j < len(seq) and seq[j].item_no is None:
            j += 1
        before = seq[i - 1].item_no if i > 0 else None
        after = seq[j].item_no if j < len(seq) else None
        fill = before if after is None else (after if before is None else
                                             (before if before == after else None))
        if fill is not None:
            for r in seq[i:j]:
                if r.description or r.qty_raw:
                    r.item_no = fill
        i = j + 1


# 標題欄外框那條線離最外側標籤最遠可以差多少（pt）。有的圖框外框緊貼著標籤
# （差不到 1pt）；有的標題欄是**一個更大的框**——除了 DRAWN/CHECKED/APPROVED
# 那組欄位，同一個外框裡還圈著公司商標與 `UNLESS OTHERWISE SPECIFIED` 那段
# 公差說明，外框離最左標籤八十幾 pt。60pt 只夠框住第一種，抓不到第二種
# 的外框，症狀是「側邊框格 / 公差說明」那兩塊被漏掉，而且不報錯
# ——`side()` 找不到線就退回標籤自己的邊界，框變小了但看起來像正常運作。
# 100 留了餘裕，同時仍然遠小於「隔了半張圖的另一條線」的距離。
_TITLE_BOX_TOL = 100.0


def _title_box(labels: dict[str, Word],
               vspans: list[tuple[float, float, float]]):
    """標題欄那張表格的外框 (x0, x1, y0, y1)。抓不到就回 None。

    ★ 為什麼需要它：標題欄裡有**沒有欄位名稱的格子**
      有的客戶圖框把保密聲明（`THIS DOCUMENT CONTAINS PROPRIETARY INFORMATION
      ...`）印在標題欄框線**裡面**、最上面那一格，那一格沒有欄位名稱。`title_block_multi()` 是
      「一個標籤找一格值」，沒有標籤的格子它永遠看不到。

    ★ 用標籤的 y 範圍去挑框線，而不是反過來
      左右框線是**縱貫整個標題欄**的長線，所以「x 在最外側標籤的外面、
      而且 y 範圍蓋住所有標籤」這個條件就能認出它們，並且順便量到整張表格
      的上下界 —— 保密聲明那一格在最上面的標籤**上方**好幾十 pt，用標籤的
      bounding box 是找不到的，用框線才行。
    """
    if not (labels and vspans):
        return None
    lx0 = min(w[0] for w in labels.values())
    lx1 = max(w[2] for w in labels.values())
    ly0 = min(w[1] for w in labels.values())
    ly1 = max(w[3] for w in labels.values())

    def side(want_left: bool):
        cands = [s for s in vspans
                 if (s[0] <= lx0 + 2 if want_left else s[0] >= lx1 - 2)
                 and s[1] <= ly0 + 2 and s[2] >= ly1 - 2
                 and abs((lx0 if want_left else lx1) - s[0]) <= _TITLE_BOX_TOL]
        if not cands:
            return None
        # 最靠近標籤的那一條才是這張表格的框線（再外面的是圖框自己的邊）。
        return min(cands, key=lambda s: abs((lx0 if want_left else lx1) - s[0]))

    left, right = side(True), side(False)
    if left is None and right is None:
        return None
    ref = left or right
    x0 = left[0] if left else lx0
    x1 = right[0] if right else lx1
    return (x0, x1, ref[1], ref[2])


# 標題欄未標示格重讀的解析度。以保密聲明那一格為例：定位那一輪
# （整條 band 壓成 1450px）把單字之間的空白全吃掉，讀成
# `THISDOCUMENTCONTAINSPROPRIETARY`；只給那一格、600dpi 就讀回
# `THIS / DOCUMENT CONTAINS / PROPRIETARY / ...`（信心 0.92~1.00）。
# 300dpi 還是黏一半（`DOCUMENTCONTAINSP`），450 也還有 `WITHOUTF`，所以是 600。
_NOTE_DPI = 600


def _title_notes(ws: list[Word], box, used: set,
                 path: str | None = None, page_no: int = 0) -> str:
    """標題欄表格裡、沒有被任何欄位名稱認領的字（照閱讀順序）。

    這就是保密聲明那一格。判準是「在表格框線內」+「沒被認領」，兩個條件都
    必要：只看前者會把已經拆成欄位的值再抄一遍，只看後者會把**表格外**的字
    （中文圖框底部那整段著作權聲明、圖上的技術要求、接腳圖說明）也收進來
    —— 那些不在使用者要的兩張表裡。

    找到之後**只對那一塊再 OCR 一次**（見 _NOTE_DPI）：這一格是整段英文
    句子，而定位那一輪的尺度會把單字之間的空白吃掉 —— 黏成一團的句子
    人讀不了，而這一格的內容正是要給人讀的。讀失敗就用原本的字，不編。
    """
    if box is None:
        return ""
    x0, x1, y0, y1 = box
    # ★ 「已認領」用座標重疊判斷，不能用「物件相等」（`w not in used`）
    #   標題欄那塊後來會再 OCR 一次補細節（`_extract_frame` 的 fine-refine
    #   那一段），新字框跟原本認領的字框座標差了零點幾 pt，物件相等一律判
    #   False，於是同一個標籤（`DRAWN:` / `PART NO.:`）跟它的值又被當成
    #   「沒人要」的字收進這裡，變成整段重複。跟兩塊字框合併時同一個問題，
    #   同一個解法（`_covers()`，見 `_merge()` 的說明）。
    rest = [w for w in ws
            if not any(_covers(u, w, overlap=0.5) for u in used)
            and x0 - 2 <= _cx(w) <= x1 + 2 and y0 - 2 <= _cy(w) <= y1 + 2]
    if not rest:
        return ""

    if path is not None:
        import fitz

        clip = fitz.Rect(min(w[0] for w in rest) - 2, min(w[1] for w in rest) - 2,
                         max(w[2] for w in rest) + 2, max(w[3] for w in rest) + 2)
        try:
            fine = pdf_ocr.words(path, page_no, dpi=_NOTE_DPI, clip=clip)
        except Exception as e:  # noqa: BLE001 -- 重讀失敗就用原本的字
            print(f"    ⚠ 標題欄未標示格重讀失敗（{type(e).__name__}: {e}）")
            fine = []
        # ★ 重讀的範圍是 `rest` 的外接矩形，不是 `rest` 那幾個字本身
        #   當「未認領的字」是散在整個標題欄裡的好幾塊（不是一整段連續文字，
        #   例如公差說明混雜在 DRAWN/CHECKED/APPROVED 那一整排格子之間）時，
        #   外接矩形會把已認領的欄位也整塊圈進去，而**重讀是對那塊矩形整片
        #   重新 OCR**，不是只重讀 rest 裡那幾個字——於是已經被認領的
        #   `DRAWN:` / `PART NO.:` 又被讀出來一次，跟公差說明混在一起變成
        #   一整段重複。所以重讀完也要用同一個「已認領」判準濾一次。
        fine = [w for w in fine if not any(_covers(u, w, 0.5) for u in used)]
        # 濾完的字數明顯比原本多才採用：少於原本代表這個尺度反而漏讀
        # （DPI 對偵測的影響是非單調的，見模組 docstring）。
        if fine and len("".join(w[4] for w in fine)) >= \
                len("".join(w[4] for w in rest)) * 0.9:
            rest = fine
    return pdf_ocr.page_text(rest)


# 信心低於這個門檻的欄位才值得重讀。讀對的欄位信心大多在 0.95 以上，
# 讀錯的（例如 `1:1` 被讀成 `11`）落在 0.9 上下——0.95 能單獨挑出那一格，
# 不會連著已經讀對的欄位一起重讀。
_LOW_CONF_THRESHOLD = 0.95


def _rescue_low_conf_cells(path: str, page_no: int, tb: dict[str, Any]) -> None:
    """信心低於門檻的欄位，單獨用它自己的矩形重讀一次。

    ★ 為什麼只挑信心低的，不是每個欄位都重讀
      例如 SCALE 讀成 `11`：定位那一輪整條 band 壓成 1450px，
      `1:1` 的冒號被吃掉、兩個 1 黏成一團；只給那一格算 DPI 重讀（同一塊只有
      十幾 pt 寬，反推出來是 300dpi 上限），讀回 `1:1`。但這件事不能對每一格
      都做——已經讀對的欄位（信心已經很高）重讀一次是純風險沒有收益，
      `_rescue_faint_numbers()` 的「一格一格重讀很貴，只在有理由懷疑時才做」
      同一個原則用在這裡。

    找不到、重讀結果是空的，或比原本更短，就維持原值 —— 寧可留著可疑的舊值，
    不拿另一次同樣不可靠的猜測去蓋掉它。
    """
    import fitz

    conf = tb.get("_conf") or {}
    cells = tb.get("_cells") or {}
    for key, score in list(conf.items()):
        if score >= _LOW_CONF_THRESHOLD or key not in cells or key not in tb:
            continue
        x0, y0, x1, y1 = cells[key]
        clip = fitz.Rect(x0 - 2, y0 - 2, x1 + 2, y1 + 2)
        try:
            fine = _ocr_block(path, page_no, clip, BLOCK_PX)
        except Exception as e:  # noqa: BLE001 -- 重讀失敗就保留原值
            print(f"    ⚠ 低信心欄位重讀失敗（{key}）：{type(e).__name__}: {e}")
            continue
        if not fine:
            continue
        text = _txt(fine)
        if len(text) < len(tb[key]) * 0.9:
            continue
        tb[key] = text
        conf[key] = min(w[5] for w in fine)


def _find_header_words(ws: list[Word], want: set[str]) -> list[Word] | None:
    """`want` 裡的字全部湊到同一列，回那一列的字框；湊不齊就回 None。

    ★ 為什麼要求「全部」而不是「湊到幾個就算」
      側邊框格的表頭常是三個分開的 OCR 字框，而圖上別處往往還有含其中一個字
      的其他標籤——單一個字的命中率很高，湊三個字都在同一列才是這個框格
      獨有的訊號。少一個字就不算，寧可漏掉這個框格，不要把不相關的位置當成它。

    ★ 為什麼要求「同一列」，不是「頁面上任何位置都湊到就算」
      三個字各自都可能在頁面上出現好幾次（表頭用的多半是常見詞）。
      用 y 相差在一個字高以內分組，確保挑到的是**同一次印刷**的那三個字，
      不是隨機湊出來的組合。
    """
    cands = [w for w in ws if w[4].strip().upper() in want]
    if len(cands) < len(want):
        return None
    rows: list[list[Word]] = []
    for w in sorted(cands, key=_cy):
        h = max(w[3] - w[1], 4.0)
        if rows and abs(_cy(w) - _cy(rows[-1][-1])) <= h * 0.8:
            rows[-1].append(w)
        else:
            rows.append([w])
    best = max(rows, key=lambda r: len({x[4].strip().upper() for x in r}),
              default=[])
    return best if len({x[4].strip().upper() for x in best}) >= len(want) else None


def _extract_frame(path: str, pg: Page, frame: Any,
                   auth: str | None) -> Drawing:
    """客戶圖框那條路。表格先讀，標題欄再讀（表格佔掉的字不能再當標籤）。"""
    pool = _merge(list(_page_text_words(path, pg.page_no)), list(pg.words),
                  self_check=False, overlap=0.5)
    # 這一頁被判定「文字層夠用」而略過 OCR（`find_page()` 的 anchor >= 4
    # 捷徑），但這個圖框自己知道圖上有一部分內容是點陣圖——材料表、公司
    # logo 都不在文字層裡，見 `Frame.ocr_boost` 的說明。這裡把同一塊 band
    # 補 OCR 一次，跟預設路徑「文字層 + OCR 取聯集」是同一個原則，只是
    # 一般路徑會自己判斷要不要 OCR，這裡是圖框自己說「我這裡一定要」。
    if frame.ocr_boost and pg.method != "ocr" and pdf_ocr.available():
        boost = _ocr_block(path, pg.page_no, pg.clip, LOCATE_PX)
        pool = _merge(pool, boost, self_check=False, overlap=0.5)
    # 少數圖框的標籤跟值黏合的方式跟其他幾種不一樣（有的黏成一個字框、
    # 有的切成三個字框），要先重新切一次才能讓下面的
    # `_label_hits_all()` 比對到——見 drawing_frames.Frame.preprocess 的說明。
    # 這一步只影響設了 preprocess 的那個圖框，其餘沿用原樣的 pool。
    if frame.preprocess is not None:
        pool = frame.preprocess(pool)

    rows: list[BomRow] = []
    rects: list[Any] = []
    for tbl in frame.tables:
        got = _frame_table(path, pg, tbl, pool)
        if got is None:
            print(f"    ⚠ {frame.name}: 表格 {tbl.name} 找不到表頭或列，略過")
            continue
        rect, trows = got
        rects.append(rect)
        rows.extend(trows)

    # ★ 標題欄的標籤要先把表格裡的字排除掉。`DESCRIPTION` / `UNIT` / `NO` /
    #   `LENGTH` / `COLOR` 這些字**同一張圖上兩張表都有**，不排除的話材料表的
    #   表頭會被當成標題欄的標籤，兩張表同時錯（見 drawing_frames 模組說明）。
    #
    # ★ 這裡要用**整頁**的 pool，不能只留 pg.clip（下方 band）那一段
    #   pool 本身已經是整頁文字層（`_page_text_words` 不帶 clip）跟 band OCR
    #   的聯集，這一行原本又把它濾回 pg.clip，等於把整頁那一半白白丟掉。
    #   後果不是「少幾個字」——是圖上除了兩張表以外的所有東西（公司名稱、
    #   保密聲明、技術要求 1~3 條、接腳圖上的文字說明）全部不見，而且
    #   `title_raw` 不會報錯，只是那些內容從來沒進去過。
    outside = [w for w in pool if not any(_in_rect(w, r) for r in rects)]
    label_table = tuple((k, re.compile(p)) for k, p in frame.title_labels)
    cands = _label_hits_all(outside, label_table)

    # 標題欄那一塊自己再 OCR 一次（與預設路徑的第 3 段同一件事）：定位那一輪
    # 是整條 band 壓成 1450px，只給標題欄那 190~420pt 就是兩三倍的每字像素數。
    # 範圍取「已經認出來的標籤」的 x 範圍，不是 outside 的 —— 後者在中文圖框
    # 會從最左邊的保密聲明一路到最右邊，反推出來只有 91 DPI，比不做更糟。
    if cands:
        lo = min(w[0] for lst in cands.values() for w in lst)
        hi = max(w[2] for lst in cands.values() for w in lst)
        fine = _refine(path, pg, lo - 12, hi + 12)
        # 這一輪是全新一批 OCR 字框，還沒被 frame.preprocess 處理過——
        # 沒有這一步的話，重新切開的黏合標籤與已經濾掉的雜訊（例如
        # 比例尺刻度）會在這裡原樣復活，跟上面第一輪的處理各做各的。
        if frame.preprocess is not None:
            fine = frame.preprocess(fine)
        if fine:
            outside = _merge(outside, fine, self_check=False)
            outside = [w for w in outside
                       if not any(_in_rect(w, r) for r in rects)]
            cands = _label_hits_all(outside, label_table) or cands
    # ★ 算標題欄下界時只看**每個標籤最下面那個候選**，也就是 title_block_multi
    #   會先試的那一個。用第一個（文件順序）會被圖上別處的同名字帶偏：
    #   圖的上半另有一個客戶簽核用的 `APPROVED BY` 時，`_title_bottom()` 從它
    #   開始沿格線往下走，在標題欄上方的大空隙就停 ——
    #   於是真正的標題欄**整塊落在下界外面**，SCALE / DESCRIPTION /
    #   CUSTOMER PART NO. 三格同時變成空的，而 PART NO. 那幾格因為值在標籤
    #   右邊同一列所以還是對的。壞一半、對一半，最難發現的那種。
    labels = {k: max(v, key=lambda w: (_cy(w), w[0])) for k, v in cands.items()}
    # `_title_bottom()` 的格線節奏只對密集 CAD 格線的圖框可信，見
    # `Frame.title_bottom_from_grid` 的說明——沒有那種格線的圖框
    # 直接退回頁底，不要拿格線節奏去猜。
    band_bottom = (_title_bottom(pg, labels) if frame.title_bottom_from_grid
                  else pg.rect.y1)
    tb = (title_block_multi(outside, cands, band_bottom,
                            list(_hspans(path, pg.page_no)))
          if cands else {"_conf": {}})

    # 出圖公司：從**整頁**（含表格裡的字，例如某些圖框的公司名就印在材料表
    # 附近）找，不只 outside——company 是落款而不是標題欄的一格，不必先扣掉
    # 表格佔用的範圍。找不到就是 None，見 drawing_frames.extract_company()。
    company = drawing_frames.extract_company(pool, frame.name)
    if company:
        tb["company"] = company

    # 標題欄表格裡沒有欄位名稱的那幾格（某些客戶圖框的保密聲明就在這裡）。
    # 只收**框線內**的字 —— 表格外的東西（技術要求、公差表、接腳圖說明、
    # 中文圖框底部那段著作權聲明）不在使用者要的兩張表裡，不收。
    notes = _title_notes(outside, _title_box(labels, list(_vspans(path, pg.page_no))),
                         tb.get("_used") or set(), path, pg.page_no)
    if notes:
        tb["note"] = notes

    # 兩張表、標題欄以外，圖上獨立存在的第三個框格（由
    # drawing_frames.SIDE_BOXES 宣告）。做法跟保密聲明相同：找到表頭 -> 用框線量出
    # 這一格 -> 收裡面沒被認領的字，只是表頭本身是三個分開的字、不在
    # title_labels 裡，所以要另外找。
    side = drawing_frames.SIDE_BOXES.get(frame.name)
    if side:
        want, side_key, _ = side
        # 用整頁的 pool 找表頭，不是 outside——這個框格緊挨著材料表，表頭
        # 的其中一個字落在材料表矩形加了 14pt padding 的範圍裡，
        # 已經被 outside 濾掉了（見 table_rect() 的 padding 說明）。搜表頭
        # 本身不受這個限制：三個字要湊在同一列才算數，不會誤觸材料表內容。
        hdr = _find_header_words(pool, set(want))
        if hdr:
            fake = {"_side": (min(w[0] for w in hdr), min(w[1] for w in hdr),
                              max(w[2] for w in hdr), max(w[3] for w in hdr),
                              "", 1.0)}
            side_box = _title_box(fake, list(_vspans(path, pg.page_no)))
            # 這裡故意用 pool 不是 outside：`outside` 已經把材料表矩形（連
            # 14pt padding）濾掉了，而這個框格的表頭第一個字剛好落在那圈
            # padding 裡——用 outside 會漏掉表頭本身，值只剩後兩個字。
            # 這個框格本來就是獨立的，不需要 outside 那層材料表
            # 排除規則。
            side_text = _title_notes(pool, side_box, tb.get("_used") or set(),
                                     path, pg.page_no)
            if side_text:
                tb[side_key] = side_text

    _rescue_low_conf_cells(path, pg.page_no, tb)

    raw_pn = tb.get(frame.part_no_key) or tb.get("part_no") or ""
    pn, ok = normalize_pn(raw_pn, auth)
    sheet_no, sheet_of = _parse_sheet_cell(tb.get("sheet") or "")

    conf = tb.get("_conf") or {}
    all_conf = list(conf.values()) + [r.conf for r in rows]
    # 客戶圖框的列一律是圖上的格線切的（row_bands 沒有「按字高分群」那條
    # 退路），所以只有 grid 這一種，不必像預設路徑那樣分兩種回報。
    d = Drawing(
        page_no=pg.page_no,
        method=_method(pg, used_grid=bool(rows)),
        title=tb.get("title") or tb.get("description"),
        part_no=pn,
        part_no_ok=ok,
        rev=_clean_rev(tb.get("rev")),
        size=tb.get("size"),
        scale=tb.get("scale"),
        sheet_no=sheet_no,
        sheet_of=sheet_of,
        pd_file_no=None,
        ocr_conf=round(min(all_conf), 4) if all_conf else None,
        title_raw=pdf_ocr.page_text(outside),
        items_raw=pdf_ocr.page_text(
            [w for w in pool if any(_in_rect(w, r) for r in rects)]),
        items=rows,
        # 底線開頭的鍵是「只用來當格子邊界」的標籤（簽核欄那一疊），不回報：
        # 它們在圖上是空格，值會是隔壁格漏過來的東西。
        title_fields={k: v for k, v in tb.items()
                      if not k.startswith("_") and isinstance(v, str) and v.strip()},
        frame=frame.name,
    )
    m = _DATE.search(tb.get("date") or "") or _DATE.search(d.title_raw)
    d.dwg_date = m.group(1) if m else None
    return d


def _clean_rev(v: str | None) -> str | None:
    """REV. 那一格偶爾會連旁邊的日期一起吃進來，只留版次本身。

    版次是 `01` 這種兩位數，而 OCR 把它讀成 `O 1`（字距被當成分隔）——
    所以先把空白收掉再做 O->0 還原，不然 `01` 會以原樣 `O 1` 入庫。
    """
    if not v:
        return None
    rev = re.compile(r"^[A-Z]?\d{1,3}[A-Z]?$|^[A-Z]$")
    # 先試整串（`O 1` 這種被字距拆開的），再試第一個 token。
    # 需要試第一個 token，是因為隔壁的 SHEET 偶爾會被 OCR 連著值讀成一整塊
    # （`SH,T NO. 1OF 1`），認不出是標籤，於是整塊落進 REV. 那一格，
    # 版次就變成 `01 SH,T NO. 1OF1`。
    for cand in (re.sub(r"\s+", "", v.strip().upper()), v.strip().upper().split()[0]):
        # 版次是 01/02/A/A1 這種短碼，所以「整串 O 換成 0 之後是純數字」時就換掉
        # ——`O1` 只可能是 `01`。這條不能拿去套自由文字（`NONE` 會變 `N0NE`）。
        if cand.replace("O", "0").isdigit():
            cand = cand.replace("O", "0")
        m = rev.match(cand)
        if m:
            return m.group()
    return v.strip()[:16]


# 自家的文件編號固定是 `<前綴>-` + 一個字母 + 數字（示意：`ABC-S00001`）。連字號被
# OCR 吃掉是最常見的錯誤之一，而這個格式是公司內部規定、不是猜的，
# 所以這一個欄位補得回來。其他自由文字（TITLE、描述）不做這件事 ——
# 那等於編造內容。展示版：前綴以 `ABC` 示意，實際前綴未收錄。
_PD_FILE = re.compile(r"ABC-?([A-Z])[-\s]?(\d{3,})", re.I)


def _clean_pd_file(v: str | None) -> str | None:
    """把 `ABC-S00001` 從那一格裡撈出來。

    用 search 而不是 match（不要求整格就是編號）：這一格旁邊常常還有沒有標籤的
    格子被一起吃進來，抽到的會是 `ABC-S00001 CUSTOMER /` 這種。編號本身的格式是
    公司規定的，撈得出來就只留編號；撈不出來才原樣保留，不自己編。
    """
    if not v:
        return None
    m = _PD_FILE.search(re.sub(r"\s+", "", v.strip().upper()))
    return f"ABC-{m.group(1)}{m.group(2)}" if m else v.strip()
