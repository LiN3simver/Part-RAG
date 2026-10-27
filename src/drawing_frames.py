"""客製圖框的特例樣式：偵測 + 表格逐格讀值。

`drawing.py` 的欄界規則是對預設圖框量出來的（語料裡絕大多數圖面用的那一套）：
標題欄在右下、材料表在左下、材料表由下往上編號、表頭是 `ITEM / DESCRIPTION /
Q'TY`。客戶或供應商自己的圖框不吃這一套，而且**不是壞在 OCR，是壞在版面假設**。
常見的差異：

    - 表頭字樣不同：中文圖框是 `序号 / 名称与规格 / 数量 / 备注`，預設那一套的
      表頭字樣一個都沒有，於是 `bom_rows()` 直接回空 list —— 材料表整張不見。
    - 列的方向相反：由上往下編號。
    - 欄名不同（`PART NUMBER` vs `PART NO.`、多一個 `UNIT` 欄），或圖上另有
      第二張表，落在預設的下方 band 之外，現行流程看不到它。
    - 單一零件規格書：**沒有材料表**（這張圖就是那個零件本身），而且標題欄的
      標籤與值被 OCR 黏成同一個字框、或被文字層切成好幾個字框，舊流程連候選
      都湊不出來，標題欄整格是空的。

所以這個模組加的是**版面詞彙與版面方向**，不是新的 OCR 技巧。判斷「是哪一種
圖框」看的是圖上印的字（如 `物料明细表`），不是檔名 —— 同一家客戶下次送新圖，
抽取照樣成立；用檔名寫死的話下次會安靜地抽不到。

展示版只保留兩個示意 frame（中文 BOM 表頭、無材料表的單一零件規格書）；
實際的圖框 profile 是依各家客戶圖面逐一整理的，不包含在展示版中。

★ 沒有材料表的規格書，是不是不該叫「圖框」
  它們的標題欄一樣有各自的版面詞彙、需要各自的規則才抽得到，跟有材料表的
  圖框是同一類問題（版面假設錯了，不是 OCR 錯了）——只是這種圖本身就是「單一
  零件規格書」，不是「組裝圖」，材料表原本就不存在，不是漏抽。`Frame.tables`
  留空 tuple 就是這個意思：這個圖框沒有材料表要抽，跟「有材料表但抽不到」
  是兩種不同的空。

★ 為什麼另開一個模組而不是把字樣加進 `drawing._TITLE_LABELS`
  `DESCRIPTION` / `UNIT` / `NO` / `LENGTH` / `COLOR` 這些字在同一張圖上
  **兩張表都有**：`DESCRIPTION` 既是材料表的表頭、也是標題欄的欄位名。
  加進共用字典的話，`_label_hits()` 的「取最右下角那一個」會讓材料表的表頭
  去當標題欄的標籤，而症狀是**標題欄少一欄、材料表多一欄**，兩張表同時錯。
  預設圖框佔絕大多數，不該為了少數客製圖框承擔這個風險。

★ 欄界用「兩個表頭之間**最靠近中點的那一條垂直格線**」，不是中點本身
  中點會切錯：中文圖框的 `名称与规格` 是置中標題，而規格文字從欄界一開始就
  有字 —— `序号` 與它的中點會落在真正欄界右邊幾 pt。差距不大，但貼著欄界的
  那一格字框中心就會有半個字掉進序號欄。格線是 CAD 畫的真值，中點只用來
  決定「該挑哪一條」。沒有格線時才退回中點（掃描圖）。

★ 列一律從格線來，而且**方向要跟著圖走**
  預設圖框的材料表第 1 列在表頭正上方（由下往上），客製圖框常常相反。
  方向猜錯的症狀不是抽不到，是**編號與內容對不起來**（第 1 列的規格掛到
  第 3 列）—— 比抽不到危險得多，所以寫在 profile 裡，不用啟發式去猜。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

Word = tuple[float, float, float, float, str, float]

@dataclass(frozen=True)
class Column:
    """表格的一欄。

    role 決定這一欄的值進 `drawing.BomRow` 的哪個欄位，可以是
    item / description / qty / unit / remark / color / length / kind；
    `extra` 是「認得出來但不單獨存」，只進 items_raw。
    """

    key: str
    pattern: str          # 比對前已經把空白收掉（OCR 常常把欄名的空白吃掉）
    role: str = "extra"

    @property
    def rx(self) -> re.Pattern[str]:
        return re.compile(self.pattern)


@dataclass(frozen=True)
class Table:
    name: str             # 進 drawing_items.table_tag
    columns: tuple[Column, ...]
    # 列在表頭的哪一邊。False = 表頭在下、列往上長（預設圖框的材料表）。
    rows_below_header: bool = True
    # 這張表要不要用它自己的矩形再 OCR 一次。中文圖框的序號與數量是
    # CAD 向量（文字層裡沒有數字），整條 band 壓成 1450px 等於 120 DPI，
    # 那些孤立的一位數會整批讀不到 —— 只給表格那一塊就是 300 DPI。
    ocr_refine: bool = True
    # 至少要認出這幾個欄名才算「找到這張表」。少於這個數就當沒有，
    # 不要拿半套欄界去切 —— 切錯的欄位不會報錯，只是值放到別欄去。
    min_headers: int = 2


@dataclass(frozen=True)
class Frame:
    name: str
    # 圖上出現這些字樣（空白收掉、轉大寫後比對）才算命中，全部都要有。
    detect: tuple[str, ...]
    title_labels: tuple[tuple[str, str], ...] = ()
    tables: tuple[Table, ...] = ()
    # 標題欄裡哪一格當 dwg_part_no。中文圖框沒有 `PART NO.`，圖號寫在
    # `DWG NO.`；不指定就沿用 drawing.py 的 part_no。
    part_no_key: str = "part_no"
    # 這個圖框的標籤跟值黏在一起或斷開的方式，跟預設圖框不一樣時才需要——
    # 在交給 `_label_hits_all()` 逐字比對之前，先把整頁的字重新切一次。
    # 預設 None（不做任何事）。見 `_split_glued()`（OCR 把「標籤:值」黏成
    # 一個框）與 `_merge_customer_pn()`（文字層把一個標籤切成三個字）。
    preprocess: Any = None
    # 標題欄格子的下界要不要用 `drawing._title_bottom()` 的格線節奏去猜。
    # 那支函式假設標題欄是密集、均勻的 CAD 格線（見它的 docstring）。鬆散
    # 排版的表單沒有那種節奏，猜出來的下界可能比標籤自己的 y 還高，於是
    # 候選下界比上界還小，`_cell_words()` 篩出空集合，整格消失。設 False
    # 就直接退回頁底（`hrules` 是空的時 `_title_bottom()` 本來就會這樣做，
    # 這裡只是把「沒有格線可信」提前講清楚，不必等它猜錯）。
    title_bottom_from_grid: bool = True
    # 這個圖框的頁面**已經**判斷用文字層就夠（`find_page()` 的 anchor >= 4
    # 捷徑），但圖上有一部分內容其實是點陣圖、不在文字層裡——要不要因此
    # 強制多補一次 OCR。預設 False（文字層夠用時不必多花那次 OCR）。
    #
    # ★ 為什麼需要這個旗標
    #   有些規格書的標題欄是真文字（走文字層捷徑合理），但材料表、公司 logo
    #   是供應商那邊以**點陣圖**插進來的。文字層路徑從頭到尾不會生出這批字，
    #   材料表就永遠是空的、公司永遠是 None，而且**不會報錯**，因為別的欄位
    #   都抽到了，看起來像是抽取成功了一半。
    ocr_boost: bool = False


# --------------------------------------------------------------------------
# 示意圖框一：中文 BOM 表頭
# --------------------------------------------------------------------------
# ★ 標籤的 regex 一律允許結尾的冒號與全形冒號：有些圖框每一個欄名都印著
#   冒號（`PART NO.:`），而 `_label_hits()` 只收空白、不收冒號。
_CN = Frame(
    name="cn_bom",
    detect=("物料明细表",),
    title_labels=(
        ("title", r"^TITLE[:：]?$"),
        # 中文圖框把圖號寫在 DWG NO.，沒有 PART NO. 這一格。
        ("part_no", r"^DWGN[O0]\.?[:：]?$"),
        ("material", r"^MATERIAL[:：]?$"),
        ("p_code", r"^P\.?C[O0]DE[:：]?$"),
        ("scale", r"^SCALE[:：]?$"),
        ("rev", r"^REV\.?[:：]?$"),
        ("size", r"^SIZE[:：]?$"),
        ("sheet", r"^SHEET[:：]?$"),
        # ★ 底線開頭 = 只當格子邊界，不回報值
        #   這幾格在圖上是**空的**（簽核欄還沒簽、PROJECTION 那格是符號不是字），
        #   但它們是隔壁那幾格的邊界，不認出來的話 TITLE / DWG NO. 會把左半邊
        #   一起吃進來。回報它們的「值」則是把隔壁漏過來的東西當成內容 ——
        #   例如 DATE 那一格會抽到整串公司名稱。所以：要認，但不要存。
        #   APPRVALS 不是打錯，是 OCR 把 APPROVALS 的一個 O 吃掉了。
        ("_approvals", r"^APPR[O0]?VALS?[:：]?$"),
        ("_date", r"^DATE[:：]?$"),
        ("_design", r"^DESIGN[:：]?$"),
        ("_approval", r"^APPR[O0]?VAL[:：]?$"),
        ("_check", r"^CHECK[:：]?$"),
        ("_projection", r"^PR[O0]JECTI[O0]N[:：]?$"),
    ),
    tables=(
        Table(
            name="BOM",
            columns=(
                Column("item", r"^序号$", "item"),
                Column("description", r"^名称与规格$", "description"),
                Column("qty", r"^数量$", "qty"),
                Column("remark", r"^备注$", "remark"),
            ),
            rows_below_header=True,
        ),
    ),
)


# --------------------------------------------------------------------------
# 示意圖框二：單一零件規格書（沒有材料表），OCR 把「標籤:值」黏成一個字框
# --------------------------------------------------------------------------
# `ITEM NO.:ABC-0001`、`DES.:<品名>` 這種格子在某些 OCR 結果裡是**同一個
# 字框**，不是「標籤」+「值」兩個框——跟預設圖框（標籤與值分開，靠幾何位置
# 配對）正好相反。`_label_hits_all()` 逐字比對一個字框的全部內容，於是這幾格
# 連候選都進不了：不是抽錯，是從一開始就沒被當成「這是一個標籤」。
#
# ★ 用正則切，不是切在固定字元位置
#   `ITEM NO.:` 與 `DES.:` 長度不一樣，而且同一批圖裡偶爾多一個空白
#   （OCR 的字距忽大忽小）。用「標籤本身的規則」去切，兩種寫法都切得對；
#   切在固定第 N 個字元的話，`DES.:` 短一截，切點會落進值的第一個字裡。
_GLUED_RE = re.compile(r"^\s*(ITEMN[O0]\.?\s*:|DES\.?\s*:)\s*(.+)$", re.I)


def _split_glued(words: list[Word]) -> list[Word]:
    """把「標籤:值」黏成一個字框的那幾格，按標籤／值的字元數比例切成兩格。

    切點只是「字元數比例」，不是真的量出來的欄界——這兩格接下來只會被當成
    普通的標籤字框去配對值（`_label_hits_all` 逐字比對、`title_block_multi`
    用幾何位置找隔壁的值），不會再切欄，所以比例切估得不夠準也不影響結果：
    只要切出來的標籤那一半仍然完整比對得上正則就夠了。
    """
    out: list[Word] = []
    for w in words:
        m = _GLUED_RE.match(w[4])
        if not m:
            out.append(w)
            continue
        label, val = m.group(1), m.group(2)
        total = len(w[4]) or 1
        cut = w[0] + (w[2] - w[0]) * (len(label) / total)
        out.append((w[0], w[1], cut, w[3], label, w[5]))
        out.append((cut, w[1], w[2], w[3], val, w[5]))
    return out


# ★ 頁面最下緣若印著一條 `10cm` 比例尺，尺上的刻度數字長得跟真正的標題欄
#   欄位值一模一樣——都是孤零零的一兩位數字。比例尺那一列跟任何欄位之間
#   沒有格線或別的標籤可以擋，`_cell_words()` 找不到下界就一路開到頁底，
#   `SHEET` 那格就會多吃到一個刻度數字——不是抽錯了，是多吃了一個跟欄位值
#   長得一樣的雜訊。判斷「這一列是比例尺」看的是 `10cm` 這個字，不是位置：
#   找到它才知道那一列在哪，直接整列丟棄。
_RULER_RE = re.compile(r"^10CM$", re.I)


def _drop_ruler(words: list[Word]) -> list[Word]:
    hit = [w for w in words if _RULER_RE.match(w[4].strip())]
    if not hit:
        return words
    y0 = min(w[1] for w in hit) - 3
    y1 = max(w[3] for w in hit) + 3
    return [w for w in words if not (y0 <= (w[1] + w[3]) / 2 <= y1)]


def _spec_preprocess(words: list[Word]) -> list[Word]:
    return _split_glued(_drop_ruler(words))


_SPEC_SHEET = Frame(
    name="spec_sheet",
    # 用規格書表尾固定印的欄名當偵測字樣，不用公司 logo：公司名稱只是落款，
    # 理由跟 `extract_company()` 那段「找到再正規化」是同一個原則——落款認不
    # 出來就該是 None，不能因為判斷圖框樣式而先假設公司對了。
    detect=("PRODUCTSPECIFICATION",),
    title_labels=(
        ("part_no", r"^ITEMN[O0]\.?:?$"),
        ("description", r"^DES\.?:?$"),
        ("scale", r"^SCALE[:：]?$"),
        ("unit", r"^UNIT[:：]?$"),
        # OCR 有時把 `DRAWING NO.` 讀成兩個分開的字框（`DRAWING`／`NO.`），
        # 只認得到 `DRAWING` 那半——半個標籤仍然值得認，理由跟 `_CN` 的
        # `_approvals` 那組同一個原則：認得到就能當隔壁欄位的邊界，認不到的話
        # 相鄰兩格會把這一整行吃進去。
        ("_drawing_no", r"^DRAWING$"),
        ("sheet", r"^SHEET[:：]?$"),
    ),
    # 這張圖本身就是那個零件，不是一份組裝圖，材料表原本就不存在。
    #
    # ★ 就算圖上印著材料表表頭，若整張表是點陣圖插進來、**沒有一條真的 CAD
    #   格線**，也刻意不做成 `Table`：`column_bounds()`／`row_bands()` 全靠
    #   向量線定欄界跟列界，硬套下去的結果是每一個字都對、但格子全錯開，看起來
    #   卻像一張正常的表——這比整格空白危險得多。材料內容改走 `title_raw`
    #   （見 ingest_drawing.build_search_text 對沒有材料表的圖框額外收
    #   title_raw 的那段），以完整、不切欄的原文形式查得到、也給模型看得到。
    tables=(),
    preprocess=_spec_preprocess,
)


# --------------------------------------------------------------------------
# 另一種前處理：文字層把一個標籤切成三個字框
# --------------------------------------------------------------------------
# `CUSTOMER PART NO.:` 在某些圖的文字層裡排版切成三段（`CUSTOMER` / `PART` /
# `NO.:`），跟 `_split_glued()` 的問題正好相反：這裡不是黏太多，是切太細——
# 單獨看每一段都湊不出完整的標籤字樣，`_label_hits_all()` 一樣連候選都進不了。
# 遇到這種版式的 frame 把它掛在 `Frame.preprocess` 上。
#
# ★ 合併規則刻意寫死這三個字，不是「同一列、間隔小於某個值就合併」
#   材料的自由文字描述裡也有很多同一列、間隔很小的英文詞（`RJ45 PLUG -
#   Generic Parts`），通用的「鄰接就合併」規則會把它們也黏成一整條長字串，
#   而且黏錯了不會報錯，跟這個模組其他地方的教訓一樣：抓太寬比抓太窄危險。
_CPN_SEQ = ("CUSTOMER", "PART", "NO.:")


def _merge_customer_pn(words: list[Word]) -> list[Word]:
    """把 `CUSTOMER` / `PART` / `NO.:` 三個字框合併成一個 `CUSTOMER PART NO.:`。

    找法：從一個文字是 `CUSTOMER` 的字框開始，往右找同一列（y 中心差 <= 2pt）
    且緊接在後面（間隔 <= 12pt，字距容差）的 `PART`，再找同樣條件的 `NO.:`。
    三個都湊到才合併；湊不到就照原樣放回去，不強行湊。
    """
    def cy(w: Word) -> float:
        return (w[1] + w[3]) / 2

    used: set[int] = set()
    by_text = [w[4].strip().upper() for w in words]
    out: list[Word] = []
    for i, w in enumerate(words):
        if i in used or by_text[i] != _CPN_SEQ[0]:
            continue
        seq = [w]
        ptr = w
        ok = True
        for want in _CPN_SEQ[1:]:
            nxt = next((j for j, t in enumerate(by_text)
                        if j not in used and t == want
                        and abs(cy(words[j]) - cy(w)) <= 2
                        and -2 <= words[j][0] - ptr[2] <= 12), None)
            if nxt is None:
                ok = False
                break
            seq.append(words[nxt])
            used.add(nxt)
            ptr = words[nxt]
        if ok:
            used.add(i)
            out.append((seq[0][0], min(x[1] for x in seq), seq[-1][2],
                       max(x[3] for x in seq), " ".join(_CPN_SEQ),
                       min(x[5] for x in seq)))
    out.extend(w for i, w in enumerate(words) if i not in used)
    return out


FRAMES: tuple[Frame, ...] = (
    _CN, _SPEC_SHEET,
)

# 沒有材料表的圖框（這張圖就是那個零件本身，或材料是自由文字/點陣圖，見各
# frame 的說明）。`ingest_drawing.build_search_text()` 用它決定要不要把
# `title_raw` 整段收進檢索文字——這幾種圖框沒有 `drawing_items` 可以扛
# 「圖上還寫了什麼」，那些內容不進 search_text 就等於沒抽到。
NO_BOM_FRAMES = frozenset(f.name for f in FRAMES if not f.tables)


# --------------------------------------------------------------------------
# 出圖公司（logo / 保密聲明落款）
# --------------------------------------------------------------------------
# 每個 frame 認的公司名稱怎麼找 + 正規化成什麼字串。
#
# ★ 為什麼是「找到再正規化」，不是直接把公司名寫死在 Frame 上
#   `frame.detect` 用的字樣（`物料明细表`…）認的是**版面樣式**，不是公司本身
#   —— 換一家客戶、剛好用同一套版面詞彙（尤其中文的「物料明细表」四個字不算
#   獨特），也會被判成同一個 frame。所以公司名稱還是要**從這份圖自己的字裡
#   找出來**，找不到就是 None，不能因為 frame 對了就假設公司也對了。
#
# ★ 正規化：圖上常常只印 logo 的裸字，而保密聲明（或同一家公司的其他圖）印著
#   全名；把裸字還原成那家公司自己用的全名，同一家公司才不會被算成兩家。
#
# 實際的規則表依各客戶圖面整理，不包含在展示版中；以下為示意。
_COMPANY_RULES: dict[str, tuple[re.Pattern[str], str | None]] = {
    # 中文公司全名本身就是要存的值，不必正規化成別的字串（第二個元素
    # None 代表「比對到的字串本身就是答案」）。
    "cn_bom": (re.compile(r"[一-鿿]{2,20}?有限公司"), None),
    # 圖上印的裸字（logo）正規化成全名。
    "spec_sheet": (re.compile(r"EXAMPLECO", re.I), "EXAMPLE CO., LTD."),
}


# 兩張表、標題欄以外，有些圖上還有獨立的小框格（例如緊挨在材料表右邊、自己
# 有邊框、但沒有任何欄位認得它的清單）——跟標題欄的保密聲明是同一種情況
# （有邊框、沒有標籤能命中它），只是位置不在標題欄裡。表頭若是幾個分開的
# OCR 字框，要全部都在同一列才算數，免得單獨出現的同一個字在別處誤觸。
#
# key -> (表頭要湊到的字, 存進 title_fields 的鍵, 圖上印的欄名)
# 例：{"<frame>": (("PART", "LIST"), "part_list", "PART LIST")}
# 實際條目依客戶圖框整理，不包含在展示版中。
SIDE_BOXES: dict[str, tuple[tuple[str, ...], str, str]] = {}


def extract_company(words: list[Word], frame_name: str) -> str | None:
    """這份圖是哪家公司出的，從圖上實際印出來的字找，不是猜的。

    找不到就回 None（例如規則只認 logo 的裸字，某一份圖萬一連這個都沒印
    出來），寧可留空，不拿「這個 frame 通常是這家公司」去填一個圖上其實沒
    印出來的名字。
    """
    rule = _COMPANY_RULES.get(frame_name)
    if not rule:
        return None
    rx, normalized = rule
    m = rx.search(" ".join(w[4] for w in words))
    if not m:
        return None
    return normalized or m.group()


# title_fields 的鍵 -> 圖上印的欄名。卡片（a2a）與 prompt（llm）共用同一份，
# 不各寫一份 —— 兩邊漂掉的話畫面上寫 `CUSTOMER PART NO.`、送給模型的卻是
# `CUSTOMER_PN`，模型引用時會寫出圖上根本沒有的欄名。
FIELD_LABEL = {
    "company": "公司",
    # 標題欄裡沒有欄位名稱的那一格。有些圖框放的是保密聲明。
    "note": "標題欄其他文字",
    "customer_pn": "CUSTOMER PART NO.",
    "description": "DESCRIPTION",
    "material": "MATERIAL",
    "p_code": "P.CODE",
    "unit": "UNIT",
    "drawn_by": "DRAWN BY",
    "checked_by": "CHECKED BY",
    "approved_by": "APPROVED BY",
    "date": "DATE",
}
# 這幾個鍵已經有自己的資料庫欄位（dwg_title / dwg_part_no / …），上游已經
# 顯示過了，重複列一次只會讓人以為圖上寫了兩次。
FIELD_ALREADY_SHOWN = ("title", "part_no", "rev", "size", "scale", "sheet")


def field_rows(title_fields: dict | None) -> list[tuple[str, str]]:
    """title_fields -> [(圖上的欄名, 值)]，已濾掉重複與空值。"""
    return [(FIELD_LABEL.get(k, k.upper()), v)
            for k, v in (title_fields or {}).items()
            if k not in FIELD_ALREADY_SHOWN and v]


def _flat(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).upper()


def detect(words: list[Word]) -> Frame | None:
    """圖上的字 -> 是哪一種客戶圖框。認不出來就回 None，走 drawing.py 的預設路徑。"""
    blob = _flat(" ".join(w[4] for w in words))
    for f in FRAMES:
        if all(_flat(k) in blob for k in f.detect):
            return f
    return None


# --------------------------------------------------------------------------
# 表頭 -> 欄界
# --------------------------------------------------------------------------
def header_row(words: list[Word], table: Table) -> dict[str, Word] | None:
    """表頭那一列：**同一個 y 帶上湊出最多欄名的那一列**。

    ★ 不能用「最上面那個」或「最下面那個」去挑每一個欄名
      這些字在同一張圖上會出現不只一次，而且方向不固定：有的圖標題欄的
      `DESCRIPTION` 在材料表表頭**上面**（取最上面就取錯），有的在**下面**
      （取最下面就取錯），所以任何「取某一端」的規則都會有一半是錯的。
      表頭真正的特徵不是位置，是**它們排成一列**：一整列同時湊出 4~6 個欄名，
      而別處的同名字孤零零只有一個。挑欄名最多的那一列，兩種都對。

    ★ 挑錯的代價：欄界會從標題欄那個字框算起，於是整張表的欄位錯開一格 ——
      規格跑到數量欄、數量欄變空的。抽不到會被 --report 標出來，錯開不會。
    """
    cands: list[tuple[str, Word]] = []
    for w in words:
        t = _flat(w[4])
        for col in table.columns:
            if col.rx.match(t):
                cands.append((col.key, w))
                break
    if not cands:
        return None

    rows: list[list[tuple[str, Word]]] = []
    for key, w in sorted(cands, key=lambda kw: (kw[1][1] + kw[1][3]) / 2):
        cy = (w[1] + w[3]) / 2
        h = max(w[3] - w[1], 4.0)
        if rows:
            ref = rows[-1][0][1]
            if abs(cy - (ref[1] + ref[3]) / 2) <= h * 0.8:
                rows[-1].append((key, w))
                continue
        rows.append([(key, w)])

    def score(row: list[tuple[str, Word]]) -> tuple[int, float]:
        keys = {k for k, _ in row}
        span = max(w[2] for _, w in row) - min(w[0] for _, w in row)
        return (len(keys), span)

    best = max(rows, key=score)
    if len({k for k, _ in best}) < table.min_headers:
        return None
    out: dict[str, Word] = {}
    for key, w in best:
        # 同一列裡同一個欄名出現兩次（OCR 把框切開）就取靠左的那個。
        if key not in out or w[0] < out[key][0]:
            out[key] = w
    return out


def _nearest(x: float, xs: list[float], tol: float) -> float | None:
    cands = [v for v in xs if abs(v - x) <= tol]
    return min(cands, key=lambda v: abs(v - x)) if cands else None


def column_bounds(table: Table, hits: dict[str, Word],
                  vrules: list[float]) -> dict[str, tuple[float, float]]:
    """每一欄的 (左界, 右界)。相鄰兩個欄名之間取最靠近中點的那條垂直格線。

    最外側兩界（第一欄的左、最後一欄的右）放寬到 ±inf —— 表格外框那條線
    CAD 常常畫在別的圖層或被長度過濾掉，而最外側不需要精準：那個方向
    沒有別的欄可以搶。
    """
    cols = [c for c in table.columns if c.key in hits]
    cols.sort(key=lambda c: hits[c.key][0])
    if not cols:
        return {}

    edges: list[float] = [float("-inf")]
    for a, b in zip(cols, cols[1:]):
        left, right = hits[a.key][2], hits[b.key][0]
        mid = (left + right) / 2
        # 容差取「兩個欄名之間那段距離的一半」：再遠就不是這兩欄之間的線了。
        gap = max(right - left, 1.0)
        edges.append(_nearest(mid, vrules, gap / 2) or mid)
    edges.append(float("inf"))
    return {c.key: (edges[i], edges[i + 1]) for i, c in enumerate(cols)}


# --------------------------------------------------------------------------
# 格線 -> 列
# --------------------------------------------------------------------------
def _median(xs: list[float]) -> float:
    return sorted(xs)[len(xs) // 2]


# 表格最外側那條格線離表頭字框最遠可以差多少（pt）。表頭置中時，外框與
# 表頭字框可以差到數十 pt，80 是寬鬆的上限；超過就當那條線不是這張表的。
OUTER_TOL = 80.0


def grid_box(hits: dict[str, Word],
             vspans: list[tuple[float, float, float]]):
    """這張表的格線外框 (x0, x1, y0, y1)。抓不到就回 None。

    ★ 上下界：**表頭字框範圍內**那幾條垂直格線的長度（取中位數）
      那幾條一定是這張表的欄界。取中位數而不是最小/最大：合併儲存格的內線會
      短一截，而表格外框偶爾會被 CAD 畫長一點，中位數對兩種雜訊都不敏感。

    ★ 左右界：表頭範圍**外面最近的那一條同高格線**
      不能改用「y 範圍跟內側那幾條一樣」去撈 —— 標題欄的外框往往跟材料表
      同高（都畫到圖框底），於是最右邊會撈到標題欄的右框線，表格矩形一路
      吃過整個標題欄，材料表的數量欄因此把整片標題欄當成數量吃進來。
      取「最近的一條」才是欄界的定義。

    ★ 最外側那兩條非取不可（不能拿表頭字框當界）：表頭是置中的，最左欄的
      左框線在表頭字框的左邊 —— 差個十幾 pt 序號欄整欄就會被切掉。
    """
    if not (hits and vspans):
        return None
    hx0 = min(w[0] for w in hits.values())
    hx1 = max(w[2] for w in hits.values())
    hy0 = min(w[1] for w in hits.values())
    hy1 = max(w[3] for w in hits.values())

    inner = [s for s in vspans
             if hx0 - 2 <= s[0] <= hx1 + 2 and s[2] > hy0 and s[1] < hy1]
    if not inner:
        return None
    top = _median([s[1] for s in inner])
    bot = _median([s[2] for s in inner])
    if bot - top < 4:
        return None

    # 「同高」= y 範圍與表格重疊八成以上。門檻不能嚴（合併儲存格的內線短一截），
    # 也不能鬆到把整頁的外框收進來。
    span = bot - top
    near = [s[0] for s in vspans
            if min(bot, s[2]) - max(top, s[1]) >= span * 0.8]
    left = [x for x in near if x <= hx0 + 2]
    right = [x for x in near if x >= hx1 - 2]
    x0 = max(left) if left and hx0 - max(left) <= OUTER_TOL else hx0 - 20
    x1 = min(right) if right and min(right) - hx1 <= OUTER_TOL else hx1 + 20
    return (x0, x1, top, bot)


def row_rules(hspans: list[tuple[float, float, float]], box) -> list[float]:
    """真的橫跨這張表的那幾條列線的 y。

    門檻是「蓋掉表格寬度的一半」，兩邊都要顧：
      要收：有些表的列線只畫過前幾欄（最後一欄放示意圖、沒有列線），
            蓋掉的寬度只比一半多一點。
      要擋：表格旁邊獨立小框格的橫線，只蓋掉表寬的四分之一左右，收進來
            會把材料表第一列切成兩列。

    ★ 幾乎重合的線要先併掉，這一步不能省
      CAD 常常把同一條列線畫兩次（每一對相距不到 1pt）。不併的話
      `row_bands()` 算出來的「典型列高」會是那個不到 1pt 的間距（一半的間距
      都是那些重複線），於是「間距突然變大就停」在第一條真的列界就觸發 ——
      一張十幾列的表只抽到 1 列。門檻 1.5pt：真正的列高遠大於此，不會誤併。
    """
    if box is None:
        return []
    x0, x1 = box[0], box[1]
    width = max(x1 - x0, 1.0)
    ys = sorted({y for y, a, b in hspans
                 if min(x1, b) - max(x0, a) >= width * 0.5})
    out: list[float] = []
    group: list[float] = []
    for y in ys:
        if group and y - group[-1] <= 1.5:
            group.append(y)
            continue
        if group:
            out.append(_median(group))
        group = [y]
    if group:
        out.append(_median(group))
    return out


def row_bands(hrules: list[float], header: Word,
              below: bool, limit: tuple[float, float] | None = None) -> list[tuple[float, float]]:
    """表頭那一列之外的資料列。`drawing._grid_bands()` 的雙向版本。

    兩道門檻與那一支相同，理由也相同（見該函式）：太靠近上一條的線是合併
    儲存格的內線不是列界；間距突然變大就是表格結束了。
    """
    if limit is not None:
        lo, hi = limit
        hrules = [y for y in hrules if lo - 2 <= y <= hi + 2]
    if below:
        lines = sorted(y for y in hrules if y >= header[3] - 2)
    else:
        lines = sorted((y for y in hrules if y <= header[1] + 2), reverse=True)
    if len(lines) < 2:
        return []
    gaps = [abs(lines[i + 1] - lines[i]) for i in range(len(lines) - 1)]
    typical = sorted(gaps)[len(gaps) // 2] or 1.0

    keep = [lines[0]]
    for y in lines[1:]:
        gap = abs(keep[-1] - y)
        if gap > max(typical * 2.5, typical + 8):
            break
        if gap < typical * 0.5:
            continue
        keep.append(y)
    if len(keep) < 2:
        return []
    pairs = [(min(keep[i], keep[i + 1]), max(keep[i], keep[i + 1]))
             for i in range(len(keep) - 1)]
    return pairs if below else list(reversed(pairs))


def table_rect(bounds: dict[str, tuple[float, float]],
               bands: list[tuple[float, float]], header: Word, box=None):
    """這張表自己的矩形：歸格看的範圍，也是 ocr_refine 要 render 的那一塊。

    ★ 有 grid_box 就一定要用它的左右界，不能用 bounds 的有限值
      `bounds` 最外側那兩界刻意是 ±inf（那個方向沒有別的欄可以搶），拿掉
      inf 之後剩下的最小值是**第一欄的右界** —— 用它當矩形左緣等於把第一欄
      整欄切掉：序號與最右邊的备注同時不見（而且不會報錯，就是空的）。
    """
    import fitz

    ys = [v for b in bands for v in b] + [header[1], header[3]]
    if box is not None:
        x0, x1 = box[0], box[1]
    else:
        finite = [v for lo, hi in bounds.values() for v in (lo, hi)
                  if abs(v) != float("inf")]
        x0 = min(finite + [header[0]])
        x1 = max(finite + [header[2]])
    # 左右各留 14pt：欄界外一點點空白是偵測模型判斷「一行字的開頭」的線索
    # （drawing._RESCUE_PADS 記著同一件事）。
    return fitz.Rect(x0 - 14, min(ys) - 3, x1 + 14, max(ys) + 3)


# --------------------------------------------------------------------------
# 逐格讀值
# --------------------------------------------------------------------------
def _join(parts) -> str:
    """一格裡的字串起來。**下一段以連字號開頭時不加空白。**

    PDF 的文字層是照排版切段的，料號因此會被切在連字號上：例如
    `ABC-0001` + `-B`。照一般規則加空白就變成 `ABC-0001 -B` —— 那個料號
    從此查不到（使用者打的是 `ABC-0001-B`）。只看「下一段開頭是連字號」
    而不順便處理「上一段結尾是連字號」，是因為前者是排版事實，後者則會把
    `28~32AWG, G/F- PLATED` 這種真的有空白的寫法黏起來。
    """
    out = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if out and not p.startswith("-"):
            out += " "
        out += p
    return out.strip()


def read_rows(words: list[Word], table: Table, hits: dict[str, Word],
              bounds: dict[str, tuple[float, float]],
              bands: list[tuple[float, float]]) -> list[dict[str, Any]]:
    """回傳每一列 {欄 key: 文字} + `_conf`。空列直接跳過（表格畫著但沒填）。

    歸格看**字框中心**，不是整框包含：OCR 的框常常比格子寬一點點，要求完全
    包含會讓長規格整段掉出來（`pdf_ocr._cell_text` 記著同一件事）。
    """
    header_lo = min(w[1] for w in hits.values())
    header_hi = max(w[3] for w in hits.values())
    out: list[dict[str, Any]] = []
    for lo, hi in bands:
        # 表頭自己那一列不是資料列。用重疊高度判斷，不是「y 大於/小於」——
        # 表頭在下的表格（由下往上編號）方向是相反的。
        if min(hi, header_hi) - max(lo, header_lo) > (header_hi - header_lo) * 0.5:
            continue
        cells: dict[str, list[Word]] = {k: [] for k in bounds}
        for w in words:
            cy = (w[1] + w[3]) / 2
            if not (lo - 1 <= cy <= hi + 1):
                continue
            cx = (w[0] + w[2]) / 2
            for key, (x0, x1) in bounds.items():
                if x0 <= cx <= x1:
                    cells[key].append(w)
                    break
        row: dict[str, Any] = {}
        got: list[Word] = []
        for key, ws in cells.items():
            if not ws:
                continue
            ws.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
            row[key] = _join(w[4] for w in ws)
            got.extend(ws)
        if not row:
            continue
        row["_conf"] = min(w[5] for w in got)
        row["_band"] = (lo, hi)
        out.append(row)
    return out


# 圈起來的序號。有些材料表用 ①②③④ 當項次，`int()` 對它們一律失敗 ——
# 留 NULL 的話那張表的每一列都沒有項次。這是字元對照，不是猜。
CIRCLED = {c: str(i) for i, c in enumerate("①②③④⑤⑥⑦⑧⑨⑩", start=1)}


def plain_digits(s: str) -> str:
    return "".join(CIRCLED.get(c, c) for c in s or "")
