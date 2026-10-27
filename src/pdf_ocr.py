"""純圖片頁的文字還原：RapidOCR 認字，PDF 自己的向量框線定欄位。

★ 來源：RFQ-RAG/src/pdf_ocr.py，整支搬過來，原本的註解與 DPI 實測數據保留。
  刻意複製而不是跨專案 import —— 與 agentkit 的處理同一原則：
  兄弟目錄的東西不裝進這個 venv，缺了只能降級，不能讓服務起不來。
  唯一的修改是 words() 多一個 clip 參數（見該函式）。

★ 為什麼不讓 OCR 自己做版面分析
  README 第 7 節記著「OCR 路徑不能沿用 pdfplumber 的表格解析（欄位邊界丟失，
  需按文字框 x 座標分群歸欄）」—— 那句話只對了一半。丟失的是**文字層**，
  不是**框線**：這批表單是 Excel 列印出來的，儲存格外框是向量圖元，
  pdfplumber 照樣抓得到完整網格（文字層只有寥寥幾個字元的頁面，
  find_tables() 仍找得到完整的材料表網格）。

  工程圖這批也吃這一套：多數圖帶有大量向量線，find_tables() 通常
  至少找得到 1 個表 —— 材料表的格線是 CAD 畫的實線。

  所以正確做法不是自己 cluster x 座標 —— 合併儲存格與整欄空白都會歸錯欄，
  而且欄數要用猜的。框線是真值：**格子來自 PDF，文字來自 OCR，用中心點
  把字放回格子裡**。歸欄這一步因此完全沒有啟發式。

★ 座標一律換算回 PDF point
  OCR 在點陣圖上做，回來的是像素座標；pdfplumber 的 bbox 是 point。
  兩邊不換算，比對永遠落空（而且會安靜地回一整張空表，不會報錯）。

成本：300 DPI 全頁約 12 秒（CPU，離線）。工程圖只要下方那條長條，
帶 clip 之後是 3.6~6.5 秒/份 —— 判斷在 drawing.find_page()。
"""
from __future__ import annotations

import os
from functools import lru_cache

# ★ 300，不是 200，也不要調到 400（量出來的，判準是「總價欄的和對不對得上表尾合計」）：
#
#   200 DPI 下小數點只有一個像素，會整個掉掉（'12.3400' 認成 '123400'）
#   或被認成逗號（'1.2345' -> '1,2345'）—— 數值直接差 10^4 倍。
#   400 DPI 反而更糟（偵測模型內部會再縮放），而且不是單調的，所以這個值
#   是量出來的不是推出來的。三檔耗時差不多，300 是白拿的。
DPI = int(os.environ.get("OCR_DPI", "300"))

# 低於這個信心的字丟掉。成本表是印刷體，正常字都在 0.9 以上；
# 0.5 以下的多半是框線或圖案被當成字。工程圖同理 —— 尺寸線與箭頭
# 很容易被偵測成 '-' 或 '1'。
MIN_SCORE = float(os.environ.get("OCR_MIN_SCORE", "0.5"))

# (x0, y0, x1, y1, text, score)，座標是 PDF point
Word = tuple[float, float, float, float, str, float]


@lru_cache(maxsize=1)
def _engine():
    """RapidOCR 實例。載入約 0.7 秒，整個 ingest 只做一次。"""
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def available() -> bool:
    """沒裝 rapidocr 時整條 OCR 路徑靜靜跳過，其餘照跑。

    與 retrieval._by_vector 同樣的降級原則：少一條通道不該讓整個管線壞掉。
    """
    try:
        _engine()
        return True
    except Exception:
        return False


def words(path: str, page_no: int, dpi: int = DPI, clip=None,
          boost: int = 0) -> list[Word]:
    """把一頁 render 成點陣圖跑 OCR，回傳 PDF point 座標的文字框。

    ★ boost（本專案新增）：先二值化再加粗一個像素，給**淡色印刷**用。
      典型情況是物料明細表裡序號欄、數量欄的 `1` 用淡灰色印，整格幾乎沒有
      深色墨點，RapidOCR 換多個 DPI **全部偵測不到** —— 不是解析度
      不夠，是對比不夠。二值化（門檻 252：只要不是純白就算墨）加上一圈膨脹之後，
      同一格就讀得到 `1`、信心接近 1。

      boost 的值就是那個門檻。0 = 不處理（預設，既有呼叫端行為不變）。

      ★ 只對**單一格**有用，整張表一起 boost 沒有用（實測 450/600dpi 都還是
        讀不到那幾個 `1`）：偵測模型在大圖裡會忽略孤立的小字。所以呼叫端要
        一格一格給，不要指望一次過。

    ★ clip（本專案新增）：只 render 頁面的一部分。工程圖是 A4/A3 橫幅，
      要的兩個表格都在最下面那條，整頁 render 是白花 2~3 倍的時間
      （全頁約 12 秒 vs 下方 30% 的 3.6~6.5 秒）。而且放大同樣的 DPI
      只算那條，記憶體也小得多。

      get_pixmap(clip=...) 回來的點陣圖原點是 clip 的左上角，所以換算回
      PDF point 之後**一定要加回 clip.x0 / clip.y0**。忘了加的症狀很惡劣：
      座標全部合法、只是整批往上平移，切欄與歸格都還「成功」，只是切在
      錯的地方 —— 不會報錯。
    """
    import fitz
    import numpy as np

    doc = fitz.open(path)
    try:
        pm = doc[page_no].get_pixmap(dpi=dpi, clip=clip)
        img = np.frombuffer(pm.samples, dtype=np.uint8).reshape(
            pm.height, pm.width, pm.n)
    finally:
        doc.close()

    if boost:
        gray = img.min(axis=2)
        flat = np.where(gray < boost, 0, 255).astype(np.uint8)
        thick = flat.copy()
        for shift in (1, -1):
            thick = np.minimum(thick, np.roll(flat, shift, axis=0))
            thick = np.minimum(thick, np.roll(flat, shift, axis=1))
        img = np.stack([thick] * 3, axis=2)

    res, _ = _engine()(img)
    scale = dpi / 72.0
    ox = float(clip.x0) if clip is not None else 0.0
    oy = float(clip.y0) if clip is not None else 0.0
    out: list[Word] = []
    for box, text, score in res or []:
        if score < MIN_SCORE or not (text or "").strip():
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        out.append((min(xs) / scale + ox, min(ys) / scale + oy,
                    max(xs) / scale + ox, max(ys) / scale + oy,
                    text.strip(), score))
    return out


def page_text(ws: list[Word]) -> str:
    """OCR 結果還原成閱讀順序的文字。

    給下游對整頁文字下正規表示式用 —— 需要的是「同一行的字要在同一行」，
    不是原始的偵測順序（RapidOCR 是由上而下但同一行內不保證由左而右）。

    分行用「y 中心落在目前這行的字高之內」，不用固定像素門檻 —— 表格列高
    在不同模板差兩倍以上，寫死會把兩列黏成一列。

    ★ 工程圖不要拿這個當主路線。兩個表格在同一個 y 帶上，只按閱讀順序排
      會把材料表某一列跟標題欄黏成一列（實測：
      `3 TERMINAL... 6PCS RoHS COMPLIANT 0.0=±.5 CHK <姓名> CBL,...`）。
      必須先按 x 切欄 —— drawing.split_bottom() 做這件事。這裡留著是給
      title_raw / items_raw 存原文用的。
    """
    if not ws:
        return ""
    items = sorted(ws, key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    lines: list[list[Word]] = [[items[0]]]
    for w in items[1:]:
        cur = lines[-1]
        cy = (w[1] + w[3]) / 2
        ref_cy = sum((x[1] + x[3]) / 2 for x in cur) / len(cur)
        ref_h = sum(x[3] - x[1] for x in cur) / len(cur)
        if abs(cy - ref_cy) <= ref_h * 0.6:
            cur.append(w)
        else:
            lines.append([w])
    return "\n".join(
        " ".join(w[4] for w in sorted(line, key=lambda w: w[0]))
        for line in lines
    )


def fill_tables(pl_page, ws: list[Word]) -> list[list[list[str | None]]]:
    """用 OCR 的字填滿 pdfplumber 找到的格線，形狀對齊 page.extract_tables()。

    形狀刻意一模一樣（含「空格子是 None 不是空字串」），下游才能同一份程式碼
    同時吃文字層與 OCR 兩條路 —— 兩份表頭判斷邏輯遲早會漂掉。

    ★ 空格子留 None，不要留 ""。下游的數字解析與空列判斷都靠 None 分辨
      「這格沒東西」與「這格有東西但不是數字」。
    """
    out: list[list[list[str | None]]] = []
    for table in pl_page.find_tables():
        rows: list[list[str | None]] = []
        for row in table.rows:
            rows.append([_cell_text(c, ws) if c else None for c in row.cells])
        if rows:
            out.append(rows)
    return out


def _cell_text(cell, ws: list[Word]) -> str | None:
    """落在這個格子裡的字，照閱讀順序串起來。

    用文字框的**中心點**判斷歸屬，不是整框包含 —— OCR 的框常常比格子寬一點點
    （尤其規格說明那種塞滿的欄），要求完全包含會讓長字串整個掉出來。
    """
    x0, y0, x1, y1 = cell
    got = [w for w in ws
           if x0 - 1 <= (w[0] + w[2]) / 2 <= x1 + 1
           and y0 - 1 <= (w[1] + w[3]) / 2 <= y1 + 1]
    if not got:
        return None
    got.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    return " ".join(w[4] for w in got).strip() or None
