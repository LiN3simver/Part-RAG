"""partrag 的 A2A 適配層。**純新增，不改這個專案任何既有行為。**

掛上去的方式（main.py 尾端已加，只有三行 try/except）：

    from src import a2a
    a2a.mount(app)

它做的事只有兩件：
  1. 把既有的 retrieval / aggregate 包成統一形狀（Hit / Block），讓 Agent Hub
     的共用前端畫得出來。
  2. 把同樣的能力包成 Anthropic tool schema，讓上游的「RD 分析 BOM Agent」
     可以直接呼叫。

★ 為什麼不是改 /api/search 的請求形狀
  這個專案的 static/index.html 與 /api/chat 都在用 {query, filters, top_k}，
  而 RFQ-RAG 用的是 {q, limit}。要共用前端就得有一個統一形狀，但那個統一
  形狀應該長在旁邊（/a2a/*），不是把已經在跑的路徑改掉。舊路徑一個字不動。

★ 列出/統計型問句的處理與 main.py 的 _find() 完全一致
  「列出所有 示例廠 的料號」這種問題要走 aggregate 的確定性 SQL，不是排序
  檢索。如果 A2A 這條路徑自己只呼叫 retrieval.search()，同一個問題從統一
  前端問只會得到一頁抽樣、從原生前端問會得到正確總數 —— 兩個介面對同一個
  問題給不同答案，比少一個功能糟糕得多。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

from dotenv import load_dotenv

# --- agentkit 引導 --------------------------------------------------------
# 預設從本專案的固定 vendor 快照匯入。AGENTKIT_PATH 若另行指定，也必須符合 lock。
# 快照缺漏或內容不同時 A2A 不掛載，readiness 會反映必要能力不可用。
#
# ★ 這一行一定要在 import agentkit **之前**。agentkit.gateway 在模組層自己
#   呼叫 load_dotenv()，而它是從**自己的檔案位置**（Agent_Hub_RAG_Frontend 底下）往上找
#   .env，找到的是 Agent_Hub_RAG_Frontend/.env，不是這個專案的。load_dotenv 不覆寫已
#   存在的環境變數，所以先載入本專案的，本專案 .env 若明確設了 LLM_* 才會
#   贏過 Agent Hub 的集中設定（見 src/llm.py 開頭同樣的說明）。
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from .delivery import verify_vendor  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_HUB = Path(os.environ.get("AGENTKIT_PATH") or _ROOT / "vendor")
verify_vendor(_ROOT, _HUB / "agentkit")
if _HUB.is_dir() and str(_HUB) not in sys.path:
    sys.path.insert(0, str(_HUB))

from agentkit import AgentSpec, Block, Hit, ToolSpec, registry  # noqa: E402
verify_vendor(_ROOT, Path(sys.modules["agentkit"].__file__).parent)

from . import aggregate, drawing_frames, grounding, llm, retrieval, sync_state  # noqa: E402
from . import toolguard  # noqa: E402
from .a2a_runtime import build_router  # noqa: E402

AGENT_ID = "partrag"
# 展示版：公司系統對接（來源 profile 設定）已移除，這裡固定用單一 profile。
_PROFILE = "default"


# --------------------------------------------------------------------------
# 檢索
# --------------------------------------------------------------------------
def _find(query: str, top_k: int, filters: dict[str, Any]) -> dict[str, Any]:
    """與 main.py 的 _find() 同一個決策，見模組 docstring。"""
    spec = aggregate.plan(query, filters)
    if spec:
        return aggregate.run(spec)
    return retrieval.search(query, filters, top_k)


# 規格欄位 -> 顯示標籤。挑的是「同一條線材改長度就會變」的那些欄，
# 因為統一前端最常被用來比較兩個相近料號。
_SPEC_ROWS = [
    ("length_mm", "長度 mm"), ("pitch_mm", "Pitch mm"), ("pitch_series", "Pitch 系列"),
    ("pins_total", "總 PIN 數"), ("pin_layout", "PIN 排列"), ("awg", "AWG"),
    ("gen", "世代"), ("speed_g", "速率 G"), ("conn_a", "接頭 A"), ("conn_b", "接頭 B"),
    ("color", "顏色"), ("rated_voltage", "額定電壓"), ("customer_pn", "客戶料號"),
    ("drawing_no", "圖號"),
]

# PLM 零組件表單欄位。這些是工程師實際會講的詞（產品小類、環保標準、
# 建立廠別），所以要跟規格分開一區顯示。
_PLM_ROWS = [
    ("prod_l1", "產品大類"), ("prod_l2", "產品中類"), ("prod_l3", "產品小類"),
    ("item_type", "料件類型"), ("eco_std", "環保標準"), ("plant", "建立廠別"),
    ("announce_cat", "公告大類"), ("item_class", "料件分類"), ("unit", "單位"),
    ("lifecycle", "Lifecycle"), ("state", "狀態"), ("major_rev", "版本"),
    ("created_by", "建立人員"), ("created_on", "建立日期"),
    ("modified_by", "修改人員"), ("modified_on", "修改日期"),
]


def _num(v: Any) -> Any:
    """600.0 -> 600。長度與 pitch 是 numeric，字面上帶 .0 只是噪音。"""
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _fmt(v: Any) -> Any:
    """list 攤成 "600, 300"；其他原樣回傳。

    ★ None 一定要原樣往上傳，不要轉成 "" 或 0：前端靠 null 判斷「未填」，
      而 0（例如 length_mm=0）是有效值。
    """
    if isinstance(v, list):
        return ", ".join(str(_num(x)) for x in v) if v else None
    return _num(v)


def _rows(row: dict, keys: list[tuple[str, str]]) -> list[list[Any]]:
    """只留下有值的列。全部 24 欄都畫出來會讓卡片變成一面欄位牆，
    但「有填 0」要留下 —— 那是資訊，不是空白。"""
    out = []
    for col, label in keys:
        v = _fmt(row.get(col))
        if v is None or v == "":
            continue
        out.append([label, v])
    return out


def _badges(row: dict) -> list[list[str]]:
    """卡片上的小標籤。

    ★ 這張表有九個欄位是 **array**，不是純量：length_mm / pins_total /
      pin_counts / pin_layout / pins_row / pitch_series / conn_types /
      customer_pn / drawing_no。原因是一條線有兩端、可能多段，
      「600mm 那一段 4P、300mm 那一段 6P」在同一列裡。
      直接 f"{row['length_mm']:g}" 會炸（實測 TypeError: unsupported format
      string passed to list.__format__），所以一律先過 _fmt()。
    """
    b: list[list[str]] = []
    for col, label, suffix in (
        ("length_mm", "L", "mm"),
        ("pins_total", "PIN", ""),
        ("pitch_series", "Pitch", ""),
        ("awg", "AWG", ""),
    ):
        v = _fmt(row.get(col))
        if v is None or v == "":
            continue
        b.append([label, f"{v}{suffix}"])
    for c in row.get("conn_types") or []:
        b.append([str(c), ""])
    if row.get("is_placeholder"):
        b.append(["占位料號", ""])
    return b


def _pdf_url(drawing_id: Any) -> str | None:
    """工程圖 PDF 的**絕對**網址。

    卡片是在 Hub（:8600）上畫的，相對路徑會打到 Hub 去，而檔案是 partrag
    自己在服務 —— 這是 Hit.links 的既有約定（見 agentkit/envelope.py）。
    拿不到自己的 base_url 就回 None，寧可少一顆按鈕，不要給一個 404 的連結。

    ★ 兩種部署，網址指向不一樣的地方（RFQ-RAG 實測踩過）
      本機開發：partrag 自己就對外開著 8000，網址直接指自己（registry 的
      base_url），Hub 打得到，瀏覽器也打得到。

      Linux 伺服器（docker compose）：partrag 只掛在 ragnet 內部網路、不對外
      開 host port，只有 Agent Hub 有對外的固定位址。這時網址若還是
      127.0.0.1:8000，瀏覽器解析到的是**使用者自己那台電腦** —— 而且不是一個
      連不上的錯誤，是安靜地連到一個誰都沒開著這個服務的地方。

      設了 A2A_HUB_PUBLIC_URL 就改指 Hub 的 /hub/file/{agent}/{doc_id} 代理端點
      （見 Agent_Hub_RAG_Frontend/main.py），由它在 ragnet 內部轉打下面的 /a2a/file/{id}，
      再把 PDF 位元組吐回瀏覽器。不設就維持本機開發直連自己的舊行為。
    """
    if drawing_id is None:
        return None
    hub = os.environ.get("A2A_HUB_PUBLIC_URL", "").strip()
    if hub:
        return f"{hub.rstrip('/')}/hub/file/{AGENT_ID}/{drawing_id}"
    try:
        return f"{registry.get(AGENT_ID).base_url}/api/drawing/file/{drawing_id}"
    except Exception:  # noqa: BLE001 -- 沒註冊到 Hub 時照樣要能回卡片
        return None


def _export_url(rel: str) -> str | None:
    """統計結果「完整清單 CSV」的絕對網址，理由同 `_pdf_url`：卡片在 Hub 上
    畫，相對路徑（`aggregate.run()` 給的 `/api/export?query=...`）會打到 Hub
    去，不是 partrag 自己。

    Hub 設了對外固定位址（`A2A_HUB_PUBLIC_URL`）時，目前沒有像 drawing 那樣
    的 `/hub/file/...` 代理端點可以轉打 `/api/export` —— 給一個瀏覽器打不到
    的網址比不給連結更糟，所以這種部署下寧可不掛連結（需要匯出時
    改用 curl 直打伺服器）。
    """
    if os.environ.get("A2A_HUB_PUBLIC_URL", "").strip():
        return None
    try:
        return f"{registry.get(AGENT_ID).base_url}{rel}"
    except Exception:  # noqa: BLE001 -- 沒註冊到 Hub 時照樣要能回卡片
        return None


# 圖上抽出來的標題欄，順序刻意照 PDF 右下角那一格由上而下、由左而右排。
# 使用者是拿卡片跟原圖對著看的，欄位順序跟圖不一樣就得一格一格找。
_TITLE_ROWS = [
    ("dwg_title", "TITLE"), ("dwg_part_no", "PART NO."), ("dwg_rev", "REV."),
    ("dwg_size", "SIZE"), ("dwg_scale", "SCALE"), ("pd_file_no", "PD./FILE NO."),
]

# 沒有材料表的圖框，「圖上原文」（title_raw）各處要截到幾個字。
#
#   _SPEC_CHARS       卡片上給人看的。留餘裕給較長的規格書，同時擋住
#                     「整頁文字倒進一張卡」的情況。
#   _SPEC_MODEL_CHARS 給模型看的（單份題）。
#   _SPEC_LIST_CHARS  給模型看的（清單題）。短，是因為那時候的預算要留給
#                     _drawing_index 的索引（見 DRAWING_INDEX 那段說明）。
_SPEC_CHARS = 3000
_SPEC_MODEL_CHARS = 2000
_SPEC_LIST_CHARS = 500

_METHOD_LABEL = {
    "text": "PDF 文字層（非 OCR）",
    "ocr": "OCR",
    "ocr_grid": "OCR + 圖上格線定列",
    # 客戶圖框常常是一半文字層一半 CAD 向量（見 drawing.find_page 的說明），
    # 兩者可信度不同，所以講清楚是混的。
    "ocr+text": "OCR + PDF 文字層",
    "ocr_grid+text": "OCR + PDF 文字層 + 圖上格線定列",
}


def _by_table(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """材料列照 table_tag 分組，保持 retrieval 給的順序（BOM 在前）。"""
    out: dict[str, list[dict[str, Any]]] = {}
    for r in items:
        out.setdefault(r.get("table_tag") or "BOM", []).append(r)
    return out


def _table_block(tag: str, rows: list[dict[str, Any]]) -> Block:
    """一張表一個 Block。欄頭照圖上那張表寫，而且**空欄不顯示** ——
    有些圖框沒有 UNIT / 备注 / COLOR，硬留一排空欄只會讓卡片看起來抽壞了。"""
    if tag == "PN_LENGTH":
        return Block(
            kind="table", title=f"P/N AND LENGTH（{len(rows)} 列）",
            head=["NO", "P/N", "LENGTH", "PACKAGE"],
            rows=[[r.get("item_no"), r.get("description") or "",
                   r.get("length_raw") or "", r.get("remark") or ""]
                  for r in rows],
        )
    extra = [(k, h) for k, h in (("unit", "UNIT"), ("color", "COLOR"),
                                 ("length_raw", "LENGTH"), ("remark", "備註"))
             if any(r.get(k) for r in rows)]
    head = ["ITEM", "DESCRIPTION", "Q'TY", *[h for _, h in extra], "廠商料號"]
    return Block(
        kind="table", title=f"材料表（圖框左下，{len(rows)} 列）", head=head,
        rows=[[r.get("item_no"), r.get("description") or "", r.get("qty_raw") or "",
               *[r.get(k) or "" for k, _ in extra],
               "、".join(r.get("vendor_pn") or [])]
              for r in rows],
    )


def _drawing_hits(found: dict[str, Any]) -> list[Hit]:
    """一份工程圖一張卡：標題欄 + 材料表 + 出處。

    ★ 材料表用 kind="table" 而不是 kv，欄頭就寫 ITEM / DESCRIPTION / Q'TY ——
      那是圖上那張表的欄頭。使用者要的是「跟原圖長一樣」，能一列一列對。

    ★ 出處（檔名 / 檔案更新時間 / 判讀方式）一定跟值放在同一張卡，而且
      PDF 連結掛在 Hit.links 上。多數圖是 OCR 讀的，O/0 混淆與被吃掉
      的連字號是已知的誤判 —— 沒有原檔可點，使用者沒辦法判斷該不該相信
      卡片上的字。這跟 rfq agent 對掃描件成本表的處理是同一個原則。
    """
    out: list[Hit] = []
    for d in found.get("drawings") or []:
        title_rows = [[label, d[col]] for col, label in _TITLE_ROWS if d.get(col)]
        if d.get("sheet_no"):
            sheet = f"{d['sheet_no']} OF {d['sheet_of']}" if d.get("sheet_of") else str(d["sheet_no"])
            title_rows.append(["SHEET", sheet])

        # 客戶圖框的其餘標題欄欄位接在後面（同一張卡，不另開一張）——
        # 使用者問的是「這張圖上寫什麼」，MATERIAL 與 TITLE 沒有理由分兩張卡。
        title_rows.extend([label, val] for label, val
                          in drawing_frames.field_rows(d.get("title_fields")))

        blocks = [Block(kind="kv", title="標題欄（圖框右下）", tone="green",
                        rows=title_rows or [["（未抽到標題欄）", ""]])]

        # 一份圖可能有兩張表（材料表 + 料號長度對照表），一張表一個 Block ——
        # 併成一張的話 12 列料號長度會看起來像 12 種材料。
        for tag, rows in _by_table(d.get("items") or []).items():
            blocks.append(_table_block(tag, rows))

        # ★ 沒有材料表的圖框，圖上的正文要自己一塊
        #   規格書型的圖框（drawing_frames.NO_BOM_FRAMES）本來就沒有材料表 ——
        #   圖上真正的內容是 `Specification:` 底下的材料與性能條列，全部只存在
        #   title_raw。上面的迴圈對這些圖跑零次，卡片只剩標題欄與出處，使用者
        #   會以為「這種圖沒抽」，而值其實早就在資料庫裡、也已餵給模型了
        #   （見 _drawing_payload 的 spec_text）。
        #
        #   有材料表的圖框刻意不加：那份 title_raw 混著標題欄的欄名與雜訊，
        #   材料表本身已經是乾淨的結構化答案，再貼一份原文只會讓人不知道
        #   該信哪一個。理由與 _drawing_payload 那邊完全相同。
        if not (d.get("items") or []) and d.get("frame") in drawing_frames.NO_BOM_FRAMES:
            spec = (d.get("title_raw") or "").strip()
            if spec:
                blocks.append(Block(
                    kind="text", tone="green",
                    # 標題要講清楚這是原文不是欄位：這段是 OCR 與文字層混排出來
                    # 的，同一行可能跨了圖上兩個不相干的區塊（規格文字的右邊
                    # 就是標題欄），逐字都對、行的對應不保證對。
                    title="圖上原文（規格／性能，OCR 混排）",
                    text=spec[:_SPEC_CHARS],
                ))

        prov: list[list[Any]] = [
            ["來源檔名", d.get("local_file")],
            ["檔案更新時間", str(d.get("doc_modified_on") or "")[:19] or "未知"],
            ["判讀方式", _METHOD_LABEL.get(d.get("extract_method") or "",
                                       d.get("extract_method") or "未知")],
        ]
        if d.get("ocr_conf") is not None:
            prov.append(["最低字元信心", d["ocr_conf"]])
        # 歸檔料號與圖上料號不一致時要講出來，不能只挑一個顯示：download_log
        # 記的是「PLM 從哪個料號的紀錄給出這個檔」，圖上印的是圖自己說的，
        # 兩者對不上正是使用者該去看原圖的訊號。
        if d.get("dwg_part_no") and d["dwg_part_no"] != d.get("item_number"):
            # part_no_ok 為真而兩邊字面不同 -> 差的是 PLM 的系統前綴（`PT-`），
            # 已經核對過是同一顆（見 drawing.normalize_pn）。這種情況不能沿用
            # 「請以原圖為準」那句話：那是叫使用者去懷疑一個已經確認過的值。
            note = ("PLM 的系統前綴，與圖上料號是同一顆" if d.get("part_no_ok")
                    else "與圖上 PART NO. 不同，請以原圖為準")
            prov.append(["★ 歸檔料號", f"{d.get('item_number')}（{note}）"])
        if not d.get("part_no_ok"):
            prov.append(["★ 注意", "PART NO. 未與 PLM 料號核對一致，此欄請以原圖為準"])
        blocks.append(Block(kind="kv", title="出處與可信度", tone="yellow", rows=prov))

        url = _pdf_url(d.get("id"))
        stamp = str(d.get("doc_modified_on") or "")[:19]
        out.append(Hit(
            id=d.get("dwg_part_no") or d.get("item_number") or "",
            title=d.get("dwg_title") or "（工程圖）",
            subtitle=f"工程圖 · {d.get('local_file') or ''}",
            badges=[["工程圖", ""]] + ([["REV.", d["dwg_rev"]]] if d.get("dwg_rev") else []),
            blocks=blocks,
            # 連結文字帶檔案更新時間，不是只有檔名 —— 同一張圖會有好幾個版本，
            # 版本錯了材料就錯了。這是 rfq agent 那條註解的同一個理由。
            links=[[f"{d.get('local_file')}（更新 {stamp or '時間未知'}）", url]] if url else [],
            raw=d,
        ))
    return out


def _hits(found: dict[str, Any]) -> list[Hit]:
    hits: list[Hit] = []

    # 工程圖排在最前面。問「…的工程圖」時圖就是答案本身，料號卡是佐證；
    # 排在後面等於要使用者先捲過一堆規格欄位才看到他問的東西。
    hits += _drawing_hits(found)

    # 問了工程圖但沒有圖：這句話要自成一張卡，不能只靠 LLM 轉述。
    if found.get("drawing_note"):
        hits.append(Hit(
            id="工程圖", title="沒有這個料號的工程圖",
            body=found["drawing_note"],
            blocks=[Block(kind="kv", tone="warn",
                          rows=[["狀態", "查無工程圖"],
                                ["目前收錄", "僅 data/pdf/ 底下的部分料號工程圖"]])],
            raw={"drawing_note": found["drawing_note"]},
        ))

    # 列出/統計型問句：總數與匯出連結必須排在第一張卡。
    # 只把抽樣的 25 筆畫出來，會讓人以為那 25 筆就是全部。
    lst = found.get("listing")
    if lst:
        tgt = (lst.get("target") or {}).get("label")
        rows: list[list[Any]] = [["符合筆數", f"{lst['total']:,}"]]
        if lst.get("shown") is not None:
            rows.append(["本頁顯示", f"{lst['shown']:,}"])
        if tgt:
            rows.append(["統計欄位", tgt])
        if lst.get("conditions"):
            def _cond_str(c: dict) -> str:
                vals = "/".join(str(v) for v in c.get("values") or [])
                return f"{c['label']}={vals}" if vals else c["label"]
            rows.append(["條件", "、".join(_cond_str(c) for c in lst["conditions"])])
        if lst.get("truncated"):
            rows.append(["★ 注意", "下方只是抽樣，不是全部"])
        export_link = None
        if lst.get("export_url"):
            export_link = _export_url(lst["export_url"])
            rows.append(["完整清單", "CSV（見下方連結）" if export_link else f"{lst['export_url']}（CSV）"])
        blocks = [Block(kind="kv", title="清單／統計", tone="good", rows=rows)]
        vals = lst.get("values") or []
        if vals:
            blocks.append(Block(
                kind="table", title=f"{tgt or '值'}分佈（前 30）", head=["值", "筆數"],
                rows=[[v.get("value"), v.get("n")] for v in vals[:30]],
            ))
        hits.append(Hit(
            id=f"統計：{found.get('query', '')}",
            title=f"共 {lst['total']:,} 筆",
            blocks=blocks, raw=lst,
            links=[["下載完整清單（CSV）", export_link]] if export_link else [],
        ))

    # 語料裡沒有的料號：一定要說出來。partrag 的檢索會給鄰近料號，
    # 沒有這張卡的話那些鄰居看起來就像「查到了」。
    for pn in found.get("not_found") or []:
        hits.append(Hit(
            id=pn, title="語料中沒有這個料號",
            body="下方為同區段的鄰近料號，不是這個料號本身。可能是 PLM 新開、尚未匯出到語料。",
            blocks=[Block(kind="kv", tone="warn", rows=[["查詢的料號", pn], ["狀態", "not_found"]])],
            raw={"item_number": pn, "found": False},
        ))

    for note in found.get("plm_notes") or []:
        hits.append(Hit(
            id=note.get("label") or "PLM 欄位說明",
            title=note.get("message") or "",
            blocks=[Block(kind="kv", tone="warn",
                          rows=[[k, v] for k, v in note.items() if k != "message"])],
            raw=note,
        ))

    # 問工程圖而且真的有圖時，就只給工程圖的卡。
    # retrieval 已經把 results 收斂成「有圖的那幾筆」，但那幾筆的規格卡在這裡
    # 仍然是雜訊：使用者問的是圖上畫什麼，不是這顆料的 Pitch 與建立廠別。
    # 沒有圖時不套用 —— 那時候規格卡是「這個料號存在、只是沒有圖」的唯一證據。
    if found.get("drawings"):
        return hits

    for r in found.get("results") or []:
        spec_rows, plm_rows = _rows(r, _SPEC_ROWS), _rows(r, _PLM_ROWS)
        blocks = []
        if spec_rows:
            blocks.append(Block(kind="kv", title="規格", tone="green", rows=spec_rows))
        if plm_rows:
            blocks.append(Block(kind="kv", title="PLM 表單欄位", tone="yellow", rows=plm_rows))
        hits.append(Hit(
            id=r.get("item_number") or "",
            title=r.get("name") or "",
            subtitle=" · ".join(
                str(x) for x in (r.get("category"), r.get("prod_l3"), r.get("state")) if x
            ),
            score=r.get("score"),
            why=r.get("match_reasons") or [],
            badges=_badges(r),
            blocks=blocks,
            raw=r,
        ))
    return hits


class LLMUnavailable(RuntimeError):
    """摘要階段 gateway 掛掉。帶結構化診斷，agentkit.server 會轉成 error 事件。"""

    def __init__(self, diag: dict[str, Any]) -> None:
        super().__init__(f"{diag['title']}：{diag['detail']}")
        self.diag = diag
        self.user_message = f"{diag['title']}：{diag['detail']}"
        self.hint = diag["fix"]


class AnswerUnverified(RuntimeError):
    """The gateway answered, but its summary must not be treated as reliable."""

    user_message = grounding.UNVERIFIED_NOTICE


def _answer(query: str, found: dict[str, Any], history: list[dict]) -> Any:
    # found 整包傳進去：裡面有 not_found / plm_notes，那是模型學會說
    # 「<料號> 不在語料裡」而不是拿最近的鄰居充當答案的依據。
    hist = [{"role": h["role"], "content": h["content"]} for h in history
            if h.get("role") in ("user", "assistant")]

    # ★ 為什麼要接 status，而不是直接把 stream_answer() 回出去
    #   stream_answer() 從不拋例外 —— gateway 掛掉時它 yield 的是
    #   "[LLM 無法使用：…]"，那是一串**長得像答案的錯誤訊息**。上游只要收到
    #   任何一個 delta 就判定 llm_available=true，Agent Hub 還會把它當成助理
    #   的正式回答存進 conversations.json，下一輪再原樣回放給模型當歷史。
    #   所以這裡改看它的 status 出口：失敗就把那串字攔下來，改拋帶診斷的
    #   例外走 error 事件，前端才畫得出「該做什麼、要不要換 API key」。
    status: dict[str, Any] = {}
    for chunk in llm.stream_answer(query, found.get("results") or [], found, hist,
                                   status=status):
        if status.get("ok") is False and chunk.startswith("[LLM 無法使用："):
            continue
        yield chunk
    if status.get("ok") is False:
        raise LLMUnavailable(llm.diagnose(status.get("error")))

    # 事實核對的結果只進 log，不進回答。
    #
    # ★ 為什麼不往前端送
    #   攔得下來的幻覺已經被重寫掉了，使用者沒有必要知道模型第一次講錯了什麼；
    #   重寫後仍失敗時只顯示核對失敗提示。中間再多一個技術性欄位
    #   只會讓畫面變吵。但**維運要看得到**：first_pass 的比例是這顆模型在這份
    #   語料上的品質指標，掉下去代表該重跑一次 `python -m src.evalrun`。
    if status.get("first_pass") is False:
        kinds = "、".join(v["kind"] for v in (status.get("violations") or [])) or "已修正"
        print(f"[a2a] 事實核對攔截：route={status.get('route')} "
              f"呼叫 {status.get('attempts')} 次，最終殘留={kinds}｜{query[:40]}")
    if status.get("verified") is False:
        raise AnswerUnverified(grounding.UNVERIFIED_NOTICE)


def _sync_summary() -> dict[str, Any]:
    return sync_state.summarize(sync_state.report(_PROFILE), _PROFILE)


def _status() -> tuple[str, dict[str, Any]]:
    try:
        f = retrieval.facets()
        detail: dict[str, Any] = {"db": True, **(f.get("stats") or {})}
    except Exception as e:
        return "not_ready", {"db": False, "error": f"{type(e).__name__}: {e}"}
    # 排程讀不到（沒套排程的 migration 就會這樣）只是加註，不改 status：
    # 檢索與問答完全不依賴排程，語料就在表裡。
    try:
        detail["sync"] = _sync_summary()
    except Exception as e:
        detail["sync"] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                          "note": "無法讀取同步狀態，請檢查資料庫與 migration"}
    # ★ 這裡刻意放**結構化**的診斷，不是只有一行字串。
    #   Agent Hub 的降級橫幅原本是把整包 detail 做 JSON.stringify 印出來，
    #   使用者看到的是巢狀引號包著的 gateway 原始 400，讀完仍然不知道
    #   「這是 LLM 的問題還是檢索的問題、我該去找誰」。llm.diagnose() 已經
    #   把「發生什麼 / 該做什麼 / 誰能修 / 要不要換新 key」拆成欄位，
    #   前端照欄位畫就好，不需要認識 gateway 的錯誤格式。
    d = llm.diagnose()
    detail["llm"] = "ok" if d["ok"] else f"{d['title']}：{d['detail']}"
    detail["llm_diag"] = d
    # LLM 掛掉只是不能摘要，檢索照樣可用 —— 這是降級，不是壞掉。
    if not d["ok"]:
        return "degraded", detail
    # ★ 某個 stage 上一輪失敗，即刻算降級 —— 不用等到 STALE_HOURS（預設 100
    #   小時）那麼久才被看見。這裡刻意放在 stale 檢查前面：同一輪裡「候選抓
    #   不到」比「最近一次成功是幾小時前」更具體、更該優先讓人看到，尤其是
    #   ingest_drawing 全軍覆沒這種——record_failure 存的 note 本身就講清楚
    #   「PDF 可能壞了或缺 OCR」，直接搬到橫幅上就是完整的診斷，不用另外編。
    #   自我修復：下一輪只要那個 stage 成功，record_success 會把 ok 蓋回
    #   true，這裡自然回到 ready，不需要另外一套「已解除」的邏輯。
    failed = {s: info for s, info in detail["sync"].get("stages", {}).items()
              if info.get("ok") is False}
    if failed:
        lines = "；".join(f"{s}：{info.get('note') or '無說明'}"
                          for s, info in failed.items())
        return "degraded", {**detail,
                            "note": f"排程有 stage 上次執行失敗：{lines}"}
    # 排程過期也是降級：查得到、但查到的可能是好幾週前的語料。note 要把小時數與
    # 門檻都講出來，Hub 的降級橫幅是直接顯示這行字的（見 llm_diag 那段的理由）。
    if not detail["sync"].get("ok", True):
        return "degraded", {**detail, "note": detail["sync"].get("note") or "同步尚未完成"}
    return "ready", detail


# --------------------------------------------------------------------------
# 工具（給上游 agent 呼叫）
# --------------------------------------------------------------------------
def _drawing_payload(d: dict, with_items: bool = True) -> dict:
    """一份工程圖給**模型**看的形狀（不是給前端畫卡片的那份）。

    只挑標題欄、材料表與出處三件事，並且把 items 壓成一列一個字串 ——
    整包 drawings 原封不動塞進 tool_result 會把 thinking 的 token 預算擠掉，
    而材料表正是這一題的答案本身，不能被 delegate._fit() 砍掉。

    ★ `with_items=False`：清單題不給材料表
      「某段時間的工程圖」撈到幾十份時，問的是「有哪些圖」。材料表一份就很長，
      展開幾份就把整批的索引擠出預算 —— 而索引才是這一題的答案
      （見 _drawing_index 與 llm._render_drawings 的同一段說明）。
    """
    # ★ 客戶料號一定要跟著回，它常常就是使用者打進來的那個編號
    #   使用者打的 `<客戶料號>` 可能只出現在標題欄的 CUSTOMER PART NUMBER 那一格，
    #   和圖上 PART NO.、歸檔料號都不同。這一欄不回的話，模型手上**沒有任何
    #   一個欄位**可以判斷「這份圖到底是不是他問的那一份」—— 遇到只差一碼的
    #   編號（ABC-0001 / ABC-0002），它只會當成同一個東西的兩種寫法。
    tf = d.get("title_fields") or {}
    return {
        "part_no": d.get("dwg_part_no") or d.get("item_number"),
        "customer_part_no": tf.get("customer_pn"),
        "filed_under": d.get("item_number"),
        "title": d.get("dwg_title"),
        "rev": d.get("dwg_rev"),
        "sheet": f"{d.get('sheet_no')} OF {d.get('sheet_of')}" if d.get("sheet_no") else None,
        "file": d.get("local_file"),
        "file_modified_on": str(d.get("doc_modified_on") or "")[:19] or None,
        "extract_method": _METHOD_LABEL.get(d.get("extract_method") or "",
                                            d.get("extract_method") or "未知"),
        "part_no_ok": d.get("part_no_ok"),
        "pdf_url": _pdf_url(d.get("id")),
        "items": [
            " | ".join(str(x) for x in (
                f"ITEM {r.get('item_no')}", r.get("description") or "",
                f"Q'TY {r.get('qty_raw') or ''}",
                *( ["廠商料號 " + "、".join(r["vendor_pn"])] if r.get("vendor_pn") else [] ),
            ))
            for r in (d.get("items") or [])[:40]
        ] if with_items else None,
        "items_note": None if with_items else (
            f"這一題是清單題，材料表沒有展開（這一份有 {len(d.get('items') or [])} 列）。"
            "使用者要看材料表時用這個料號或檔名再呼叫一次 search_drawings。"
        ),
        # 這幾種圖框（廠商規格書、客戶的自由文字規格書）沒有材料表，
        # 圖上真正的內容——Specification / Performance / Cable Length
        # Tolerances、材料自由文字——全部只在 title_raw 裡。不附的話模型除了
        # 標題欄那幾個欄位就無話可答，使用者問「Insertion Force 是多少」
        # 只能得到「這份圖沒有材料表」，但答案其實就在圖上，只是沒被扛過來。
        # 有材料表的圖框刻意不附：那份 title_raw 混著標題欄雜訊，材料表本身
        # 已經是乾淨的結構化答案，重複塞一份雜訊只會擠掉 thinking 的預算。
        #
        # ★ 清單題也要附，只是截短 —— 不能整段拿掉
        #   舊寫法是 `if with_items and ...`，於是命中超過 3 份圖時這一欄就是
        #   None。而「哪幾份圖的 Insertion force 是 20N」正好同時是清單題**又**
        #   只有這一欄答得出來：材料表可以被截掉（那是「這一份裡有什麼」，清單
        #   題不問這個），但 spec_text 對這幾種圖框就是它的材料表，截掉等於這
        #   一題無解。截短到 _SPEC_LIST_CHARS 是折衷：規格條列是由重要到次要
        #   往下排，前段就涵蓋 Material 與 Performance 的頭幾條。
        "spec_text": (d.get("title_raw") or "").strip()[
                        :_SPEC_MODEL_CHARS if with_items else _SPEC_LIST_CHARS]
                    if d.get("frame") in drawing_frames.NO_BOM_FRAMES
                    else None,
    }


# 工具回傳裡「詳細展開」與「只留一行索引」各給幾份。
#
# ★ 為什麼一定要有索引
#   只回前幾份 + total_drawings 的話，被截掉的圖在 tool_result 裡沒有任何痕跡。
#   跨兩個月的清單題照更新時間新到舊排，前幾份全落在較新的月份，模型就會回
#   「沒有較早那個月的圖」—— 而使用者畫面上就有那個月的卡片。
#   一行索引很短，幾十份也容得進 delegate.py 的預算；材料表才是該被截的東西。
DRAWING_DETAIL = 3          # 展開幾份（清單題只展開標題欄與出處，不含材料表）
DRAWING_LIST_OVER = 3       # 幾份以上算清單題（與 llm.DRAWING_SUMMARY_MIN 同一個門檻）
DRAWING_INDEX = 60          # 索引最多幾份

# 工具回傳的自我預算。
#
# ★ 為什麼是我們自己收，而不是去改 agentkit 的裁剪順序
#   delegate.py 的 RESULT_CHARS 是 6,000：超過就先砍它 `_TRIMMABLE` 裡的清單
#   筆數，砍不動才**字元截斷**——而字元截斷會把 JSON 切在字串中間，實測的
#   後果不是「少講一點」而是整段答案變成數字亂碼（見它的 _fit 註解）。
#   它的 `_TRIMMABLE` 不認得 `drawings` / `drawings_index`，所以這一題會一路
#   掉到字元截斷。但 agentkit 是四個 agent 共用而且**不進本專案版控**
#   （`vendor/` 是複製品，重新 vendor 一次就蓋掉，理由同 src/toolguard.py），
#   所以裁剪寫在這裡：順序由我們決定 —— 材料表先砍，索引留到最後。
TOOL_BUDGET = 5600


def _is_listing(total: int) -> bool:
    """這是「有哪些圖」的清單題嗎。是的話材料表一律不展開 ——
    材料表是下一題（見 llm._detail_count 與 _drawing_payload 的同一段說明）。"""
    return total >= DRAWING_LIST_OVER


def _detail_payloads(dwgs: list[dict]) -> list[dict]:
    """要展開的那幾份。清單題固定三份、只有標題欄與出處；
    圖少的時候照舊給材料表。三份是因為模型被要求「最多舉三份例子」，
    而每個例子都需要可以照抄的檔名、更新時間與連結。"""
    listing = _is_listing(len(dwgs))
    n = DRAWING_DETAIL if listing else len(dwgs)
    return [_drawing_payload(d, with_items=not listing) for d in dwgs[:n]]


def _drawing_index(drawings: list[dict]) -> list[str]:
    """每一份圖一行字串：這一題到底有哪些圖。只有料號與更新時間。

    ★ 為什麼是字串而不是 dict
      dict 的欄名（item_number / file / file_modified_on）每一筆都要重複一次，
      幾十筆下來比三份完整材料表還大 —— 而它承載的資訊只有三格。

    ★ 為什麼不放檔名（理由與 llm._drawing_scope 那段相同）
      索引的職責是證明「那幾份圖存在」，料號加時間就夠。放了檔名，模型就會拿
      索引當引用來源：截短的被原樣抄進答案（使用者看到半截檔名），完整的又會
      被它「修正」（`Z0000XX0XXXX-0_XX_V01.pdf` 的 X 換成同一行的料號，寫出一個
      不存在的檔名，整題的正確答案就被核對整包擋掉）。
      要引用的那幾份在 `drawings` 裡整份展開，檔名與 pdf_url 都是完整的。
    """
    return [
        f"{d.get('item_number')} | {str(d.get('doc_modified_on') or '')[:16] or '更新時間未知'}"
        for d in drawings[:DRAWING_INDEX]
    ]


def _drawing_scope(found: dict, dwgs: list[dict]) -> dict:
    """工具回傳裡與「有哪些圖」有關的那幾格，兩支工具共用。"""
    shown = min(DRAWING_DETAIL if _is_listing(len(dwgs)) else len(dwgs), len(dwgs))
    out: dict[str, Any] = {
        "total_drawings": len(dwgs),
        "detail_shown": shown,
        "drawings_index": _drawing_index(dwgs),
    }
    buckets = found.get("drawing_buckets") or []
    if buckets:
        # 份數是資料庫算出來的，不要模型自己去數 drawings_index（那份有上限）。
        out["time_buckets"] = [
            {"label": b["label"], "start": b["start"], "end": b["end"], "n": b["n"]}
            for b in buckets
        ]
    if found.get("time_query"):
        out["time_filter"] = found["time_query"]["describe"]
    # 檔名裡有一串 X 的通用圖。理由與文案見 llm.PLACEHOLDER_NOTE ——
    # 實測模型會拿同一行的料號去把 X 換掉，寫出一個不存在的檔名。
    if llm._has_placeholder_name(dwgs):
        out["filename_note"] = llm.PLACEHOLDER_NOTE
    if len(dwgs) > shown:
        out["note_truncated"] = (
            f"共 {len(dwgs)} 份圖，其中只有前 {shown} 份附上標題欄與出處"
            "（清單題不展開材料表）；"
            f"其餘 {len(dwgs) - shown} 份在 drawings_index 裡一份一行"
            "（料號 | 檔案更新時間；那份索引刻意不給檔名與網址）。"
            "**index 裡有的就是語料裡有的圖**，time_buckets 份數 >0 的時間區間"
            "一律不可以說「沒有」。\n"
            "這一題要回的是總結：總數 + 各區間份數 + 最多舉三份例子"
            "（每份只寫檔名與檔案更新時間，**不要寫連結** —— index 裡沒有網址，"
            "而網址結尾是資料庫流水號、跟第幾份無關，推算出來的會指到另一份圖）。\n"
            "★ 不要總結這批圖的規格、接頭型號或產品類型 —— 手上只有檔名與更新"
            "時間，沒有這些圖的規格。要看某一份的材料表就用它的料號再呼叫一次。"
        )
    return out


def _size(v: Any) -> int:
    return len(json.dumps(v, ensure_ascii=False, default=str))


def _trim_tool_out(out: dict) -> dict:
    """把工具回傳收在 TOOL_BUDGET 內，**自己決定砍的順序**。

    ★ 順序：材料表 -> 索引尾巴。索引是最後才碰的東西
      被砍掉的圖在模型眼裡就等於不存在（「某個月沒有工程圖」這種幻覺就是
      這樣長出來的），而材料表少一份只是少講一份的細節，使用者再問那個
      料號就拿得到。
    """
    while _size(out) > TOOL_BUDGET and len(out.get("drawings") or []) > 1:
        out["drawings"] = out["drawings"][:-1]
        out["detail_shown"] = len(out["drawings"])
    idx = out.get("drawings_index") or []
    while _size(out) > TOOL_BUDGET and len(out["drawings_index"]) > 8:
        keep = max(8, len(out["drawings_index"]) - 4)
        out["drawings_index"] = out["drawings_index"][:keep]
    dropped = len(idx) - len(out.get("drawings_index") or [])
    if dropped > 0:
        out["index_truncated"] = (
            f"索引只列得下 {len(out['drawings_index'])} 份（篇幅有限），"
            f"另有 {dropped} 份沒有列出來 —— 它們一樣存在，"
            "份數請一律以 time_buckets / total_drawings 為準。"
        )
    return out


def _t_search(a: dict) -> dict:
    found = _find(a["query"], int(a.get("top_k", 10)), a.get("filters") or {})
    dwgs = found.get("drawings") or []
    return _trim_tool_out({
        "query": found.get("query"),
        "total_candidates": found.get("total_candidates"),
        "listing": found.get("listing"),
        "not_found": found.get("not_found") or [],
        # 工程圖一定要跟著回。這一路的 found 本來就帶著它（問句提到工程圖時），
        # 少了它模型會拿「parts 語料裡沒有這顆料」當成「查不到這張圖」——
        # 而圖與它的材料表就在同一次檢索結果裡。
        "drawings": _detail_payloads(dwgs),
        **_drawing_scope(found, dwgs),
        "drawing_note": found.get("drawing_note"),
        # 見 _t_drawings 的同名欄位。
        "drawing_absent": found.get("drawing_absent") or [],
        # 給模型看的精簡列。整列 58 欄塞進 tool_result 會把 thinking 的
        # token 預算擠掉（實測 thinking 本身就要 600~900）。
        "results": [
            {k: _fmt(r.get(k)) for k in (
                "item_number", "name", "category", "length_mm", "pins_total",
                "pitch_series", "awg", "conn_a", "conn_b", "prod_l3",
                "item_type", "eco_std", "state", "score",
            )}
            for r in (found.get("results") or [])
        ],
    })


def _t_get_part(a: dict) -> dict:
    pn = a["item_number"]
    row = retrieval.get_part(pn)
    if row is None:
        near = retrieval.search(pn, {}, 8)
        return {
            "found": False, "item_number": pn,
            "hint": "此料號可能是 PLM 新開、尚未進到語料匯出檔",
            "nearby": [
                {"item_number": r["item_number"], "name": r["name"]}
                for r in near["results"]
            ],
        }
    return {"found": True, "row": row}


def _t_similar(a: dict) -> dict:
    pn = a["item_number"]
    rows = retrieval.find_similar(pn, int(a.get("top_k", 10)))
    return {
        "item_number": pn,
        "results": [
            {k: _fmt(r.get(k)) for k in ("item_number", "name", "length_mm",
                                         "pins_total", "conn_a", "conn_b", "score")}
            for r in rows
        ],
    }


def _t_drawings(a: dict) -> dict:
    """工程圖專用檢索：標題欄 + 材料表 + 出處。

    ★ 為什麼不能叫模型用 search_parts 查圖
      retrieval 的工程圖通道有一道閘門：問句沒提到「工程圖／材料表／BOM」就
      整條路徑不啟動（見 retrieval._wants_drawing，那道閘門是為了擋自由文字的
      材料描述誤命中，不能拿掉）。使用者原句有「工程圖」三個字，但模型改寫成
      工具參數之後通常只剩規格字串（例如 "HSG PH4.2 2*3P V0 nylon66 black
      <廠商料號>"）—— 閘門關著、drawings 是空的，模型於是回「語料內無此料號」，
      而使用者的畫面上正畫著那張圖的材料表。
      這支工具自己把關鍵詞補回去，「呼叫它就是要查圖」這件事不必靠模型記得措辭。
    """
    q = (a.get("query") or "").strip()
    if "工程圖" not in q and "材料表" not in q:
        q = f"{q} 工程圖"
    found = retrieval.search(q, {}, int(a.get("top_k", 5)))
    dwgs = found.get("drawings") or []
    return _trim_tool_out({
        "ok": True,
        "query": q,
        "drawings": _detail_payloads(dwgs),
        **_drawing_scope(found, dwgs),
        # 問句點名了、但語料裡沒有那一份圖的編號。模型看到這個欄位有值時，
        # 唯一正確的回答是「沒有這一份」——不可以拿 drawings 裡編號相鄰的那份代答
        # （見 grounding 的 wrong_drawing_subject）。
        "absent": found.get("drawing_absent") or [],
        "note": found.get("drawing_note") or (
            None if dwgs else
            "沒有找到符合的工程圖。目前只收錄部分料號的工程圖，"
            "查不到不代表這個零件不存在，只代表它的圖不在收錄範圍裡。"
        ),
    })


def _t_count(a: dict) -> dict:
    q = a["query"]
    spec = aggregate.plan(q)
    if not spec:
        return {
            "ok": False,
            "error": "這不是清單／統計型問句",
            "hint": "請用「列出所有…」「…共有幾筆」的問法，或改用 search_parts",
        }
    out = aggregate.run(spec, sample=0)
    return {"listing": out.get("listing"), "query": q}


TOOLS = [
    ToolSpec(
        name="search_parts",
        description=(
            "用自然語言查成品線材料號（partrag 語料）。支援規格條件（長度 600mm、"
            "30AWG、PH4.2 3P、MCIO 8i）、料號、客戶料號反查，以及 PLM 零組件表單"
            "欄位（產品小類、環保標準、建立廠別、公告大類、建立人員）。"
            "★ 這份語料是**成品線材**，不是 BOM 子件。"
            "膠帶、乾燥劑、端子、防塵蓋這類元件請改問「材料建議 Agent」——"
            "BOM 子件的料號幾乎都不在這份語料裡。"
            "回傳的 not_found 列出語料確實沒有的料號，不要把鄰近料號當成它。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "自然語言查詢"},
                "top_k": {"type": "integer", "description": "回幾筆，預設 10"},
            },
            "required": ["query"],
        },
        handler=_t_search,
    ),
    ToolSpec(
        name="get_part",
        description=(
            "用完整料號查該料號的全部欄位（規格 + PLM 表單欄位）。"
            "查不到會回 found=false 與同區段的鄰近料號，"
            "**鄰近料號不是答案**，不要拿它當成查詢的那一個。"
        ),
        input_schema={
            "type": "object",
            "properties": {"item_number": {"type": "string", "description": "完整料號，例 Z0000XX00000-1"}},
            "required": ["item_number"],
        },
        handler=_t_get_part,
    ),
    ToolSpec(
        name="find_similar_parts",
        description=(
            "給一個既有料號，找語料裡最像的其他料號。"
            "用途是「這條線改長度／改 PIN 數有沒有現成的」。"
            "注意：相似是語料層的相似（品名 + 規格欄位），不代表可以互換。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "item_number": {"type": "string"},
                "top_k": {"type": "integer", "description": "預設 10"},
            },
            "required": ["item_number"],
        },
        handler=_t_similar,
    ),
    ToolSpec(
        name="search_drawings",
        description=(
            "查**工程圖**（收錄的 PDF 的標題欄與材料表）。"
            "問「哪一張圖上有 XXX」「這顆料的工程圖」「材料表裡有沒有 YYY」一律用這支，"
            "不要用 search_parts —— search_parts 查的是成品線材語料的規格欄位，"
            "圖上的材料表不在那裡面。"
            "★ 圖上的材料表包含膠帶、標籤、端子、HSG 這類 BOM 子件，"
            "它們雖然不在成品線材語料裡，但**在圖上就是查得到**，"
            "這種問題屬於我自己的能力範圍，不要轉給別的 agent。"
            "回傳每份圖的 TITLE / PART NO. / REV. / 材料表逐列 / 檔名 / 檔案更新時間 / PDF 連結。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "料號、圖上關鍵字或廠商料號，例 Z0000XX00000-1、<廠商料號>"},
                "top_k": {"type": "integer", "description": "預設 5"},
            },
            "required": ["query"],
        },
        handler=_t_drawings,
    ),
    ToolSpec(
        name="count_parts",
        description=(
            "清單／統計型問句的確定性答案（走 SQL，不是排序檢索）。"
            "例「示例廠 的線束共有幾筆」「列出所有公告大類」。"
            "回傳 total 是**全庫符合筆數**，不是抽樣；超過可回傳量時會給 CSV 匯出連結。"
        ),
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "用「列出所有…」「…共有幾筆」的問法"}},
            "required": ["query"],
        },
        handler=_t_count,
    ),
]


SYSTEM = """你是 OEM/ODM 估價流程裡的「相似搜尋建議 Agent」，資料來源是公司 PLM
匯出的**成品線材**語料（partrag）。

你能做的事：用規格／料號／客戶料號／PLM 表單欄位找料號，查單一料號的完整欄位，
找相似料號，回答清單與統計型問題，以及**查工程圖**（標題欄與材料表）。

你必須遵守的限制：

* 這份語料只有**成品線材**。BOM 子件（膠帶、標籤、金屬沖壓件、焊錫、
  塑膠件）幾乎都不在裡面。
  被問到這類元件要直說「不在我的語料範圍」，並指向材料建議 Agent。

* ★ 上一條只適用於**料號語料**，不適用於工程圖。工程圖的材料表上就是印著
  HSG、端子、膠帶、標籤這些子件，「哪一張圖上有 ○○」是我自己答得出來的題目：
  先看手上的檢索結果，沒有就呼叫 search_drawings（不要用 search_parts 查圖，
  那是兩份不同的資料）。真的查不到才說明「收錄的圖裡沒有」，
  而且要講清楚那是圖的收錄範圍問題，不是這個零件不存在。
  手上的檢索結果已經有工程圖時，**直接照它回答**，不要再去問任何人。

* 不要把答得出來的問題推給別的 agent。只有在使用者要的東西確實在別人手上
  （例如報價、開模判斷、BOM 子件的料號建議）時才提一句，而且是補充，
  不是拿來取代回答。

* 「語料裡沒有這個料號」和「找到相似的料號」是兩件事。回傳的 not_found 有值時
  必須明講那個料號查不到，不可以拿鄰近料號代替。

* ★ **使用者沒有說「相關 / 相似 / 類似 / 替代」，就不要端出不符合他條件的東西。**
  查不到就說查不到 —— 那是一個完整的答案，而且是使用者要的那個答案。
  拿一批「看起來很像、但條件對不上」的料號填版面，比空手更糟：他無法一眼看出
  哪幾筆是真的符合，於是整份回答都不能用。要給近似的東西之前先問他。

* ★ 但**上面那條不是拒答的理由**。這兩件事要分清楚：
  - 檢索結果**是空的** -> 說查不到。
  - 檢索結果**有東西** -> 就要拿它回答，一筆都不准漏講。
  使用者說了「相關／相似／替代」時更是如此：那正是他要近似料號的意思，
  這時候端出不完全符合的才是對的，並且要標明哪幾筆完全符合、哪幾筆只是相近。
  手上有 10 筆卻回「資料中沒有符合的」是**錯的答案**，比多給幾筆更糟。

* ★ **檢索結果是一個視窗，不是全部。** 出現「全庫符合…共 N 筆」時，
  第一句就要講出 N，並說明列出的是相關度最高的前幾筆。使用者手上那一顆
  沒有出現在清單裡，往往只是因為它排在第 30 名 —— 講出總數他就知道要
  再加條件，不講他會以為系統查不到。

* ★ **不要把工具回傳的 JSON 當成答案送出去。** 那是給你讀的中間資料。
  回答一律是寫給人看的繁體中文：講結論、附完整料號、必要時列表。

* 欄位值 0 與未填是兩件事。0 是有效值，未填才是沒有資料。

* 成本欄位（mat_cost_usd 等）在這份語料裡大量缺漏，不要用它回答價格問題。

回答用繁體中文，提到某一筆時附上完整料號讓人可以自己去查。"""


# 統一前端把每一條畫成一顆可以直接點的 chip。
# 展示版：範例問句清單（由語料挑出）已移除。
EXAMPLES: list = []


def _context(query: str, found: dict[str, Any]) -> str:
    """工具迴圈要看的「我這次檢索到什麼」。見 llm.build_context()。

    沒有這個鉤子時，工具那條路只拿得到 found["results"]，於是同一句工程圖問句
    勾了「用自己的工具」反而回「語料內無此料號」—— 而卡片上那張材料表是對的。
    工程圖、清單統計、查無料號都不在 results 裡。
    """
    return llm.build_context(query, found)


SPEC = AgentSpec(
    agent_id=AGENT_ID,
    tools=TOOLS,
    system=SYSTEM,
    examples=EXAMPLES,
    search_fn=_find,
    hits_fn=_hits,
    answer_stream=_answer,
    context_fn=_context,
    status_fn=_status,
    can_delegate=False,  # 葉節點：沒有下游可以呼叫
)


def _mount_files(app: Any) -> None:
    """GET /a2a/file/{drawing_id} —— 工程圖 PDF 原檔，瀏覽器直接開。

    ★ 為什麼不能只留 main.py 那支 /api/drawing/file/{id}
      Agent Hub 的代理端點 /hub/file/{agent}/{doc_id} 打的是下游的
      **/a2a/file/{doc_id}**（寫死在 agentkit/client.py 的 file_bytes()，
      不吃各專案自己的原生路徑）。少了這一支，Linux 上那顆「開原圖」按鈕
      就只剩「指回 partrag 自己」一條路，而 partrag 在伺服器上不對外開 port。

    ★ 規則與 /api/drawing/file/{id} 一字不差：檔名只從 DB 的 local_file 取、
      不接受呼叫端傳路徑，取出來還要 resolve() 確認仍在該 profile 的 PDF 目錄
      底下。參數是整數 id 正是為了讓檔名永遠不經過網路。
    """
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    @app.get("/a2a/file/{drawing_id}", tags=["a2a"])
    def drawing_file(drawing_id: int) -> FileResponse:
        with retrieval.pool().connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT local_file, source_profile FROM partrag.drawings WHERE id = %s",
                (drawing_id,),
            )
            row = cur.fetchone()
        if not row:
            raise HTTPException(404, f"找不到工程圖 {drawing_id}")

        # 展示版：公司系統對接（依 source_profile 解析 PDF 目錄）已移除。
        base = (_ROOT / "data" / "pdf").resolve()
        path = (base / row[0]).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            # 「查無這筆」與「這台機器上沒有這個檔案」是兩件事：搬機器時
            # data/pdf 忘了一起帶過來就是走到這裡，訊息要講得出是哪一種。
            raise HTTPException(404, f"工程圖 {drawing_id} 的檔案不在 {base} 底下")
        return FileResponse(
            path,
            media_type="application/pdf",
            headers={"Content-Disposition":
                     f"inline; filename*=UTF-8''{quote(path.name)}"},
        )


def mount(app: Any) -> bool:
    """把 /a2a/* 掛到既有的 app 上。失敗只印警告，主服務照舊。"""
    try:
        # ★ 掛在 include_router **之前**：工具迴圈是在請求進來時才 import 的，
        #   但 handler 的包裝必須在第一個請求之前完成，否則那一題的工具輸出
        #   不會進核對上下文，模型查到的真資料會被判成幻覺。
        toolguard.wrap_handlers(TOOLS)
        if toolguard.install():
            print("[a2a] 工具模式已掛上事實核對（src/toolguard.py）")
        app.include_router(build_router(SPEC, app))
        _mount_files(app)
        return True
    except Exception as e:
        print(f"[a2a] 掛載失敗，partrag 仍照舊運作：{type(e).__name__}: {e}")
        return False
