"""data/pdf/ 的工程圖 -> partrag.drawings / drawing_items。

Run:
    .venv/Scripts/python.exe -m src.ingest_drawing              # 全量
    .venv/Scripts/python.exe -m src.ingest_drawing --limit 5    # 先看 5 份
    .venv/Scripts/python.exe -m src.ingest_drawing --only Z0000XX00000-1
    .venv/Scripts/python.exe -m src.ingest_drawing --force      # 忽略去重重解析
    .venv/Scripts/python.exe -m src.ingest_drawing --report     # 產人工核對 CSV
    .venv/Scripts/python.exe -m src.ingest_drawing --apply-report  # 收回人工補的編號

人工補編號（OCR 讀不出材料列編號時）：
    --report 會另外產 data/drawing_item_fix.csv，只列出 item_no 是 NULL 的材料列；
    人對著原圖把 item_no_fixed 那一欄填好，--apply-report 收回去。
    收回去的值存在 partrag.drawing_item_fix（獨立一張表），**每次重抽都會自動再
    套一次** —— 因為 ingest() 會把 drawing_items 整批刪掉重插，值放在那張表上
    每重抽一次就會消失一次，而驗證一次抽取規則的改動往往就要重抽好幾輪。

料號一律讀 data/pdf/download_log.csv（由下載端產生；展示版：公司系統對接已移除），
**不從檔名猜** —— PLM 回的檔名跟料號經常毫無關係。doc_modified_on /
file_md5 / 原始檔名也直接沿用 log 的值，不重算：那些是下載當下用 md5 對
PLM 檔案表查出來的，重算只會多一次出錯的機會。

可中斷續跑：去重鍵是 local_file，比對 file_md5，沒變就跳過。OCR 每份要
數十秒，整批動輒一小時以上，不能每次全部重跑。
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from . import drawing, drawing_frames
from .extract import cjk_bigrams
# 展示版：公司系統對接已移除（PDF 下載端與來源設定模組不包含在內）。

for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() != "utf-8":
        _stream.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
PG_DSN = os.environ["PG_DSN"]
ROOT = Path(__file__).resolve().parent.parent
REPORT_CSV = ROOT / "data" / "drawing_extract_report.csv"
# 人工補編號的工作單。--report 產出（只列缺編號的那幾列），人填完
# --apply-report 收回去，存進 partrag.drawing_item_fix。
FIX_CSV = ROOT / "data" / "drawing_item_fix.csv"
FIX_COLS = ("source_profile", "local_file", "item_number",
            "description", "qty_raw", "item_no_fixed", "note")

# 展示版：預設來源名與 PDF 目錄（原本由來源設定模組提供）。
DEFAULT_PROFILE = "default"
DEFAULT_PDF_DIR = Path("data") / "pdf"

# source_profile 排第一：去重鍵是 (source_profile, local_file)，兩個來源各自有
# 一份同名的圖時才不會互相覆蓋。
INSERT_COLS = (
    "source_profile item_number local_file plm_filename file_md5 file_bytes "
    "doc_modified_on "
    "dwg_title dwg_part_no part_no_ok dwg_rev dwg_size dwg_scale "
    "sheet_no sheet_of pd_file_no dwg_date "
    "page_no extract_method ocr_conf title_raw items_raw "
    "frame title_fields search_text embedding"
).split()
CONFLICT_COLS = ("source_profile", "local_file")


def load_log(pdf_dir: Path) -> list[dict]:
    """download_log.csv 是「哪個檔案對應哪個料號」的唯一權威來源。"""
    path = pdf_dir / "download_log.csv"
    if not path.exists():
        print(f"✗ 找不到 {path}（由下載端產生）")
        raise SystemExit(2)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # 展示版：公司系統對接已移除，log 欄位格式檢查（對照下載端的欄名）一併拿掉。
        rows = [r for r in reader if (r.get("本機檔名") or "").strip()]
    return rows


def build_search_text(d: drawing.Drawing) -> str:
    """圖上的字，串成一條給 tsvector / trgm / 向量共用的檢索文字。

    刻意包含 vendor_pn：「哪些料號的工程圖用了 <廠商端子料號> 端子」要靠它命中，
    而廠商料號在 parts 那邊完全不存在（那是廠商的料號，不是自家料號）。

    ★ 客戶圖框多出來的欄位也要進來，否則抽到了等於沒抽
      多料號圖上各料號的長度欄、中文圖框的备注、UNIT 欄 ——
      這些是使用者真的會問的值，而 drawings 這條通道的文字檢索與向量
      **只看 search_text**。抽出來放在 drawing_items 卻沒進這裡的話，
      要嘛整份圖檢索不到，要嘛檢索到了但答案裡沒有那個值。
      title_fields 同理（MATERIAL / P.CODE / CUSTOMER PART NUMBER）。

    ★ 沒有材料表的圖框（`drawing_frames.NO_BOM_FRAMES`）額外收 `title_raw`
      廠商規格書（`Material:` / `Performance:` 那一長串）或材料自由文字
      都不是 drawing_items 裡的列——這些圖框本來就沒有材料表可以扛這些
      內容，值全部只存在 `title_raw`。不收的話「這顆規格的 Insertion Force
      是多少」這種問題永遠查不到，即使 `drawing.extract()` 早就讀出來了。
      有材料表的圖框刻意不收（`title_raw` 混著標題欄雜訊，訊噪比低，而且
      材料表已經有 drawing_items 扛正確答案，重複收一次只會把 tsvector
      灌水、稀釋真正的關鍵字）。
    """
    bits: list[str] = []
    for v in (d.title, d.pd_file_no, d.part_no):
        if v:
            bits.append(str(v))
    bits.extend(v for v in d.title_fields.values() if v)
    for r in d.items:
        if r.description:
            bits.append(r.description)
        bits.extend(r.vendor_pn)
        bits.extend(v for v in (r.unit, r.remark, r.color, r.length_raw) if v)
    if d.frame in drawing_frames.NO_BOM_FRAMES and d.title_raw:
        bits.append(d.title_raw)
        bits.append(deglue_enum(d.title_raw))

    return _with_bigrams(" ".join(bits))


# OCR 把條列標號跟後面的第一個字黏成一個 token：`a.Insertion` / `2.RJ45` /
# `3-1.Material`。CAD 圖上這是排版問題，但進了 tsvector 就是檢索問題 ——
# PostgreSQL 的 parser 把 `a.Insertion` 判成一個 host/file 型 token
# `a.insertion`，於是 `websearch_to_tsquery('simple','insertion force')` 對這
# 份圖**回零列**。search_text 裡明明有這個詞、
# 使用者也照著圖上抄，就是查不到，而且沒有任何錯誤訊息。
#
# ★ 只**追加**拆開後的詞，不動原文
#   跟 `_with_bigrams()` 同一個原則：原文是給 word_similarity 與 embedding
#   看的，改寫它等於改掉另外兩條通道的輸入。而且「什麼時候該拆」沒有安全的
#   通則 —— `CAT.6` 拆開就毀了型號、`SIGN.R` 拆開簽核人就對不回去。追加的
#   話拆錯最多只是多幾個沒人查的 token，拆漏了也維持現狀，兩邊都不會退步。
#
# ★ 標號限 3 碼、後面的字至少 3 碼，是為了不去碰型號與日期
#   `CAT.6` 的 `.6` 開頭是數字（要求 `[A-Za-z]` 起頭）擋掉；`SIGN.R` 的尾巴
#   只有 1 碼擋掉。剩下會命中的 `22.MAY`（日期）拆開無害 —— 它只是多放出一個
#   `MAY` token，而 `22.MAY.2010` 原文照樣留著。
_GLUED_ENUM = re.compile(
    r"(?<![0-9A-Za-z.])([0-9A-Za-z]{1,3}(?:-[0-9]{1,2})?)\.([A-Za-z][0-9A-Za-z]{2,})")


def deglue_enum(text: str | None) -> str:
    """`a.Insertion` -> `a. Insertion`，只回被拆開的那幾段，原文不含在內。"""
    if not text:
        return ""
    return " ".join(f"{m.group(1)}. {m.group(2)}"
                    for m in _GLUED_ENUM.finditer(text))


def _with_bigrams(blob: str) -> str:
    """原文 + 中文字元二元組，與 `extract.build_search_text()`（parts 用的那支）
    同一個做法。

    ★ 沒有這一段的話，工程圖對中文是全盲的
      `drawings.tsv` 是 `to_tsvector('simple', search_text)`，而 PostgreSQL 的
      simple parser 把一整串中文當成**一個 token**。所以材料表上寫著
      「香蕉头端子」，查「香蕉头端子」卻一份都命不中（ILIKE 命中得到，
      tsv 與 trigram 都是 0 —— pg_trgm 對中文同樣沒有鑑別力）。
      parts 早就這樣做了，drawings 這條線漏掉，而客戶圖的材料表**大半是中文**。
    """
    blob = (blob or "").strip()
    if not blob:
        return ""
    return " ".join([blob, *cjk_bigrams(blob)])


def embed(text: str):
    """跟 parts 同一顆 BGE-M3（本機 CPU，資料不出公司）。

    模型載不起來就回 None —— 少一個向量只是少一條通道，不該讓整批 OCR
    的結果落不了地。與 retrieval._by_vector 同一個降級原則。
    """
    if not text.strip():
        return None
    try:
        from .retrieval import embed_query

        return embed_query(text[:2000])
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ 向量產生失敗（{type(e).__name__}: {e}），embedding 留空")
        return None


def _ts(v: str) -> str | None:
    return v.strip() or None


def existing(cur, profile: str) -> dict[str, str]:
    cur.execute("SELECT local_file, coalesce(file_md5,'') FROM partrag.drawings "
                "WHERE source_profile = %s", (profile,))
    return {r[0]: r[1] for r in cur.fetchall()}


def ingest(rows: list[dict], pdf_dir: Path, force: bool, profile: str,
           deadline: float | None = None) -> tuple[int, int, int]:
    """deadline 是 time.time() 的絕對時間戳（排程的 --budget 用）。

    OCR 每份數十秒，整批動輒一小時以上 —— 沒有時間上限的話，排程會壓在深夜
    的 CPU 上跑到天亮。停在整份的邊界，每份入庫後就 commit，下一輪靠 md5
    去重自然接續（這跟原本的可中斷續跑是同一個機制）。
    """
    done = skipped = failed = 0
    with psycopg.connect(PG_DSN, autocommit=False) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            have = {} if force else existing(cur, profile)

            for i, r in enumerate(rows, 1):
                if deadline and time.time() > deadline:
                    print(f"  ⏱ 時間預算用完，停在 {i}/{len(rows)}，"
                          f"剩下的留給下一輪")
                    break
                local = r["本機檔名"].strip()
                path = pdf_dir / local
                pn = r["料號"].strip()

                if not path.exists():
                    print(f"[{i}/{len(rows)}] {pn} → 檔案不存在（{local}），略過")
                    failed += 1
                    continue
                # 去重比的是 md5，不是「有沒有這一列」—— 圖改版重下之後
                # 檔名可能一樣但內容不同，只看檔名會把新版當成已經做過。
                if local in have and have[local] == r["md5"].strip():
                    skipped += 1
                    continue

                t = time.time()
                try:
                    d = drawing.extract(str(path), pn)
                except Exception as e:  # noqa: BLE001 -- 一份壞掉不該停整批
                    print(f"[{i}/{len(rows)}] {pn} → 解析失敗 {type(e).__name__}: {e}")
                    failed += 1
                    continue
                if d is None:
                    print(f"[{i}/{len(rows)}] {pn} → 找不到圖框頁，略過")
                    failed += 1
                    continue

                text = build_search_text(d)
                vals = [
                    profile,
                    pn, local, _ts(r.get("PLM原始檔名", "")), _ts(r.get("md5", "")),
                    int(r["bytes"]) if (r.get("bytes") or "").strip().isdigit() else None,
                    _ts(r.get("檔案更新時間", "")),
                    d.title, d.part_no, d.part_no_ok, d.rev, d.size, d.scale,
                    d.sheet_no, d.sheet_of, d.pd_file_no, d.dwg_date,
                    d.page_no, d.method, d.ocr_conf, d.title_raw, d.items_raw,
                    d.frame, Jsonb(d.title_fields),
                    text, embed(text),
                ]
                ph = ",".join(["%s"] * len(INSERT_COLS))
                cur.execute(
                    f"INSERT INTO partrag.drawings ({','.join(INSERT_COLS)}) "
                    f"VALUES ({ph}) "
                    # 重跑時覆蓋既有那一列。ON CONFLICT DO NOTHING 會讓
                    # --force 完全沒有效果，而且是安靜地沒有效果。
                    f"ON CONFLICT ({','.join(CONFLICT_COLS)}) DO UPDATE SET "
                    + ",".join(f"{c}=EXCLUDED.{c}" for c in INSERT_COLS
                               if c not in CONFLICT_COLS)
                    + ", ingested_at = now() RETURNING id",
                    vals,
                )
                did = cur.fetchone()[0]
                cur.execute("DELETE FROM partrag.drawing_items WHERE drawing_id = %s",
                            (did,))
                for row in d.items:
                    cur.execute(
                        "INSERT INTO partrag.drawing_items "
                        "(drawing_id,item_no,description,qty_raw,qty_num,"
                        " vendor_pn,kind,ocr_conf,"
                        " table_tag,unit,remark,color,length_raw) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (did, row.item_no, row.description, row.qty_raw,
                         drawing.qty_num(row.qty_raw), row.vendor_pn, row.kind,
                         round(row.conf, 4),
                         row.table_tag, _ts(row.unit), _ts(row.remark),
                         _ts(row.color), _ts(row.length_raw)),
                    )
                # 人工核對過的編號在這裡補回來 —— 上面剛把 drawing_items
                # 整批刪掉重插，不套的話每重抽一次就要重核一次。
                apply_fixes(cur, profile, did, local)
                conn.commit()
                done += 1
                flag = "" if d.part_no_ok else "  ⚠ PART NO. 未與 PLM 料號核對一致"
                print(f"[{i}/{len(rows)}] {pn} → {len(d.items)} 列材料、"
                      f"{d.method}、conf={d.ocr_conf}、{time.time()-t:.0f}s{flag}")
    return done, skipped, failed


# ---------------------------------------------------------------------------
# 人工補編號
# ---------------------------------------------------------------------------
# 描述的正規化。SQL 端與 Python 端必須算出同一個字串，不然人工值會對不上而且
# 不會報錯 —— 只是那一列永遠套不上，看起來像「填了沒用」。
_DESC_KEY_SQL = "upper(regexp_replace(coalesce(description,''), '[^0-9A-Za-z]', '', 'g'))"


def _desc_key(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", s or "").upper()


def apply_fixes(cur, profile: str, drawing_id: int, local_file: str) -> int:
    """把這份圖既有的人工編號套回剛重插的材料列。

    ★ 每次入庫都要呼叫，不是只在 --apply-report 的時候。ingest() 重抽時會把
      drawing_items 整批刪掉重插，人工值不在這裡補回來，就等於每重抽一次
      就要重核一次 —— 那這整套東西沒有意義。
    """
    cur.execute(
        f"""UPDATE partrag.drawing_items i
               SET item_no = f.item_no
              FROM partrag.drawing_item_fix f
             WHERE i.drawing_id = %s
               AND f.source_profile = %s
               AND f.local_file = %s
               AND f.desc_key = {_DESC_KEY_SQL.replace('description', 'i.description')}
               AND i.item_no IS DISTINCT FROM f.item_no""",
        (drawing_id, profile, local_file),
    )
    return cur.rowcount


def write_fix_csv(cur, out: Path) -> int:
    """缺編號的材料列，一列一行，`item_no_fixed` 留白給人填。

    只列 item_no IS NULL 的列 —— 這張表是工作單，不是資料匯出。把全部材料列
    都倒出來，真正要補的那幾列就淹掉了。
    """
    cur.execute(
        """SELECT d.source_profile, d.local_file, d.item_number,
                  i.description, i.qty_raw
             FROM partrag.drawing_items i
             JOIN partrag.drawings d ON d.id = i.drawing_id
            WHERE i.item_no IS NULL
            ORDER BY d.source_profile, d.local_file, i.id"""
    )
    rows = cur.fetchall()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIX_COLS)
        for r in rows:
            w.writerow([*r, "", ""])       # item_no_fixed / note 留白
    return len(rows)


def apply_report(path: Path) -> int:
    """讀人填好的工作單 -> 存進 drawing_item_fix -> 立刻套用到 drawing_items。

    ★ 只吃有填 `item_no_fixed` 的列，其他整列跳過（人只補得動幾列很正常）。
    ★ 存一份到 drawing_item_fix 才是重點：之後每一次重抽都會自動再套一次
      （見 apply_fixes），人不必再核對第二遍。
    """
    if not path.exists():
        print(f"✗ 找不到 {path}\n"
              f"  先跑：.venv/Scripts/python.exe -m src.ingest_drawing --report")
        return 2
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and set(FIX_COLS) - set(rows[0]):
        print(f"✗ {path} 少了欄位：{sorted(set(FIX_COLS) - set(rows[0]))}\n"
              f"  這張表要用 --report 產出來的那一份，不要自己重打欄位標題")
        return 2

    todo, bad = [], 0
    for r in rows:
        raw = (r.get("item_no_fixed") or "").strip()
        if not raw:
            continue
        if not raw.isdigit():
            print(f"  ⚠ item_no_fixed 不是整數，略過：{raw!r}"
                  f"（{(r.get('local_file') or '')[:40]}）")
            bad += 1
            continue
        todo.append((r["source_profile"].strip(), r["local_file"].strip(),
                     _desc_key(r.get("description") or ""), int(raw),
                     (r.get("note") or "").strip() or None))
    if not todo:
        print(f"  {path} 裡沒有填任何 item_no_fixed，沒有東西可以套用")
        return 0

    with psycopg.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO partrag.drawing_item_fix
                   (source_profile, local_file, desc_key, item_no, note)
               VALUES (%s,%s,%s,%s,%s)
               ON CONFLICT (source_profile, local_file, desc_key) DO UPDATE
                   SET item_no = EXCLUDED.item_no,
                       note    = EXCLUDED.note,
                       fixed_at = now()""",
            todo,
        )
        # 立刻套用，不要等下一次重抽 —— 人剛填完就該看得到結果。
        applied = 0
        for profile, local in {(t[0], t[1]) for t in todo}:
            cur.execute(
                "SELECT id FROM partrag.drawings "
                "WHERE source_profile = %s AND local_file = %s", (profile, local)
            )
            for (did,) in cur.fetchall():
                applied += apply_fixes(cur, profile, did, local)
        conn.commit()

    print(f"✓ 收下 {len(todo)} 筆人工編號，實際套用到 {applied} 列"
          + (f"，{bad} 筆格式不對已略過" if bad else ""))
    if applied < len(todo):
        print("  套不上的那幾筆是描述對不上（重抽後判讀變了）——"
              " 再跑一次 --report 看它們還在不在工作單上")
    return 0


def report(out: Path) -> int:
    """每份圖一列的人工核對表。

    存在的理由是「OCR 的精度上限只能被量測，不能被消除」：O/0 混淆與被吃掉的
    連字號都還在，part_no_ok / ocr_conf / 材料列數三欄合起來就是一張
    「哪幾份要調原圖來看」的清單。
    """
    with psycopg.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT d.item_number, d.local_file, d.dwg_part_no, d.part_no_ok,
                      d.dwg_title, d.dwg_rev, d.dwg_size, d.dwg_scale,
                      d.sheet_no, d.sheet_of, d.pd_file_no,
                      d.extract_method, d.ocr_conf, d.page_no,
                      d.doc_modified_on,
                      count(i.id) AS n_items,
                      count(*) FILTER (WHERE i.item_no IS NULL) AS n_unnumbered,
                      count(*) FILTER (WHERE i.vendor_pn <> '{}') AS n_vendor,
                      -- 擺最後而不是最前：前面幾欄的位置被 report() 的
                      -- r[3] / r[15] 直接索引，插在中間會安靜地錯位。
                      d.source_profile
               FROM partrag.drawings d
               LEFT JOIN partrag.drawing_items i ON i.drawing_id = d.id
               GROUP BY d.id
               ORDER BY d.part_no_ok, d.ocr_conf NULLS FIRST, d.item_number"""
        )
        rows = cur.fetchall()
        cols = [c.name for c in cur.description]

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)

    bad = sum(1 for r in rows if not r[3])
    empty = sum(1 for r in rows if not r[15])
    print(f"✓ {len(rows)} 份寫入 {out}")
    print(f"  PART NO. 未核對一致：{bad} 份（這幾份的料號欄要看原圖）")
    print(f"  完全沒抽到材料列：{empty} 份")

    # 缺編號的那幾列另外一張工作單。兩張分開：上面那張是「哪幾份要調原圖」，
    # 這張是「哪幾列要填什麼」，填完用 --apply-report 收回去。
    with psycopg.connect(PG_DSN) as conn, conn.cursor() as cur:
        n_fix = write_fix_csv(cur, FIX_CSV)
    print(f"  缺編號的材料列：{n_fix} 列 -> {FIX_CSV}")
    if n_fix:
        print("     填好那一欄 item_no_fixed 之後："
              ".venv/Scripts/python.exe -m src.ingest_drawing --apply-report")
    return 0


def rebuild_search_text() -> int:
    """把既有 drawings 的 search_text 依現在的規則重算（tsv 是 GENERATED，會跟著更新）。

    ★ 為什麼可以不重跑 OCR
      search_text 的每一個來源都已經是資料庫裡的欄位：標題欄三欄在 drawings，
      材料表的描述與廠商料號在 drawing_items。重解析一份 PDF 要 OCR 一分鐘上下，
      重算字串是一次 UPDATE —— 而 `_with_bigrams()` 這種「檢索文字的組法變了」
      的改動之後一定還會有，每次都重跑一小時 OCR 是沒有道理的。

    embedding 刻意**不動**：它是 BGE-M3 對原文算的，bigram 是給 tsvector 用的
    人工 token，塞進向量只會讓語意空間變髒。
    """
    load_dotenv()
    with psycopg.connect(os.environ["PG_DSN"]) as conn, conn.cursor() as cur:
        cur.execute(
            # 這一段的欄位必須跟 build_search_text() 一模一樣，包含客戶圖框
            # 多出來的那幾欄（unit / remark / color / length_raw / title_fields
            # ）—— 兩邊漂掉的症狀是「重算之後某些值查不到了」。
            #
            # ★ spec：沒有材料表的圖框要收 title_raw，條件與 build_search_text()
            #   最後那三行一字不差。這一欄漏掉的話，規格書的 Material /
            #   Performance 全部只存在 title_raw，重算一次就從 search_text
            #   消失，而 `drawings` 這條通道的 tsv 與 trgm 都只看 search_text
            #   —— 圖還在、欄位還在、查詢照打，就是零命中，不報任何錯。
            #
            # ★ 每一欄都包 nullif(...,'')，不是只靠 concat_ws 跳過 NULL
            #   concat_ws 跳過的是 NULL，**空字串它照放**，於是每一個空欄都在
            #   結果裡留下一個多餘空白。Python 那邊是 `if v` 濾掉再 join，兩邊
            #   因此永遠對不齊：幾乎每份圖都只差在空白數量，token 一個不差。
            #   後果不是查錯，是每跑一次重算都回報「幾乎全部已更新」——
            #   那個數字本來是用來看「這次改動影響了誰」的，全部都動等於沒有訊息。
            """SELECT d.id,
                      concat_ws(' ', nullif(d.dwg_title, ''),
                                nullif(d.pd_file_no, ''), nullif(d.dwg_part_no, ''),
                                (SELECT string_agg(v, ' ')
                                   FROM jsonb_each_text(d.title_fields) AS t(k, v)
                                  WHERE nullif(v, '') IS NOT NULL)
                               ) AS head,
                      coalesce(string_agg(
                          nullif(concat_ws(' ', nullif(i.description, ''),
                                    nullif(array_to_string(i.vendor_pn, ' '), ''),
                                    nullif(i.unit, ''), nullif(i.remark, ''),
                                    nullif(i.color, ''), nullif(i.length_raw, '')), ''),
                          ' ' ORDER BY i.table_tag, i.item_no), '') AS body,
                      CASE WHEN d.frame = ANY(%s) THEN d.title_raw END AS spec
                 FROM partrag.drawings d
                 LEFT JOIN partrag.drawing_items i ON i.drawing_id = d.id
                GROUP BY d.id, d.dwg_title, d.pd_file_no, d.dwg_part_no,
                         d.title_fields, d.frame, d.title_raw""",
            (sorted(drawing_frames.NO_BOM_FRAMES),),
        )
        rows = cur.fetchall()
        n = 0
        for did, head, body, spec in rows:
            text = _with_bigrams(
                " ".join(x for x in (head, body, spec, deglue_enum(spec)) if x).strip())
            cur.execute(
                "UPDATE partrag.drawings SET search_text = %s WHERE id = %s"
                "  AND search_text IS DISTINCT FROM %s",
                (text, did, text),
            )
            n += cur.rowcount
        conn.commit()
    print(f"✓ search_text 重算完成：{len(rows)} 份圖，實際更新 {n} 份")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="工程圖標題欄 + 材料表批次入庫")
    ap.add_argument("--rebuild-search-text", action="store_true",
                    help="不碰 PDF，只用資料庫既有欄位重算 search_text（換了檢索"
                         "文字的組法時用，例如補中文 bigram）")
    ap.add_argument("--profile", default=None,
                    help="來源設定名，入庫時標成哪個來源")
    ap.add_argument("--limit", type=int, default=0, help="只處理前 N 份")
    ap.add_argument("--only", default="", help="只處理這個料號")
    ap.add_argument("--force", action="store_true", help="忽略去重，重新解析")
    ap.add_argument("--budget-min", type=float,
                    help="最多跑幾分鐘就停，剩下的留給下一輪（排程用）")
    ap.add_argument("--report", action="store_true", help="不入庫，只產人工核對 CSV")
    ap.add_argument("--apply-report", action="store_true",
                    help=f"讀 {FIX_CSV.name} 裡人填好的 item_no_fixed，"
                         "寫回資料庫並記住（之後重抽會自動再套用）")
    ap.add_argument("-o", "--outdir", default=None)
    args = ap.parse_args()

    if args.rebuild_search_text:
        return rebuild_search_text()
    if args.report:
        return report(REPORT_CSV)
    if args.apply_report:
        return apply_report(FIX_CSV)

    # 展示版：公司系統對接已移除，來源設定改成固定預設值。
    profile = args.profile or DEFAULT_PROFILE
    pdf_dir = Path(args.outdir) if args.outdir else DEFAULT_PDF_DIR
    if not pdf_dir.is_absolute():
        pdf_dir = ROOT / pdf_dir

    rows = load_log(pdf_dir)
    if args.only:
        want = args.only.strip().upper()
        rows = [r for r in rows if r["料號"].strip().upper() == want]
        if not rows:
            print(f"✗ download_log.csv 裡沒有料號 {args.only}")
            return 2
    if args.limit:
        rows = rows[: args.limit]

    if not drawing.pdf_ocr.available():
        print("⚠ 沒有 rapidocr，只有帶文字層的圖抽得出來（多數工程圖是掃描檔）")
        print("  .venv/Scripts/python.exe -m pip install rapidocr_onnxruntime")

    t = time.time()
    print(f"→ {len(rows)} 份待處理，來源 {pdf_dir}（profile={profile}）\n")
    deadline = time.time() + args.budget_min * 60 if args.budget_min else None
    done, skipped, failed = ingest(rows, pdf_dir, args.force, profile, deadline)
    print(f"\n✓ 入庫 {done} 份，略過（內容未變）{skipped} 份，失敗 {failed} 份，"
          f"共 {time.time()-t:.0f} 秒")
    print(f"  人工核對表：.venv/Scripts/python.exe -m src.ingest_drawing --report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
