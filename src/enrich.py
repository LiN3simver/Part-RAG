"""Backfill the PLM columns, and import the PLM master export when it exists.

Two jobs, both idempotent:

1.  Recompute every derived column (產品大類/中類/小類, 料件類型, 環保標準, pitch
    series, pin layout, connector list) and rewrite `search_text` so the PLM
    wording lands in the tsvector. `tsv` is a GENERATED column, so it reindexes
    itself. Embeddings are NOT touched -- re-running the BGE-M3 pass over the
    whole corpus costs hours on CPU and buys nothing here, because the PLM
    vocabulary is matched lexically and by attribute, not semantically.

2.  Import 料件主檔 fields that cannot be derived from a part number and a
    description -- 建立廠別, 公告大類, 批號管控, 保存天數, 成本/工時. Imported
    values are authoritative and overwrite derived ones. Rows in the export that
    are not yet in the corpus are inserted, which is how a part created in PLM
    today becomes searchable before the next full CSV dump.

Run:
    .venv/Scripts/python.exe -m src.enrich                        # backfill only
    .venv/Scripts/python.exe -m src.enrich --plm-csv <plm_export.csv>

匯出檔來自 PLM 端（展示版：公司系統對接已移除），這兩步是
排程的 stage ①② —— 見 src/schedule.py。手動跑這一支跟排程跑的是同一支函式，
差別只有排程會傳 max_reembed 上限進來（理由見 DEFAULT_MAX_REEMBED）。
"""
from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from tqdm import tqdm

from .extract import (
    build_search_text,
    categorize,
    extract_attrs,
    is_placeholder,
    normalize,
    split_item,
)
from . import taxonomy as T

load_dotenv()
PG_DSN = os.environ["PG_DSN"]
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
# ★ 這個預設值必須與 ingest.py / retrieval.py 完全一致（三處都是 256）。
#   它決定截斷長度，也就決定算出來的向量 —— 語料用一個值、查詢用另一個值，
#   兩邊就在不同的向量空間裡比對，而症狀只是「排名怪怪的」，不會報錯。
#   這裡原本寫 96 而另兩處寫 256：目前 .env 有明確設 EMBED_MAX_SEQ=96 所以三者
#   實際相同，但那等於「正確性靠一行環境變數撐著」，換台機器忘了設就靜靜壞掉。
MAX_SEQ = int(os.getenv("EMBED_MAX_SEQ", "256"))
# 模型的 revision 不在這裡定義：載入一律走 retrieval.load_embedder，設定跟著它走。
# 在這裡再放一份 EMBED_REVISION 只會變成沒人讀的死設定 —— 改了它不會有任何效果，
# 而那比沒有更糟。

# item_class is NOT here (unlike item_subclass/unit): it was believed constant
# across the corpus from a single sample row, and a PLM diagnostic later
# proved it actually varies (A 物料/B 零件/C 半成品/D 成品/E 商品). It is now
# purely import-only, like plant/announce_cat -- backfill must leave it alone
# or every re-run would stamp every row back to a wrong constant.
DERIVED_COLS = (
    "search_text plm_text prod_l1 prod_l2 prod_l3 tax_source item_type eco_std "
    "pin_layout pins_total pins_row pitch_series conn_types has_connector "
    "is_coaxial rated_voltage pin_counts length_mm awg gen pitch_mm speed_g "
    "color customer_pn drawing_no conn_a conn_b has_to_pattern "
    "item_subclass unit"
).split()


def derive(item_number: str, name: str) -> dict[str, Any]:
    """Everything computable from (item_number, name) alone."""
    nn = normalize(name)
    a = extract_attrs(name)
    c = T.classify(item_number, nn)
    plm_text = T.build_plm_text(c, a)

    row: dict[str, Any] = {
        "item_number": item_number,
        "search_text": build_search_text(nn, plm_text),
        "plm_text": plm_text,
        "prod_l1": c.l1 if c.l1 != "其他" else None,
        "prod_l2": c.l2 if c.l2 != "其他" else None,
        "prod_l3": c.l3 if c.l3 != "其他" else None,
        "tax_source": c.source,
        "eco_std": a["eco_std"],
        "item_type": a["item_type"],
        "pin_layout": a["pin_layout"],
        "pins_total": a["pins_total"],
        "pins_row": a["pins_row"],
        "pitch_series": a["pitch_series"],
        "conn_types": a["conn_types"],
        "has_connector": a["has_connector"],
        "is_coaxial": a["is_coaxial"],
        "rated_voltage": a["rated_voltage"],
        # recomputed too: the pitch fix (PH=2.54 was reported as PH2.0) and the
        # pin-layout split change these for rows already in the table
        "pin_counts": a["pin_counts"],
        "length_mm": a["length_mm"],
        "awg": a["awg"],
        "gen": a["gen"],
        "pitch_mm": a["pitch_mm"],
        "speed_g": a["speed_g"],
        "color": a["color"],
        "customer_pn": a["customer_pn"],
        "drawing_no": a["drawing_no"],
        "conn_a": a["conn_a"],
        "conn_b": a["conn_b"],
        "has_to_pattern": a["has_to_pattern"],
    }
    # Corpus-wide constants asserted from the PLM form (see CORPUS_CONSTANTS).
    # Written so the form's own fields display, but plm_synced stays false --
    # these were not read from PLM and an import will overwrite them.
    for col, (value, _why) in T.CORPUS_CONSTANTS.items():
        row[col] = value
    return row


def backfill(batch: int = 2000) -> None:
    with psycopg.connect(PG_DSN, autocommit=False) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT item_number, name FROM partrag.parts ORDER BY item_number")
            src = cur.fetchall()
        print(f"[backfill] {len(src)} rows")

        sql = (
            "UPDATE partrag.parts SET "
            + ", ".join(f"{c} = %({c})s" for c in DERIVED_COLS)
            + " WHERE item_number = %(item_number)s"
        )
        with conn.cursor() as cur:
            buf: list[dict] = []
            for r in tqdm(src, desc="derive", unit="row"):
                buf.append(derive(r["item_number"], r["name"]))
                if len(buf) >= batch:
                    cur.executemany(sql, buf)
                    conn.commit()
                    buf.clear()
            if buf:
                cur.executemany(sql, buf)
                conn.commit()
            cur.execute("ANALYZE partrag.parts")
            conn.commit()


# ---------------------------------------------------------------------------
# PLM master import
# ---------------------------------------------------------------------------
# The export's headers are the form's own labels. English aliases are accepted
# so a hand-written CSV works too.
# 展示版：完整的表頭對照表是依實際 PLM 表單整理的，不包含在展示版裡；
# 這裡只留幾筆示意（正式標籤 + 簡體變體 + 英文別名的三種寫法）。
PLM_CSV_MAP: dict[str, str] = {
    "料號": "item_number", "item_number": "item_number",
    "品名及規格": "name", "name": "name",
    # PLM 端存的是使用者 id，匯出時已換成顯示名稱。
    "建立人員": "created_by", "建立人员": "created_by", "created_by": "created_by",
    "建立日期": "created_on", "created_on": "created_on",
}
NUMERIC_COLS = {
    "shelf_days", "lot_size", "std_seconds",
    "mat_cost_usd", "outsource_cost_usd", "labor_cost_usd", "mfg_cost_usd",
}
INT_COLS = {"generation"}
DATE_COLS = {"created_on", "modified_on", "release_date", "effective_date"}

# SQL Server's CONVERT(...,120) gives "2026-04-01 10:33:00"; the PLM form itself
# displays "2026/4/1"; a hand-edited CSV round-tripped through Excel can give
# "2026/4/1 上午 10:33". All three land on the same timestamp.
_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y/%m/%d",
)
_CJK_MERIDIEM = re.compile(r"\s*(上午|下午|AM|PM)\s*", re.I)


def _parse_date(s: str) -> datetime | None:
    """None on anything unparseable -- a bad date must not abort a full import."""
    raw = s.replace("年", "-").replace("月", "-").replace("日", " ").strip()
    pm = bool(re.search(r"下午|PM", raw, re.I))
    raw = _CJK_MERIDIEM.sub(" ", raw).strip()
    raw = re.sub(r"\.\d+$", "", raw)  # drop fractional seconds
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        if pm and dt.hour < 12:
            dt = dt.replace(hour=dt.hour + 12)
        return dt
    return None


def _coerce(col: str, val: Any) -> Any:
    s = str(val).strip()
    if s in ("", "nan", "NaT", "NULL", "N/A", "N／A", "-", "--"):
        return None
    if col in DATE_COLS:
        return _parse_date(s)
    if col in INT_COLS:
        try:
            return int(float(s))
        except ValueError:
            return None
    if col in NUMERIC_COLS:
        try:
            return float(s)
        except ValueError:
            return None
    return s


# The export usually comes out of SSMS on a zh-TW box, which writes Big5 unless
# told otherwise, and pandas' default utf-8 read then dies on the first 中文 cell.
_ENCODINGS = ("utf-8-sig", "utf-8", "cp950", "big5hkscs", "gb18030")


def _read_csv(path: Path) -> pd.DataFrame:
    # ★ pandas 在函式裡才 import
    #   它只被這一支用到（讀 PLM 匯出的 CSV），但 enrich 是 write.py 的相依，
    #   而 write.py 是 main.py 的相依 —— 放在模組層等於「跑一個只回答問題的
    #   API 服務也要先載入 pandas」。Linux 容器裡那是數十 MB，而查詢路徑
    #   一次都不會用到它。
    #   型別註解不受影響：檔案開頭有 from __future__ import annotations，
    #   所以簽名裡的 pd.DataFrame 是字串，不會在 import 時求值。
    import pandas as pd

    last: Exception | None = None
    for enc in _ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False)
        except UnicodeDecodeError as e:
            last = e
            continue
        if enc != "utf-8-sig":
            print(f"[plm] {path.name} 以 {enc} 讀取（非 UTF-8，建議匯出時指定 UTF-8）")
        return df
    raise SystemExit(f"[plm] 無法判斷 {path} 的編碼：{last}")


# 一輪匯入最多重算幾筆 embedding。超過就整批只改文字（reembed=False），並在
# 回傳的 stats 裡講清楚 —— 這個上限是給**無人看著的排程**用的：
#   一次 PLM 大批改名（改編碼規則、批次補上規格）在這裡會變成「載入 BGE-M3
#   然後在 CPU 上算幾千筆」，凌晨兩點沒有人會發現排程跑了六小時還沒結束，
#   而下一輪的 advisory lock 只會安靜地放棄。
# 只改文字是安全的中間狀態：舊向量還在（不是 NULL，見 _apply_rename 的說明），
# dense 通道照常運作，只是那幾筆的語意還是舊的；補救就是人在白天跑一次
# `-m src.enrich --skip-backfill --plm-csv <同一份檔案>`，那條路沒有上限。
# 手動呼叫（CLI）預設 None＝不設限，行為與以前完全一樣。
DEFAULT_MAX_REEMBED = 500


def import_plm(csv_path: Path, reembed: bool = True,
               max_reembed: int | None = None) -> dict[str, Any]:
    df = _read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    unknown = [c for c in df.columns if c not in PLM_CSV_MAP]
    if unknown:
        print(f"[plm] ignoring unmapped columns: {unknown}")
    cols = {c: PLM_CSV_MAP[c] for c in df.columns if c in PLM_CSV_MAP}
    if "item_number" not in cols.values():
        raise SystemExit("[plm] CSV 缺少「料號」欄")

    records: list[dict[str, Any]] = []
    for _, raw in df.iterrows():
        rec: dict[str, Any] = {}
        for src_col, dst in cols.items():
            v = _coerce(dst, raw[src_col])
            if v is not None:
                rec[dst] = v
        if rec.get("item_number"):
            # A CSV that has been through Excel or concatenated by hand can carry
            # a BOM or zero-width space inside a cell, not just at the top of the
            # file. Left in, the part number matches nothing and the row is
            # treated as a brand-new part -- which then fails on the missing 品名.
            rec["item_number"] = (
                rec["item_number"].strip(" \t﻿​ ").upper()
            )
            records.append(rec)
    print(f"[plm] {len(records)} rows from {csv_path}")

    with psycopg.connect(PG_DSN, autocommit=False) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT item_number FROM partrag.parts")
            have = {r[0] for r in cur.fetchall()}

        new = [r for r in records if r["item_number"] not in have]
        upd = [r for r in records if r["item_number"] in have]

        if new:
            missing_name = [r["item_number"] for r in new if not r.get("name")]
            if missing_name:
                raise SystemExit(
                    "[plm] 新料號必須提供「品名及規格」："
                    + ", ".join(missing_name[:5])
                )
            _insert_new(conn, new)

        # 改名要先做：_apply_rename 會依新品名重算分類與屬性，而 _apply_master
        # 帶進來的匯入值是權威的，必須蓋在推導值之上 —— 順序反了，匯入檔給的
        # 產品中類會被依品名推導出來的值覆蓋掉。
        renamed = _renamed(conn, upd) if upd else []
        capped = False
        if renamed:
            if reembed and max_reembed is not None and len(renamed) > max_reembed:
                capped = True
                reembed = False
                print(f"[plm] ⚠ 品名變更 {len(renamed)} 筆超過單輪上限 "
                      f"{max_reembed} —— 這一輪只改文字，embedding 留著舊值。"
                      f"補算：-m src.enrich --skip-backfill --plm-csv {csv_path}")
            print(f"[plm] 品名變更 {len(renamed)} 筆 —— 連 name_norm / search_text /"
                  f" 分類 / 屬性{'' if reembed else '（不含 embedding）'} 一起重算")
            _apply_rename(conn, renamed, reembed=reembed)

        if upd:
            _apply_master(conn, upd, label="update")
        if new:
            # The derived pass runs after insert so the new rows get taxonomy,
            # then the master values are re-applied on top as authoritative.
            _apply_master(conn, new, label="new")
        conn.commit()
    print(f"[plm] inserted={len(new)} updated={len(upd)} renamed={len(renamed)}")
    # 回傳而不是只印：排程要把這幾個數字記進 partrag.sync_state，
    # 「這輪匯進了幾筆」才不是只存在於某個沒人會去翻的 log 檔裡。
    return {"rows": len(records), "inserted": len(new), "updated": len(upd),
            "renamed": len(renamed), "reembed_capped": capped}


def _insert_new(conn: psycopg.Connection, rows: list[dict]) -> None:
    """Insert parts that exist in PLM but not yet in the corpus CSV.

    Embedded here rather than left NULL: a single new row costs one forward pass,
    and without it the part is invisible to the dense channel. Placeholder rows
    (領號用 reservation stubs) are the exception -- ingest.py's bulk pass skips
    them for the same reason the HNSW index is partial and excludes them: they
    carry no real spec, so embedding them burns compute for nothing and their
    only correct entry point is exact item_number, which needs no vector at all.
    A real PLM export can include a brand-new part that IS a placeholder stub
    (e.g. 領號用=><客戶名稱>), so this path must apply the same skip or those
    rows quietly end up in the dense index with everything else.
    """
    # 模型載入走 retrieval.load_embedder（鎖 revision + 先試本機快取），
    # 執行緒數也由那支設在 onnxruntime 的 SessionOptions 上 —— 見該函式的說明。
    from .retrieval import load_embedder

    payload = []
    for r in rows:
        item, name = r["item_number"], r["name"]
        d = derive(item, name)
        base, variant = split_item(item)
        p4, cat = categorize(item)
        d.update(
            {
                "name": name,
                "name_norm": normalize(name),
                "base_pn": base,
                "variant": variant,
                "prefix4": p4,
                "category": cat,
                "is_placeholder": is_placeholder(name),
                "embedding": None,
            }
        )
        payload.append(d)

    targets = [p for p in payload if not p["is_placeholder"]]
    if targets:
        model = load_embedder("cpu")
        model.max_seq_length = MAX_SEQ
        vecs = model.encode(
            [p["name_norm"] for p in targets],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        for p, v in zip(targets, vecs):
            p["embedding"] = v

    cols = list(payload[0].keys())
    sql = (
        f"INSERT INTO partrag.parts ({','.join(cols)}) "
        f"VALUES ({','.join('%(' + c + ')s' for c in cols)}) "
        "ON CONFLICT (item_number) DO NOTHING"
    )
    with conn.cursor() as cur:
        cur.executemany(sql, payload)


def _renamed(conn: psycopg.Connection, rows: list[dict]) -> list[dict]:
    """匯入檔帶了品名、而且與庫裡不同的那些列。

    品名變更以前是**靜默忽略**的：_apply_master 把 name 排除在 SET 之外，而
    backfill() 又從庫裡的舊 name 重新推導，所以 name / name_norm / search_text /
    embedding 全部永遠停在初次入庫的值。PLM 那邊改了品名（例如把長度從 420mm
    改成 450mm），系統會一直用舊規格回答，而且沒有任何症狀 —— 查得到、有分類、
    看起來完全正常。
    """
    wanted = [r["item_number"] for r in rows if (r.get("name") or "").strip()]
    if not wanted:
        return []
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT item_number, name FROM partrag.parts WHERE item_number = ANY(%s)",
            (wanted,),
        )
        old = {r["item_number"]: r["name"] for r in cur.fetchall()}
    out = []
    for r in rows:
        new = (r.get("name") or "").strip()
        prev = old.get(r["item_number"])
        # 前後空白的差異不算改名 —— 匯出檔經過 Excel 一輪就會多出來，
        # 而把它當改名會讓每次匯入都重算一整批 embedding。
        if new and prev is not None and prev.strip() != new:
            out.append({**r, "_old_name": prev})
    return out


def _apply_rename(
    conn: psycopg.Connection, rows: list[dict], reembed: bool = True
) -> None:
    """品名真的變了：改 name，然後把所有從 name 算出來的東西一起重算。

    ★ 為什麼要連 embedding 一起重算，不能只改 name
      dense 通道比對的是 name_norm 的向量。只改 name 而留著舊向量，等於
      「顯示新品名、但語意上還是舊產品」—— 查新規格找不到它，查舊規格反而找得到。
      這比不更新更糟，因為畫面上看起來已經更新了。

    ★ 為什麼 is_placeholder 也要重算
      佔號列（領號用=>客戶名稱）改成實際品名，是這個語料裡真實發生的事 ——
      領了號之後才填規格。不重算的話那顆料號會永遠被排除在語意檢索之外。
      反向（實料改回佔號）同理，這時 embedding 要清成 NULL，與 ingest 的規則一致。

    ★ 成本只落在真的改名的那幾列
      整批重算要在 CPU 上跑好幾個小時；一次匯入通常只有個位數到數十列改名。
      真的碰到大批改名時可以 --no-reembed 先把文字改對，之後單獨補向量。
    """
    payload = []
    for r in rows:
        item, name = r["item_number"], r["name"].strip()
        d = derive(item, name)
        d.update({
            "name": name,
            "name_norm": normalize(name),
            "is_placeholder": is_placeholder(name),
            "embedding": None,
        })
        payload.append(d)

    targets = [p for p in payload if not p["is_placeholder"]] if reembed else []
    if targets:
        from .retrieval import load_embedder

        model = load_embedder("cpu")
        model.max_seq_length = MAX_SEQ
        print(f"[plm/rename] 重算 {len(targets)} 筆 embedding")
        vecs = model.encode(
            [p["name_norm"] for p in targets],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(targets) > 32,
        )
        for p, v in zip(targets, vecs):
            p["embedding"] = v

    cols = [c for c in payload[0] if c != "item_number"]
    if not reembed:
        # 不重算時**不要**把 embedding 寫成 NULL —— 那會讓這幾列從 dense 通道
        # 整個消失。留著舊向量至少還是「舊產品的語意」，而且下一次 --reembed-stale
        # 補得回來；寫 NULL 就只剩人工翻查才找得到。
        cols = [c for c in cols if c != "embedding"]
    sql = (
        "UPDATE partrag.parts SET "
        + ", ".join(f"{c} = %({c})s" for c in cols)
        + " WHERE item_number = %(item_number)s"
    )
    with conn.cursor() as cur:
        cur.executemany(sql, payload)

    for r in rows[:10]:
        print(f"[plm/rename] {r['item_number']}\n"
              f"    舊：{r['_old_name']}\n    新：{r['name'].strip()}")
    if len(rows) > 10:
        print(f"[plm/rename] …其餘 {len(rows) - 10} 筆")


def _apply_master(
    conn: psycopg.Connection, rows: list[dict], label: str, batch: int = 2000
) -> None:
    """Write the imported master values, overriding derived ones.

    Grouped by which columns the row actually carries, so each group goes out as
    one executemany instead of one round trip per row (which is what a full PLM
    export cost before this); an empty cell in the CSV is why the groups
    are needed at all -- _coerce drops it, so not every row has the same keys.
    """
    groups: dict[tuple[str, ...], list[dict]] = {}
    for r in rows:
        # `name` 仍然排除在這裡 —— 但不再是「忽略品名變更」，而是因為改名要連
        # name_norm / search_text / 分類 / 屬性 / embedding 一起重算，那是
        # _apply_rename 的工作。混在這條路裡只改 name 會留下一個
        # 「顯示新品名、語意還是舊產品」的列。`_old_name` 是 _renamed 加的內部欄位。
        sets = {k: v for k, v in r.items()
                if k not in ("item_number", "name", "_old_name")}
        if not sets:
            continue
        sets["plm_synced"] = True
        sets["item_number"] = r["item_number"]
        key = tuple(sorted(k for k in sets if k != "item_number"))
        groups.setdefault(key, []).append(sets)

    with conn.cursor() as cur:
        for cols, payload in groups.items():
            sql = (
                "UPDATE partrag.parts SET "
                + ", ".join(f"{c} = %({c})s" for c in cols)
                + " WHERE item_number = %(item_number)s"
            )
            for i in tqdm(
                range(0, len(payload), batch),
                desc=f"plm/{label}[{len(cols)}欄]",
                unit="batch",
            ):
                cur.executemany(sql, payload[i : i + batch])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plm-csv", type=Path, help="料件主檔匯出 CSV")
    ap.add_argument("--skip-backfill", action="store_true")
    ap.add_argument("--no-reembed", action="store_true",
                    help="品名變更時只改文字，不重算 embedding（大批改名時先跑這個）")
    args = ap.parse_args()

    if not args.skip_backfill:
        backfill()
    if args.plm_csv:
        import_plm(args.plm_csv, reembed=not args.no_reembed)

    with psycopg.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT count(*),
                      count(prod_l2),
                      count(*) FILTER (WHERE prod_l2 IS NULL),
                      count(*) FILTER (WHERE pitch_series <> '{}'),
                      count(*) FILTER (WHERE pin_layout   <> '{}'),
                      count(eco_std), count(item_type),
                      count(*) FILTER (WHERE plm_synced),
                      count(created_by), count(distinct created_by),
                      count(plant)
               FROM partrag.parts"""
        )
        (tot, l2, no_l2, pitch, layout, eco, itype, synced,
         cby, cby_n, plant) = cur.fetchone()
    print(
        f"[done] rows={tot} 有產品中類={l2} 未分類={no_l2} "
        f"pitch={pitch} pin_layout={layout} 環保標準={eco} 料件類型={itype} "
        f"PLM已匯入={synced} 建立人員={cby}（{cby_n} 人）建立廠別={plant}"
    )


if __name__ == "__main__":
    main()
