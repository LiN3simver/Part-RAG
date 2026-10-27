"""把 PLM 已經作廢／改號的料號標成作廢 —— 標記，不刪除（is_obsolete 欄位）。

    python -m src.obsolete                                  # 只看現況，什麼都不改
    python -m src.obsolete --stale                           # 列出「沒有建立日期」的可疑列
    python -m src.obsolete --stale --yes                     # 真的標記它們
    python -m src.obsolete --items Z0000XX00000-1,Z0000XX00001-1 --yes
    python -m src.obsolete --items-file data/obsolete.txt --yes
    python -m src.obsolete --missing-from data/plm_export.csv --full-export --yes
    python -m src.obsolete --restore --items Z0000XX00000-1 --yes

★ 三種判準，可靠度差很多，所以是三個不同的旗標而不是一個自動模式

  --stale         `created_on IS NULL`。這一欄由匯出 SQL 的 p.created_on 直接給，
                  所以「這一輪匯入碰得到的列都會有值」。沒有值代表這顆料號在現行
                  PLM 匯出裡完全不存在 —— 抽樣回 PLM 查過
                  都是查無任何版次。這是目前最可靠的判準，但它只抓得到
                  **匯入過至少一次之後才消失**的料號。

  --items         人給的清單。最可靠，也最不會誤傷，但要人先知道是哪幾顆。

  --missing-from  拿一份匯出檔當全集，不在檔裡的就算作廢。**這是最危險的判準**，
                  因為主檔匯出只涵蓋成品，
                  匯出來的本來就只是全庫的一個子集。拿它當全集會把 A 物料、
                  B 零件、C 半成品整批標掉。所以：
                    - 一定要另外加 --full-export，明講「我知道這是全量」
                    - 而且有比例安全閥（見 MAX_RATIO），超過就中止

★ 預設 dry-run。不加 --yes 一定不改，只把「會標什麼」印出來。理由：
  這個動作會讓料號從檢索結果裡消失，而「消失」在使用者那端跟「壞掉」
  長得一模一樣。

★ 標記是可還原的（--restore）。誤標一次就要 restore 一次，這比誤刪要重跑
  整庫向量化好太多 —— 這也是選 soft delete 的主要理由。
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() != "utf-8":
        _stream.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
PG_DSN = os.environ["PG_DSN"]

# --missing-from 一次最多可以標掉全庫的多少比例。超過就中止而不是照做：
# 匯出檔漏了半個庫（SSMS 匯出中斷、WHERE 條件寫錯、檔案只有表頭）跟
# 「這半個庫真的都作廢了」在資料上長得一模一樣，而前者遠比後者常見。
MAX_RATIO = 0.05

REASON_STALE = "現行 PLM 匯出查無此料號（created_on 為空）"
REASON_MISSING = "不在指定的全量匯出檔內"
REASON_MANUAL = "人工指定"


def _counts(cur) -> dict:
    cur.execute(
        """SELECT count(*) AS total,
                  count(*) FILTER (WHERE is_obsolete) AS obsolete,
                  count(*) FILTER (WHERE NOT is_obsolete) AS active,
                  count(*) FILTER (WHERE created_on IS NULL AND NOT is_obsolete)
                      AS stale_candidates
           FROM partrag.parts"""
    )
    return cur.fetchone()


def _stale(cur) -> list[str]:
    cur.execute(
        "SELECT item_number FROM partrag.parts "
        "WHERE created_on IS NULL AND NOT is_obsolete ORDER BY item_number"
    )
    return [r["item_number"] for r in cur.fetchall()]


def _missing_from(cur, csv_path: Path) -> list[str]:
    """料號在庫裡、但不在這份匯出檔裡。

    只讀料號那一欄，用 enrich 的同一組編碼候選與同一套清洗（BOM、零寬空格、
    大寫）—— 兩邊對料號的處理只要差一個字元，整個庫都會被判成作廢。
    """
    from .enrich import PLM_CSV_MAP, _read_csv

    df = _read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    col = next((c for c in df.columns if PLM_CSV_MAP.get(c) == "item_number"), None)
    if not col:
        raise SystemExit(f"[obsolete] {csv_path} 缺少「零組件料號」欄")
    in_csv = {
        str(v).strip(" \t﻿​　").upper()
        for v in df[col]
        if str(v).strip()
    }
    print(f"[obsolete] 匯出檔 {csv_path.name} 有 {len(in_csv)} 個料號")
    cur.execute("SELECT item_number FROM partrag.parts WHERE NOT is_obsolete")
    return sorted(r["item_number"] for r in cur.fetchall()
                  if r["item_number"] not in in_csv)


def _apply(cur, items: list[str], reason: str, restore: bool) -> int:
    if restore:
        cur.execute(
            """UPDATE partrag.parts
               SET is_obsolete = false, obsoleted_on = NULL, obsolete_reason = NULL
               WHERE item_number = ANY(%s) AND is_obsolete""",
            (items,),
        )
    else:
        cur.execute(
            """UPDATE partrag.parts
               SET is_obsolete = true, obsoleted_on = %s, obsolete_reason = %s
               WHERE item_number = ANY(%s) AND NOT is_obsolete""",
            (datetime.now(), reason, items),
        )
    return cur.rowcount


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale", action="store_true",
                    help="標記 created_on 為空的列（現行 PLM 匯出查無）")
    ap.add_argument("--items", help="逗號分隔的料號")
    ap.add_argument("--items-file", type=Path, help="一行一個料號")
    ap.add_argument("--missing-from", type=Path, metavar="CSV",
                    help="不在這份全量匯出檔裡的就算作廢（需搭配 --full-export）")
    ap.add_argument("--full-export", action="store_true",
                    help="確認 --missing-from 給的是全量匯出，不是 D 成品那類子集")
    ap.add_argument("--restore", action="store_true", help="把指定料號改回有效")
    ap.add_argument("--max-ratio", type=float, default=MAX_RATIO)
    ap.add_argument("--yes", action="store_true", help="真的寫入（預設只列出）")
    args = ap.parse_args()

    with psycopg.connect(PG_DSN, autocommit=False) as conn, \
            conn.cursor(row_factory=dict_row) as cur:
        c = _counts(cur)
        print(f"[現況] 全庫 {c['total']} 列：有效 {c['active']}、"
              f"已標作廢 {c['obsolete']}；沒有建立日期的可疑列 {c['stale_candidates']}")

        items: list[str] = []
        reason = REASON_MANUAL
        if args.stale:
            items = _stale(cur)
            reason = REASON_STALE
        if args.items:
            items += [x.strip().upper() for x in args.items.split(",") if x.strip()]
        if args.items_file:
            items += [ln.strip().upper()
                      for ln in args.items_file.read_text(encoding="utf-8").splitlines()
                      if ln.strip() and not ln.startswith("#")]
        if args.missing_from:
            if not args.full_export:
                raise SystemExit(
                    "[obsolete] --missing-from 必須加 --full-export。\n"
                    "  主檔匯出只涵蓋成品，"
                    "匯出來的只是全庫的一個子集；\n"
                    "  拿子集當全集會把 A 物料 / B 零件 / C 半成品整批標成作廢。"
                )
            items += _missing_from(cur, args.missing_from)
            reason = REASON_MISSING

        items = sorted(set(items))
        if not items:
            print("[obsolete] 沒有要處理的料號（沒給判準，或該判準下一筆都沒有）")
            return

        # 安全閥。--items 是人一顆一顆給的，不套這個限制。
        if not args.restore and (args.stale or args.missing_from):
            ratio = len(items) / max(c["total"], 1)
            if ratio > args.max_ratio:
                raise SystemExit(
                    f"[obsolete] 中止：這一輪會標掉 {len(items)} 列 "
                    f"（全庫的 {ratio:.1%}，上限 {args.max_ratio:.0%}）。\n"
                    "  這個數量級通常代表判準本身有問題（匯出檔不完整、"
                    "WHERE 條件寫錯），不是真的有這麼多料號作廢。\n"
                    "  確認過就用 --max-ratio 放寬。"
                )

        verb = "還原" if args.restore else "標記作廢"
        print(f"[obsolete] 準備{verb} {len(items)} 列，理由：{reason}")
        for pn in items[:20]:
            print(f"    {pn}")
        if len(items) > 20:
            print(f"    …其餘 {len(items) - 20} 筆")

        if not args.yes:
            print("\n[dry-run] 什麼都沒改。確認以上清單無誤後加 --yes 執行。")
            return

        n = _apply(cur, items, reason, args.restore)
        conn.commit()
        print(f"[obsolete] 已{verb} {n} 列")
        c = _counts(cur)
        print(f"[現況] 有效 {c['active']}、已標作廢 {c['obsolete']}")


if __name__ == "__main__":
    main()
