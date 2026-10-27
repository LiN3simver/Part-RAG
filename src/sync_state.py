"""partrag.sync_state 的讀寫，給 src/schedule.py 與 /api/health 共用。

★ last_run 與 last_success 分開記。候選料號不是每筆都掛得到圖，一輪抓不到
  新 PDF 是常態，不能拿「這輪 rows_done = 0」當失敗、也不能拿「有跑過」當成功
  —— 兩件事要能分開查，否則排程壞掉三週跟正常運作在畫面上長得一樣。
  做法沿用 RFQ-RAG 的 src/sync_state.py。

★ 每一列的鍵是 (stage, source_profile)：測試來源與正式來源各記各的進度，
  切換來源後看到的「上次成功時間」才是這個來源自己的。

advisory_lock() 擋兩輪排程重疊：拿不到鎖就直接放棄，不排隊等 —— OCR 是 CPU
飽和型工作，兩輪一起跑只會互相拖累，不會更快。
"""
from __future__ import annotations

import os
import zlib
from contextlib import contextmanager

from .db import pool

# 兩條線：料號主檔（plm_*，每天都有新料號，成本以分鐘計）與工程圖（其餘三個，
# 要打外部 API 又要跑 OCR，成本以小時計）。順序＝schedule.ALL_STAGES 的執行順序，
# 理由（便宜又關鍵的先跑，別被 OCR 吃光 --budget）見 src/schedule.py 的 docstring。
STAGES = ["plm_export", "plm_import", "candidates", "fetch_pdf", "ingest_drawing"]

# 最終寫入 stage 超過這麼久沒成功，就算「資料過期」：main.py 的 /api/health 標
# ok:false、a2a.py 的 /a2a/health 標 degraded。
# ★ 常數放在這裡而不是各自的模組裡，是因為兩個健康檢查端點如果各存一份門檻，
#   調了一邊沒調另一邊時，看板上會出現「/api/health 說壞了、/a2a/health 說
#   正常」這種自己打自己的狀態，而且沒有人看得出哪一邊才對。
# ★ 這個門檻看的是「排程本身有沒有在跑」（last_success），不是「來源資料有沒有
#   新東西」（watermark）—— 候選料號抓不到新圖是常態，混在一起判斷會把正常
#   運作誤報成過期（理由同本模組開頭 last_run/last_success 分家那段）。
STALE_HOURS = float(os.environ.get("SYNC_STALE_HOURS", "100"))
PARTS_STALE_HOURS = float(os.environ.get("SYNC_PARTS_STALE_HOURS", str(STALE_HOURS)))
DRAWING_STALE_HOURS = float(os.environ.get("SYNC_DRAWING_STALE_HOURS", str(STALE_HOURS)))


def touch_start(stage: str, profile: str) -> None:
    with pool().connection() as conn:
        conn.execute(
            "INSERT INTO partrag.sync_state (stage, source_profile, last_run) "
            "VALUES (%s, %s, now()) "
            "ON CONFLICT (stage, source_profile) DO UPDATE SET last_run = now(), "
            "ok = NULL, note = '執行中', rows_done = 0, rows_failed = 0, rows_skipped = 0",
            (stage, profile),
        )
        conn.commit()


def record_success(stage: str, profile: str, rows_done: int = 0, watermark=None,
                   note: str | None = None, rows_skipped: int = 0) -> None:
    with pool().connection() as conn:
        conn.execute(
            "UPDATE partrag.sync_state SET last_success = now(), rows_done = %s, "
            "ok = true, note = %s, watermark = coalesce(%s, watermark), "
            "rows_failed = 0, rows_skipped = %s, last_finished = now() "
            "WHERE stage = %s AND source_profile = %s",
            (rows_done, note, watermark, rows_skipped, stage, profile),
        )
        conn.commit()


def record_failure(stage: str, profile: str, note: str, *, rows_done: int = 0,
                   rows_failed: int = 0, rows_skipped: int = 0) -> None:
    with pool().connection() as conn:
        conn.execute(
            "UPDATE partrag.sync_state SET ok = false, note = %s, rows_done = %s, "
            "rows_failed = %s, rows_skipped = %s, last_finished = now() "
            "WHERE stage = %s AND source_profile = %s",
            (note, rows_done, rows_failed, rows_skipped, stage, profile),
        )
        conn.commit()


def get_watermark(stage: str, profile: str):
    with pool().connection() as conn:
        row = conn.execute(
            "SELECT watermark FROM partrag.sync_state "
            "WHERE stage = %s AND source_profile = %s", (stage, profile)
        ).fetchone()
    return row[0] if row else None


def report(profile: str | None = None) -> list[dict]:
    """給 /api/health 與 schedule --status 用：每個 stage 現在的狀態。"""
    where, args = "", []
    if profile:
        where, args = "WHERE source_profile = %s", [profile]
    with pool().connection() as conn:
        rows = conn.execute(
            "SELECT stage, source_profile, watermark, last_run, last_success, "
            "rows_done, ok, note, rows_failed, rows_skipped, last_finished, "
            "extract(epoch FROM now() - last_success) / 3600.0 AS hours_since_success "
            f"FROM partrag.sync_state {where} ORDER BY source_profile, stage", args
        ).fetchall()
    cols = ["stage", "source_profile", "watermark", "last_run", "last_success",
            "rows_done", "ok", "note", "rows_failed", "rows_skipped", "last_finished",
            "hours_since_success"]
    return [dict(zip(cols, r)) for r in rows]


def summarize(rows: list[dict], profile: str) -> dict:
    """Terminal stages define freshness; upstream success cannot hide stale data."""
    stages = {}
    for row in rows:
        hours = row.get("hours_since_success")
        stages[row["stage"]] = {
            key: row.get(key) for key in ("last_run", "last_success", "last_finished", "rows_done",
                                         "rows_failed", "rows_skipped", "ok", "note")
        }
        stages[row["stage"]]["hours_since_success"] = float(hours) if hours is not None else None
    freshness = {}
    for capability, stage, threshold in (("parts", "plm_import", PARTS_STALE_HOURS),
                                         ("drawings", "ingest_drawing", DRAWING_STALE_HOURS)):
        hours = stages.get(stage, {}).get("hours_since_success")
        freshness[capability] = {"stage": stage, "hours_since_success": hours,
                                 "stale_threshold_hours": threshold,
                                 "stale": hours is None or hours > threshold}
    failed = [name for name, row in stages.items() if row["ok"] is False]
    running = [name for name, row in stages.items() if row["ok"] is None and row["last_run"] is not None]
    stale_stages = [value["stage"] for value in freshness.values() if value["stale"]]
    hours = [row["hours_since_success"] for row in stages.values() if row["hours_since_success"] is not None]
    detail = []
    if not rows:
        detail.append("排程一次都還沒跑過")
    if failed:
        detail.append("stage 執行失敗：" + "；".join(f"{name}：{stages[name]['note'] or '無說明'}" for name in failed))
    if stale_stages:
        detail.append("資料未更新或已過期：" + "、".join(stale_stages))
    if running:
        detail.append("執行中或尚未完成：" + "、".join(running))
    return {"profile": profile, "ok": not failed and not stale_stages and not running,
            "stale": bool(stale_stages), "stale_threshold_hours": STALE_HOURS,
            "hours_since_last_success": min(hours) if hours else None,
            "freshness": freshness, "failed_stages": failed, "running_stages": running,
            "stages": stages, "note": "；".join(detail) or None}


def _lock_key(name: str) -> int:
    # advisory lock 的 key 是 32-bit signed int，crc32 夠用，不必另外配一張表存 key。
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


@contextmanager
def advisory_lock(name: str):
    """拿不到鎖就 yield False，呼叫端直接收工，不要排隊等下一輪的鎖釋放。"""
    key = _lock_key(name)
    with pool().connection() as conn:
        got = conn.execute("SELECT pg_try_advisory_lock(%s)", (key,)).fetchone()[0]
        try:
            yield got
        finally:
            if got:
                conn.execute("SELECT pg_advisory_unlock(%s)", (key,))
                conn.commit()
