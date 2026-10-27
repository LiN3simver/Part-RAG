"""排程主入口：料號主檔與工程圖兩條線，一輪跑完五個 stage。

    python -m src.schedule                          # 五個 stage 都跑一輪
    python -m src.schedule --profile prod           # 換一組來源設定
    python -m src.schedule --stages plm_export,plm_import
    python -m src.schedule --budget 180             # 這輪最多跑 180 分鐘就收工
    python -m src.schedule --status                 # 只看各 stage 的進度，不執行
    python -m src.schedule --dry-run                # 只列這輪會跑哪些，不執行
    python -m src.schedule --daily-at 01:00         # 常駐：睡到每天 01:00 跑一輪

觸發者有兩個，跑的是同一支：
    Windows   工作排程器 -> 批次檔 -> 這支跑一輪就結束
    Linux/容器 compose 的 scheduler service -> 這支帶 --daily-at 常駐
理由與時區的坑見 daily() 上面那段。

五個 stage 就是手動那五步，一模一樣的函式：

  ── 料號主檔（每天都可能有新料號／改品名，這條線的成本以分鐘計）──
    ① plm_export      PLM 主檔匯出 -> data/plm_export*.csv
                      （原子替換 + 筆數安全閥）
    ② plm_import      該 CSV -> partrag.parts（enrich.import_plm，冪等）

  ── 工程圖（要打外部 API 又要跑 OCR，這條線的成本以小時計）──
    ③ candidates      MSSQL 跑候選 SQL -> data/<profile>/plm_pdf_candidates.csv
    ④ fetch_pdf       逐筆打工程圖下載 API -> PDF + download_log.csv
                      （續跑靠 log）
    ⑤ ingest_drawing  PDF -> partrag.drawings / drawing_items
                      （ingest_drawing.ingest，續跑靠 md5）

這支只負責「呼叫誰、跑多久、失敗了怎麼辦」，不重新實作任何邏輯 —— 每個 stage
的行為與各自單獨執行完全一樣。做法對齊 RFQ-RAG 的 src/schedule.py。

★ 為什麼主檔那兩步排在工程圖前面
  --budget 是整輪的總預算，用完就停在下一個 stage 之前。工程圖那條線的 OCR
  每份 30~45 秒，很容易把整份預算吃光；排在後面的 stage 於是「每一輪都輪不到」。
  而主檔匯入是這個系統最基本的新鮮度來源（新料號、改品名、改成本），成本只有
  幾分鐘 —— 讓便宜又關鍵的那條線先跑完，昂貴又可續跑的那條線用剩下的預算。
  兩條線之間沒有資料依賴，順序純粹是預算分配。

★ 一個 stage 炸掉要記下來然後跑下一個，不能整輪中止。兩組例外，各自只影響
  自己那條線（早期版本是直接 break，那會讓工程圖連不上庫時連帶跳過主檔匯入）：
    plm_export 失敗 -> 跳過 plm_import  （匯出檔沒更新，重匯一次同一份檔案
                                          不會有新東西，只是白白重寫整份主檔）
    candidates 失敗 -> 跳過 fetch_pdf / ingest_drawing
                                         （拿舊 CSV 去打 API 只會重抓已經抓過
                                          的料號，白花額度）

★ 用 advisory lock 擋兩輪排程重疊：拿不到鎖就直接結束，不排隊等 —— OCR 是
  CPU 飽和型工作，兩輪一起跑只會互相拖累，不會更快。--status / --dry-run
  不需要拿鎖（不寫入任何東西）。

★ --budget 是整輪的總預算，不是每個 stage 各自的。真正會吃掉預算的是
  ④（每筆 API 0.3 秒起跳）跟 ⑤（OCR 每份 30~45 秒），兩者都收 deadline，
  用剩下的預算依序往下切；到點就停在整份的邊界，剩下的留給下一輪。
  ①②**不收 deadline**：一次匯入是一個 transaction，切一半沒有意義，
  而它整體只要幾分鐘，不是會撐爆預算的那一種。

★ 這支**不會**刪任何東西，也不會標作廢。「PLM 查不到的料號要不要標作廢」是
  src/obsolete.py 的事，它預設 dry-run 且判準的可靠度差很多（--missing-from
  最危險），刻意不接進排程 —— 每天自動跑一次「拿匯出檔當全集」的判準，等於
  每天給自己一次把整個庫標掉的機會。「確認正式資料進得來之後再刪測試資料」
  同理，刻意留給人手動執行。
"""
from __future__ import annotations

import argparse
import os
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from typing import Any

from . import ingest_drawing, sync_state

# 展示版：公司系統對接已移除 —— 來源設定（PLM 主檔匯出、候選 SQL、工程圖下載
# API 的連線資訊）與對應的匯出／下載模組不在展示範圍內。Source 只留型別名。
Source = Any


def _load_source(profile: str | None) -> Source:
    raise NotImplementedError("展示版：公司系統對接已移除")

LOCK_NAME = "partrag_schedule"
ALL_STAGES = ["plm_export", "plm_import",
              "candidates", "fetch_pdf", "ingest_drawing"]

# 某個 stage 失敗時，同一條線上還沒跑的哪些要跳過。見模組 docstring。
SKIP_ON_FAILURE = {
    "plm_export": ["plm_import"],
    "candidates": ["fetch_pdf", "ingest_drawing"],
}


def _remaining_min(deadline: float | None) -> str:
    if not deadline:
        return "無上限"
    return f"剩 {max(0.0, (deadline - time.time()) / 60):.0f} 分鐘"


@dataclass
class StageResult:
    done: int
    failed: int = 0
    skipped: int = 0
    complete: bool = True
    note: str | None = None


def _run_stage(name: str, profile: str, fn) -> int | None:
    """統一的「記錄開始 -> 跑 -> 記錄成功/失敗 -> 繼續下一個」外殼。

    fn 回傳筆數，或 (筆數, 給 --status 看的一句話)。
    回傳這輪處理了幾筆；失敗回 None（呼叫端用它判斷要不要往下跑）。

    ★ SystemExit 也要接住。enrich 這類模組是先寫成 CLI 的，遇到
      「CSV 缺欄位」「新料號沒有品名」一律 raise SystemExit —— 那是 BaseException
      的子類，不在 `except Exception` 裡。漏接的話整個排程行程直接結束，
      後面的 stage 一個都不跑，而 sync_state 上只留著 touch_start 寫的
      last_run，看起來像「跑到一半機器被關掉」。
    """
    from . import observability
    started = time.monotonic()
    observability.event("sync_stage_started", stage=name, profile=profile)
    print(f"\n=== {name} ===", flush=True)
    sync_state.touch_start(name, profile)
    try:
        out = fn()
        if isinstance(out, StageResult):
            if out.failed or not out.complete:
                note = out.note or (f"成功 {out.done}，失敗 {out.failed}，略過 {out.skipped}；"
                                    + ("尚未完成，下一輪續跑" if not out.complete else "請重試失敗文件"))
                sync_state.record_failure(name, profile, note, rows_done=out.done,
                                          rows_failed=out.failed, rows_skipped=out.skipped)
                observability.event("sync_stage_finished", stage=name, profile=profile, completed=False,
                                    rows_done=out.done, rows_failed=out.failed, rows_skipped=out.skipped,
                                    duration_ms=round((time.monotonic() - started) * 1000))
                return None
            sync_state.record_success(name, profile, rows_done=out.done,
                                      rows_skipped=out.skipped, note=out.note)
            observability.event("sync_stage_finished", stage=name, profile=profile, completed=True,
                                rows_done=out.done, rows_failed=0, rows_skipped=out.skipped,
                                duration_ms=round((time.monotonic() - started) * 1000))
            return out.done
        rows, note = out if isinstance(out, tuple) else (out, None)
        sync_state.record_success(name, profile, rows_done=rows or 0, note=note)
        observability.event("sync_stage_finished", stage=name, profile=profile, completed=True,
                            rows_done=rows or 0, rows_failed=0, rows_skipped=0,
                            duration_ms=round((time.monotonic() - started) * 1000))
        return rows or 0
    except (Exception, SystemExit) as e:  # noqa: BLE001 -- 一個 stage 炸掉不該讓整輪中止
        observability.event("sync_stage_finished", stage=name, profile=profile, completed=False,
                            error_type=type(e).__name__,
                            duration_ms=round((time.monotonic() - started) * 1000))
        msg = f"{type(e).__name__}: {e}"[:500]
        sync_state.record_failure(name, profile, msg)
        print(f"  [{name}] 失敗，記錄後繼續：{type(e).__name__}（詳細原因見同步狀態）", flush=True)
        return None


# ---------------------------------------------------------------------------
# 五個 stage
# ---------------------------------------------------------------------------
def _do_plm_export(src: Source) -> tuple[int, str]:
    # 展示版：公司系統對接已移除（原本呼叫 PLM 主檔匯出，原子替換 + 筆數安全閥）
    raise NotImplementedError("展示版：公司系統對接已移除")


def _do_plm_import(src: Source) -> tuple[int, str]:
    """把匯出檔匯進 partrag.parts。

    ★ 刻意只跑 import_plm，不跑 enrich.backfill()
      backfill 是「拿庫裡現有的品名把所有推導欄位重算一遍」，整批全庫要跑
      好幾分鐘，而它的輸入（品名）除了改名之外根本沒變 —— 而改名那條路
      _apply_rename 已經自己重算過了。每天重算一次全庫只是每天把同一批值
      寫回同一個地方。真的改了 extract.py / taxonomy.py 的推導規則時，
      那是人要主動跑一次 `-m src.enrich` 的時機，不是排程的工作。

    ★ 匯入完不呼叫 retrieval.invalidate_facets()
      排程是**另一個行程**，在這裡失效的是這個行程自己的快取，服務那邊
      完全看不到（README「狀態查詢與冷啟動」講的就是這件事：能主動失效的
      只有 write.py 那一條，因為它跟服務同一個行程）。服務端靠
      FACETS_TTL_SECONDS（預設 300 秒）自己過期，凌晨匯入、五分鐘後看板
      就是新的數字，不需要為此重啟服務。
    """
    from . import enrich

    if not src.export_csv.exists():
        raise RuntimeError(f"找不到主檔匯出檔 {src.export_csv}，"
                           f"plm_export 這個 stage 沒跑成功")
    st = enrich.import_plm(src.export_csv, reembed=True,
                           max_reembed=enrich.DEFAULT_MAX_REEMBED)
    note = (f"新增 {st['inserted']:,} 更新 {st['updated']:,} "
            f"改名 {st['renamed']:,}")
    if st["reembed_capped"]:
        # 這一句一定要留在 sync_state 上：改名的文字已經進去了、向量還是舊的，
        # 而這件事在檢索結果上完全看不出來（查得到、有分類、看起來正常）。
        note += (f"（改名超過 {enrich.DEFAULT_MAX_REEMBED} 筆上限，embedding 未重算，"
                 f"請人工跑 -m src.enrich --skip-backfill --plm-csv {src.export_csv.name}）")
    return st["inserted"] + st["updated"], note


def _do_candidates(src: Source) -> int:
    # 展示版：公司系統對接已移除（原本在 MSSQL 跑候選 SQL、寫出候選 CSV）
    rows: list = []
    if not rows:
        # 空清單不是例外，但要記進 note 讓 --status 看得到 —— 「查得到但是 0 筆」
        # 跟「連不上所以沒查」在畫面上必須是兩件事。
        raise RuntimeError("候選清單是空的：確認該年度真的有 BOM Released + 已發行的料號")
    return len(rows)


def _do_fetch_pdf(src: Source, delay: float, deadline: float | None) -> StageResult:
    if not src.candidates_csv.exists():
        raise RuntimeError(f"找不到候選清單 {src.candidates_csv}，"
                           f"candidates 這個 stage 沒跑成功")
    # 展示版：公司系統對接已移除（原本逐筆打下載 API，收 delay 限速與 deadline，
    # 回傳 new / skipped / deadline_stop 三個數字）
    raise NotImplementedError("展示版：公司系統對接已移除")


def _do_ingest_drawing(src: Source, deadline: float | None) -> StageResult:
    rows = ingest_drawing.load_log(src.pdf_dir)
    done, skipped, failed = ingest_drawing.ingest(rows, src.pdf_dir, False,
                                                  src.profile, deadline)
    print(f"  入庫 {done} 份，略過（內容未變）{skipped} 份，失敗 {failed} 份")
    return StageResult(done, failed, skipped, complete=done + failed + skipped == len(rows))


# ---------------------------------------------------------------------------
def status(profile: str | None) -> int:
    rows = sync_state.report(profile)
    if not rows:
        print("partrag.sync_state 沒有任何紀錄 —— 排程一次都還沒跑過。")
        print("（若是還沒建表：先跑 python -m src.migrate）")
        return 0
    # sync_state.report() 是照 stage 名排序的（給 /api/health 用，那邊不在乎順序）。
    # 這張表是給人看「這一輪跑到哪」的，照執行順序列才讀得下去 —— 照字母排的話
    # plm_export/plm_import 會掉到最後面，看起來像是工程圖跑完才輪到主檔。
    rows.sort(key=lambda r: (r["source_profile"],
                             ALL_STAGES.index(r["stage"])
                             if r["stage"] in ALL_STAGES else len(ALL_STAGES)))
    print(f"{'profile':8} {'stage':16} {'上次成功':20} {'幾小時前':>8} "
          f"{'筆數':>6}  ok  note")
    for r in rows:
        ls = r["last_success"].strftime("%Y-%m-%d %H:%M:%S") if r["last_success"] else "—"
        hrs = f"{r['hours_since_success']:.1f}" if r["hours_since_success"] is not None else "—"
        note = (r["note"] or "")[:60]
        print(f"{r['source_profile']:8} {r['stage']:16} {ls:20} {hrs:>8} "
              f"{r['rows_done'] or 0:>6}  {'✓' if r['ok'] else '✗'}   {note}")
    return 0


def run(src: Source, stages: list[str], delay: float,
        budget_min: float | None, dry_run: bool) -> int:
    if dry_run:
        print(f"dry-run：{src.describe()}")
        print(f"這輪會跑：{' -> '.join(stages)}")
        print(f"主檔 SQL   : {src.export_sql}")
        print(f"主檔 CSV   : {src.export_csv}"
              f"{'（已存在，會被原子替換）' if src.export_csv.exists() else '（還沒產生）'}")
        print(f"候選 SQL   : {src.candidates_sql}")
        print(f"候選 CSV   : {src.candidates_csv}"
              f"{'（已存在）' if src.candidates_csv.exists() else '（還沒產生）'}")
        print(f"PDF 目錄   : {src.pdf_dir}"
              f"{'（已存在）' if src.pdf_dir.exists() else '（還沒建立）'}")
        print(f"下載紀錄   : {src.log_path}"
              f"{'（已存在）' if src.log_path.exists() else '（還沒產生）'}")
        print(f"DWG API   : {src.dwg_api_url}")
        print(f"這輪上限   : 候選 {src.candidates} 筆、PDF {src.limit} 份、"
              f"{budget_min or '無'} 分鐘")
        return 0

    from . import observability
    with sync_state.advisory_lock(LOCK_NAME) as got, observability.job_context():
        if not got:
            print("拿不到鎖，可能有另一輪排程還在跑，這次直接放棄"
                  "（不排隊等，見本檔 docstring）")
            return 0

        deadline = time.time() + budget_min * 60 if budget_min else None
        t0 = time.time()
        print(f"→ {src.describe()}；預算 {budget_min or '無上限'}")

        # 前面某個 stage 失敗、因此這輪不該再跑的 stage。用集合而不是 break：
        # 主檔與工程圖是兩條獨立的線，其中一條斷了不該連累另一條。
        skip: set[str] = set()
        failed_stages: list[str] = []
        budget_exhausted = False

        for name in stages:
            if deadline and time.time() > deadline:
                budget_exhausted = True
                print(f"\n時間預算用完，{name} 及之後的 stage 這輪不跑，留給下一輪",
                      flush=True)
                break
            if name in skip:
                print(f"\n=== {name} ===\n  這輪跳過（同一條線上的前一個 stage 失敗）",
                      flush=True)
                continue
            if name == "plm_export":
                ok = _run_stage("plm_export", src.profile,
                                lambda: _do_plm_export(src)) is not None
            elif name == "plm_import":
                ok = _run_stage("plm_import", src.profile,
                                lambda: _do_plm_import(src)) is not None
            elif name == "candidates":
                ok = _run_stage("candidates", src.profile,
                                lambda: _do_candidates(src)) is not None
            elif name == "fetch_pdf":
                ok = _run_stage("fetch_pdf", src.profile,
                                lambda: _do_fetch_pdf(src, delay, deadline)) is not None
            elif name == "ingest_drawing":
                ok = _run_stage("ingest_drawing", src.profile,
                                lambda: _do_ingest_drawing(src, deadline)) is not None
            else:
                print(f"不認識的 stage：{name}，略過")
                continue
            if not ok:
                failed_stages.append(name)
            if not ok and name in SKIP_ON_FAILURE:
                skip.update(SKIP_ON_FAILURE[name])
                print(f"  {name} 失敗 —— 這輪連帶跳過："
                      f"{'、'.join(SKIP_ON_FAILURE[name])}")
            print(f"  （整輪已跑 {(time.time() - t0)/60:.1f} 分鐘，"
                  f"預算{_remaining_min(deadline)}）")

        print(f"\n整輪耗時 {(time.time() - t0)/60:.1f} 分鐘")
        print("進度：.venv/Scripts/python.exe -m src.schedule --status")
        observability.event("sync_job_result", completed=not (failed_stages or budget_exhausted),
                            failed_stages=failed_stages, budget_exhausted=budget_exhausted)
    return 1 if failed_stages or budget_exhausted else 0


# ---------------------------------------------------------------------------
# --daily-at：自己當常駐排程器（Linux/Docker 用）
# ---------------------------------------------------------------------------
# Windows 那邊的觸發者是工作排程器（批次檔），容器裡沒有那種東西。三條路：
#   host 的 cron 打 docker compose exec  -> 觸發器住在 repo 外面，換一台機器
#                                           就得有人記得重設一次，而忘了設的症狀
#                                           是「語料安靜地停在某一天」
#   image 裡裝 cron/supercronic         -> 多一個要維護的套件與一份 crontab 檔，
#                                           而且 cron 的環境變數與 log 導向是
#                                           另一組獨立的坑
#   這一支（--daily-at）                -> 排程器就是 schedule.py 自己，跟著
#                                           docker compose 一起起來、一起停，
#                                           log 直接進 docker compose logs
# 選第三條：觸發方式跟著 repo 走，`docker compose up -d` 就有排程，沒有任何
# 「還要記得去某台機器上設一件事」的步驟。
#
# ★ 時區是這件事最容易安靜出錯的地方
#   容器預設是 UTC，「01:00」會變成台北時間早上九點 —— 正好是上班時間，OCR
#   把 CPU 吃滿。而且這件事不會報錯，只會讓人覺得「系統白天怎麼特別慢」。
#   所以：compose 有 TZ=Asia/Taipei，image 裝了 tzdata，而下面啟動時一定會把
#   解析後的時區與下一輪的實際時刻印出來，對不上一眼就看得到。
def _tz_banner() -> str:
    now = datetime.now().astimezone()
    return f"{now.tzname()}（UTC{now.strftime('%z')}）現在時刻 {now:%Y-%m-%d %H:%M:%S}"


def _check_tz() -> None:
    """TZ 設了卻查不到時區資料時要吵，不能安靜地退回 UTC。

    python:3.12-slim 這類精簡映像檔不一定帶 tzdata；缺了的話 libc 會把
    TZ=Asia/Taipei 當成無效值直接用 UTC，沒有任何錯誤訊息 —— 症狀是排程
    每天晚八小時跑，而 log 上的時間看起來都很正常。
    """
    tz = os.environ.get("TZ", "").strip()
    if not tz:
        return
    try:
        ZoneInfo(tz)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"✗ TZ={tz} 但這個環境找不到對應的時區資料（{type(e).__name__}）——\n"
            f"  再跑下去排程會安靜地用 UTC，也就是每天晚 8 小時。\n"
            f"  image 要裝 tzdata（見 Dockerfile），或把 TZ 拿掉明講就是用 UTC。")


def _seconds_until(hhmm: str) -> float:
    try:
        hh, mm = (int(x) for x in hhmm.split(":", 1))
        target_ok = 0 <= hh <= 23 and 0 <= mm <= 59
    except ValueError:
        target_ok = False
    if not target_ok:
        raise SystemExit(f"--daily-at 要是 HH:MM（24 小時制），拿到的是「{hhmm}」")
    now = datetime.now()
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def daily(src: Source, stages: list[str], delay: float,
          budget_min: float | None, at: str) -> int:
    """睡到每天的 `at`，跑一輪，再睡。這支永遠不會自己結束。"""
    _check_tz()
    print(f"[daily] 常駐排程器啟動：每天 {at} 跑一輪 "
          f"{' -> '.join(stages)}，預算 {budget_min or '無上限'} 分鐘")
    print(f"[daily] 時區 {_tz_banner()}")
    print("[daily] 容器重啟只影響「還沒到點的那一輪」——已經跑完的進度記在 "
          "partrag.sync_state，不會重跑（見各 stage 的續跑機制）", flush=True)
    while True:
        wait = _seconds_until(at)
        nxt = datetime.now() + timedelta(seconds=wait)
        print(f"\n[daily] 下一輪 {nxt:%Y-%m-%d %H:%M:%S}"
              f"（{wait / 3600:.1f} 小時後）", flush=True)
        time.sleep(wait)
        print(f"\n{'=' * 60}\n[daily] {datetime.now():%Y-%m-%d %H:%M:%S} 開始這一輪\n"
              f"{'=' * 60}", flush=True)
        try:
            result = run(src, stages, delay, budget_min, dry_run=False)
            if result:
                print("[daily] 這一輪未全部完成，請查看 --status；下一輪會續跑", flush=True)
        except Exception:  # noqa: BLE001
            # 這個迴圈死掉＝排程從此再也不跑，而容器還活著、health 還是綠的。
            # run() 內部已經逐個 stage 接過例外了，會走到這裡的是它外面那層
            # （拿鎖、連不上 partrag 自己的資料庫…）。印完照睡，明天再試。
            traceback.print_exc()
            print("[daily] 這一輪整個失敗，明天同一時間再試", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Part-RAG 排程主入口（料號主檔 + 工程圖）")
    ap.add_argument("--profile", default=None,
                    help="來源設定名，不給就用 .env 的 "
                         "SOURCE_PROFILE")
    ap.add_argument("--stages", default=",".join(ALL_STAGES),
                    help=f"逗號分隔，預設全部：{','.join(ALL_STAGES)}")
    ap.add_argument("--delay", type=float, default=0.3,
                    help="每筆 API 請求間隔秒數（限速，打的是生產 API）")
    ap.add_argument("--budget", type=float,
                    help="整輪最多跑幾分鐘就收工（不給就跑到全部 stage 做完）")
    ap.add_argument("--status", action="store_true",
                    help="只印各 stage 的進度，不執行任何 stage")
    ap.add_argument("--dry-run", action="store_true",
                    help="只列這輪會用哪組設定、跑哪些 stage，不執行")
    ap.add_argument("--daily-at", metavar="HH:MM",
                    help="常駐模式：睡到每天這個時刻跑一輪，跑完繼續睡（給 Linux/"
                         "Docker 用，Windows 那邊的觸發者是工作排程器 + 批次檔）")
    a = ap.parse_args()

    if a.status:
        return status(a.profile)

    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        raise SystemExit(f"不認識的 stage：{unknown}，可用的是 {ALL_STAGES}")

    src = _load_source(a.profile)
    if a.daily_at:
        if a.dry_run:
            # 常駐模式配 --dry-run 會變成「每天準時什麼都不做」，而且看起來很正常。
            raise SystemExit("--daily-at 與 --dry-run 不能一起用："
                             "那會變成每天準時印一份設定然後什麼都不做")
        return daily(src, stages, a.delay, a.budget, a.daily_at)
    return run(src, stages, a.delay, a.budget, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
