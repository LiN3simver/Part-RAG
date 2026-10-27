"""跑分器：把問法庫的每一題跑過一遍，算通過率，把失敗變成教訓。

展示版：問法庫（每題綁著語料裡的真實料號與事實）不包含在內，下面的 Case /
Expect 只保留形狀，讓斷言邏輯可讀；ALL / SUITES 是空的。

    python -m src.evalrun                          # 檢索層，不需要 LLM
    python -m src.evalrun --layer answer --record  # 連答案一起核對，寫進資料庫
    python -m src.evalrun --suite drawing -v       # 只跑一組，印出每一題
    python -m src.evalrun --case <case-id>         # 只跑一題（除錯用）
    python -m src.evalrun --facts                  # 先確認問法庫綁的語料事實還在
    python -m src.evalrun --layer answer --no-lessons   # 量沒有教訓時的基準線

★ 為什麼直接呼叫 src/a2a.py 的 _find()，而不是自己組一份分流
  分流是這個系統最會漂的地方（同一份清單曾經漂成兩份）。跑分如果自己實作
  一次「什麼問題走哪條路」，那它量的就是**跑分器的分流**，線上那份改壞了
  它照樣全綠。所以這裡呼叫的是線上那一支本人。

★ 兩層分開跑，而且預設只跑第一層
  retrieval 層不呼叫 gateway：它問的是「該找到的東西找到了嗎、分流對不對、
  前 5 名有幾筆真的符合條件」。這一層跑得完、跑得快、gateway 掛掉照樣跑
  —— 而且很多幻覺的根因其實在檢索端，**這一層就抓得到**。
  answer 層才呼叫 LLM，問的是「模型有沒有講出資料裡沒有的東西」。
  兩層的通過率必須分開看：檢索 60% 而回答 100%，代表模型很老實地覆述了
  一堆錯東西 —— 那是最危險的一種「全綠」。

★ 只有 answer 層的失敗會變成教訓
  教訓是塞給模型看的，而模型改變不了檢索。把「該找到 <料號> 卻沒找到」
  寫成教訓，只會讓教訓庫長滿模型無法執行的句子，並且製造「已經處理了」的錯覺。
  檢索層的失敗一律回報成程式缺陷，印在報表最後面等人去修。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

from . import grounding, lessons, llm
from .db import pool


# --------------------------------------------------------------------------
# 展示版：問法庫已移除，這裡只留題目的形狀。
# --------------------------------------------------------------------------
@dataclass
class Expect:
    """一題的期望。空值＝不檢查那一項。"""
    route: str | None = None                       # search | drawing | listing
    hit: tuple[str, ...] = ()                       # 這些料號都要在結果裡
    hit_any: tuple[str, ...] = ()                   # 至少一個在結果裡
    forbid: tuple[str, ...] = ()                    # 這些不准出現
    attr: tuple[tuple[str, Any], ...] = ()          # 前 attr_top 名至少 attr_min 筆符合
    attr_top: int = 5
    attr_min: int = 1
    not_found: tuple[str, ...] = ()                 # 要被標成「語料中沒有」
    drawings: bool | None = None
    drawings_n: int | None = None
    match_total_min: int | None = None
    all_rows: tuple[tuple[str, Any], ...] = ()      # 嚴格模式：每一筆都要符合
    min_results: int | None = None
    listing_total_min: int | None = None
    listing_values_min: int | None = None
    result_dates: tuple[tuple[str, str, str], ...] = ()   # (欄位, 起, 迄)
    drawing_dates: tuple[tuple[str, str, str], ...] = ()
    answer_must: tuple[str, ...] = ()               # 答案至少提到其中一個
    answer_must_not: tuple[str, ...] = ()


@dataclass
class Case:
    id: str
    route: str
    question: str
    expect: Expect = field(default_factory=Expect)
    note: str = ""


class ES:
    """問法庫的介面（展示版為空）。"""
    ALL: list[Case] = []
    SUITES: dict[str, list[Case]] = {}

    @staticmethod
    def facts() -> dict[str, str]:
        return {}


# 依「今天」動態綁定期望的題目（相對時間題）。展示版用示意 id。
_REL_LAST_MONTH = "search-last-month"         # 上個月修改的料號
_REL_DRAWING_3M = "drawing-last-3-months"     # 最近三個月更新的工程圖

PASS, FAIL, DIM = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m", "\033[90m"
RESET = "\033[0m"


def route_of_found(found: dict[str, Any]) -> str:
    """實際走到哪條路 —— 從 _find() 的回傳形狀反推。

    形狀就是事實：listing 只有 aggregate.run() 會給，
    drawings/drawing_note 只有 wants_drawing 開了才有。
    """
    if found.get("listing"):
        return "listing"
    if found.get("drawings") or found.get("drawing_note"):
        return "drawing"
    return "search"


def _find(question: str, top_k: int = 10) -> dict[str, Any]:
    from .a2a import _find as a2a_find  # 延遲載入：a2a 會拉進 agentkit
    return a2a_find(question, top_k, {})


def _same(a: Any, b: Any) -> bool:
    """數字用數值比、其他用字串比。length_mm 在 DB 是 Decimal('600')、
    JSON 化之後是 600.0、問法庫寫的是 600 —— 三個都是同一個長度，
    用字串比會全部對不上。"""
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def _has_attr(row: dict[str, Any], col: str, want: Any) -> bool:
    v = row.get(col)
    if v is None:
        return False
    if isinstance(v, (list, tuple)):
        return any(_same(x, want) for x in v)
    return _same(v, want)


# --------------------------------------------------------------------------
# 檢索層斷言
# --------------------------------------------------------------------------
def check_retrieval(case: Case, found: dict[str, Any]) -> list[dict[str, str]]:
    e = case.expect
    rows = found.get("results") or []
    items = [r.get("item_number") for r in rows]
    bad: list[dict[str, str]] = []

    def fail(check: str, detail: str) -> None:
        bad.append({"check": check, "detail": detail})

    actual = route_of_found(found)
    if e.route and actual != e.route:
        fail("route", f"應走 {e.route}，實際走 {actual}")

    for pn in e.hit:
        if pn not in items:
            fail("hit", f"{pn} 不在前 {len(items)} 名：{items[:5]}")

    if e.hit_any and not any(pn in items for pn in e.hit_any):
        fail("hit_any", f"{'、'.join(e.hit_any)} 一個都沒進前 {len(items)} 名")

    for pn in e.forbid:
        if pn in items:
            fail("forbid", f"{pn} 不該出現在結果裡")

    for col, want in e.attr:
        n = sum(1 for r in rows[: e.attr_top] if _has_attr(r, col, want))
        if n < e.attr_min:
            sample = "；".join(f"{r.get('item_number')} {str(r.get('name'))[:28]}"
                              for r in rows[:3])
            fail("attr", f"前 {e.attr_top} 名只有 {n} 筆 {col}={want}"
                         f"（要 {e.attr_min}）｜{sample}")

    for pn in e.not_found:
        if pn not in (found.get("not_found") or []):
            fail("not_found", f"{pn} 沒有被標成「語料中沒有」")

    if e.drawings is not None:
        got = bool(found.get("drawings"))
        if got != e.drawings:
            fail("drawings", f"應{'有' if e.drawings else '沒有'}工程圖，實際"
                             f"{'有' if got else '沒有'}（{len(found.get('drawings') or [])} 份）")

    if e.drawings_n is not None:
        n = len(found.get("drawings") or [])
        if n != e.drawings_n:
            fail("drawings_n", f"應剛好 {e.drawings_n} 份工程圖，實際 {n} 份")

    if e.match_total_min is not None:
        mt = (found.get("match_total") or {}).get("total")
        if not mt or mt < e.match_total_min:
            fail("match_total",
                 f"沒有回報全庫符合筆數（或太少）：{mt}，要 ≥ {e.match_total_min}"
                 "｜使用者看不到母體多大就會把「前 10 筆」當成全部")

    # 嚴格模式：端出去的**每一筆**都要符合，不是前幾名符合就好。
    for col, want in e.all_rows:
        bad_rows = [r.get("item_number") for r in rows if not _has_attr(r, col, want)]
        if bad_rows:
            fail("all_rows",
                 f"{len(bad_rows)}/{len(rows)} 筆不符合 {col}={want}："
                 + "、".join(str(x) for x in bad_rows[:4])
                 + "｜使用者沒說「相似」就不該端出這些")

    if e.min_results is not None and len(rows) < e.min_results:
        fail("min_results", f"只回 {len(rows)} 筆，要 {e.min_results} 筆")

    listing = found.get("listing") or {}
    if e.listing_total_min is not None:
        total = listing.get("total") or 0
        if total < e.listing_total_min:
            fail("listing_total", f"全庫總筆數 {total}，要 ≥ {e.listing_total_min}")
    if e.listing_values_min is not None:
        vals = listing.get("values") or []
        if len(vals) < e.listing_values_min:
            fail("listing_values",
                 f"值分佈只有 {len(vals)} 個，要 ≥ {e.listing_values_min}"
                 f"（kind={listing.get('kind')}）")

    for kind, records, bounds in (("result_dates", rows, e.result_dates),
                                  ("drawing_dates", found.get("drawings") or [], e.drawing_dates)):
        for column, start, end in bounds:
            outside = [r.get("item_number") for r in records
                       if not r.get(column) or not start <= str(r[column])[:10] < end]
            if outside:
                fail(kind, f"{len(outside)} 筆 {column} 不在 [{start}, {end})")

    return bad


def bind_relative_expectations(case: Case, today: date) -> Case:
    """Independent calendar/SQL oracle: the clock changing must not break fixed corpus tests."""
    if case.id not in {_REL_LAST_MONTH, _REL_DRAWING_3M}:
        return case
    month_index = today.year * 12 + today.month - 1
    offset = 1 if case.id == _REL_LAST_MONTH else 3
    start_index = month_index - offset
    start = date(start_index // 12, start_index % 12 + 1, 1)
    end = today.replace(day=1) if case.id == _REL_LAST_MONTH else today + timedelta(days=1)
    with pool().connection() as conn, conn.cursor() as cur:
        if case.id == _REL_LAST_MONTH:
            cur.execute("SELECT count(*) FROM partrag.parts WHERE NOT is_placeholder AND NOT is_obsolete "
                        "AND modified_on >= %s AND modified_on < %s", (start, end))
            count = cur.fetchone()[0]
            expectation = replace(case.expect, min_results=min(1, count),
                result_dates=(("modified_on", start.isoformat(), end.isoformat()),))
        else:
            cur.execute("SELECT count(*) FROM partrag.drawings WHERE doc_modified_on >= %s "
                        "AND doc_modified_on < %s", (start, end))
            count = cur.fetchone()[0]
            expectation = replace(case.expect, drawings=count > 0, drawings_n=count,
                drawing_dates=(("doc_modified_on", start.isoformat(), end.isoformat()),))
    return replace(case, expect=expectation)


# --------------------------------------------------------------------------
# 答案層
# --------------------------------------------------------------------------
def answer_of(case: Case, found: dict[str, Any], with_lessons: bool
              ) -> dict[str, Any]:
    """這一題的答案。"""
    res = llm.answer_verified(case.question, found.get("results") or [], found,
                              history=None, with_lessons=with_lessons,
                              learn=False)  # 教訓由 run() 統一寫，避免寫兩次
    res["llm"] = True
    return res


def check_answer(case: Case, found: dict[str, Any], res: dict[str, Any],
                 judge: bool = False) -> list[dict[str, str]]:
    e = case.expect
    bad: list[dict[str, str]] = []
    text = res.get("text") or ""

    if not res.get("ok"):
        bad.append({"check": "llm", "detail": str(res.get("error"))})
        return bad
    if not text.strip():
        bad.append({"check": "llm", "detail": "模型沒有產生文字"})
        return bad

    # 事實核對（block 級）。這一項就是「幻覺」的操作型定義。
    for v in res.get("violations") or []:
        if v.get("severity") == "block":
            bad.append({"check": f"grounding:{v['kind']}", "detail": v["detail"]})

    low = text.lower()
    if e.answer_must and not any(m.lower() in low for m in e.answer_must):
        bad.append({"check": "answer_must",
                    "detail": "答案沒有提到 " + " / ".join(e.answer_must)})
    for m in e.answer_must_not:
        if m.lower() in low:
            bad.append({"check": "answer_must_not", "detail": f"答案不該出現「{m}」"})

    if judge and res.get("llm"):
        ctx = llm.build_prompt(case.question, found.get("results") or [], found,
                               with_lessons=False)
        verdict = llm.judge(case.question, ctx, text)
        if verdict["ok"] is False:
            bad.append({"check": "judge",
                        "detail": "；".join(verdict["issues"][:3]) or "裁判判定不合格"})
        elif verdict["ok"] is None:
            bad.append({"check": "judge_unavailable", "detail": "裁判未完成，無法宣稱驗收通過"})

    return bad


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def run(cases: list[Case], layer: str, *, judge: bool = False,
        with_lessons: bool = True, verbose: bool = False,
        threshold: float = 0.90, progress=None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    today = datetime.now().date()

    def checkpoint():
        if progress is not None:
            progress(summarize(rows, layer, threshold, t0, today,
                               expected_total=len(cases), complete=False))

    checkpoint()

    for case in cases:
        started = time.perf_counter()
        stage = "facts"
        try:
            case = bind_relative_expectations(case, today)
            stage = "find"
            found = _find(case.question)
        except Exception as exc:  # noqa: BLE001
            rows.append({"case": case, "ok": False, "answer": "",
                         "failed": [{"check": stage, "detail": type(exc).__name__}],
                         "detail": {}, "ms": round((time.perf_counter() - started) * 1000, 2),
                         "retrieval_ok": False, "answer_ok": None,
                         "lessons": []})
            checkpoint()
            continue

        failed = check_retrieval(case, found)
        retrieval_ok = not failed
        answer_ok = None
        answer_text, used_lessons = "", []

        if layer == "answer":
            try:
                res = answer_of(case, found, with_lessons)
                answer_text = res.get("text") or ""
                used_lessons = res.get("lessons") or []
                answer_failed = check_answer(case, found, res, judge=judge)
            except Exception as exc:  # One failed gateway call must not discard the run.
                answer_failed = [{"check": "answer_exception", "detail": type(exc).__name__}]
            answer_ok = not answer_failed
            failed += answer_failed

        rows.append({
            "case": case,
            "ok": not failed,
            "retrieval_ok": retrieval_ok,
            "answer_ok": answer_ok,
            "answer": answer_text,
            "failed": failed,
            "lessons": used_lessons,
            "detail": {
                "route": route_of_found(found),
                "items": [r.get("item_number") for r in (found.get("results") or [])][:5],
                "not_found": found.get("not_found") or [],
                "drawings": len(found.get("drawings") or []),
            },
            "ms": round((time.perf_counter() - started) * 1000, 2),
        })
        checkpoint()
        if verbose:
            r = rows[-1]
            print(f"  [{PASS if r['ok'] else FAIL}] {case.id} {case.question[:38]}"
                  + (f"  — {r['failed'][0]['detail'][:70]}" if r["failed"] else ""))
        else:
            sys.stdout.write("." if rows[-1]["ok"] else "x")
            sys.stdout.flush()

    if not verbose:
        print()
    result = summarize(rows, layer, threshold, t0, today,
                       expected_total=len(cases), complete=True)
    if progress is not None:
        progress(result)
    return result


def summarize(rows, layer, threshold, t0, today, *, expected_total, complete):
    passed = sum(1 for r in rows if r["ok"])
    total = len(rows)
    by_route = {}
    for route in dict.fromkeys(r["case"].route for r in rows):
        group = [r for r in rows if r["case"].route == route]
        count = sum(r["ok"] for r in group)
        ordered = sorted(r["ms"] for r in group)
        by_route[route] = {"total": len(group), "passed": count,
                           "pass_rate": count / len(group),
                           "ok": count / len(group) >= threshold,
                           "latency_ms": {f"p{p}": ordered[math.ceil(len(group) * p / 100) - 1]
                                          for p in (50, 95, 99)}}
    return {
        "rows": rows,
        "total": total,
        "passed": passed,
        "pass_rate": (passed / total) if total else 0.0,
        "threshold": threshold,
        "ok": complete and total == expected_total and total > 0 and all(group["ok"] for group in by_route.values()),
        "complete": complete,
        "expected_total": expected_total,
        "by_route": by_route,
        "retrieval_passed": sum(r["retrieval_ok"] for r in rows),
        "answer_checked": sum(r["answer_ok"] is not None for r in rows),
        "answer_passed": sum(r["answer_ok"] is True for r in rows),
        "seconds": round(time.perf_counter() - t0, 2),
        "layer": layer,
        "evaluation_date": today.isoformat(),
    }


def write_report(result: dict[str, Any], path: Path, *, suite: str,
                 judge: bool, with_lessons: bool) -> None:
    """Persist a portable result without DB writes or secret-bearing configuration."""
    from . import delivery
    from . import retrieval
    files = delivery.build_files()
    cases = [asdict(r["case"]) for r in result["rows"]]
    artifact = {**result, "format": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "suite": suite, "judge": judge, "with_lessons": with_lessons,
                "app_commit": delivery.git_revision(delivery.ROOT),
                "source_sha256": delivery.json_digest(files),
                "cases_sha256": hashlib.sha256(json.dumps(cases, ensure_ascii=False,
                    sort_keys=True).encode("utf-8")).hexdigest(),
                "embedding": {"model": retrieval.EMBED_MODEL, "revision": retrieval.EMBED_REVISION,
                              "max_seq": retrieval.MAX_SEQ, "threads": retrieval.EMBED_THREADS,
                              "loaded": retrieval.embedding_ready()},
                "model": llm.MODEL if result["layer"] == "answer" else None,
                "rows": [{**row, "case": case} for row, case in zip(result["rows"], cases)]}
    delivery.write_json(path, artifact)


def learn_from(result: dict[str, Any]) -> tuple[int, int]:
    """答案層的失敗 -> 教訓；掛著教訓而且通過的題 -> streak +1。

    回 (新增/更新幾條, 驗證通過幾條)。檢索層的失敗刻意不進來（見模組 docstring）。
    """
    recorded, verified = 0, 0
    for r in result["rows"]:
        case: Case = r["case"]
        kinds = [f["check"].split(":", 1)[1] for f in r["failed"]
                 if f["check"].startswith("grounding:")]
        for kind in dict.fromkeys(kinds):
            rule = lessons.rule_for(kind)
            if rule and lessons.record(case.route, rule, question=case.question,
                                       wrong=next(f["detail"] for f in r["failed"]
                                                  if f["check"].endswith(kind)),
                                       source="eval"):
                recorded += 1
        if r["ok"] and r["lessons"]:
            lessons.mark_pass(r["lessons"])
            verified += len(r["lessons"])
    return recorded, verified


def save(result: dict[str, Any], suite: str) -> str | None:
    """寫進 partrag.eval_runs / eval_cases。表不在就跳過（回 None）。"""
    if not lessons.available():
        print("⚠ partrag.eval_runs 不存在，這一輪沒有寫進資料庫（先跑資料庫遷移）")
        return None
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{result['layer']}"
    try:
        with pool().connection() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO partrag.eval_runs
                       (run_id, finished_at, suite, layer, model, total, passed,
                        pass_rate, threshold, ok)
                   VALUES (%s, now(), %s, %s, %s, %s, %s, %s, %s, %s)""",
                (run_id, suite, result["layer"],
                 llm.MODEL if result["layer"] == "answer" else None,
                 result["total"], result["passed"], round(result["pass_rate"], 4),
                 result["threshold"], result["ok"]),
            )
            cur.executemany(
                """INSERT INTO partrag.eval_cases
                       (run_id, case_id, route, question, ok, failed, detail, answer, ms)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [(run_id, r["case"].id, r["case"].route, r["case"].question, r["ok"],
                  json.dumps(r["failed"], ensure_ascii=False),
                  json.dumps(r["detail"], ensure_ascii=False, default=str),
                  (r["answer"] or None), round(r["ms"])) for r in result["rows"]],
            )
            conn.commit()
        return run_id
    except Exception as e:  # noqa: BLE001
        print(f"⚠ 跑分結果寫入失敗（不影響本次結果）：{type(e).__name__}: {e}")
        return None


def report(result: dict[str, Any]) -> None:
    by_route: dict[str, list[dict]] = {}
    for r in result["rows"]:
        by_route.setdefault(r["case"].route, []).append(r)

    print("\n" + "=" * 66)
    print(f"跑分結果（{result['layer']} 層，{result['seconds']} 秒）")
    print("=" * 66)
    for route, rs in by_route.items():
        p = sum(1 for r in rs if r["ok"])
        rate = p / len(rs)
        mark = "✅" if rate >= result["threshold"] else "❌"
        print(f"  {mark} {route:<14} {p:>3}/{len(rs):<3} = {rate:6.1%}")

    fails = [r for r in result["rows"] if not r["ok"]]
    if fails:
        print(f"\n--- 失敗 {len(fails)} 題 ---")
        for r in fails:
            print(f"\n  {r['case'].id}｜{r['case'].question}")
            if r["case"].note:
                print(f"    {DIM}（{r['case'].note}）{RESET}")
            for f in r["failed"]:
                print(f"    ✗ {f['check']}：{f['detail']}")

    rate = result["pass_rate"]
    print("\n" + "=" * 66)
    print(f"總計：{result['passed']}/{result['total']} = {rate:.1%}"
          f"（門檻 {result['threshold']:.0%}）"
          f" {'✅ 通過' if result['ok'] else '❌ 未達門檻'}")
    print("門檻套用到每條路徑；總平均不會掩蓋單一路徑退步。")
    if result["layer"] == "answer":
        print(f"檢索通過 {result['retrieval_passed']}/{result['total']}；"
              f"答案通過 {result['answer_passed']}/{result['answer_checked']}"
              f"（未完成檢索 {result['total'] - result['answer_checked']} 題未核對答案）")


def check_facts() -> bool:
    """問法庫綁的語料事實還在不在。語料換一批之後這裡會先炸，
    而不是讓整批題目一起變紅、看起來像系統壞了。"""
    from . import retrieval
    ok = True
    print("問法庫依賴的語料事實：")
    for label, value in ES.facts().items():
        if label.startswith("料號"):
            exists = retrieval.get_part(value) is not None
            want = "不存在" not in label
            good = exists == want
            print(f"  [{PASS if good else FAIL}] {label:<18} {value}"
                  f"（{'在' if exists else '不在'}語料裡）")
            ok &= good
        else:
            print(f"  [{DIM}skip{RESET}] {label:<18} {value}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Part-rag 問答跑分")
    ap.add_argument("--suite", default="all",
                    choices=["all", *ES.SUITES.keys()])
    ap.add_argument("--layer", default="retrieval", choices=["retrieval", "answer"])
    ap.add_argument("--case", help="只跑這一題（填題目 id）")
    ap.add_argument("--limit", type=int, help="每組只跑前 N 題")
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--judge", action="store_true", help="加上 LLM 語意裁判（慢）")
    ap.add_argument("--no-lessons", action="store_true",
                    help="不掛教訓庫，量沒有教訓時的基準線")
    ap.add_argument("--record", action="store_true", help="寫進資料庫並更新教訓庫")
    ap.add_argument("--facts", action="store_true", help="只檢查語料事實")
    ap.add_argument("--output", type=Path, help="另存 JSON 報告，不需 --record 或寫入 DB")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if not 0 < a.threshold <= 1:
        ap.error("--threshold 必須介於 0（不含）與 1 之間")
    if a.limit is not None and a.limit <= 0:
        ap.error("--limit 必須大於 0")

    if a.facts:
        return 0 if check_facts() else 1

    if a.case:
        cases = [c for c in ES.ALL if c.id == a.case]
        if not cases:
            print(f"沒有這一題：{a.case}")
            return 2
    else:
        pool_ = ES.ALL if a.suite == "all" else ES.SUITES[a.suite]
        if a.limit:
            by: dict[str, list[Case]] = {}
            for c in pool_:
                by.setdefault(c.route, []).append(c)
            cases = [c for rs in by.values() for c in rs[: a.limit]]
        else:
            cases = list(pool_)

    print(f"跑 {len(cases)} 題（{a.suite} / {a.layer} 層"
          f"{' / 不掛教訓' if a.no_lessons else ''}{' / 開裁判' if a.judge else ''}）")
    progress = None
    if a.output:
        from . import delivery
        source_before = delivery.json_digest(delivery.build_files())

        def progress(result):
            if delivery.json_digest(delivery.build_files()) != source_before:
                raise RuntimeError("source changed during evaluation")
            write_report(result, a.output, suite=a.suite, judge=a.judge,
                         with_lessons=not a.no_lessons)

    res = run(cases, a.layer, judge=a.judge, with_lessons=not a.no_lessons,
              verbose=a.verbose, threshold=a.threshold, progress=progress)
    report(res)
    if a.output:
        print(f"JSON 報告：{a.output}")

    if a.record:
        run_id = save(res, a.suite)
        if run_id:
            print(f"\n已寫入 partrag.eval_runs / eval_cases：{run_id}")
        if a.layer == "answer":
            rec, ver = learn_from(res)
            print(f"教訓庫：新增/更新 {rec} 條，驗證通過 {ver} 條"
                  f"（python -m src.lessons 看內容）")

    return 0 if res["ok"] else 1


if __name__ == "__main__":  # python -m src.evalrun
    for _s in (sys.stdout, sys.stderr):
        if getattr(_s, "encoding", "").lower() != "utf-8":
            _s.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
