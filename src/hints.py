"""答不出來的時候，告訴使用者**怎麼問才問得到**。

★ 為什麼「查不到」還不夠
  「查不到」是誠實的，但它把整個負擔丟回給使用者：他不知道是自己問錯、
  條件太嚴、還是語料真的沒有。例如「給我修改日期為三月跟五月的工程圖」
  拿到的是「查不到任何…」，而真正的情況是**那兩個月都有圖**、只是系統
  當時把日期拿去比對圖上的文字。使用者看到那句話只會放棄，不會想到換個問法。

  所以每一種「答不出來」都要附上**可以直接點下去執行的下一句**。不是
  「請提供更多資訊」這種話 —— 那跟沒講一樣 —— 而是照著這份語料現在真的有的
  東西，給出兩三句具體的問句。

★ 建議必須是**用這份語料算出來的**，不能寫死
  「試試 2026 年 5 月」這種建議，只有在五月真的有資料時才不是另一個謊。
  所以日期類的建議會去查該欄位實際涵蓋的區間與有資料的月份；
  沒有資料的月份不會出現在建議裡。

★ 只在真的答不出來時才給
  有結果卻附一段「你可以這樣問」，等於在暗示這次的答案不好。
  build() 對有結果的查詢一律回 None。
"""
from __future__ import annotations

import time
from typing import Any

from psycopg.rows import dict_row

from .db import pool

# 語料的日期涵蓋範圍。TTL 與 values.py 的值索引同一個量級 —— 這是「資料庫現在
# 有什麼」而不是設定，排程灌完新資料五分鐘內就該反映出來。
_TTL = 300.0
_cache: tuple[float, dict[str, Any]] | None = None


def coverage() -> dict[str, Any]:
    """三個日期欄位各自涵蓋到哪裡，以及工程圖有資料的月份清單。"""
    global _cache
    now = time.time()
    if _cache and now - _cache[0] < _TTL:
        return _cache[1]
    out: dict[str, Any] = {}
    try:
        with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT min(doc_modified_on)::date AS lo,
                          max(doc_modified_on)::date AS hi,
                          count(*) FILTER (WHERE doc_modified_on IS NULL) AS unknown,
                          count(*) AS total
                     FROM partrag.drawings"""
            )
            out["drawing"] = dict(cur.fetchone() or {})
            cur.execute(
                """SELECT to_char(doc_modified_on, 'YYYY-MM') AS m, count(*) AS n
                     FROM partrag.drawings WHERE doc_modified_on IS NOT NULL
                    GROUP BY 1 ORDER BY 1 DESC LIMIT 8"""
            )
            out["drawing_months"] = [dict(r) for r in cur.fetchall()]
            cur.execute(
                """SELECT min(created_on)::date AS c_lo, max(created_on)::date AS c_hi,
                          min(modified_on)::date AS m_lo, max(modified_on)::date AS m_hi
                     FROM partrag.parts"""
            )
            out["parts"] = dict(cur.fetchone() or {})
    except Exception as e:  # noqa: BLE001 -- 給不出建議不該讓回答壞掉
        print(f"[hints] 讀不到日期涵蓋範圍（略過建議）：{type(e).__name__}: {e}")
        out = {}
    _cache = (now, out)
    return out


def _months_line(limit: int = 4) -> str:
    ms = (coverage().get("drawing_months") or [])[:limit]
    if not ms:
        return ""
    return "、".join(f"{m['m']}（{m['n']} 份）" for m in ms)


def _empty(found: dict[str, Any]) -> bool:
    return not (found.get("results") or found.get("drawings")
                or (found.get("listing") or {}).get("total"))


def gap(found: dict[str, Any]) -> str | None:
    """這一題**哪裡答不出來**。None＝答得出來，不需要建議。

    ★ 「有東西回傳」不等於「答得出來」
      問了工程圖但那顆沒有圖時，results 裡仍有那一筆料號（留給「沒有圖」那句話
      當主詞）；點名的料號不在語料裡時，results 裡是鄰近料號。兩種情況畫面上
      都不是空的，但**使用者要的東西都沒拿到** —— 這正是最需要告訴他
      「換個方式就問得到」的時刻。只看空不空會把它們全部漏掉。
    """
    tq = found.get("time_query")
    rows = found.get("results") or []
    dwgs = found.get("drawings") or []
    total = (found.get("listing") or {}).get("total")
    if tq and not dwgs and not rows and not total:
        return "time"
    if found.get("drawing_note") and not dwgs:
        return "drawing"
    if found.get("not_found"):
        return "not_found"
    if found.get("strict_dropped") and not rows:
        return "strict"
    if _empty(found):
        return "empty"
    return None


def build(query: str, found: dict[str, Any] | None) -> str | None:
    """-> 「你可以這樣問」那一段，或 None（這一題答得出來，不需要建議）。"""
    found = found or {}
    kind = gap(found)
    if kind is None:
        return None

    tq = found.get("time_query")
    is_drawing = bool(found.get("drawing_note") or found.get("drawings")) or (
        tq and "工程圖" in (tq.get("describe") or ""))
    lines: list[str] = []
    asks: list[str] = []

    # --- 1. 有時間條件但查不到 -------------------------------------------
    if kind == "time":
        rng = "、".join(s["label"] for s in tq.get("spans") or [])
        cov = coverage()
        if is_drawing:
            d = cov.get("drawing") or {}
            lines.append(
                f"我是照**工程圖檔案在 PLM 的更新時間**去篩的（{rng}），"
                f"這個範圍內沒有圖。"
            )
            if d.get("lo") and d.get("hi"):
                lines.append(
                    f"目前收錄的 {d.get('total')} 份圖，更新時間分佈在 "
                    f"{d['lo']} ～ {d['hi']}"
                    + (f"，另有 {d['unknown']} 份沒有時間戳。" if d.get("unknown") else "。")
                )
            months = _months_line()
            if months:
                lines.append(f"有資料的月份：{months}。")
                first = (cov.get("drawing_months") or [{}])[0].get("m")
                if first:
                    y, m = first.split("-")
                    asks.append(f"{y} 年 {int(m)} 月的工程圖")
            asks.append("找最近三筆工程圖給我")
        else:
            p = cov.get("parts") or {}
            who = "建立日期" if tq.get("field") == "created" else "修改日期"
            lines.append(f"我是照料號的**{who}**去篩的（{rng}），這個範圍內沒有料號。")
            lo = p.get("c_lo") if tq.get("field") == "created" else p.get("m_lo")
            hi = p.get("c_hi") if tq.get("field") == "created" else p.get("m_hi")
            if lo and hi:
                lines.append(f"語料裡的{who}分佈在 {lo} ～ {hi}。")
            if not tq.get("explicit_field"):
                other = "建立" if tq.get("field") == "modified" else "修改"
                lines.append(
                    f"另外，你沒有明講要看建立還是修改，我預設用**{who}** ——"
                    f"要看另一個的話請在問句裡寫「{other}日期」。")
                asks.append(f"列出 {rng} {other}的所有料號")
            asks.append(f"列出 {rng} {who[:2]}的所有料號共有幾筆")

    # --- 2. 問了工程圖但沒有圖 -------------------------------------------
    elif kind == "drawing":
        lines.append("這個條件在目前收錄的工程圖裡找不到"
                     "（工程圖只涵蓋一小部分料號）。")
        asks += ["這個料號的規格是什麼？",
                 "工程圖材料表有 LABEL 的料號有哪些？",
                 "找最近三筆工程圖給我"]

    # --- 3. 條件太嚴，被濾掉了 -------------------------------------------
    elif kind == "strict":
        n = found["strict_dropped"]
        lines.append(
            f"有 {n} 筆料號**部分符合**你的條件，但沒有一筆全部符合，所以我沒有列出來。")
        asks.append(query + "，類似的也可以")
        asks.append("（把其中一個條件拿掉再問一次，例如不指定長度）")

    # --- 4. 點名的料號不在語料裡 -----------------------------------------
    elif kind == "not_found":
        pn = found["not_found"][0]
        lines.append(f"{pn} 不在語料裡（可能是 PLM 剛開、還沒進到每天的匯出檔）。")
        asks += [f"跟 {pn} 類似的料號有哪些？", "用規格找：PH4.2 2*6P HSG 的料號"]

    # --- 5. 其他查不到 ----------------------------------------------------
    else:
        lines.append("這句話我沒有找到對得上的資料。")
        asks += ["600mm 的 MCIO 8i 線，30AWG，有哪些料號？",
                 "列出所有公告大類",
                 "找有香蕉头端子的工程圖",
                 "你可以做什麼？"]

    if not asks:
        return None
    body = "".join(lines)
    ask = "\n".join(f"- 「{a}」" for a in list(dict.fromkeys(asks))[:4])
    return f"{body}\n\n**你可以改成這樣問**（照抄就能執行）：\n{ask}"


def prompt_block(query: str, found: dict[str, Any] | None) -> str:
    """給 llm.build_prompt / build_context 用的那一段。

    ★ 為什麼要把建議**原文**交給模型，而不是叫模型自己想
      模型想出來的建議會是「請提供更多資訊」或一個查不到的日期 —— 它不知道
      語料涵蓋到哪個月。這裡算出來的每一句都是照現在的資料產生的，
      所以指示是「照抄」，不是「參考」。
    """
    hint = build(query, found)
    if not hint:
        return ""
    return (
        "★ 這一題查不到資料。請先誠實說明查不到，然後把下面這段**原文照抄**"
        "放在回答最後（可以調整標點，但建議的問句一個字都不要改）：\n\n"
        + hint + "\n\n"
    )
