"""教訓庫：把「上次答錯的地方」變成下次 prompt 裡的一條硬規則。

★ 這一支要解的問題
  SYSTEM 那段規則是**人寫的、靜態的**。它涵蓋的是我們想得到的錯法；跑分跑出來
  的失敗，涵蓋的是模型實際會犯的錯法。兩者的交集比想像中小 —— 而且每次換模型
  就換一批。所以錯誤要能被收集、歸納、餵回去，這一支就是那條迴路：

      跑分/線上核對失敗 -> record() 寫成一條規則 -> 下次同類問題 for_query()
      把它挑出來塞進 prompt -> 該題通過 -> mark_pass() 累積 streak
      -> 連續通過夠多輪就退役（不再吃 token）-> 再犯就被 record() 叫回來

★ 一條教訓能修什麼、不能修什麼（這件事寫在這裡，因為最容易搞混）
  能修的是**模型的表達與取材**：編料號、不附工程圖來源、把「查無」講成「找到了」、
  把某個欄位的值拿去回答另一個欄位。
  不能修的是**檢索本身**：該檢索到的東西沒進來，塞再多教訓也變不出來。所以
  src/evalrun.py 只從 answer 層的失敗產生教訓；retrieval 層失敗一律回報成
  程式要改的缺陷，不會被寫成教訓假裝解決了。這條界線一旦破，教訓庫會迅速長成
  一堆「請記得要找到 Z0000XX00000-1」這種對模型毫無意義的句子。

★ 教訓表還沒建也要能跑
  所有讀取端遇到表不存在時回空清單，寫入端安靜跳過 —— 問答不依賴教訓庫，
  它是加分項不是必要條件（與 src/sync_state.py 對 sync_state 的處理同一原則）。
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from typing import Any

from psycopg.rows import dict_row

from .db import pool
from .extract import normalize, parse_query, space_cjk_ascii

# 一條教訓連續通過幾輪之後退役。
#
# ★ 為什麼不是「通過一次就拿掉」
#   通過一次可能是運氣（同一題換個問法就再犯）。也不能永不退役：每條教訓都吃
#   token，而且互相稀釋注意力 —— 三十條「不要編造料號」的效果遠低於三條。
RETIRE_STREAK = int(os.getenv("LESSON_RETIRE_STREAK", "3"))

# 一次最多塞幾條進 prompt。實測 prompt 每多 1,000 字，這顆會 thinking 的模型
# 就多花可觀的時間，而排在第 7 條之後的規則幾乎不影響輸出。
MAX_IN_PROMPT = int(os.getenv("LESSON_MAX_IN_PROMPT", "6"))

_available: tuple[float, bool] | None = None
_AVAIL_TTL = 60.0


def available() -> bool:
    """partrag.lessons 這張表在不在。快取 60 秒 —— 每題問一次 information_schema
    是白花的往返，而 migration 不會在一分鐘內套兩次。"""
    global _available
    now = time.time()
    if _available and now - _available[0] < _AVAIL_TTL:
        return _available[1]
    try:
        with pool().connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT to_regclass('partrag.lessons') IS NOT NULL")
            ok = bool(cur.fetchone()[0])
    except Exception:
        ok = False
    _available = (now, ok)
    return ok


def invalidate() -> None:
    global _available
    _available = None


# 詞級 stopword：這些詞每一句都有，當觸發詞等於「全部命中」。
_STOP = frozenset({
    "料號", "料号", "有哪些", "什麼", "什么", "查詢", "查询", "請問", "请问",
    "我要", "可以", "怎麼", "怎么", "如何", "這個", "这个", "THE", "AND", "FOR",
})


def terms_of(question: str) -> list[str]:
    """一個問句的觸發詞。要能跨問法穩定，所以取的是**規格 token 與領域詞**，
    不是整句斷詞。

    「600mm 的 MCIO 8i 線」與「我要找 MCIO 8i、長度 600mm 的」抽出來的是同一組
    ['MCIO', '600MM', ...]，所以掛在前者上的教訓後者也吃得到 —— 這正是
    「換個問法就再犯一次」要擋掉的東西。
    """
    n = space_cjk_ascii(normalize(question or ""))
    out: list[str] = []

    p = parse_query(question or "")
    out += [str(x) for x in p.get("pitch_series") or []]
    out += [str(x) for x in p.get("conn_types") or []]
    if p.get("awg"):
        out.append(f"{p['awg']}AWG")
    if p.get("gen"):
        out.append(f"GEN{p['gen']}")
    for v in p.get("length_mm") or []:
        out.append(f"{v:g}MM")
    # 料號與客戶料號**不當觸發詞**：它們是「這一題」的東西，掛上去的教訓永遠
    # 不會有第二題命中，等於寫進去就死掉。而教訓要管的是「同一類問題」，
    # 不是「同一顆料」。所以這裡要明確排除 —— 下面那條拉丁 token 的規則
    # 會抓到 Z0000XX00000-1 這種料號，光靠註解擋不住它。
    skip = {str(x).upper() for x in (*(p.get("item_numbers") or []),
                                     *(p.get("codes") or []))}

    # 領域關鍵詞：只收有辨識度的中文詞與拉丁 token。
    for tok in re.findall(r"[A-Z][A-Z0-9.+\-]{1,}", n):
        if len(tok) >= 2 and tok not in _STOP and not tok.isdigit() and tok not in skip:
            out.append(tok)
    for word in ("工程圖", "材料表", "標題欄", "清單", "統計", "相似", "替代",
                 "客戶料號", "建立人員", "廠別", "公告大類", "作廢", "成本", "工時"):
        if word in (question or "") or normalize(word) in n:
            out.append(word)

    return sorted(dict.fromkeys(t for t in out if t and t not in _STOP))[:12]


def _key(route: str, rule: str) -> str:
    return hashlib.md5(f"{route}\n{rule.strip()}".encode()).hexdigest()


def record(route: str, rule: str, *, question: str = "", wrong: str = "",
           terms: list[str] | None = None, source: str = "eval") -> str | None:
    """記下一條教訓（或把既有那條的 misses 加一、streak 歸零、取消退役）。

    回傳 key；表不存在時回 None。
    """
    rule = (rule or "").strip()
    if not rule or not available():
        return None
    key = _key(route, rule)
    tt = terms if terms is not None else terms_of(question)
    try:
        with pool().connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO partrag.lessons
                       (key, route, trigger_terms, rule, wrong, question, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (key) DO UPDATE SET
                    -- 觸發詞取聯集：同一條規則在新的問法下再犯一次，代表那組新的
                    -- 詞也該掛上它，否則下次那個問法仍然吃不到這條教訓。
                    trigger_terms = ARRAY(
                        SELECT DISTINCT unnest(partrag.lessons.trigger_terms || EXCLUDED.trigger_terms)
                    ),
                    misses     = partrag.lessons.misses + 1,
                    streak     = 0,
                    retired    = false,
                    wrong      = COALESCE(EXCLUDED.wrong, partrag.lessons.wrong),
                    updated_at = now()
                """,
                (key, route, tt, rule, wrong or None, question or None, source),
            )
            conn.commit()
        return key
    except Exception as e:  # noqa: BLE001
        print(f"[lessons] record 失敗（略過）：{type(e).__name__}: {e}")
        return None


def for_query(question: str, route: str | None = None,
              limit: int = MAX_IN_PROMPT) -> list[dict[str, Any]]:
    """這一題該掛哪幾條教訓。回 [{key, rule, route, overlap}]，最相關的在前。

    排序是「觸發詞重疊數 > 犯過幾次 > 新的優先」。重疊 0 的只有在
    trigger_terms 是空陣列（該 route 全域適用）時才會進來。
    """
    if not available():
        return []
    tt = terms_of(question)
    routes = ["*"] + ([route] if route else [])
    try:
        with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT key, route, rule,
                       cardinality(ARRAY(SELECT unnest(trigger_terms)
                                         INTERSECT SELECT unnest(%s::text[]))) AS overlap,
                       cardinality(trigger_terms) AS n_terms, misses
                  FROM partrag.lessons
                 WHERE NOT retired AND route = ANY(%s)
                """,
                (tt, routes),
            )
            rows = [dict(r) for r in cur.fetchall()]
    except Exception as e:  # noqa: BLE001
        print(f"[lessons] for_query 失敗（略過）：{type(e).__name__}: {e}")
        return []

    picked = [r for r in rows if r["overlap"] > 0 or r["n_terms"] == 0]
    picked.sort(key=lambda r: (-r["overlap"], -r["misses"]))
    picked = picked[:limit]

    if picked:
        try:
            with pool().connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE partrag.lessons SET hits = hits + 1, last_hit_at = now()"
                    " WHERE key = ANY(%s)",
                    ([r["key"] for r in picked],),
                )
                conn.commit()
        except Exception:  # noqa: BLE001
            pass  # 統計欄位失敗不該影響回答
    return picked


def as_block(picked: list[dict[str, Any]]) -> str:
    """挑出來的教訓寫成 prompt 裡的一段。

    ★ 標題寫「過去在這類問題上犯過的錯」而不是「注意事項」：模型對「這是你犯過
      的錯」的遵從度明顯高於一般提醒，而且這句話是真的 —— 每一條後面都有一次
      實際失敗的紀錄。
    """
    if not picked:
        return ""
    lines = ["★ 過去在這類問題上犯過的錯（這一題必須避免，違反者視同答錯）："]
    lines += [f"{i}. {r['rule']}" for i, r in enumerate(picked, 1)]
    return "\n".join(lines)


def mark_pass(keys: list[str]) -> None:
    """掛著這些教訓跑，而且該題通過了。streak 到門檻就退役。"""
    if not keys or not available():
        return
    try:
        with pool().connection() as conn, conn.cursor() as cur:
            cur.execute(
                """UPDATE partrag.lessons
                      SET streak = streak + 1,
                          verified_at = now(),
                          updated_at = now(),
                          retired = (streak + 1) >= %s
                    WHERE key = ANY(%s)""",
                (RETIRE_STREAK, keys),
            )
            conn.commit()
    except Exception as e:  # noqa: BLE001
        print(f"[lessons] mark_pass 失敗（略過）：{type(e).__name__}: {e}")


def report(limit: int = 50) -> list[dict[str, Any]]:
    if not available():
        return []
    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT route, rule, misses, hits, streak, retired,
                      trigger_terms, created_at, verified_at
                 FROM partrag.lessons
                ORDER BY retired, misses DESC, created_at DESC LIMIT %s""",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


# 違規種類 -> 要寫進教訓庫的那一句規則。
#
# ★ 為什麼是固定文案而不是把 Violation.fix 原樣存進去
#   fix 是**針對那一題**的（「請移除 Z0000XX00000-1」），存起來對下一題毫無用處，
#   而且會讓教訓庫長出幾百條只差一個料號的規則。教訓要的是**通則**。
RULES: dict[str, str] = {
    "unknown_item": "只能寫出「檢索結果」區塊裡實際列出的料號。"
                    "任何沒有出現在該區塊的料號一律不得寫出來，"
                    "包括你認為相似、相鄰或印象中存在的料號。",
    "unknown_file": "工程圖檔名一律從「工程圖」區塊原封不動複製，"
                    "一個字元都不要改，也不要憑格式推出一個看起來很像的檔名。",
    "unsupported_spec": "長度、AWG、GEN、pitch 這些規格數值一律照抄檢索結果，"
                        "不要換算、取整或推估；資料裡沒有就說沒有。",
    "missing_source": "引用工程圖時，回答最後必須獨立一行寫出來源："
                      "檔名 ＋ 檔案更新時間 ＋ 連結，三者缺一不可。",
    "not_found_ignored": "出現「查無此料號」區塊時，第一句就要明講該料號不在語料中，"
                         "再把鄰近料號標示為僅供參考，不可以直接拿它們當答案。",
    "wrong_drawing_subject":
        "使用者點名某個料號／客戶料號的工程圖時，先確認「工程圖」區塊裡那一份"
        "的標題欄（PART NO. / CUSTOMER P/N）或歸檔料號真的就是他問的那一個。"
        "編號只差一碼（ABC-0001-01 / ABC-0002-01）的圖是不同的圖，"
        "材料表就算一模一樣也不可以互相代答；沒有那一份就直接說沒有。",
    "raw_dump": "回答一律寫成給人看的繁體中文，不要把工具回傳的 JSON 原樣貼出來，"
                "也不要留下 \\uXXXX 這種未解碼的字元。",
    "unknown_link": "工程圖的連結一律從資料裡原封不動複製。連結裡的數字是資料庫"
                    "流水號，不是「清單的第幾份」，絕對不可以推算 ——"
                    "沒有附連結的那幾份就只寫檔名與檔案更新時間。",
    "denied_present": "時間條件分成好幾段（「三月跟五月」）時，先照系統給的"
                      "「各時間區間各有幾份」講出每一段的份數，再舉例。"
                      "份數 >0 的區間一律不可以說「沒有」「查無」——"
                      "系統只展開得下前幾份的標題欄，沒有展開不等於不存在。",
    "refused_with_results": "檢索結果有東西就一定要用它回答，不可以說「查不到」。"
                            "不完全符合條件時要逐筆講清楚差在哪裡 ——"
                            "「有這些但不完全一樣」與「什麼都沒有」是兩個不同的答案。",
}


def rule_for(kind: str) -> str:
    return RULES.get(kind, "")


if __name__ == "__main__":  # python -m src.lessons
    import io
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if not available():
        print("partrag.lessons 不存在 —— 先跑 python -m src.migrate 建表")
        raise SystemExit(1)
    rows = report()
    if not rows:
        print("教訓庫是空的（跑一次 python -m src.evalrun --layer answer --record）")
        raise SystemExit(0)
    print(f"共 {len(rows)} 條\n")
    for r in rows:
        flag = "（已退役）" if r["retired"] else ""
        print(f"[{r['route']}] 犯 {r['misses']} 次 / 引用 {r['hits']} 次 / "
              f"連續通過 {r['streak']}{flag}")
        print(f"    {r['rule']}")
        print(f"    觸發詞：{'、'.join(r['trigger_terms']) or '(全域)'}\n")
