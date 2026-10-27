"""工具模式的事實核對：讓「勾了用自己的工具」那條路也走 grounding。

★ 為什麼需要獨立這一支
  這個 agent 有**兩條**產生答案的路徑，而它們在不同的地方收尾：
    ① 純摘要      a2a._answer -> llm.stream_answer -> llm.answer_verified
    ② 工具迴圈    agentkit.delegate.stream_delegated（前端勾「用自己的工具」）
  核對本來只接在 ①。而使用者實際上幾乎都開著 ② —— 於是實測到的幻覺
  （把工具的原始 JSON 當答案吐出來、自己編一份「最近三筆工程圖」排行榜）
  一次都沒有被攔到。核對只保護一條路，等於沒有核對：使用者不會知道
  自己這次走的是哪一條。

★ 為什麼用 monkey-patch，而不是改 agentkit
  `vendor/agentkit/` 是 Agent_Hub_RAG_Frontend 的固定快照，以 lock 記錄並納入版控。
  共用套件的正本由 Hub 維護；partrag 特定的核對規則仍放在這裡，
  避免下次更新快照時混入 Part-RAG 專屬修改，或把規則強加給其他 Agent。
  所以掛鉤寫在這裡，install() 由 src/a2a.py 在 mount 時呼叫一次，
  影響範圍就是這個 agent 自己。真的要下放成共用能力時，再把它提到 agentkit
  並刪掉這一支 —— 屆時 install() 回 False（偵測到已經有人掛過）就好。

★ 核對的「上下文」必須包含工具拿到的東西
  ① 那條路的上下文就是 prompt 本身。② 這條路的模型還會自己去呼叫
  search_drawings / count_parts，拿到一批 found 裡沒有的資料。只拿 found
  去核對的話，模型**查到的真東西**會被判成幻覺，然後被叫去把正確答案刪掉。
  所以工具的回傳值要一併記下來：handler 在 wrap_handlers() 裡被包過一層，
  每次呼叫把輸出放進 thread-local，guarded() 在事件迴圈裡立刻收走（見 _local）。
"""
from __future__ import annotations

import json
import threading
from typing import Any, Iterator

from . import grounding, hints, lessons, llm

# 工具剛剛回傳了什麼。**thread-local，而且立刻被抽走**。
#
# ★ 為什麼不是 contextvar，也不是全域變數
#   - 全域變數：同一個行程會同時服務多個請求，A 的核對會讀到 B 的工具輸出。
#   - contextvar：這一支是在**產生器**裡跑的，而 Starlette 把同步產生器丟進
#     threadpool 逐次 next()。實測 record() 在工作執行緒裡拿到的是 default
#     （None），於是一次都沒記到 —— 症狀是核對把模型**用工具查到的真料號**
#     全部判成幻覺，然後把一個正確答案換成「找不到」。這比不核對更糟。
#
#   現在的做法不依賴任何一種傳播：agentkit 的迴圈是
#       _dispatch(...) -> yield EV_TRACE{phase:"result"}
#   兩件事在**同一次 next()、同一條執行緒**裡發生。所以 handler 把輸出放進
#   thread-local，guarded() 一看到那則 trace 事件就立刻 drain() 收走，
#   收進屬於這一次請求的區域變數。跨執行緒、跨請求都不會混到。
_local = threading.local()

# 一題最多讓模型產出幾次答案（第一次 + 重寫）。
#
# ★ 為什麼是 3
#   使用者要求「給模型檢查過三次，若真沒有就回覆找不到」。這個數字也剛好是
#   實測的收斂點：第二次改不掉的，第三次通常也改不掉 —— 因為那代表資料本身
#   撐不起這個答案，而不是模型寫得不好。三次之後就不要再試，直接把
#   「查不到」講清楚，那比第四次嘗試誠實。
MAX_ATTEMPTS = 3

# 工具輸出寫進核對上下文時，每一支最多留幾個字元。
TOOL_CHARS = 6000


def record(name: str, out: Any) -> None:
    """工具的 handler 包裝層呼叫它。放進 thread-local，等 drain() 收走。"""
    try:
        body = out if isinstance(out, str) else json.dumps(
            out, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        body = str(out)
    box = getattr(_local, "pending", None)
    if box is None:
        box = _local.pending = []
    box.append((name, body[:TOOL_CHARS]))


def drain() -> list[tuple[str, str]]:
    """把這條執行緒上還沒收走的工具輸出取出來並清空。"""
    box = getattr(_local, "pending", None)
    if not box:
        return []
    _local.pending = []
    return box


def wrap_handlers(tools: list[Any]) -> None:
    """把每一支工具的 handler 包一層，讓它的回傳值進得了核對上下文。

    包裝是冪等的（看得到 _partrag_wrapped 就跳過），因為 mount() 在測試裡
    會被呼叫不只一次。
    """
    for t in tools:
        fn = getattr(t, "handler", None)
        if fn is None or getattr(fn, "_partrag_wrapped", False):
            continue

        def make(inner):  # noqa: ANN001 -- 閉包，型別無意義
            def wrapped(args: dict) -> dict:
                out = inner(args)
                record(getattr(inner, "__name__", "tool"), out)
                return out
            wrapped._partrag_wrapped = True  # type: ignore[attr-defined]
            return wrapped

        t.handler = make(fn)


def _context(query: str, found: dict[str, Any],
             tools: list[tuple[str, str]] | None = None) -> str:
    """模型這一輪**有可能讀到**的全部東西：一開始的檢索 + 每一次工具回傳。"""
    parts = [llm.build_context(query, found, with_lessons=False)]
    for name, body in tools or []:
        parts.append(f"【工具 {name} 回傳】\n{body}")
    return "\n\n".join(p for p in parts if p)


def _no_answer(found: dict[str, Any], violations: list, query: str = "") -> str:
    """三次都沒過時要送出去的那一段。

    ★ 這一段刻意**不經過模型**
      模型已經證明它在這一題上會編。再問第四次只是換一種編法。這裡只講
      兩件確定的事：查到了什麼、沒查到什麼，其餘留白讓使用者自己決定下一步。
    """
    lines = ["我沒有辦法給出一個能用資料佐證的答案，所以直接說明現況："]
    n = len(found.get("results") or [])
    d = len(found.get("drawings") or [])
    if n:
        lines.append(f"- 檢索到 {n} 筆料號，已列在下方卡片，請直接參考。")
    if d:
        lines.append(f"- 檢索到 {d} 份工程圖，已列在下方卡片。")
        # ★ 三次不過≠什麼都不能講
        #   典型情況是模型把通用圖檔名裡的 XXX 換成料號，使用者卻只拿到
        #   「我沒辦法給出答案」—— 而「各時間區間各幾份」是資料庫算出來的
        #   確定值，跟模型講錯的那個檔名毫無關係。把確定的事實留下來，
        #   使用者至少知道問題問對了、東西真的在。
        buckets = [b for b in (found.get("drawing_buckets") or []) if b.get("n")]
        if buckets:
            lines.append("  其中 " + "、".join(
                f"{b['label']} {b['n']} 份" for b in buckets) + "（資料庫直接統計）。")
    if not n and not d:
        lines.append("- 這個條件在語料裡**查不到任何符合的資料**。")
    if found.get("not_found"):
        lines.append("- 語料中沒有這些料號："
                     + "、".join(found["not_found"][:5]))
    if found.get("strict_dropped"):
        lines.append(f"- 另有 {found['strict_dropped']} 筆**不完全符合**你點名的條件，"
                     "已經濾掉；要看近似的請說「相似」或「類似」。")
    lines.append("- 自動核對未通過的項目：" + "；".join(
        v.detail for v in violations[:3]))
    lines.append("請換個條件再問一次，或告訴我要放寬哪一項。")
    return "\n".join(lines)


def _repair(spec: Any, query: str, context: str, answer: str,
            feedback: str) -> str:
    """把違規項回饋給模型，要它重寫一次。**不掛工具** —— 這一步要的是
    「照著已經拿到的資料重講」，不是再查一輪。"""
    msgs = [
        {"role": "user", "content": f"{query}\n\n【可以引用的全部資料】\n{context}"},
        {"role": "assistant", "content": answer},
        {"role": "user", "content": feedback},
    ]
    text, err = llm._complete(msgs, system=getattr(spec, "system", "") or llm.SYSTEM)
    if err:
        print(f"[toolguard] 重寫失敗（保留原答案）：{err}", flush=True)
        return ""
    return text


def _finish(spec: Any, req: Any, found: dict[str, Any], answer: str,
            tools: list[tuple[str, str]] | None = None
            ) -> Iterator[tuple[str, dict]]:
    from agentkit.envelope import EV_DELTA

    if not answer:
        # 迴圈自己就沒產出文字（gateway 掛了、或被 _forced_answer 接手過）。
        # 那條路已經送過自己的說明，這裡不要再疊一句。
        return

    query = getattr(req, "query", "")
    context = _context(query, found, tools)
    route = llm.route_of(found)

    first = grounding.check(answer, context, found, query)
    # best＝目前為止違規最少的那一版（要送出去的）；
    # last＝最近一次的產出（下一輪重寫要接在它後面，對話才會往前走）。
    best_answer, best_v = answer, first
    last_answer, last_v = answer, first
    attempts = 1

    # ★ 真的試滿三次，不要「沒變好就放棄」
    #   舊寫法是一旦某次重寫沒有讓違規變少就 break，於是 MAX_ATTEMPTS=3 實際上
    #   常常只跑兩次。而模型第二次沒改對、第三次改對是會發生的 —— 使用者要的
    #   就是「給模型檢查三次」。沒變好的那一版**不採用**（best 不動），
    #   但仍然當成對話的一部分餵回去，否則下一輪拿到一模一樣的輸入，
    #   當然產生一模一樣的輸出。
    while grounding.blocking(best_v) and attempts < MAX_ATTEMPTS:
        rewritten = _repair(spec, query, context, last_answer,
                            grounding.feedback(grounding.blocking(last_v)))
        attempts += 1
        if not rewritten:
            break
        new_v = grounding.check(rewritten, context, found, query)
        last_answer, last_v = rewritten, new_v
        if len(grounding.blocking(new_v)) < len(grounding.blocking(best_v)):
            best_answer, best_v = rewritten, new_v

    answer, v = best_answer, best_v
    blocked = grounding.blocking(v)
    if blocked:
        # 三次（或中途放棄）之後仍然過不了 -> 不送模型那段話。
        yield EV_DELTA, {"t": _no_answer(found, blocked, query)}
    else:
        yield EV_DELTA, {"t": answer + grounding.note(
            [x for x in v if x.severity == grounding.WARN])}

    if grounding.blocking(first):
        kinds = "、".join(x.kind for x in grounding.blocking(first))
        print(f"[toolguard] 事實核對攔截：route={route} 產生 {attempts} 次，"
              f"第一次的違規={kinds}，最終{'仍未通過' if blocked else '已修正'}"
              f"｜{query[:40]}", flush=True)
        if llm.LEARN_RUNTIME:
            for x in grounding.blocking(first):
                rule = lessons.rule_for(x.kind)
                if rule:
                    lessons.record(route, rule, question=query, wrong=x.detail,
                                   source="runtime")


def install() -> bool:
    """把核對掛到 agentkit 的工具迴圈上。回傳有沒有真的掛上去。"""
    try:
        from agentkit import delegate
        from agentkit.envelope import EV_DELTA  # guarded() 用它認出文字事件
    except Exception as e:  # noqa: BLE001
        print(f"[toolguard] agentkit 不可用，工具模式沒有核對：{e}")
        return False

    orig = getattr(delegate, "stream_delegated", None)
    if orig is None or getattr(orig, "_partrag_guard", False):
        return False

    def guarded(spec: Any, req: Any, found: dict[str, Any]
                ) -> Iterator[tuple[str, dict]]:
        parts: list[str] = []
        tools: list[tuple[str, str]] = []
        drain()  # 這條執行緒上如果還留著上一次的殘渣，先丟掉
        try:
            for ev, payload in orig(spec, req, found):
                # 每一次交還控制權就把剛剛那支工具的輸出收走（同一條執行緒、
                # 同一次 next()，所以收得到而且不會收到別人的）。
                tools += drain()
                # ★ 文字**全部攔下來**，一個字都不先送。
                #   核對要看完整答案才做得了；而一旦有一個字到了前端就收不回來，
                #   使用者會同時看到「已經印出來的幻覺」與「後來補上的更正」，
                #   那比不核對更糟。代價是感受上「想比較久」，不是「先講錯再改口」。
                if ev == EV_DELTA and payload.get("t"):
                    parts.append(payload["t"])
                    continue
                yield ev, payload
            tools += drain()
            yield from _finish(spec, req, found, "".join(parts).strip(), tools)
        except GeneratorExit:
            # 使用者按了停止。把已經產生的原樣送出去，不做核對（來不及了），
            # 也不要吞掉 —— 空白畫面比未核對的半段話更難解釋。
            raise
        finally:
            drain()

    guarded._partrag_guard = True  # type: ignore[attr-defined]
    delegate.stream_delegated = guarded
    return True
