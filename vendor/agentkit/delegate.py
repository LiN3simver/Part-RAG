"""委派迴圈：讓一個 agent 在回答時去呼叫下游 agent 的工具。

這是「A2A」真正發生的地方。原生 tool-calling 迴圈，不做規則路由 ——
gateway 支援原生 tool use，退回規則路由只會讓
「先查相似料號、再查它的 BOM、再選料」這種鏈變成寫死的 if/else。

    使用者 -> bom agent：「這條 600mm MCIO 線要哪些料？」
      step1  partrag__search_parts   {"query": "MCIO 600mm"}
      step2  material__suggest_material {"role": "STA", ...}
      -> 用兩邊的結果作答

★ 為什麼每一步都要吐 trace 事件
  委派鏈失敗的方式不只有「答錯」：下游沒開、下游回 not_ready、模型呼叫了
  參數不對的工具，這三種在最終文字上長得一模一樣（都是「查不到」）。
  把每一步的 agent / 工具 / 參數 / ok / 耗時送到前端，才有辦法分辨到底是
  資料真的沒有，還是有個服務忘了啟動。

★ 只掛下游（registry 的 downstream），不掛上游。
  bom 可以呼叫 material / partrag；material 不能回頭呼叫 bom。
  這是防環的最便宜做法：圖是單向的，就不可能繞回來。
"""
from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterator
from typing import Any

from . import client as peer_client
from . import gateway, registry
from .envelope import EV_DELTA, EV_ERROR, EV_RESULTS, EV_TRACE

# 一次工具結果餵給模型的上限。下游可能回幾百筆，整包塞進去會把 thinking
# 的預算擠掉（實測 thinking 就要 600~900 token），反而拿不到答案。
RESULT_CHARS = 6000
# trace 事件裡給前端看的摘要長度
PREVIEW_CHARS = 400
# ★ 連續幾次工具呼叫失敗就停止嘗試、直接逼問答案。
#   與 gateway.MAX_STEPS（步數上限）互補：MAX_STEPS 管「總共可以試多久」，
#   這個管「一直撞牆時多快放棄」。三次連續失敗代表這條路走不通，
#   而每多試一步就多一次 thinking（實測 600~900 token）。
MAX_CONSECUTIVE_FAILS = int(os.environ.get("A2A_MAX_CONSECUTIVE_FAILS", "3"))

# ★ 委派模式**單獨**的步數上限。沒設就沿用 gateway.MAX_STEPS（行為不變）。
#
#   為什麼需要跟 gateway.MAX_STEPS 分開：同一個旋鈕在服務兩種成本相反的工作。
#     「只用自己的工具」 一步很快，而且分流是確定性的、資料本來就在
#                        context 裡 —— 多跑幾步多半是白繞（模型可能連呼叫
#                        一串工具才作答，比不用工具慢好幾倍），所以要壓低。
#     「委派下游」       每次 ask_<agent> 要幾十秒（下游要跑自己的 LLM
#                        迴圈），而且**每一個子問題都需要自己的一步**。
#
#   一個旋鈕只能二選一：壓低會讓兩段式問句（「BOM 有什麼？還有工程圖嗎？」）
#   問不完，調高則讓不委派的那條路多繞好幾十秒。所以拆成兩個。
MAX_STEPS_DELEGATE = int(os.environ.get("A2A_MAX_STEPS_DELEGATE", "0"))


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + f"…（截斷，原長 {len(s)} 字）"


def _dumps(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, default=str)


# 工具結果裡「可以砍筆數」的清單欄位。砍筆數而不是砍字元，理由見 _fit()。
_TRIMMABLE = ("results", "nearby", "roles", "comparables", "values", "hits", "breakdown")


def _fit(v: Any, budget: int) -> str:
    """把工具結果塞進預算內，**砍筆數不砍字元**。

    ★ 為什麼不能直接 s[:budget]
      寬表工具（一次回好幾列 × 幾十欄）被字元截斷會在 JSON 中間切斷，
      模型收到的是一段語法壞掉、數字與 null 密集的殘骸。結果不是「少講
      一點」而是**整段答案崩掉**：輸出變成數字亂碼，還把截斷標記當成內容抄進去。
      所以一律砍「筆數」——少三筆案例模型還是能正確回答，
      半個 JSON 只會讓它開始編。
    """
    s = _dumps(v)
    if len(s) <= budget:
        return s
    if isinstance(v, dict):
        for key in _TRIMMABLE:
            items = v.get(key)
            if isinstance(items, list) and len(items) > 1:
                keep = max(1, len(items) // 2)
                return _fit(
                    {**v, key: items[:keep],
                     f"{key}_truncated": f"原有 {len(items)} 筆，只給前 {keep} 筆"},
                    budget,
                )
    # 沒有可砍的清單（例如單筆但欄位極多）才退回字元截斷
    return _clip(s, budget)


# 主題命中時，先幫模型撈回來的下游資料上限（總量，會平分給命中的下游）。
PREFETCH_CHARS = int(os.environ.get("A2A_PREFETCH_CHARS", "3000"))


def _topic_hit(query: str, tags: list[str]) -> list[str]:
    """問句命中了下游的哪幾個主題關鍵詞。

    ★ ASCII 關鍵詞（`8D` / `CAR`）要看字界，中文才用純子字串比對。
      不看字界的話料號裡的字母會誤中 —— 料號內含 `CAR` 這種字母串
      （例如 `ABCAR-0001`）會讓每一題都去 prefetch 一次。
    """
    q = query or ""
    ql = q.lower()
    out: list[str] = []
    for raw in tags:
        t = (raw or "").strip()
        if len(t) < 2:
            continue
        if t.isascii():
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(t.lower())}(?![A-Za-z0-9])", ql):
                out.append(t)
        elif t in q:
            out.append(t)
    return out


def _flatten_hit(h: dict[str, Any]) -> str:
    """把一張卡片壓成一段純文字。kv 的標籤/值與 text 區塊都要留。

    ★ 只取 hits 的 title/body 不夠：下游最有價值的內容常常在 blocks 裡
      （car8d 的 D3 根本原因、partrag 的材料表都是 text/table 區塊）。
      壓平之後模型才讀得到。
    """
    parts = [f"【{h.get('id') or ''}】{h.get('title') or ''}".strip()]
    if sub := h.get("subtitle"):
        parts.append(str(sub))
    if body := h.get("body"):
        parts.append(str(body))
    for b in h.get("blocks") or []:
        kind = b.get("kind")
        title = b.get("title") or ""
        if kind == "kv":
            rows = "；".join(
                f"{k}={'未填' if v is None else v}" for k, v in (b.get("rows") or []))
            if rows:
                parts.append(f"{title}：{rows}")
        elif kind == "text":
            if txt := (b.get("text") or "").strip():
                parts.append(f"{title}：{txt}")
        elif kind == "table":
            head = "/".join(str(x) for x in (b.get("head") or []))
            rows = " | ".join(
                "/".join(str(x) for x in r) for r in (b.get("rows") or [])[:12])
            if rows:
                parts.append(f"{title}（{head}）：{rows}")
    return "\n".join(p for p in parts if p)


def _prefetch_peers(
    query: str, peer_ids: list[str]
) -> tuple[list[dict[str, Any]], str, list[tuple[str, str, list[dict[str, Any]]]]]:
    """問句命中下游主題時，**先替模型把資料撈回來**。

    回 (trace 事件, 要塞進問句的文字, [(agent_id, 名稱, 卡片)…])。

    ★ 第三個回傳值是**給使用者看的卡片**，不是給模型的。
      光把資料塞進問句，畫面上只會有上游自己的卡片 —— 使用者看到答案在講
      客訴單的根本原因，底下的卡片卻是三張詢價單，對不起來也沒辦法核對。
      所以下游的卡片要一起送到畫面，跟直接問那個 agent 看到的一樣。

    ★★ 為什麼一定要做這件事（靠提示詞解不掉）

      典型失敗：問「<料號> 有被客訴過嗎？」，上游語料剛好也有這顆料的紀錄，
      模型於是判定「我有資料」，**一個工具都沒呼叫**，直接回「檢索結果中
      未包含任何客訴記錄」—— 而客訴 agent 就掛在它的工具清單裡，而且真的有資料。
      換一顆上游語料裡查不到的料號就會正常委派，
      也就是說**失敗條件是「自己碰巧有這個料號」**。

      試過在 system 裡加一條明確規則（「碰巧有這個料號不等於能回答這個主題」，
      而且把下游的主題關鍵詞也列了出來）—— 本地小模型**照樣不呼叫**。
      很長的 system 裡多一條規則改變不了它的行為。

      所以改成確定性的：命中主題就先打下游的 `/a2a/search`，
      把結果直接放進**問句**裡。模型不必決定要不要問 —— 資料已經在眼前。

    ★ 為什麼用 `/a2a/search` 而不是 `ask_<agent>`
      前者是每個 agent 都有的統一檢索、**不經 LLM**（亞秒級）；
      後者要跑下游自己的工具迴圈與 LLM（以十秒計）。這是每一題都可能發生的
      事前動作，不能花十秒。下游的 `/a2a/search` 自己會從問句認出料號／編號
      （那是確定性的正則，不是猜的）。

    ★ 撈不到東西也要講。空手而回時塞一句「已經問過它、它也沒有」——
      不講的話模型會回到「我沒有這個欄位所以不知道」，那正是要避免的答法。
    """
    traces: list[dict[str, Any]] = []
    chunks: list[str] = []
    cards: list[tuple[str, str, list[dict[str, Any]]]] = []
    matched: list[tuple[str, dict[str, Any], list[str]]] = []
    for aid in peer_ids:
        c = peer_client.card(aid)
        if c.get("status") == "offline" or not (c.get("capabilities") or {}).get("search"):
            continue
        if tags := _topic_hit(query, c.get("tags") or []):
            matched.append((aid, c, tags))
    if not matched:
        return traces, "", cards

    budget = max(600, PREFETCH_CHARS // len(matched))
    for aid, c, tags in matched:
        name = c.get("name", aid)
        t0 = time.perf_counter()
        try:
            out = peer_client.search(aid, query, top_k=3)
        except Exception as e:      # noqa: BLE001 -- 事前動作失敗不該讓整題掛掉
            traces.append({"phase": "prefetch", "agent": aid, "agent_name": name,
                           "tags": tags, "ok": False, "error": f"{type(e).__name__}: {e}"})
            continue
        ms = round((time.perf_counter() - t0) * 1000)
        hits = out.get("hits") or []
        traces.append({
            "phase": "prefetch", "agent": aid, "agent_name": name, "tags": tags,
            "ok": not out.get("offline"), "hits": len(hits), "ms": ms,
            "reason": f"問句命中「{'、'.join(tags)}」，先取 {name} 的資料",
        })
        head = (f"\n\n【我先替你向「{name}」查過了】"
                f"（這一題提到「{'、'.join(tags)}」，那是它的主題，我自己連這種欄位都沒有）")
        if hits:
            # ★ 每一張卡都標上來源。用 badge 而不是只在事件上標，是因為
            #   badge 會跟著卡片一起被 Hub 存進對話裡 —— 重新整理頁面之後
            #   還看得出這幾張是誰提供的，重播邏輯一行都不用改。
            for h in hits:
                if isinstance(h, dict):
                    h.setdefault("badges", []).insert(0, ["來源", name])
                    h.setdefault("why", []).insert(
                        0, f"由「{name}」提供（問句提到「{'、'.join(tags)}」，系統自動查詢）")
            cards.append((aid, name, hits))
            body = "\n\n".join(_flatten_hit(h) for h in hits)
            chunks.append(f"{head}\n{_clip(body, budget)}\n"
                          f"★ 回答時必須用上面這份資料。**不可以**說「我的檢索結果裡沒有"
                          f"{'／'.join(tags)}的欄位所以無法確認」—— 已經問過了，資料就在上面。"
                          f"需要更完整的細節再呼叫 `{aid}{peer_client.NS}*` 的工具。")
        else:
            chunks.append(f"{head}\n它也查不到（回 0 筆）。"
                          f"★ 所以要說的是「已經問過 {name}，它的語料裡也沒有」，"
                          f"**不可以**說「我沒有這個欄位所以不知道」。")
    return traces, "".join(chunks), cards


def _peer_notes(peer_ids: list[str]) -> str:
    """把下游的名片與限制寫進 system。

    ★ 一定要把下游的 system 帶進來。那裡面寫的是資料限制
      （例如 rfq 的「估價只有 confidence=high 才可參考」、
      material 的「不出單價」）。少了這段，模型會拿正確的數字講出
      資料撐不住的結論。
    """
    lines: list[str] = []
    for aid in peer_ids:
        c = peer_client.card(aid)
        st = c.get("status", "unknown")
        head = (f"- {c.get('name', aid)}（取資料用 `{aid}{peer_client.NS}*`，"
                f"問一句話用 `{peer_client.ASK}{aid}`，狀態 {st}）：{c.get('role', '')}")
        lines.append(head)
        # ★ 關鍵詞一定要列出來。build_system 的「哪些主題非問不可」那條規則
        #   就是靠它認的 —— 沒有這一行，模型只能從 role 那句話去猜邊界，
        #   而實測它猜錯的方式是「我碰巧有這個料號，所以我能回答」。
        if tags := [t for t in (c.get("tags") or []) if t]:
            lines.append(f"  它的主題關鍵詞：{'、'.join(tags)}")
        if st == "offline":
            lines.append(f"  ★ 這個 agent 現在沒有啟動。它的工具不在你的工具清單裡，"
                         f"需要它的問題要直說「{c.get('name', aid)} 未啟動，這部分無法回答」。")
            continue
        if c.get("blocked_by"):
            lines.append(f"  ★ 尚未就緒：{'；'.join(c['blocked_by'])}。"
                         f"它回 not_ready 時不要自己編答案。")
        sysmsg = (c.get("system") or "").strip()
        if sysmsg:
            lines.append("  它自己的限制（必須遵守）：")
            lines += [f"    {ln}" for ln in _clip(sysmsg, 900).splitlines() if ln.strip()]
    return "\n".join(lines)


def build_system(spec: Any, peer_ids: list[str]) -> str:
    own = (spec.system or "").strip()
    notes = _peer_notes(peer_ids)
    ref = registry.get(spec.agent_id)
    parts = [own] if own else [f"你是「{ref.name}」：{ref.role}。"]
    if notes:
        parts.append(
            "你可以呼叫下面這些下游 agent 的工具來補足自己沒有的資料。\n"
            f"{notes}\n\n"
            "呼叫規則：\n"
            "- ★ 先看自己手上的資料。上面「我自己的檢索結果」或自己的工具就足以回答時，"
            "**直接回答，不要呼叫任何下游** —— 委派不是每題都要做的儀式，"
            "多問一輪要多花好幾秒與好幾倍 token，答案還可能被無關的資料帶偏。\n"
            "- 只有在缺的那塊資料自己真的沒有時才委派，而且只問缺的那一塊，"
            "不要把整題轉包出去。\n"
            # ★★ 這一條是上面那條的必要配套，兩條要一起讀。
            #   上游語料碰巧有這個料號時，模型會判定「我有資料」而不呼叫客訴
            #   agent，直接回「沒有相關欄位，無法確認」（見 _prefetch_peers）。
            #   只寫「自己有資料就別問人」而不界定什麼叫「有資料」，
            #   模型就會把「有這個料號」當成「能回答這個主題」。
            "- ★★ 但「自己手上有資料」指的是**能回答這一題的資料**，"
            "不是「碰巧也有這個料號／編號」。上面每個下游都列了它的主題關鍵詞，"
            "那些主題是你**連欄位都沒有**的領域 —— 問題落在那些主題上時，"
            "不論你自己查到多少東西，都要呼叫對應的下游。\n"
            "  自己查到了那個料號、但要問的是別人的主題 -> 照樣要問，"
            "然後把兩邊的結果合起來回答。\n"
            "  ★ 這種情況**絕對不要**寫成「我的檢索結果裡沒有這個欄位，所以無法確認」"
            "—— 那不是誠實，是漏問。真的問過而對方也沒有，才可以那樣說。\n"
            "- 需要別人的資料就直接呼叫工具，不要猜。\n"
            f"- 知道要哪一筆資料 -> 用 `<agent>{peer_client.NS}<工具>`；"
            f"問題屬於對方專業但不確定用哪支工具 -> 用 `{peer_client.ASK}<agent>` "
            "把整句問題交給它（它看不到你的對話，問題要自帶料號與規格）。\n"
            "- 工具回 {\"ok\": false} 或 not_ready 時，如實說明是哪一個 agent、"
            "為什麼拿不到，**絕對不要用猜的料號或數字補位**。\n"
            "- 引用下游的結果時要說是哪個 agent 提供的，方便使用者自己去查。"
        )
    else:
        # peer_ids 空 = 獨立模式（使用者只開了「用自己的工具」）。
        # 這段要寫出來：不講的話模型會以為自己什麼都能做，
        # 缺資料時傾向自己編一個數字，而不是說「這部分要別的 agent」。
        parts.append(
            "你現在是**獨立模式**：只有自己的工具，沒有任何其他 agent 可以呼叫。\n"
            "- 需要計算或查表就用自己的工具，不要憑印象給數字。\n"
            "- 缺的那塊資料要別的 agent 才有時，直說「這部分需要 ○○ Agent，"
            "目前沒有開放委派」，不要自己補一個。"
        )
    parts.append(
        "工具呼叫的紀律（違反的話使用者會完全看不到答案）：\n"
        "- 只能呼叫工具清單裡列出的名字，不要自己發明工具（例如 `summarize`——"
        "沒有這種工具，摘要是你自己讀資料後直接用文字回答的事，不是呼叫工具）。\n"
        "- ★ 拿到足夠回答的資料就**立刻用文字作答**，不要再呼叫任何工具。"
        "工具呼叫的次數有上限，用完了你就沒有機會講話，使用者會拿到一片空白。\n"
        "- ★ 同一支工具不要反覆改寫查詢重試。第一次查到了就用第一次的結果；"
        "查不到就直接說「查不到」並說明是哪裡沒有資料，換個寫法再查通常不會有新東西。\n"
        "- 工具回的資料不完整時，就用已經有的部分作答並講明哪一塊缺，"
        "不要為了補齊而一直查下去。\n"
        "- ★ 答案要寫成**給人看的段落文字**，不可以包成 JSON、不要輸出 "
        "{\"summary\": …} 這種信封，也不要用 \\u 逃逸中文。直接寫字就好。"
    )
    parts.append("回答用繁體中文，直接講結論，不要客套開場白。")
    return "\n\n".join(parts)


def _resolve_bare(spec: Any, tool: str, allowed: list[str]) -> str | None:
    """模型漏掉命名空間時，把裸工具名對回**唯一**擁有它的下游。回 agent_id。

    ★ 為什麼要補這個（反覆出現的模式）
      模型在步數快用完時常會吐一次 `search_parts`（沒有 `partrag__` 前綴），
      而那確實是 partrag 的工具名。舊行為是回「未知的工具」——
      白燒一整步（還多一次 thinking），而它想做的事其實
      完全合法，只是名字少了前綴。

      掛給模型的名字一律帶前綴，所以裸名一定是筆誤，不是另一個意思。
      修正一個**唯一解**的筆誤比退件划算；有兩個以上的下游都有同名工具時
      不猜，照舊退件（下面 len(hits) != 1 那條）。

    ★ 只在 allowed（使用者這一題同意的下游）裡找。不能因為「猜得出來」
      就把資料送去一個沒被授權的 agent —— 那會繞過獨立模式那道鎖。
    """
    hits = [aid for aid in allowed
            if any(t.get("name") == tool and t.get("ready", True)
                   for t in (peer_client.card(aid).get("tools") or []))]
    return hits[0] if len(hits) == 1 else None


def _dispatch(spec: Any, name: str, args: dict[str, Any],
              peers_on: bool = True,
              allowed: list[str] | None = None) -> tuple[str, dict[str, Any]]:
    """執行一次工具呼叫。回 (執行者 agent_id, 結果)。

    peers_on=False（獨立模式）時任何指向別的 agent 的呼叫一律拒絕。模型本來
    就看不到那些工具，這裡是第二道鎖：使用者沒同意就不能有任何資料離開這個
    agent，不能只靠「模型應該不會去叫」。
    """
    if not peers_on and (name.startswith(peer_client.ASK) or peer_client.split_ns(name)[0]):
        return spec.agent_id, {
            "ok": False,
            "error": "目前是獨立模式（使用者沒有勾選「委派下游」），不能呼叫其他 agent。"
                     "只能用自己的工具，或直說這部分要對方才有。",
        }
    # ask_<agent>：把整句問題交給下游那個 agent（agent 之間的「講話」介面）
    if name.startswith(peer_client.ASK):
        aid = name[len(peer_client.ASK):]
        if aid not in registry.BY_ID:
            return spec.agent_id, {"ok": False, "error": f"未知的 agent：{aid}"}
        if aid not in registry.get(spec.agent_id).downstream:
            # 圖是單向的（見模組 docstring 的防環說明）。模型偶爾會想往上問，
            # 直接擋掉並說明，比讓它繞成 A->B->A 好。
            return spec.agent_id, {
                "ok": False,
                "error": f"{aid} 不是 {spec.agent_id} 的下游，不能呼叫（避免 A→B→A 繞回來）",
                "allowed": list(registry.get(spec.agent_id).downstream),
            }
        q = (args.get("question") or "").strip()
        if not q:
            return aid, {"ok": False, "error": "question 不可為空"}
        return aid, peer_client.ask_agent(aid, q)

    aid, tool = peer_client.split_ns(name)
    if aid is None:
        t = spec.tool(tool)
        if t is None:
            # 自己沒有這支 —— 先看是不是漏了命名空間（見 _resolve_bare）
            if peers_on and (peer := _resolve_bare(spec, tool, allowed or [])):
                out = peer_client.call_tool(peer, tool, args)
                if isinstance(out, dict):
                    out = {**out, "note_from_hub":
                           f"你呼叫的 `{tool}` 少了命名空間，已當成 "
                           f"`{peer}{peer_client.NS}{tool}` 執行。"
                           f"下次請寫完整的名字。"}
                return peer, out
            own_names = sorted(x.name for x in spec.tools if x.ready)
            return spec.agent_id, {
                "ok": False,
                "error": f"未知的工具：{name}（不存在，不要再呼叫這個名字）",
                "available_tools": own_names,
            }
        if not t.ready:
            return spec.agent_id, {"ok": False, "error": f"工具 {tool} 尚未可用",
                                   "reason": t.not_ready_reason}
        try:
            out = t.handler(args)
        except Exception as e:
            return spec.agent_id, {"ok": False, "error": f"{type(e).__name__}: {e}"}
        return spec.agent_id, out if isinstance(out, dict) and "ok" in out else {"ok": True, "result": out}
    return aid, peer_client.call_tool(aid, tool, args)


def _forced_answer(spec: Any, req: Any, own_ctx: str,
                   collected: list[tuple[str, str]]) -> Iterator[tuple[str, dict]]:
    """把工具的框架整個拆掉，用一則普通的問答逼出文字答案。

    ★ 為什麼「最後一輪不掛工具」還不夠
      不掛 tools，但歷史裡還是一長串 assistant.tool_use / user.tool_result
      的來回 —— 這顆模型會照著那個樣式繼續生下一個工具呼叫，而且因為沒有
      tools 參數，它是用「純文字寫 OpenAI 格式 JSON」的方式生（見 gateway
      docstring 第 6 點）。我們把那段 JSON 濾掉之後就什麼都不剩，
      使用者拿到的還是空白答案。

      所以這裡不是「少給工具」，而是**換一個對話**：只有一則 user 訊息，
      內容是「原本的問題 + 已經查到的資料」，歷史裡一個 tool_use 都沒有。
      沒有可以模仿的樣式，模型才會真的把結論寫出來。
    """
    lines = [req.query, ""]
    if own_ctx.strip():
        lines.append(own_ctx.strip())
    if collected:
        lines.append("以下是為了回答這一題、已經查到的資料：")
        # 預算平均分給每一次呼叫，免得第一次的結果就把後面的擠掉
        per = max(600, RESULT_CHARS // max(1, len(collected)))
        for name, body in collected:
            lines.append(f"\n【{name} 的結果】\n{_clip(body, per)}")
    else:
        lines.append("（沒有任何工具成功回傳資料。）")
    lines.append(
        "\n請直接用繁體中文把上面那個問題的答案寫出來。\n"
        "★ 你現在沒有任何工具可以呼叫，也不要輸出 JSON —— 只能寫文字。\n"
        "★ 資料裡有的就照實講；資料不足的部分明講是哪一塊缺，不要自己補數字。"
    )

    # system 也要換掉：原本那份寫滿了「怎麼呼叫工具」，留著等於還在提示它呼叫。
    ref = registry.get(spec.agent_id)
    own = (spec.system or "").strip() or f"你是「{ref.name}」：{ref.role}。"
    system = (own + "\n\n現在是**收尾**：資料都查完了，你的工作只剩下用文字把結論講清楚。"
                    "不要呼叫工具，不要輸出 JSON，用繁體中文直接講結論。")

    got = False
    for kind, payload in gateway.stream_step(system, [{"role": "user", "content": "\n".join(lines)}], None):
        if kind == "text":
            got = True
            yield EV_DELTA, {"t": payload}
        elif kind == "error":
            if isinstance(payload, dict):
                yield EV_ERROR, {"error": f"{payload['title']}：{payload['detail']}",
                                 "hint": payload["fix"], "diag": payload}
            else:
                yield EV_ERROR, {"error": payload}
            return
    if not got:
        # 連拆掉框架都問不出一句話。這時候只能誠實講，不要留空白讓人以為壞了。
        yield EV_DELTA, {
            "t": "（模型跑完了工具但始終沒有產出文字答案。"
                 "上方軌跡與下方卡片是已經查到的真實資料，請直接參考。）"
        }


def stream_delegated(spec: Any, req: Any, found: dict[str, Any]) -> Iterator[tuple[str, dict]]:
    """跑工具迴圈，yield (event, payload)。事件用 envelope 的統一詞彙。

    ★ 下游只有 `req.delegate` 為真時才掛上來。
      `req.use_tools` 是「獨立但可以用自己的工具」：同一個迴圈，只是
      allowed 為空 -> peer_tools() 回空、build_system() 也不會寫進任何
      下游的名片與呼叫規則。模型看不到別人的工具，就不可能呼叫別人 ——
      這比在 _dispatch 裡擋要可靠，模型不會去試一個它沒看到的東西。
    """
    ok, detail = gateway.available()
    if not ok:
        yield EV_DELTA, {"t": f"（LLM gateway 不可用：{detail}。以上為檢索結果，未呼叫任何工具。）"}
        return

    ref = registry.get(spec.agent_id)
    peers_on = bool(getattr(req, "delegate", False) and getattr(spec, "can_delegate", False))
    allowed = [p for p in ref.downstream if not req.peers or p in req.peers] if peers_on else []
    tools = [t.for_model() for t in spec.tools if t.ready] + peer_client.peer_tools(allowed)
    system = build_system(spec, allowed)
    # 委派時走自己那份預算（沒設就是 gateway.MAX_STEPS），理由見 MAX_STEPS_DELEGATE
    max_steps = MAX_STEPS_DELEGATE if (peers_on and MAX_STEPS_DELEGATE) else gateway.MAX_STEPS

    # 自己檢索到的東西先給模型當背景，它才知道還缺什麼要去問下游。
    #
    # ★ 有 spec.context_fn 就用它，不要只挑 found["results"]。
    #   results 只是 found 的一格：partrag 的工程圖在 drawings、清單統計在
    #   listing、查無料號在 not_found。只餵 results 的後果不是「講得少一點」，
    #   而是模型明明有資料卻說查不到 —— 而同一份資料已經被 hits_fn 畫成卡片
    #   擺在它的答案旁邊了（見 AgentSpec.context_fn 的說明）。
    own_ctx = ""
    hits = found.get("results") or []
    ctx_fn = getattr(spec, "context_fn", None)
    if ctx_fn is not None:
        try:
            text = (ctx_fn(req.query, found) or "").strip()
        except Exception as e:  # noqa: BLE001 -- 背景講不出來也要能繼續跑工具
            text = ""
            print(f"[delegate] context_fn 失敗，退回 results：{type(e).__name__}: {e}")
        if text:
            # 這段是自然語言不是 JSON，_fit 砍不了筆數，用字元截斷（見 _fit）
            own_ctx = f"\n\n我自己的檢索結果：\n{_clip(text, RESULT_CHARS)}"
    if not own_ctx and hits:
        # 同樣走 _fit：自己的檢索結果一樣可能是幾十欄乘八列
        own_ctx = (f"\n\n我自己的檢索結果（{len(hits)} 筆）：\n"
                   f"{_fit({'results': hits[:8]}, RESULT_CHARS)}")
    elif not own_ctx and found.get("unavailable"):
        own_ctx = f"\n\n（我自己沒有可用的語料：{found.get('note', '')}）"

    # ★★ 問句命中下游主題時，**先替模型把下游的資料撈回來**，直接塞進問句。
    #   理由見 _prefetch_peers()：模型在「自己碰巧有這個料號」的時候
    #   一個工具都不呼叫，而光在 system 裡加規則改不掉。
    #   放在**問句**裡而不是 system 裡是刻意的 —— 這類模型對最後一則
    #   user 訊息的注意力遠高於很長的 system（同 _forced_answer 的觀察）。
    peer_ctx = ""
    if allowed:
        pre_traces, peer_ctx, pre_cards = _prefetch_peers(req.query, allowed)
        for t in pre_traces:
            yield EV_TRACE, {"step": 0, **t}
        # ★ 下游的卡片也要送到畫面上，跟直接問那個 agent 看到的一樣。
        #   `append: True` 是給呼叫端的訊號：**這一包是加上去的，不要蓋掉
        #   前面那一包**（server.py 在跑這個迴圈之前已經送過上游自己的
        #   results 了）。Hub 與前端都認這個欄位。
        for aid, name, hits in pre_cards:
            yield EV_RESULTS, {
                "hits": hits,
                "raw": {"prefetched_from": aid, "query": req.query},
                "append": True,
                "agent": aid,
                "agent_name": name,
            }

    messages: list[dict[str, Any]] = []
    for h in req.history[-8:]:  # 只帶最近幾輪，歷史太長會擠掉 thinking 預算
        if h.get("role") in ("user", "assistant") and (h.get("content") or "").strip():
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": req.query + own_ctx + peer_ctx})

    # 已經執行過的呼叫簽名（工具名＋參數）。見下面重複呼叫的處理。
    seen: set[str] = set()
    # 每一次成功的工具呼叫拿到什麼。逼問最後那一輪要靠它把資料重講一次。
    collected: list[tuple[str, str]] = []
    # 模型到底有沒有講出過一個字。沒有的話結尾一定要走 _forced_answer()。
    answered = False
    # ★ 連續失敗的工具呼叫數。連續 MAX_CONSECUTIVE_FAILS 次就不再給工具，
    #   直接去逼問最後一次。
    #
    #   為什麼 seen（相同參數不重複執行）擋不住這件事：那一招只認**一模一樣**
    #   的簽名。模型查不到東西時的另一種傾向是「把參數微調一下再試」——
    #   料號多補一個 -1、把 max_level 從 12 改成 11 —— 每一次簽名都不同，
    #   所以每一次都會真的去打一次 DB 或下游，然後拿到同一種失敗。
    #   五步全部燒在這上面，而每一步都要重新 thinking 一次。
    #
    #   連續三次失敗代表的是「這條路走不通」，不是「再試一次就會好」。
    #   歸零條件是**任何一次成功**，所以偶發的逾時不會累積成死刑。
    consecutive_fails = 0

    # ★ 重複呼叫的次數。**刻意跟 consecutive_fails 分開算。**
    #
    #   常見情形：下游第一次就回了 ok=true 與完整資料，模型接著用
    #   **一模一樣的參數**又呼叫好幾次，每次都被 seen 擋下並回「你已經問過了」。
    #   舊版把那幾次算成 consecutive_fails，於是：
    #     - 軌跡上顯示「連續 3 次工具呼叫失敗」——**什麼都沒有失敗**，
    #       而且 agent 欄標的是上游自己（擋下來的是這裡），讀起來像下游壞了
    #     - 白燒好幾步與好幾次 thinking
    #
    #   重複呼叫跟「工具失敗」是兩件不同的事：資料已經在 collected 裡了，
    #   模型是卡住不是查不到。
    #
    # ★★ 處置是「把那支工具從模型的清單裡拿掉」，**不是中止整個迴圈**。
    #
    #   舊版第二次重複就 break 去逼問答案。以兩段式問句
    #   （「<料號> 底下的 bom 有什麼？還有工程圖嗎？」）為例，那個處置本身
    #   就是失敗原因：
    #     step1 bom__get_bom  ok
    #     step2 bom__get_bom  同參數重複
    #     step3 bom__get_bom  同參數重複 -> 中止
    #   問句的第二段（工程圖）從來沒有被問到，而**步數預算大半沒用到**。
    #   也就是說殺死這一題的是這個中止，不是步數不夠、也不是下游有問題。
    #
    #   為什麼「移除工具」比「回一句你已經問過了」有效：後者是提示，而這類
    #   模型對提示無效（見 _prefetch_peers()：很長的 system 裡多一條規則
    #   改變不了它的行為）。移除之後它是**看不到**那支工具，
    #   而本模組的防環也正是靠同一個道理 —— 「模型看不到的東西不可能呼叫」
    #   比在 _dispatch 裡擋可靠。剩下的步數於是留給還沒回答的那一段：
    #   它只能改用別的工具（例如 ask_bom 去問工程圖）或直接作答。
    #
    #   ★ 計數是**逐工具**的，不是整題一個總數。卡住的是某一支工具，
    #     移除也只移除那一支；別的工具還在正常工作不該被連坐。
    #   ★ ask_<agent> 永不移除：那是「不確定用哪支工具就整句問對方」的
    #     退路，把退路拿掉等於把跨 agent 的能力關掉（見 client.ask_agent）。
    #   ★ 歸零條件跟 consecutive_fails 一樣是那支工具**自己成功一次**：
    #     換了參數又查到東西代表它在正常工作，不是卡在原地。
    #   ★ 為什麼是 2 而不是 1（試過 1，被否決了，不要再改）
    #     「第一次重複就收回」看起來可以省掉一步白燒，而且那句「你已經問過了」
    #     確實幾乎不會改變模型的下一步。但那一步的容忍
    #     額度換到的不是提醒，是**換參數重試的空間**：
    #         step2 search_drawings{料號}  ok（查無）
    #         step3 search_drawings{料號}  重複        <- 設 1 就在這裡收回
    #         step4 search_drawings{品名}  ok  <- 換成品名才找到相關圖
    #     那往往是最有價值的一次結果，設 1 就拿不到了。
    #     「重複一次、然後換個寫法查到東西」是有生產力的模式，不能連坐。
    #     所以維持 2：卡住的代價是兩步，換來換參數重試不會被誤殺。
    repeats_by_tool: dict[str, int] = {}
    MAX_REPEATS = 2

    step = 0
    while step < max_steps:
        step += 1
        final = None
        for kind, payload in gateway.stream_step(system, messages, tools):
            if kind == "text":
                if (payload or "").strip():
                    answered = True
                yield EV_DELTA, {"t": payload}
            elif kind == "error":
                # payload 是 gateway.classify() 的診斷 dict（舊版是純字串，
                # 兩種都收）。攤成 error / hint / diag 三欄，前端才畫得出
                # 「發生什麼、該做什麼、要不要換一把新 API key」。
                if isinstance(payload, dict):
                    yield EV_ERROR, {
                        "error": f"{payload['title']}：{payload['detail']}",
                        "hint": payload["fix"],
                        "diag": payload,
                    }
                else:
                    yield EV_ERROR, {"error": payload}
                return
            else:
                final = payload

        if final is None:
            return

        # ★ 判斷「要不要執行工具」一定要看 content 裡有沒有 tool_use block，
        #   **不能看 stop_reason**。
        #   公司 gateway 會回四個 block（空 text / thinking / text / tool_use）而
        #   stop_reason 給的是 **'end_turn'**，不是 Anthropic 規格的 'tool_use'。
        #   照規格判斷的話症狀非常誤導：模型講了「我來呼叫材料建議 Agent」
        #   然後就停住，trace 一片空白，看起來像模型不肯用工具，
        #   其實工具呼叫已經生成了、只是被我們丟掉。
        calls = [b for b in final.content if getattr(b, "type", "") == "tool_use"]
        if not calls:
            note = gateway.empty_answer_note(final)
            if note:
                yield EV_DELTA, {"t": note}
                return
            # 沒有工具呼叫、也沒講出任何文字 —— 多半是整段回應都是被我們濾掉的
            # 工具呼叫 JSON（見 gateway 第 6 點）。拆掉框架再問一次。
            if not answered:
                yield from _forced_answer(spec, req, own_ctx + peer_ctx, collected)
            return
        messages.append({"role": "assistant", "content": gateway._assistant_content(final.content)})

        results: list[dict[str, Any]] = []
        for c in calls:
            args = c.input if isinstance(c.input, dict) else {}
            yield EV_TRACE, {"step": step, "phase": "call", "id": c.id,
                             "tool": c.name, "args": args}
            t0 = time.perf_counter()
            # ★ 一模一樣的呼叫不再執行第二次。
            #   模型查不到東西時的傾向是「換個寫法再查一次」，但參數常常
            #   原封不動 —— 同一支工具、同一組參數、同一份結果，白白燒掉一步。
            #   直接回一句「你已經問過了」比讓它再等一次 DB 有用：它拿到的是
            #   新資訊（該停了），而不是同一份讓它繼續糾結的空結果。
            sig = f"{c.name}:{json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)}"
            repeated = sig in seen
            if repeated:
                aid, out = spec.agent_id, {
                    "ok": False,
                    "error": "你剛才已經用一模一樣的參數呼叫過這個工具了，結果就在上面。"
                             "不要重複呼叫：改用已經拿到的資料直接作答，"
                             "或是換一組真的不同的參數。"
                             "★ 問句還有沒有回答到的部分，就去問那一部分。",
                }
            else:
                seen.add(sig)
                aid, out = _dispatch(spec, c.name, args, peers_on, allowed)
            ms = round((time.perf_counter() - t0) * 1000)
            body = _dumps(out)
            yield EV_TRACE, {
                "step": step, "phase": "result", "id": c.id, "tool": c.name,
                "agent": aid, "agent_name": registry.get(aid).name if aid in registry.BY_ID else aid,
                "ok": bool(out.get("ok")), "offline": bool(out.get("offline")),
                "repeated": repeated,
                "ms": ms, "preview": _clip(body, PREVIEW_CHARS),
            }
            if out.get("ok"):
                consecutive_fails = 0        # ★ 成功一次就歸零
                repeats_by_tool.pop(c.name, None)   # ★ 同上，見宣告處的說明
                collected.append((c.name, body))
            elif repeated:
                # ★ 不算 consecutive_fails：沒有任何工具失敗，是模型卡住了。
                #   算進去會讓軌跡寫「連續 N 次工具呼叫失敗」而下游其實好得很。
                n = repeats_by_tool[c.name] = repeats_by_tool.get(c.name, 0) + 1
                # 重複到上限就把這支工具從模型的清單裡拿掉（理由見宣告處）。
                # ask_ 是退路，永不移除。
                if n >= MAX_REPEATS and not c.name.startswith(peer_client.ASK):
                    kept = [t for t in tools if t.get("name") != c.name]
                    if len(kept) < len(tools):
                        tools = kept
                        yield EV_TRACE, {
                            "step": step, "phase": "drop_tool", "tool": c.name,
                            "reason": f"同一組參數重複呼叫 {n} 次，資料已在手上；"
                                      "把這支工具收回，剩下的步數留給問句還沒"
                                      "回答到的部分",
                            "tools_left": len(tools),
                        }
            else:
                consecutive_fails += 1
            results.append({
                "type": "tool_result",
                "tool_use_id": c.id,
                "content": _fit(out, RESULT_CHARS),
                **({"is_error": True} if not out.get("ok") else {}),
            })

        # ★ 連續失敗太多次：不要再給工具了，直接去逼問答案。
        #   放在這裡（送出 tool_result 之前就 break）是刻意的 ——
        #   _forced_answer() 會拆掉工具框架、把 collected 裡已經查到的東西
        #   重講一次，所以使用者拿到的是「我查到這些、這幾條路不通」，
        #   而不是一個空白畫面或是第四、第五次同樣的失敗。
        if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
            yield EV_TRACE, {
                "step": step, "phase": "abort",
                "reason": f"連續 {consecutive_fails} 次工具呼叫失敗，停止嘗試並直接作答",
            }
            break

        # ★ 工具全被收回了才收工。這是「移除工具」那條路的終點，不是
        #   「重複太多次」的懲罰 —— 只要還有一支工具在，就讓它繼續試，
        #   因為問句可能還有沒回答到的部分（見 repeats_by_tool 的說明）。
        #   另外 tools 空陣列不能送給 Messages API，這裡也是那個保護。
        if not tools:
            yield EV_TRACE, {
                "step": step, "phase": "abort",
                "reason": "所有工具都因為重複呼叫被收回，資料已取得，直接作答",
            }
            break

        # 快用完步數時先講一聲。放在同一則 user 訊息的最後一個 block ——
        # tool_result 之後可以接 text，但不能再多送一則 user 訊息
        # （Messages API 的角色必須交替）。
        if step == max_steps - 1:
            results.append({
                "type": "text",
                "text": "（這是最後一次可以呼叫工具的機會，下一輪就要交答案了。"
                        "請用目前手上的資料直接把結論寫出來。）",
            })
        messages.append({"role": "user", "content": results})

    # 步數用完，模型從頭到尾沒交出answer —— 拆掉工具框架逼問最後一次。
    # 這是「一定會有東西給使用者」的保證：資料都查到了，不能讓畫面空白。
    yield from _forced_answer(spec, req, own_ctx + peer_ctx, collected)
