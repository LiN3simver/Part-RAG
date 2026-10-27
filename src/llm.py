"""Local LLM gateway client.

The gateway at LLM_BASE_URL speaks the Anthropic Messages API, so the official
`anthropic` SDK drives it directly -- only base_url and auth differ.

Two behaviours observed against the live endpoint that shape this module:
  1. It returns `thinking` content blocks before the `text` block, so answer
     extraction must filter by block type rather than reading content[0].
  2. Thinking consumes the token budget, so max_tokens must be generous or the
     reply comes back empty with stop_reason="max_tokens".

It is a local model (currently an open-weight build; LLM_MODEL_FALLBACK is another
model, used only if the primary drops out of the gateway's model list) behind
an Anthropic-compatible shim, so Claude-only request fields (thinking,
output_config.effort, fallbacks) are NOT sent -- a shim will usually reject
them.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Iterator

import anthropic
import httpx
from dotenv import load_dotenv

from . import drawing_frames, grounding, hints, lessons
from . import taxonomy as T

load_dotenv()
# ★ LLM_BASE_URL / LLM_API_KEY / LLM_MODEL / LLM_MODEL_FALLBACK 集中放在
#   Agent_Hub_RAG_Frontend/.env（四個 agent 共用一份）。load_dotenv 不覆寫已存在的環境
#   變數，所以本專案 .env 若明確設了值還是贏；平常換模型/換 key 只需要
#   改 Agent_Hub_RAG_Frontend/.env 後重啟服務，不必動這個專案。
_HUB_ENV = Path(__file__).resolve().parents[2] / "Agent_Hub_RAG_Frontend" / ".env"
if _HUB_ENV.is_file():
    load_dotenv(_HUB_ENV)

BASE_URL = os.getenv("LLM_BASE_URL", "")
API_KEY = os.getenv("LLM_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "")
# 只有 MODEL 真的不在 gateway 的模型表裡時才切這顆，見 _effective_model()。
MODEL_FALLBACK = os.getenv("LLM_MODEL_FALLBACK", "")
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
# 狀態探測（diagnose）用的逾時。答題的 _client() 是 120 秒，因為 thinking 很慢；
# 「你還活著嗎」不該用同一個數字問。
PROBE_TIMEOUT = float(os.getenv("LLM_PROBE_TIMEOUT", "10"))

# 探測結果的快取秒數。
#
# ★ 為什麼一定要快取
#   diagnose(None) 不是讀本地設定，它**真的送一則 messages.create 到 gateway**
#   （內容 "ping"）。而它的兩個呼叫端是狀態查詢：main.health() 與 a2a._status()。
#   看板每 10 秒輪詢一次，就是一天 8,640 次計費請求，內容全是 "ping"。
#   而 diagnose() 自己有一個分類叫 rate_limit —— 輪詢夠密的話，這支端點會親手
#   製造出它自己回報的那個錯誤，而且看板上完全看不出來源是自己。
#
# ★ 為什麼 60 秒是對的數字
#   gateway 60 秒前是活的，現在幾乎確定還是活的；真的剛掛掉，晚 60 秒知道也不會
#   改變任何人的決策（LLM 掛掉是降級不是壞掉，檢索照樣回得出結果）。
#   換了新 key 想立刻確認的話走 invalidate_probe()，不必等 TTL。
PROBE_TTL = float(os.getenv("LLM_PROBE_TTL", "60"))

# 答完之後要不要做事實核對、抓到就重寫一次。
#
# ★ 為什麼預設開，即使它會讓失敗的那一題多花一次呼叫
#   核對本身是純字串比對（src/grounding.py），成本是零；只有**真的抓到幻覺**
#   的那一題才會多打一次 gateway。而幻覺那一題原本的輸出價值是負的 ——
#   它看起來像答案，使用者不會回頭去對卡片。多花五秒換掉一個錯答案，
#   在工程與採購的使用情境裡沒有懸念。
VERIFY = os.getenv("LLM_VERIFY", "1") not in ("0", "false", "False", "")
# 核對抓到的錯要不要寫進教訓庫（下一題的 prompt 會帶上）。
LEARN_RUNTIME = os.getenv("LESSON_LEARN_RUNTIME", "1") not in ("0", "false", "False", "")
# 語意層的 LLM 裁判。預設關：它每題多一次 gateway 呼叫，而它抓的那類問題
# （答非所問、規格張冠李戴）比字面幻覺少得多。跑分時用 --judge 打開。
JUDGE = os.getenv("LLM_JUDGE", "0") not in ("0", "false", "False", "")

_probe_cache: tuple[float, dict[str, Any]] | None = None

SYSTEM = """你是線材與連接器料號查詢助手，服務對象是工程與採購人員。

規則：
- 只根據「檢索結果」內的資料回答，不要臆測或補充資料中沒有的規格。
- 每次提到料號都要完整寫出 item_number，包含所有後綴；即使在系列摘要中，
  也不可省略 -1／-2 等後綴或改寫成前綴加 -*。要比較變體時，逐一寫出完整料號。
- 若結果中有多筆規格相近者，明確指出它們的差異（pitch 系列、PIN 排列、長度、AWG、GEN）。
  注意 2*3P（雙排共 6 PIN）與 3P（單排 3 PIN）是不同產品，不可混稱。
- 若出現「查無此料號」區塊：先明確說出該料號不在語料中（通常是 PLM 剛開、尚未進到匯出檔），
  再把「同系列鄰近料號」當成參考料號列出，不要假裝找到了。
- 若出現「PLM 欄位說明」區塊：照該說明回答該欄位，不要拿名稱模糊比對的結果硬湊。
- **資料涵蓋範圍**：只要區塊中出現「目前系統尚未包含該資料」，就照該句說明是哪個欄位／
  哪個值沒有。不要改用相近的料號、相似的人名或其他欄位代答，也不要自行推測。
  這比給出看似合理但錯誤的答案重要。
- 一個問題可能同時問到「有資料」和「尚未包含」的欄位（例如標準成本五欄只匯入其中兩欄）。
  這種情況要逐欄回答：有值的欄位照檢索結果答，沒有的欄位才說尚未包含，不要因為其中一欄
  缺資料就整題不答。
- 檢索結果每一列的欄位都標了 PLM 表單上的名稱（如「料件分類 D 成品」「料件類型 製造件」）。
  只能用標籤相符的那個值回答，不要拿別的欄位的值代答；該列沒有那個標籤才算沒有資料。
- 若出現「清單／統計結果」區塊：那是資料庫直接統計的數字，不是相似度排序。
  務必先講總筆數，再講分佈，最後才列出範例料號；並明確說明列出的是前幾筆、
  完整清單要用匯出連結。不要把範例當成全部，也不要自己加總或改寫筆數。
- 這是多輪對話。追問時要延續前文的主題，但**只能引用本輪「檢索結果」裡的料號**；
  前幾輪提過但這輪沒檢索到的料號，不可以當成事實再次陳述。
- **標了「★已作廢」的料號要主動講出來**，不能只當成一般料號回答規格。它是
  PLM 已經作廢或改號的料號，不可以推薦為可用料或替代料；使用者要替代料時
  請他明講。作廢料號只會在使用者點名問它的時候出現在結果裡。
- 用繁體中文回答，簡潔、條列、務實。不要重複整段原始描述。

工程圖（只在出現「工程圖」區塊時適用）：
- **你的角色是總結，不是轉錄。** 前端已經把標題欄與材料表照原圖畫成表格顯示，
  所以不要逐欄、逐列把那張表再抄一次；要講的是表格看不出來的東西：
  這是什麼線、材料怎麼組成、哪裡不可靠。
- **引用工程圖的任何內容時，一定要附上該 PDF 的來源：檔名、連結、檔案更新時間，
  三者缺一不可。** 沒有「工程圖」區塊就不要編造來源，也不要說「請參考工程圖」。
- 查不到工程圖時就說查不到，**不要拿相似料號或鄰近料號的圖代答**，
  也不要改去比較那些料號的規格。使用者要相似料號時會自己講。
  「本系統未收錄」不代表這個料號在公司沒有工程圖，請保留查詢範圍。
- 標題欄的 **TITLE 與 PART NO. 是最權威的兩欄**。簽核人（D.R. / CH.K / APV.D）
  本來就沒有抽出來，使用者問起就說沒有這項資料。
- 判讀方式是 OCR、或標了「PART NO. 未與 PLM 料號核對一致」時，要在該欄後面註明
  「此欄為 OCR 判讀，請以原圖為準」，不要當成確定值。特別是英數字裡的
  O 與 0、以及被吃掉的連字號，是已知且未解決的誤判。
- **圖上的 TITLE 不要用檢索結果的品名代答，反過來也不行**：TITLE 與 PLM
  品名是各自獨立的來源欄位，只能分別引用。資料未明列時，不可宣稱 TITLE 是
  客戶命名、由誰設計或提供，也不可推測客戶或用途。
- 材料表的數量欄，`A/R`（As Required）是有效數量，意思是「依需求」，
  不是 0、也不是沒有資料。問「用了幾 PCS」時要照實說是 A/R。
- 材料名稱中的英文、型號與數字只能引用 DESCRIPTION 的原文片段，不能縮寫型號、
  補字或自行修正 OCR。概括分類可以，但不能推算分類筆數，或把不同 ITEM 的數量
  合併成「各 1 或 4 PCS」。提到數量時逐一標明 ITEM、原始 Q'TY 與 UNIT；
  表格列數不等於物料總件數。"""


def _client() -> anthropic.Anthropic:
    from .gateway_compat import guard_client
    return guard_client(anthropic.Anthropic(
        base_url=BASE_URL,
        auth_token=API_KEY,  # -> Authorization: Bearer <key>
        timeout=120.0,
        max_retries=1,
    ))


def models() -> list[str]:
    """Model ids this key may actually use, or [] if the gateway won't say.

    Not part of the Anthropic SDK surface: the gateway serves an OpenAI-style
    /v1/models, and its own 400 for a bad model name tells you to call it. An
    empty list is a real answer here -- it is what the gateway returns once the
    model behind LLM_MODEL is retired or the key loses access to it.
    """
    if not (BASE_URL and API_KEY):
        return []
    try:
        r = httpx.get(
            BASE_URL.rstrip("/") + "/v1/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=15.0,
        )
        r.raise_for_status()
        return [m["id"] for m in (r.json().get("data") or []) if m.get("id")]
    except Exception:
        return []


_MODEL_CACHE_TTL = 300.0
_model_cache: tuple[float, str] | None = None


def _effective_model() -> str:
    """實際要打的模型 id。預設 MODEL；只有 MODEL 真的不在 models() 清單裡、
    且 MODEL_FALLBACK 在清單裡時才切過去（換模型當天新模型還沒上線，或被
    暫時下架的安全網）。查不到清單就不猜，照樣用 MODEL 讓錯誤如實回報。
    快取 5 分鐘，邏輯與 agentkit.gateway._effective_model 相同。"""
    global _model_cache
    now = time.time()
    if _model_cache and now - _model_cache[0] < _MODEL_CACHE_TTL:
        return _model_cache[1]
    chosen = MODEL
    if MODEL_FALLBACK and MODEL_FALLBACK != MODEL:
        avail = models()
        if avail and MODEL not in avail and MODEL_FALLBACK in avail:
            chosen = MODEL_FALLBACK
            print(f"[llm] {MODEL} 不在 /v1/models 清單，改用 fallback {MODEL_FALLBACK}")
    _model_cache = (now, chosen)
    return chosen


# --------------------------------------------------------------------------
# Gateway 故障分類
#
# ★ 為什麼要分類，而不是把原始錯誤原樣往前端丟
#   gateway 的 400 實際長這樣：
#     {"error":{"message":"400: {'error': 'anthropic_messages: Invalid model name
#      passed in model=<model>. Call `/v1/models` to view
#      available models for your key.'}","type":"None","param":"None","code":"400"}}
#   Agent Hub 前端把整包 detail 用 JSON.stringify 印出來，使用者讀完三層巢狀
#   引號之後仍然不知道兩件最重要的事：**這是誰的問題、我現在該做什麼**。
#
# ★ 以下是歷史實測回應；需同時看模型清單，不能單靠狀態碼推論授權：
#     GET  /v1/models                 200 {"object":"list","data":[]}
#         -> 這把 key 名下一顆模型都沒有
#     POST /v1/messages 亂打的模型名    404 {"detail":"unknown model: ..."}
#         -> gateway 根本不認識這個名字
#     POST /v1/messages 設定的模型名    400 Invalid model name ... for your key
#         -> 模型名被生成端拒絕，還需核對 /v1/models
#     POST /v1/messages 亂打的 key      401 {"detail":"invalid API key"}
#         -> 認證本身就過不了
#   模型名列在 /v1/models 卻仍被生成端拒絕，也可能是 Gateway 的路由、
#   模型綁定或後端狀態不一致，不能只憑 400 斷言需要換 key。
#
# ★ quota 與 rate_limit 是刻意分開的兩類，因為處置完全相反：
#   429 是「這一分鐘打太密」，等一下重試就好，換 key 沒有用；
#   額度/預算用盡（訊息裡有 budget / quota / exceeded）才是真的要換一把新 key。
#   分不清楚的話，每次 LLM 出問題都會變成「再去要一把 key」。
# --------------------------------------------------------------------------
OWNER_ADMIN = "gateway 管理者"
OWNER_LOCAL = "Agent_Hub_RAG_Frontend 的 .env"

# kind -> 這個故障算不算「換一把新 API key 就會好」。前端據此決定要不要
# 把「請跟 gateway 管理者換新 key」講出來 —— 講在不該講的地方，
# 使用者會為了 429 去要一把新 key，然後發現一樣不能用。
NEEDS_NEW_KEY = {"auth", "quota", "no_model_access"}


def _err_text(err: BaseException | str | None) -> str:
    if err is None:
        return ""
    return err if isinstance(err, str) else f"{type(err).__name__}: {err}"


def _diag(kind: str, title: str, detail: str, fix: str, owner: str,
          raw: str = "", models_seen: list[str] | None = None) -> dict[str, Any]:
    return {
        "kind": kind,
        "ok": kind == "ok",
        "title": title,
        "detail": detail,
        "fix": fix,
        "owner": owner,
        "needs_new_key": kind in NEEDS_NEW_KEY,
        "model": _effective_model(),
        "base_url": BASE_URL,
        "models": models_seen or [],
        "raw": raw,
    }


def diagnose(err: BaseException | str | None = None) -> dict[str, Any]:
    """把 gateway 的狀態或一個錯誤翻成「發生什麼 / 該做什麼 / 誰能修」。

    err=None：探測 gateway（給 /api/health 與 /a2a/card 的狀態檢查用），結果
    快取 PROBE_TTL 秒，見下方。
    err=例外或錯誤字串：只做分類，不再打網路 —— 答題答到一半掛掉時是這條路，
    那時候再多送一次請求只是讓使用者多等。這條**不快取**：它分類的是這一次
    真的發生的錯誤，拿上一次的答案回覆等於答錯題。
    """
    if err is not None:
        return _classify(err)

    global _probe_cache
    if _probe_cache is not None and time.time() - _probe_cache[0] <= PROBE_TTL:
        return _probe_cache[1]
    out = _classify(None)
    _probe_cache = (time.time(), out)
    return out


def _gateway_model_unavailable(raw: str, models_seen: list[str]) -> dict[str, Any]:
    return _diag(
        "gateway_model_unavailable", "Gateway 列出的模型無法生成",
        f"/v1/models 列出 {_effective_model()}，但生成 API 仍拒絕這個名稱。"
        "模型清單與生成端的可用狀態不一致。",
        "請 Gateway 管理者核對模型名稱對應、API key 的模型綁定與後端狀態，"
        "並用 /v1/messages 確認能生成；不能只改回清單中的同一個名稱。",
        OWNER_ADMIN, raw, models_seen,
    )


def invalidate_probe() -> None:
    """下一次 diagnose() 重新探測。換了 key 想立刻確認時用。"""
    global _probe_cache
    _probe_cache = None


def _classify(err: BaseException | str | None) -> dict[str, Any]:
    if not (BASE_URL and API_KEY and MODEL):
        missing = "、".join(n for n, v in (
            ("LLM_BASE_URL", BASE_URL), ("LLM_API_KEY", API_KEY), ("LLM_MODEL", MODEL)
        ) if not v)
        return _diag(
            "not_configured", "LLM 尚未設定", f"缺少 {missing}",
            "在 Agent_Hub_RAG_Frontend/.env（各 agent 共用）補齊這幾個變數後重啟服務", OWNER_LOCAL,
        )

    if err is None:
        try:
            # ★ 探測用獨立的短逾時 client，不是 _client()（120 秒、會重試一次）。
            #   這條路是 /api/health 與 /a2a/card 在走的：gateway 只是塞住而不是
            #   拒絕的時候，狀態查詢不該跟著等兩分鐘。答題那條路仍然用 _client()，
            #   那裡的長逾時是必要的（thinking 很慢）。
            model = _effective_model()
            anthropic.Anthropic(
                base_url=BASE_URL, auth_token=API_KEY,
                timeout=PROBE_TIMEOUT, max_retries=0,
            ).messages.create(
                model=model, max_tokens=16,
                messages=[{"role": "user", "content": "ping"}],
            )
            return _diag("ok", "LLM 可用", f"{model} @ {BASE_URL}", "", "")
        except BaseException as e:  # noqa: BLE001 -- 分類本身不該再拋
            err = e

    raw = _err_text(err)
    low = raw.lower()
    code = getattr(err, "status_code", None) if isinstance(err, BaseException) else None

    if code == 401 or "invalid api key" in low or "authenticationerror" in low:
        return _diag(
            "auth", "API key 無效或已被撤銷",
            "gateway 回 401，目前設定的這把 key 連認證都過不了。",
            "跟 gateway 管理者換一把新的 API key，填進 .env 的 LLM_API_KEY 後重啟服務。",
            OWNER_ADMIN, raw,
        )

    if code == 429 or "rate limit" in low or "too many requests" in low:
        return _diag(
            "rate_limit", "呼叫太密集，被 gateway 限流",
            "gateway 回 429。這是每分鐘的頻率上限，不是總量用完。",
            "等幾十秒再送一次就好。★ 這種情況換新 key 沒有用。",
            OWNER_ADMIN, raw,
        )

    if any(w in low for w in ("budget", "quota", "exceeded", "insufficient", "credit")):
        return _diag(
            "quota", "這把 key 的額度用完了",
            "gateway 的訊息裡提到額度／預算已用盡 —— 這才是「用到一定次數就要換 key」的那種情況。",
            "跟 gateway 管理者要一把新的 API key（或請他把這把的額度加回來），"
            "換進 .env 的 LLM_API_KEY 後重啟服務。",
            OWNER_ADMIN, raw,
        )

    if code == 404 or "unknown model" in low:
        avail = models()
        if _effective_model() in avail:
            return _gateway_model_unavailable(raw, avail)
        return _diag(
            "unknown_model", f"gateway 不認識模型名稱 {_effective_model()}",
            "gateway 回 404 unknown model —— 這個名字在它的模型表裡根本不存在，多半是打錯字。",
            ("先請 gateway 管理者給正確的模型名稱。" if not avail
             else "改成 Agent_Hub_RAG_Frontend/.env 的 LLM_MODEL＝" + "、".join(avail[:10])
                  + " 其中之一，四個 agent 都會一起生效。"),
            OWNER_LOCAL if avail else OWNER_ADMIN, raw, avail,
        )

    if "invalid model name" in low or (code == 400 and "model" in low):
        avail = models()
        if _effective_model() in avail:
            return _gateway_model_unavailable(raw, avail)
        if avail:
            return _diag(
                "wrong_model", f"這把 key 不能用 {_effective_model()}，但還有別的可以用",
                "gateway 回 400 說這個模型名不被接受，而 /v1/models 仍列得出其他模型。",
                "把 Agent_Hub_RAG_Frontend/.env 的 LLM_MODEL 改成 " + "、".join(avail[:10])
                + " 其中之一後重啟 Hub 與各 agent。",
                OWNER_LOCAL, raw, avail,
            )
        return _diag(
            "no_model_access", "這把 API key 名下沒有任何可用模型",
            f"gateway 認得 {_effective_model()}（亂打的模型名會回 404，這裡回的是 400），"
            "但 /v1/models 對這把 key 回的是空清單 —— 也就是 key 還活著、"
            "模型授權卻被收走了（額度到期、或管理者改了這把 key 的模型綁定）。",
            "跟 gateway 管理者要一把新的 API key，或請他把這顆模型重新綁回這把 key。"
            "★ 改 Agent_Hub_RAG_Frontend/.env 的 LLM_MODEL 沒有用，因為這把 key 一顆模型都沒有。",
            OWNER_ADMIN, raw,
        )

    if isinstance(err, BaseException) and isinstance(err, anthropic.APIConnectionError):
        return _diag(
            "offline", "連不上 LLM gateway",
            f"打不到 {BASE_URL}。gateway 沒開、或這台機器連不到那個網段。",
            "確認 gateway 服務有起來、且本機連得到 LLM_BASE_URL 那台主機。",
            OWNER_ADMIN, raw,
        )
    if any(w in low for w in ("connect", "timeout", "timed out", "unreachable")):
        return _diag(
            "offline", "連不上 LLM gateway 或逾時",
            f"打 {BASE_URL} 的時候連線失敗或逾時。",
            "確認 gateway 有起來；若是逾時，模型可能正在排隊，稍後重試。",
            OWNER_ADMIN, raw,
        )

    if "max_tokens" in low:
        return _diag(
            "token_budget", "模型的思考佔滿了 token 預算",
            f"這顆模型的 thinking 區塊算進 max_tokens，目前是 {MAX_TOKENS}。",
            "調高 .env 的 LLM_MAX_TOKENS 後重啟服務。", OWNER_LOCAL, raw,
        )

    return _diag(
        "other", "LLM gateway 回了未分類的錯誤", raw or "（沒有錯誤訊息）",
        "把下面的原始訊息貼給 gateway 管理者。", OWNER_ADMIN, raw,
    )


def available() -> tuple[bool, str]:
    """Cheap liveness probe used by /api/health. 一行摘要版的 diagnose()。"""
    d = diagnose()
    if d["ok"]:
        return True, "ok"
    return False, f"{d['title']}：{d['detail']} → {d['fix']}"


# Rendered on every row, WITH the PLM form's own label. The labels are not
# decoration: an unlabelled value from one field (料件類型) gets read as the
# answer to a neighbouring field (料件分類), and an unlabelled category value is
# not recognised as 產品中類, so the model falls back to "目前系統尚未包含該資料"
# about a value sitting in the line it was given.
_ALWAYS: tuple[tuple[str, str], ...] = (
    ("item_type", "料件類型"),
    ("eco_std", "環保標準"),
    ("plant", "建立廠別"),
    ("announce_cat", "公告大類"),
    ("shelf_days", "保存天數"),
    ("lifecycle", "Lifecycle"),
    ("state", "狀態"),
    ("created_by", "建立人員"),
)

# Already rendered by the block above or by the hand-written spec bits, so a
# focus column naming one of them must not be printed twice.
_RENDERED = {c for c, _ in _ALWAYS} | {
    "item_number", "name", "prod_l2", "prod_l3", "pitch_series", "pin_layout",
    "pins_total", "length_mm", "awg", "gen", "speed_g", "created_on",
    "customer_pn", "drawing_no",
}

# A focus list longer than this stops being focus. Guards the token budget in
# the pathological case of a query that names a dozen fields.
MAX_FOCUS = 8


def _fmt(v: Any) -> str:
    """One PLM value as the model should read it. Decimal('700') must not reach
    the prompt as "700.0000" and a bool must not reach it as "True"."""
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, (int, float)):
        return f"{v:g}"
    if isinstance(v, list):
        return "/".join(str(x) for x in v)
    s = str(v)
    # ISO timestamp -> date. The time of day never changes an answer.
    return s[:10] if re.match(r"^\d{4}-\d{2}-\d{2}[ T]", s) else s


def _render(results: list[dict], limit: int = 10, focus: tuple[str, ...] = ()) -> str:
    """Only retrieved rows are sent to the model -- never the whole CSV.

    Kept tight on purpose. An earlier version rendered all eleven PLM fields
    plus the match reasons per row, which pushed the prompt from ~1.3k to ~2.5k
    characters; the gateway then spent the entire max_tokens budget on its
    `thinking` block and returned no text at all. `match_reasons` in particular
    is for the UI, not the model.

    But a fixed field list is what made "批號管控是什麼" unanswerable: the value
    was imported for nearly every row and simply never rendered. So the list is now
    fixed PLUS query-driven -- `focus` carries the columns this query actually
    named (retrieval reports them in plm_notes / field_matches), which keeps the
    prompt short for ordinary spec questions and complete for field questions.
    """
    extra = [c for c in dict.fromkeys(focus) if c not in _RENDERED][:MAX_FOCUS]
    lines = []
    for i, r in enumerate(results[:limit], 1):
        bits = [f"[{i}] {r['item_number']}", r["name"]]
        # 作廢是**答案的一部分**，不是附註，所以擺在品名的正後面而不是排到那一長串
        # PLM 欄位裡。作廢列只會從料號精確比對那條路進來（相似度那幾路已經濾掉），
        # 也就是說使用者是點名問這一顆的 —— 這時候「它已經作廢了」正是他要的答案。
        if r.get("is_obsolete"):
            why = r.get("obsolete_reason") or ""
            bits.append("★已作廢" + (f"（{why}）" if why else ""))
        tri = " / ".join(x for x in (r.get("prod_l2"), r.get("prod_l3")) if x)
        if tri:
            bits.append("產品中類/小類 " + tri)
        if r.get("pitch_series"):
            bits.append("/".join(r["pitch_series"]))
        if r.get("pin_layout"):
            bits.append("PIN 排列 " + "/".join(r["pin_layout"]))
        elif r.get("pins_total"):
            bits.append("/".join(f"{v}P" for v in r["pins_total"]))
        if r.get("length_mm"):
            bits.append("/".join(f"{v:g}mm" for v in r["length_mm"]))
        for f, label in (("awg", "AWG"), ("gen", "GEN"), ("speed_g", "速率G")):
            if r.get(f):
                bits.append(f"{r[f]}{label}" if f == "awg" else f"{label}{r[f]}")
        for f, label in _ALWAYS:
            if r.get(f) is not None and r.get(f) != "":
                bits.append(f"{label} {_fmt(r[f])}")
        # The fields this particular question asked about. 0 is a real answer for
        # a cost or an hour count, so the emptiness test is `is None`, not truthiness.
        for col in extra:
            v = r.get(col)
            if v is not None and v != "" and v != []:
                bits.append(f"{T.field_label(col)} {_fmt(v)}")
        # Date only: the time of day never changes an answer and costs tokens on
        # every one of the 25 rendered rows.
        if r.get("created_on"):
            bits.append("建立日期 " + str(r["created_on"])[:10])
        if r.get("customer_pn"):
            bits.append("客戶料號 " + ",".join(r["customer_pn"][:2]))
        if r.get("drawing_no"):
            bits.append("圖號 " + ",".join(r["drawing_no"][:2]))
        if r.get("is_placeholder"):
            bits.append("(佔號列，非實際產品)")
        lines.append(" | ".join(bits))
    return "\n".join(lines)


def _focus_columns(found: dict) -> tuple[str, ...]:
    """PLM columns this query named, in the order retrieval reported them.

    ★ 時間條件篩的是哪一欄，那一欄就要被印出來
      「上個月修改的料號」檢索是對的（`modified_on` 落在 7 月），但 _render()
      的固定欄位裡只有 `created_on`，於是每一列都印好幾年前的建立日期 ——
      模型看到的是「問七月修改、資料全是多年前」，這個矛盾它無法從 prompt 裡
      解開，只能自己編。
      篩選欄位與呈現欄位必須是同一欄，否則使用者無從驗證答案。
    """
    cols: list[str] = []
    for src in ("plm_notes", "field_matches"):
        for entry in found.get(src) or []:
            col = entry.get("column")
            if col and col not in cols:
                cols.append(col)
    tq = found.get("time_query") or {}
    if tq.get("field") == "modified":
        cols.insert(0, "modified_on")     # 擺最前面：它是這一題的判斷依據
    return tuple(dict.fromkeys(cols))


def _join(blocks: list[str], tail: str = "") -> str:
    return "\n\n".join([*blocks, tail] if tail else blocks)


def pick_lessons(question: str, found: dict | None) -> list[dict[str, Any]]:
    """這一題要掛哪幾條教訓。

    ★ 為什麼要暴露成一支公開函式，而不是讓 build_prompt 自己去撈
      `for_query()` 有副作用：它會把挑中那幾條的 `hits` 加一（那是「這條教訓被
      用了幾次」的統計）。answer_verified() 既要把教訓寫進 prompt、又要回報
      「這一題掛了哪幾條」給跑分用，各自撈一次的話同一題會被記成用了兩次，
      教訓的引用次數就變成一個看起來很忙、實際上是重複計數的數字。
      所以撈一次、把結果傳下去。
    """
    try:
        return lessons.for_query(question, route_of(found))
    except Exception as e:  # noqa: BLE001  教訓庫壞掉不該讓問答壞掉
        print(f"[llm] 教訓庫讀取失敗（略過）：{type(e).__name__}: {e}")
        return []


def _tail(question: str, found: dict | None,
          picked: list[dict[str, Any]] | None, with_lessons: bool) -> str:
    """prompt 的收尾：先「你可以這樣問」，再「以前犯過的錯」。

    ★ 為什麼統一在這裡組，而不是在各個分支裡各加一次
      build_prompt 有四個 return（工程圖 / 沒有圖 / 清單 / 一般），build_context
      也有四個。「查不到就給建議」如果寫在分支裡，就會漏掉最需要它的兩種 ——
      「問了圖但那顆沒有圖」與「點名的料號不在語料裡」**都不是空結果**
      （前者留著那顆料號當主詞，後者回的是鄰近料號），所以它們根本走不到
      「無符合資料」那一支。掛在收尾上，八個 return 一次覆蓋。
      答得出來的題目 hints.build() 回 None，這一段就是空字串。
    """
    parts = [hints.prompt_block(question, found)]
    if with_lessons:
        parts.append(_lesson_tail(question, found, picked))
    return "\n\n".join(p for p in parts if p)


def _lesson_tail(question: str, found: dict | None,
                 picked: list[dict[str, Any]] | None = None) -> str:
    """教訓區塊。**擺在 prompt 的最後**，不是最前面。

    ★ 位置是刻意的
      這一段講的是「不要做什麼」，而它要管的是模型看完資料之後的行為。放在
      資料前面，中間隔著三千字的檢索結果，等於在它動筆前很久講的一句話；
      放在最後一段，它是模型下筆前讀到的最後一件事。實測差別最大的是
      「不要把鄰近料號當答案」這條 —— 那正是它讀完一長串鄰近料號之後的衝動。
    """
    return lessons.as_block(pick_lessons(question, found) if picked is None else picked)


def route_of(found: dict | None) -> str:
    """這一題實際走到哪條路 —— 從檢索結果反推，不重跑分流。

    教訓庫是按路徑掛的（工程圖的教訓不該塞進清單題的 prompt），而分流的事實
    在 found 裡已經是既成的：有圖或問了圖就是 drawing，有 listing 就是 listing。
    再跑一次分流等於為了知道剛才做過什麼而重跑一次 SQL。
    """
    found = found or {}
    if found.get("drawings") or found.get("drawing_note"):
        return "drawing"
    if found.get("listing"):
        return "listing"
    return "search"


def build_prompt(question: str, results: list[dict], found: dict | None = None,
                 with_lessons: bool = True,
                 picked: list[dict[str, Any]] | None = None) -> str:
    """`found` is the full retrieval payload. It carries not_found and plm_notes,
    which the model needs in order to say "this number is not in the corpus"
    rather than silently answering with the nearest neighbour.

    `with_lessons` 掛上教訓庫裡與這一題相關的規則（src/lessons.py）。關掉它的
    唯一用途是跑分時要量「沒有教訓的基準線是多少」—— 教訓有沒有用，
    要靠開關兩邊的通過率差來回答，不是靠感覺。
    `picked` 是已經撈好的教訓（見 pick_lessons 的註解），沒給就自己撈。
    """
    found = found or {}
    tail = _tail(question, found, picked, with_lessons)
    blocks: list[str] = [f"使用者問題：{question}"]

    missing = found.get("not_found") or []
    if missing:
        blocks.append(
            "查無此料號（語料中不存在，可能是 PLM 新開但尚未進到匯出檔）：\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n下方若有「同系列鄰近料號」，僅供參考，不是使用者要的那一筆。"
        )

    notes = [n for n in (found.get("plm_notes") or []) if n.get("answer")]
    if notes:
        blocks.append(
            "PLM 欄位說明：\n" + "\n".join(f"- {n['answer']}" for n in notes)
        )

    # 問到圖的時候，圖**就是**答案，不是附註。所以有圖就整段只給圖，
    # 連「檢索結果」那一塊都不附。
    #
    # ★ 這不只是排版問題，是這一題的成本與正確性：
    #   - 附上料號列會讓 prompt 長好幾倍，gateway 是會 thinking 的模型，
    #     整題時間幾乎全花在生成（檢索本身不到一秒）。只留圖之後短得多。
    #   - 模型看得到 parts 的品名，就會拿它去補圖上沒讀到的欄位 —— 圖上的
    #     TITLE 與 parts 品名是不同來源的欄位（SYSTEM 有明講，
    #     但最有效的做法是根本不要把它放進同一個 prompt）。
    if found.get("drawings"):
        # 索引與份數統計排在材料表**前面**：這一段是「有哪些圖」的答案，
        # 材料表是「某一份圖裡有什麼」的答案，而且它不能被字元截斷切掉
        # （見 _drawing_scope）。
        scope = _drawing_scope(found)
        if scope:
            blocks.append(scope)
        blocks.append(_render_drawings(found["drawings"]))
        # 一次問兩個料號、只有一個有圖時，這句話是唯一會講出另一個沒有圖的地方
        # —— 沒有它，模型看到的就只有那份圖，另一個料號會被靜靜地當成也答完了。
        if found.get("drawing_note"):
            blocks.append(found["drawing_note"])
        # ★ 工程圖這一支**最需要**講清楚時間是依哪個欄位篩的
        #   `doc_modified_on` 是「檔案在 PLM 的更新時間」，不是圖上畫的日期、
        #   也不是料號的建立日期。使用者問「某個月的工程圖」拿到幾份，
        #   不講的話他不知道這幾份是依什麼挑的 —— 而他下一次問「建立日期」時
        #   會以為是同一個東西。
        blocks.append(_time_note(found) + _drawing_task(found))
        return _join(blocks, tail)

    # 問了工程圖但沒有圖：這就是整題的答案，不要再附檢索結果。
    #
    # ★ 附上去的話模型一定會拿它們回答。問「<料號> 的工程圖」而這顆
    #   料號沒有圖時，回來的會是隔壁料號的規格 —— 使用者要的是「有沒有圖」，
    #   得到的卻是一份看起來很像答案的東西。檢索那邊已經把 results 收斂成
    #   「問句點名的那顆料號本身」（見 retrieval.search 的 wants_drawing 段），
    #   這裡再把它擋在 prompt 外，兩層都不給模型代答的機會。
    if found.get("drawing_note"):
        blocks.append(found["drawing_note"])
        blocks.append(_time_note(found))
        blocks.append(
            "以上就是這一題的全部資料。請直接回答「查不到這個料號的工程圖」，"
            "並說明目前只收錄部分料號的工程圖。"
            "**不要**改用相似料號、鄰近料號或 parts 的品名規格代答，"
            "也不要列出任何其他料號 —— 使用者沒有要求時，"
            "給相似的東西比直說查不到更糟。"
            "最後可以用一句話告訴使用者：需要相似料號的話請明講。"
        )
        return _join(blocks, tail)

    listing = found.get("listing")
    if listing:
        blocks.append(_time_note(found) + _render_listing(listing, results))
        return _join(blocks, tail)

    if not results:
        blocks.append(_time_note(found))
        # 查不到的時候，最有用的東西不是「查不到」三個字，而是**下一句該怎麼問**。
        # hints 給的每一個建議都是照語料現在真的有的資料算出來的（見 src/hints.py）。
        blocks.append("檢索結果：（無符合資料）\n"
                      + (hints.prompt_block(question, found)
                         or "請說明查不到，並建議如何調整查詢條件。"))
    else:
        blocks.append(
            f"檢索結果（共 {len(results)} 筆，已依相關度排序）：\n"
            f"{_render(results, focus=_focus_columns(found))}\n\n"
            + _time_note(found)
            + _match_total_note(found)
            + "請根據以上資料回答。"
        )
    return _join(blocks, tail)


def _time_note(found: dict) -> str:
    """「我是照哪個欄位、哪一段時間篩的」。

    ★ 有結果的時候也要講
      使用者問「某個月的工程圖」，系統照 `doc_modified_on` 篩 —— 那是
      **檔案在 PLM 的更新時間**，不是圖上畫的日期、也不是料號的建立日期。
      不講清楚的話，使用者拿到幾份圖卻不知道是依什麼挑的，
      而下一次他問「建立日期」時會以為是同一個東西。
    """
    tq = found.get("time_query")
    if not tq:
        return ""
    line = f"★ 時間條件：{tq['describe']}。回答時要講出這是依哪個欄位篩的。\n"
    column = "doc_modified_on" if found.get("drawings") or found.get("drawing_note") else \
        "created_on" if tq.get("field") == "created" else "modified_on"
    for span in tq.get("spans") or []:
        if span.get("start") and span.get("end"):
            line += f"　資料庫篩選：{column} >= {span['start']} 且 < {span['end']}（結束日不含）。\n"
    line += ("　相對日期已由程式解析並套用資料庫篩選，請依這個區間與結果回答；"
             "不要用模型記憶中的年份重新解讀，也不要聲稱系統不支援已套用的日期條件。\n")
    if not tq.get("explicit_field") and not (found.get("drawings")
                                             or found.get("drawing_note")):
        line += ("　（使用者沒明講建立還是修改，這裡用的是修改日期；"
                 "請在回答裡提一句，讓他知道可以指定另一個。）\n")
    return line + "\n"


def _match_total_note(found: dict) -> str:
    """「全庫共 N 筆符合，以下是相關度最高的幾筆」。

    ★ 沒有這一句，一個正確的答案會被讀成幻覺
      「找有 PH4.2 2*6P HSG 的成品」全庫符合的可能有近百筆，畫面上只給前幾筆。
      使用者手上剛好是排在後面那一顆，於是他的結論是「它沒找到我說的東西」。
      檢索沒有錯，是回答沒有交代母體 —— 排序檢索只給一個視窗，不講視窗多大，
      就等於在暗示視窗就是全部。講了之後同一份結果就變成可以操作的答案：
      使用者知道要再加條件，或者改用「列出所有…」拿完整清單。
    """
    n = len(found.get("results") or [])
    out = ""
    mt = found.get("match_total") or {}
    total = mt.get("total")
    if total and total > n:
        conds = "、".join(mt.get("conditions") or []) or "這些條件"
        out += (
            f"★ 全庫符合【{conds}】的料號共 **{total}** 筆，上面只是相關度最高的"
            f" {n} 筆。\n"
            "回答時**必須先講出這個總數**，並說明列出的是前幾筆；"
            "使用者要完整清單時請他改問「列出所有…」。"
            "不要把這幾筆講成「符合條件的全部料號」。\n\n"
        )
    # ★ 使用者要的是「相關/相似」時，上面那批**刻意**包含不完全符合的列。
    #   不講的話模型會自己去比對條件，發現對不上，然後回一句「查不到」——
    #   「<料號> 有哪些相似料號」就會在十張卡片旁邊寫出
    #   「資料中沒有符合您查詢條件的相似料號」。
    if n and retrieval_wants_similar(found):
        out += (
            "★ 使用者要的是**相關／相似**的料號，所以上面這批**刻意包含"
            "不完全符合條件的**。請照它們回答，並標明哪幾筆完全符合、"
            "哪幾筆只是相近（差在長度／PIN 排列／pitch 哪一項）。"
            "**不可以因為它們不完全符合就說查不到。**\n\n"
        )
    return out


def retrieval_wants_similar(found: dict) -> bool:
    """這一題是不是使用者主動要近似的東西。判準與 retrieval 同一支函式，
    避免兩邊對「什麼叫要相似」有兩種看法。"""
    from . import retrieval

    return retrieval.wants_similar(str(found.get("query") or ""))


def build_context(question: str, found: dict | None = None,
                  with_lessons: bool = True) -> str:
    """工具迴圈（src/a2a.py 的 context_fn）要看的「我這次檢索到什麼」。

    ★ 為什麼不直接把 build_prompt() 拿去用
      build_prompt 是**純摘要**那條路的提示，每一段的收尾都在說「以上就是這一題的
      全部資料」「請直接回答查不到」—— 因為那條路真的沒有下一步。工具迴圈有下一步：
      第一次檢索沒撈到東西時，模型該去呼叫 search_drawings 換個問法，而不是被提示
      叫去認輸。所以這裡共用的是「怎麼把資料寫成字」的那幾個 renderer，
      不共用那些收尾指令。

    ★ 但**有**工程圖時要明講「不用再呼叫工具」。這一題的資料已經到齊了，
      再跑一輪工具只是多花好幾秒與好幾倍 token，答案還可能被別的圖帶偏。
    """
    found = found or {}
    tail = _tail(question, found, None, with_lessons)
    blocks: list[str] = []

    missing = found.get("not_found") or []
    if missing:
        blocks.append(
            "查無此料號（語料中不存在，可能是 PLM 新開但尚未進到匯出檔）：\n"
            + "\n".join(f"- {m}" for m in missing)
        )

    notes = [n for n in (found.get("plm_notes") or []) if n.get("answer")]
    if notes:
        blocks.append("PLM 欄位說明：\n" + "\n".join(f"- {n['answer']}" for n in notes))

    if found.get("drawings"):
        # 索引清單插在工程圖區塊**前面**，但要等別的區塊都組好才知道還剩多少
        # 預算給它（見 CONTEXT_BUDGET）——所以先記住位置，最後 insert 回來。
        at = len(blocks)
        blocks.append(_render_drawings(found["drawings"]))
        if found.get("drawing_note"):
            blocks.append(found["drawing_note"])
        # ★ 時間條件在工具那條路一樣要講（這裡原本漏了）
        #   勾「用自己的工具」時模型收到的背景就是這一段。少了 _time_note，
        #   它不知道這批圖是依 doc_modified_on 篩出來的，於是會自己去圖上的
        #   文字裡找日期 —— 那正是 src/timerange.py 開頭記的那個病灶。
        blocks.append(
            "★ 這就是使用者要的工程圖，**直接照它回答，不需要再呼叫任何工具**。\n"
            + _time_note(found)
            + _drawing_task(found)
        )
        # 索引**先整份放進去**，超了才依序讓位：教訓庫 -> 索引行數
        # （順序的理由見 CONTEXT_BUDGET）。
        scope = _drawing_scope(found)
        if scope:
            blocks.insert(at, scope)
        out = _join(blocks, tail)
        if len(out) > CONTEXT_BUDGET:
            tail = _clip_lines(tail, max(0, len(tail) - (len(out) - CONTEXT_BUDGET)))
            out = _join(blocks, tail)
        if len(out) > CONTEXT_BUDGET and scope:
            # 教訓全砍完還是太長（圖非常多）-> 這時才動索引，並且它自己會
            # 寫出「其餘 N 份沒有列進清單…不是不存在」。
            blocks[at] = _drawing_scope(
                found, budget=len(scope) - (len(out) - CONTEXT_BUDGET))
            out = _join(blocks, tail)
        return out

    if found.get("drawing_note"):
        blocks.append(found["drawing_note"])
        blocks.append(_time_note(found))
        blocks.append(
            "以上是用使用者原句檢索的結果。使用者要找的如果是**圖上的某個零件**"
            "（HSG、端子、膠帶、標籤這類），那要用 search_drawings 拿零件的關鍵字或"
            "廠商料號再查一次，不要用這句話直接回答查不到。"
            "**不要**改用相似料號或 parts 的品名規格代答。"
        )
        return _join(blocks, tail)

    listing = found.get("listing")
    if listing:
        blocks.append(_time_note(found)
                      + _render_listing(listing, found.get("results") or []))
        return _join(blocks, tail)

    blocks.append(_time_note(found))
    results = found.get("results") or []
    if results:
        blocks.append(
            f"檢索結果（共 {len(results)} 筆，已依相關度排序）：\n"
            f"{_render(results, focus=_focus_columns(found))}\n\n"
            + _match_total_note(found)
        )
    else:
        blocks.append(
            "檢索結果：（無符合資料）\n"
            "這只是用使用者原句跑的第一次檢索，不是結論 —— 問工程圖請用 "
            "search_drawings，其他請用自己的工具換個關鍵字再查一次。\n"
            + hints.prompt_block(question, found)
        )
    return _join(blocks, tail)


# 工程圖那一題要模型做什麼。
#
# ★ 這裡刻意**不是**「逐列列出 ITEM / DESCRIPTION / Q'TY」。前端（Agent Hub 的
#   統一卡片、src/a2a.py 的 _drawing_hits）已經把標題欄與材料表原封不動畫成
#   表格擺在回答旁邊了 —— 模型再抄一遍，使用者看到的是同一份資料連續兩次，
#   而且抄的那一份還比表格難讀（描述被 OCR 黏成一長串）。
#   模型該做的是表格做不到的事：講重點、講風險。
DRAWING_TASK = (
    "以上就是這一題的全部資料，請**只根據工程圖區塊**回答。\n"
    "★ 先確認主詞：使用者問的那個編號，必須真的出現在這份圖的標題欄"
    "（PART NO. / CUSTOMER PART NO.）或它的歸檔料號裡。對不上就直說"
    "「這份圖不是 <他問的編號>，語料裡沒有那一份」，"
    "**不要**把這份圖的標題欄、版本或材料表講成他問的那一份 ——"
    "編號只差一碼的圖（ABC-0001 / ABC-0002）是兩份不同的圖，"
    "材料表就算逐字相同也一樣。\n"
    "★ 使用者的畫面上已經有一張照原圖畫的「標題欄 + 材料表」表格，"
    "所以**不要把標題欄逐欄複述、也不要把材料表逐列抄一遍**，"
    "那些欄位值他看得到。請改寫成一段總結，內容是：\n"
    "1. 這是什麼東西：照圖上的 TITLE 原文講，順帶提 REV. 與 SHEET，"
    "一兩句就好。客戶、命名者、用途僅在資料明列時說明，不能推測。\n"
    "2. 材料構成：引用資料明列的表格列數，再概括可確認的類別"
    "（線材／連接器／標籤／膠帶…），不要自行統計各類筆數。"
    "一般摘要只列類別，不附英文型號或規格；使用者明確問某項材料的型號時，"
    "寫「ITEM n」加 DESCRIPTION 的連續原文，"
    "型號不可刪除中間字詞、縮寫、補字或修正 O/0（例如 PH3.O 不可改成 PH3.0）。"
    "使用者問數量才逐一引用該 ITEM 的 Q'TY 與 UNIT，不能合併不同列的數量。\n"
    "3. 要注意的地方：照引用資料的 Q'TY 統計說明 A/R（依需求）項目，不能自行重算。"
    "Q'TY 只有 PCS 等單位時沒有固定件數，不能說是固定數量。"
    "判讀方式若是 OCR 就提醒該欄以原圖為準、PART NO. 沒核對一致也要講。"
    "沒有這類問題就說明「無」，不要硬湊。\n"
    "4. 最後獨立一行寫出來源：檔名 ＋ 檔案更新時間 ＋ 連結。"
    "這一行是引用工程圖的硬性要求（同一張圖會有好幾個版本，版本錯了材料就錯了），"
    "**不算在「不要重複表格」的範圍內**，再短也要寫。\n"
    "不要提到工程圖以外的料號或規格。第 1~3 點合計控制在 200 字以內。"
)

# 撈到很多份圖時要模型做什麼。
#
# ★ 為什麼要跟 DRAWING_TASK 分開
#   DRAWING_TASK 是為「一份圖」寫的：講這是什麼東西、材料構成、風險。幾十份圖
#   照它做出來的答案是幾十段那種東西 —— 而使用者要的是「各月份各幾份，
#   最新的是哪幾張」。照單份任務做，模型會逐筆列前幾份，然後對沒看到的
#   其餘份數下錯的結論（「沒有某個月的」）。
#   問「這段時間有哪些圖」時，**份數與分佈才是答案**，材料表是下一題。
DRAWING_LIST_TASK = (
    "以上就是這一題的全部資料。這一題撈到很多份圖，使用者要的是**總結**，"
    "不是逐份的標題欄與材料表：\n"
    "1. 第一句講總數，以及每一段時間區間各有幾份 —— 照上面「各時間區間各有幾份」"
    "那一行的數字，不要自己去數清單。\n"
    "2. **一定要舉例：至少一份、最多三份**（這是硬性要求，不可以因為「使用者"
    "只問有哪些」就整段省掉——使用者要能接著追問其中某一份）。每一份寫"
    "**料號 ＋ 檔案更新時間**；要寫檔名的話只能用上面整份展開的那幾份"
    "（它們的檔名是完整的），原封不動照抄。\n"
    "   ★ 這一題**不要寫連結**。使用者的畫面上每一份圖都有一張可以直接點開的"
    "卡片，而清單裡的圖在這裡沒有附網址 —— 網址結尾那串數字是資料庫流水號，"
    "跟「第幾份」無關，**推算出來的連結會指到另一份圖**。\n"
    "3. ★ **不要總結這批圖的規格、接頭型號或產品類型。** 這一題手上只有檔名與"
    "更新時間，沒有這批圖的規格 —— 在這裡寫出「接頭以 PH2.5、PH3.0 系列為主」"
    "之類的話，那些值在資料裡完全不存在。要講組成請使用者縮小範圍後再問。\n"
    "4. 最後告訴使用者可以再加什麼條件縮小範圍（料號、接頭、長度，或把時間區間"
    "縮短），以及要看某一份的標題欄或材料表時直接問那一份的料號或檔名。\n"
    "★ **不可以說某一段時間沒有圖**：份數 >0 的區間就是有，"
    "上面沒有展開它的標題欄不代表它不存在。\n"
    "全部控制在 200 字以內 —— 使用者要的是總結。"
)


def _drawing_task(found: dict) -> str:
    """一份圖講細節、很多份講總結。判準只看份數。"""
    n = len(found.get("drawings") or [])
    return DRAWING_LIST_TASK if n >= DRAWING_SUMMARY_MIN else DRAWING_TASK

# 一份圖最多渲染幾列材料。一般材料表遠低於這個數，這個上限幾乎不會踩到，
# 存在只是為了擋住「一次問到十個料號」時把 max_tokens 吃光的病態情況 ——
# gateway 的 thinking block 一旦沒有預算，回來的是空字串（見模組 docstring）。
DRAWING_ITEMS = 30
DRAWING_MAX = 4

# 沒有材料表的圖框（規格書型那幾種），「圖上原文」給模型幾個字。
# 清單題那一份短，理由與 DRAWING_ITEMS 同一個：那時候的預算要留給索引清單。
DRAWING_SPEC_CHARS = 2000
DRAWING_SPEC_LIST_CHARS = 500

# 圖多到幾份就改成「總結 + 索引清單」而不是逐份展開。
#
# ★ 這兩個數字擋的是「跨月份清單題」的幻覺
#   跨兩個月的清單題若只展開前幾份（照更新時間新到舊，全落在較新的月份），
#   其餘只留一句「另有 N 份未列出」，模型就會回「沒有較早那個月的工程圖」——
#   它看到的東西支持這句話，而使用者的畫面上就有那個月的卡片。
#   所以圖一多就反過來：**展開的份數變少，讓每一份都至少在索引清單裡留一行**。
#   材料表是「這一份圖裡有什麼」的答案，索引是「到底有哪些圖」的答案，
#   而這一類問句問的是後者。
DRAWING_SUMMARY_MIN = 3         # 幾份以上改用「總結」型任務（DRAWING_LIST_TASK）
# 清單題展開幾份的**標題欄與出處**（沒有材料表，見 _render_drawings）。
# 3 是因為任務指令要它「最多舉三份例子」，而每個例子都得有可以照抄的完整
# 檔名與更新時間 —— 索引那一行刻意不放檔名（見 _drawing_scope）。
DRAWING_LIST_DETAIL = 3
DRAWING_INDEX_MAX = 60          # 索引清單最多列幾份

# build_context() 這條路的自我預算。
#
# ★ 為什麼要自己先收在預算內
#   工具迴圈把這段背景做**字元截斷**（RESULT_CHARS＝6,000，見
#   vendor/agentkit/delegate.py 的 _clip）。而這一段的尾巴是任務指令
#   （「不可以說某段時間沒有圖」「講出每段份數」）—— 超長時被切掉的正是
#   那幾句話，而且不會有任何錯誤訊息。所以這裡自己先收：**砍索引清單的
#   行數，不砍尾巴**。留 150 字餘裕（教訓庫那一段是照題目挑的，長度會浮動）。
#
# ★ 誰先讓位是有順序的：教訓 -> 索引，而且**索引排在最後**
#   這兩段搶同一份預算，而教訓庫**會隨時間長大**：教訓一變長，索引就少幾行
#   —— 清單是新到舊排的，掉的是最舊的那幾份。等於「模型犯過的錯越多，它下一次
#   拿到的事實就越少」，而少掉的正好是最容易被誤判成「沒有」的那幾份。方向是反的。
#   教訓是提醒（少一條仍然答得出來），索引是事實（少一份圖就是下一個
#   「三月沒有」），所以先砍教訓。
CONTEXT_BUDGET = 5990


def _clip_lines(text: str, cap: int) -> str:
    """截到 cap 字以內，**只在行邊界截**——半條教訓比沒有那條更糟。"""
    if len(text) <= cap:
        return text
    out: list[str] = []
    used = 0
    for line in text.splitlines():
        if used + len(line) + 1 > cap:
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(out)


def _detail_count(total: int) -> int:
    """這一題要展開幾份圖。

    圖少（1~2 份）＝逐份細講，含材料表，上限 DRAWING_MAX。
    圖多＝清單題，只展開三份的標題欄與出處當引用來源，材料表全部不展開。
    """
    if total < DRAWING_SUMMARY_MIN:
        return min(total, DRAWING_MAX)
    return DRAWING_LIST_DETAIL



# 材料表可能出現的欄，順序照圖上那張表。自家圖框只有前三欄，客戶圖框才有
# 後面那些（見 src/drawing_frames.py）。
_BOM_COLS = (("description", "DESCRIPTION"), ("qty_raw", "Q'TY"),
             ("unit", "UNIT"), ("color", "COLOR"),
             ("length_raw", "LENGTH"), ("remark", "備註"))
_PNL_COLS = (("description", "P/N"), ("length_raw", "LENGTH"),
             ("remark", "PACKAGE"))


def _group_items(items: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in items:
        out.setdefault(r.get("table_tag") or "BOM", []).append(r)
    return out


def _item_layout(tag: str, rows: list[dict]) -> tuple[str, tuple[str, ...]]:
    """(那一段的抬頭, 要列出來的欄)。**整欄都空的就不寫進欄頭。**

    這一點不是排版偏好：欄頭寫著 `UNIT` 而每一列都空白，模型會推論「這張圖
    的單位欄沒填」——而有些圖框上根本沒有這一欄。少講比講錯好。
    """
    spec = _PNL_COLS if tag == "PN_LENGTH" else _BOM_COLS
    used = [(k, h) for k, h in spec if any(r.get(k) for r in rows)] or [spec[0]]
    heads = " | ".join(h for _, h in used)
    if tag == "PN_LENGTH":
        return (f"料號長度對照表 P/N AND LENGTH（NO | {heads}）",
                tuple(k for k, _ in used))
    return (f"材料表（ITEM | {heads}）", tuple(k for k, _ in used))


def _render_drawings(drawings: list[dict]) -> str:
    """工程圖區塊。

    每一份圖的抬頭一定同時寫**檔名、檔案更新時間、連結**三件事，因為 SYSTEM
    要求引用時三者缺一不可 —— 提示裡沒給，模型就只能編或漏。時間欄的來源是
    用檔案 md5 對 PLM 檔案表查出來的更新時間，
    不是本機下載時間（那對所有檔案都一樣，沒有鑑別力）。

    最後一行的「判讀方式」是刻意寫成人話而不是丟 extract_method 代碼：
    模型要據此決定要不要加「請以原圖為準」，讀得懂比精確重要。

    ★ `with_items=False`：只給標題欄與出處，不給材料表
      清單型的問句（「某段時間的工程圖」撈到幾十份）問的是「有哪些圖」。
      材料表回答的是「這一份圖裡有什麼」，一份就很長 —— 展開幾份就把
      整批的索引清單擠出預算，而那份清單才是這一題的答案。
      標題欄＋出處留著，因為引用一份圖需要檔名、更新時間、連結三件事。
    """
    out: list[str] = []
    detail = _detail_count(len(drawings))
    with_items = len(drawings) < DRAWING_SUMMARY_MIN
    for d in drawings[:detail]:
        stamp = str(d.get("doc_modified_on") or "更新時間未知")[:19]
        out.append(
            f"工程圖（料號 {d['item_number']}，來源檔名 {d['local_file']}，"
            f"檔案更新時間 {stamp}，連結 {d.get('pdf_url') or '無'}）"
        )
        tb = [
            ("TITLE", d.get("dwg_title")),
            ("PART NO.", d.get("dwg_part_no")),
            ("REV.", d.get("dwg_rev")),
            ("SIZE", d.get("dwg_size")),
            ("SCALE", d.get("dwg_scale")),
            ("SHEET", _sheet(d)),
            ("PD./FILE NO.", d.get("pd_file_no")),
            # 客戶圖框多出來的欄位（MATERIAL / P.CODE / CUSTOMER PART NO. /
            # DRAWN BY …）。不放進來的話，圖上明明有的欄位，模型只能說沒有。
            # 欄名用 drawing_frames 那份對照表，與卡片上顯示的完全一致 ——
            # 兩邊不一致的話模型會引用出圖上根本沒有的欄名。
            *drawing_frames.field_rows(d.get("title_fields")),
        ]
        shown = " ｜ ".join(f"{k} {v}" for k, v in tb if v not in (None, ""))
        out.append(f"  標題欄：{shown or '（未抽到）'}")

        # 一份圖可能有兩張表：材料表，加上客戶圖框才有的料號長度對照表。
        # 兩張表要分開寫、而且要標明是哪一張 —— 併在一起的話「這份圖用了
        # 幾種材料」會被 12 列料號長度算進去。
        tables = _group_items(d.get("items") or []) if with_items else {}
        for tag, rows in tables.items():
            head, cols = _item_layout(tag, rows)
            out.append(f"  {head}列數：{len(rows)} 列（表格列數，不是物料總件數）")
            if tag == "BOM":
                ar = [r for r in rows if str(r.get("qty_raw") or "").strip().upper() == "A/R"]
                unknown = [r for r in rows if r.get("qty_num") is None
                           and str(r.get("qty_raw") or "").strip().upper() in ("", "PC", "PCS")]
                def item_labels(items):
                    return "、".join(str(r.get("item_no") if r.get("item_no") is not None else "?") for r in items)
                out.append(f"  Q'TY 原值為 A/R：{len(ar)} 列"
                           + (f"；ITEM {item_labels(ar)}" if ar else "")
                           + "（依需求，不是固定件數）。")
                if unknown:
                    out.append(f"  Q'TY 缺少數值或僅有單位：ITEM {item_labels(unknown)}；"
                               "不能視為固定數量，也不能自行補成 1 PCS。")
            out.append(f"  {head}：")
            for r in rows[:DRAWING_ITEMS]:
                no = r.get("item_no")
                vals = " | ".join(str(r.get(c) or "") for c in cols)
                out.append(f"    {no if no is not None else '?'} | {vals}")
            if len(rows) > DRAWING_ITEMS:
                out.append(f"    …其餘 {len(rows) - DRAWING_ITEMS} 列未列出")
        no_bom = d.get("frame") in drawing_frames.NO_BOM_FRAMES
        if not with_items and not no_bom:
            n_items = len(d.get("items") or [])
            out.append(
                f"  材料表：這一題是清單題，材料表沒有展開（這一份有 {n_items} 列）。"
                "要看材料表請使用者指定這個料號或檔名再問一次。"
                if n_items else
                "  材料表：這一題是清單題，材料表沒有展開。"
            )
        elif not tables and no_bom:
            # 這幾種圖框本來就沒有材料表（這張圖就是那個零件本身，或
            # 材料是自由文字/點陣圖），不要講成「沒抽到」——那句話會讓
            # 模型以為漏了什麼去道歉，而圖上真正的內容（規格、性能、
            # 材料自由文字）就在下面這段 title_raw 原文裡。
            #
            # ★ 清單題（with_items=False）也走這一支，只是截短
            #   舊寫法把清單題整個攔在上面那個分支，回一句「材料表沒有展開」
            #   就結束 —— 對這幾種圖框那句話還是假的（它們沒有材料表可以
            #   展開），而且把唯一有內容的那一欄一起吞掉了。「哪幾份圖的
            #   Insertion force 是 20N」既是清單題、又只有這一段答得出來。
            spec = (d.get("title_raw") or "").strip()[
                :DRAWING_SPEC_CHARS if with_items else DRAWING_SPEC_LIST_CHARS]
            out.append(
                "  這份圖沒有材料表（圖上本來就沒有，不是漏抽）。"
                "以下是圖上除了標題欄以外的原文（OCR／文字層混排，"
                "行與行之間的對應關係不保證正確，但每一個詞都是圖上真的印著的）："
                + ("（清單題，只給前段）" if not with_items else "") + "\n"
                f"  {spec or '（這份圖除了標題欄沒有其他文字）'}"
            )
        elif not tables:
            out.append("  材料表：（這份圖沒有抽到材料表）")

        method = {"text": "PDF 文字層（非 OCR，準確）",
                  "ocr": "OCR",
                  "ocr_grid": "OCR（表格列以圖上的格線定位）",
                  "ocr+text": "OCR 與 PDF 文字層併用（文字層那部分準確）",
                  "ocr_grid+text": "OCR 與 PDF 文字層併用，表格列以圖上的格線定位",
                  }.get(
            d.get("extract_method") or "", d.get("extract_method") or "未知")
        pn_ok = ("PART NO. 已與 PLM 料號核對一致"
                 if d.get("part_no_ok")
                 else "PART NO. 未與 PLM 料號核對一致，此欄請以原圖為準")
        conf = d.get("ocr_conf")
        tail = f"，最低字元信心 {conf}" if conf is not None else ""
        out.append(f"  判讀方式：{method}（{pn_ok}{tail}）")
    if len(drawings) > detail:
        # ★ 這句話一定要指回索引清單
        #   舊寫法只寫「另有 N 份未列出」。「未列出」被模型讀成「我不知道那些是
        #   什麼」，接著它就用手上這幾份去回答「有沒有三月的圖」—— 而正確答案
        #   在清單裡。現在把它指回上面那份清單，並且明講沒展開不等於不存在。
        out.append(
            f"（上面只展開了 {detail} 份的標題欄"
            + ("與材料表" if with_items else "")
            + f"；其餘 {len(drawings) - detail} 份的標題欄沒有展開，"
            "但每一份都在前面那份「這一題的全部工程圖」清單裡，一樣是語料裡真的有的圖）"
        )
    # 圖少的時候沒有索引清單那一段，這句話就得掛在這裡（見 PLACEHOLDER_NOTE）。
    if len(drawings) < DRAWING_SUMMARY_MIN and _has_placeholder_name(drawings[:detail]):
        out.append(PLACEHOLDER_NOTE)
    return "\n".join(out)


# 檔名裡的系列佔位（`Z0000XX0XXXX-0_XX_V01.pdf`）。
#
# ★ 這一行擋的是工具模式最難看的一種失敗
#   那份圖是**一份涵蓋整個料號系列的通用圖**，檔名就是這樣印的。模型看到同一
#   行的料號是 `Z0000XX01234-0`，就把 `XXXX` 換成 `1234` 寫出一個不存在的檔名
#   —— 它以為自己在修 OCR 沒讀好的地方（系統的提示到處在講「OCR 會錯、O/0
#   會混淆」，這個推論對它很自然）。`unknown_file` 正確擋下了，但重寫後它仍
#   堅持那個「修正」，於是整題的正確答案被整包丟掉。講一句話比擋十次便宜。
_PLACEHOLDER_NAME_RE = re.compile(r"X{2,}")

PLACEHOLDER_NOTE = (
    "★ 有幾份圖的檔名裡有一串 X（例如 `Z0000XX0XXXX-0_XX_V01.pdf`）——"
    "那是**一份涵蓋整個料號系列的通用圖**，檔名就是這樣印的，不是 OCR 沒讀好。"
    "引用時原封不動照抄，**不要拿同一行的料號把 X 換掉**：換掉之後那個檔名"
    "在系統裡並不存在。"
)


def _has_placeholder_name(drawings: list[dict]) -> bool:
    return any(_PLACEHOLDER_NAME_RE.search(d.get("local_file") or "")
               for d in drawings)


def _drawing_scope(found: dict, budget: int | None = None) -> str:
    """「這一題到底有哪些圖」——總份數、各時間區間的份數，以及每一份一行的索引。

    ★ 為什麼這一段要排在工程圖區塊**前面**
      工具迴圈那條路的 context 是字元截斷的（RESULT_CHARS＝6,000，見
      vendor/agentkit/delegate.py）。排在材料表後面的話，圖一多就被切掉 ——
      而被切掉的正是「某個月有 N 份」這個唯一能擋掉幻覺的事實。

    ★ 份數寫「資料庫直接統計」是刻意的
      模型看到清單會自己去數，而清單有 60 份的上限。講明份數不是它數出來的，
      它就不會拿「我數到的」去覆蓋掉一個正確的總數。
    """
    drawings = found.get("drawings") or []
    if len(drawings) < DRAWING_SUMMARY_MIN:
        return ""

    lines = [f"★ 這一題的全部工程圖：共 **{len(drawings)} 份**。"]

    buckets = [b for b in (found.get("drawing_buckets") or [])]
    if buckets:
        lines.append(
            "　各時間區間各有幾份（資料庫直接統計，不是從下面的清單數出來的）："
            + "、".join(f"{b['label']} **{b['n']} 份**" for b in buckets)
            + "。"
        )
        empty = [b["label"] for b in buckets if not b["n"]]
        if empty:
            lines.append(f"　其中這幾段確實一份都沒有：{'、'.join(empty)}。")

    warn = ("★ 這份清單上的每一份圖都是語料裡真的有的。**份數 >0 的時間區間一律不可以"
            "說「沒有」「查無」**——沒有展開標題欄不等於不存在。")
    if _has_placeholder_name(drawings):
        warn += "\n" + PLACEHOLDER_NOTE

    rows: list[str] = []
    # 預算是給清單用的：抬頭那幾行（份數統計）與收尾那句警告一定要留住。
    room = None if budget is None else \
        budget - sum(len(x) + 1 for x in lines) - len(warn) - 80
    # ★ 索引只給料號與更新時間：**不放檔名，也不放連結**
    #   它的職責只有一個 —— 證明「那一段時間確實有這些圖」，料號加時間就夠了。
    #   放檔名有兩個壞處：
    #     1. 幾十個檔名很佔篇幅，只能從索引的行數裡扣 —— 等於用「證據」換
    #        「引用資訊」，而被扣掉的行永遠是清單最尾端、也就是最舊的那幾份。
    #     2. 半個檔名是誘餌。截短版（`…_R…`）會被模型原樣抄進答案，使用者看到
    #        的是半截檔名；完整版又會被它「修正」——`Z0000XX0XXXX-0_XX_V01.pdf`
    #        的 X 被換成同一行的料號，寫出一個不存在的檔名（見 PLACEHOLDER_NOTE）。
    #   要引用的那幾份在下面整份展開，檔名、更新時間、連結都是完整的。
    for i, d in enumerate(drawings[:DRAWING_INDEX_MAX], 1):
        stamp = str(d.get("doc_modified_on") or "更新時間未知")[:19]
        row = f"　  {i}. {d.get('item_number')} ｜ {stamp}"
        if room is not None and sum(len(x) + 1 for x in rows) + len(row) > room:
            break
        rows.append(row)
    # 預算再緊也要看得到幾份，否則這一段等於沒寫。
    if not rows:
        rows = [f"　  {i}. {d.get('item_number')} ｜ "
                f"{str(d.get('doc_modified_on') or '')[:19]}"
                for i, d in enumerate(drawings[:8], 1)]

    lines.append("　清單（料號 ｜ 檔案更新時間）。它的用途是證明這些圖存在，"
                 "所以只給這兩格：**這裡沒有檔名也沒有網址，兩者都不要自己推**。"
                 "要寫檔名或連結請用下面整份展開的那幾份；使用者想看清單裡某一份的"
                 "細節時，請他報那個料號再問一次：")
    lines += rows
    if len(drawings) > len(rows):
        lines.append(f"　  …其餘 {len(drawings) - len(rows)} 份沒有列進這份清單"
                     "（篇幅有限），但一樣在上面那個總份數與各區間份數裡，"
                     "**不是不存在**。")
    lines.append(warn)
    return "\n".join(lines)


def _sheet(d: dict) -> str | None:
    a, b = d.get("sheet_no"), d.get("sheet_of")
    if a is None:
        return None
    return f"{a} OF {b}" if b else str(a)


def _render_listing(listing: dict, results: list[dict]) -> str:
    """The 清單／統計 block.

    Deliberately leads with the total and the distribution rather than the rows.
    A "列出所有" question over tens of thousands of parts cannot be answered by rows -- the
    number and the shape of the set ARE the answer, and the rows are an example.
    Handing the model 25 rows and no total invites it to imply the list is
    complete, which is the exact failure this path exists to prevent.
    """
    cond = "、".join(
        c["label"] + (" = " + "/".join(c["values"]) if c["values"] else "")
        for c in (listing.get("conditions") or [])
    ) or "（無額外條件，等於整個語料庫）"

    if listing.get("empty"):
        return (
            "清單／統計結果：\n"
            f"- 查詢條件：{cond}\n"
            f"- 符合筆數：0\n"
            f"- {listing.get('message')}\n\n"
            "請直接回答「目前系統尚未包含該資料」並說明是哪個條件沒有資料，不要用相近資料代答。"
        )

    lines = [
        "清單／統計結果（資料庫直接統計，非相似度排序）：",
        f"- 查詢條件：{cond}",
        f"- 符合筆數：{listing['total']}",
    ]
    if listing.get("excluded_placeholder"):
        lines.append(
            f"- 另有 {listing['excluded_placeholder']} 筆為領號用佔號料號，已排除"
        )

    if listing["kind"] == "values" and listing.get("values"):
        label = (listing.get("target") or {}).get("label") or "值"
        lines.append(
            f"- {label}共 {len(listing['values'])} 種"
            f"（每種後面的數字是料號筆數，不是種類數）："
        )
        lines += [
            f"    {v['value']}：{v['n']} 筆" for v in listing["values"][:grounding.LIST_VALUE_LIMIT]
        ]
        remaining = len(listing["values"]) - grounding.LIST_VALUE_LIMIT
        if remaining > 0:
            lines.append(f"    …其餘 {remaining} 種未列出")
            lines.append("只提供以上部分值，請交代未展開的種類數，不要補寫未提供的名稱。")
        elif not listing.get("count_only"):
            lines.append(f"以上已完整提供 {len(listing['values'])} 種{label}；請逐一列出值與料號筆數。")
        # The requested values are the answer. Unrelated distributions and
        # sample part rows previously led the model to select OTHERS alone.
        if listing.get("export_url"):
            lines.append(f"- 相關料號完整 {listing['total']} 筆清單 CSV：{listing['export_url']}")
        return "\n".join(lines)

    for b in listing.get("breakdown") or []:
        # The unit is spelled out on every number: given "<人名> 2" the model
        # read the 2 as a headcount and answered "2 人" for a 2-ROW group.
        top = "、".join(f"{i['value']} {i['n']} 筆" for i in b["items"][:8])
        lines.append(f"- {b['label']}分佈：{top}")

    if results and not listing.get("count_only"):
        # Fewer rows the larger the set. 25 alphabetically-first rows out of
        # tens of thousands are not a sample of anything, and each one costs ~200 characters
        # of description -- which is the budget the gateway's `thinking` block
        # then does not have. The count and the distribution are the answer here.
        cap = LIST_ROWS if listing["total"] <= 200 else 10
        shown = results[:cap]
        # Said unconditionally, "不是全部" made the model add a "非完整清單"
        # disclaimer to a reply that had just listed all 2 of 2 matching parts --
        # and then invite the user to download a CSV link that was not offered.
        if len(shown) >= listing["total"]:
            lines.append(f"- 以下是全部 {len(shown)} 筆（依{listing.get('order') or '料號'}排序）：")
        else:
            lines.append(
                f"- 以下只是其中 {len(shown)} 筆"
                f"（依{listing.get('order') or '料號'}排序），不是全部："
            )
        lines.append(_render_rows_compact(shown))
    if listing.get("export_url"):
        lines.append(
            f"- 完整 {listing['total']} 筆清單請下載 CSV：{listing['export_url']}"
        )
    return "\n".join(lines)


# Rows rendered into a listing prompt, when the whole set is small enough that
# listing it is actually the answer.
LIST_ROWS = 25
NAME_CLIP = 70


def _render_rows_compact(results: list[dict]) -> str:
    """Row lines for a listing block: identity and classification only.

    _render() is right for a ranked answer, where the model must explain how ten
    near-identical parts differ. In a listing the rows are an illustration of a
    counted set, and a full 300-character 導體規格 description per row buys
    nothing while crowding out the counts above it.
    """
    out = []
    for i, r in enumerate(results, 1):
        name = r["name"]
        if len(name) > NAME_CLIP:
            name = name[:NAME_CLIP] + "…"
        bits = [f"[{i}] {r['item_number']}", name]
        if r.get("prod_l2"):
            bits.append(r["prod_l2"])
        if r.get("created_by"):
            bits.append(f"建立{r['created_by']}")
        if r.get("created_on"):
            bits.append(str(r["created_on"])[:10])
        if r.get("state"):
            bits.append(str(r["state"]))
        out.append(" | ".join(bits))
    return "\n".join(out)


def _messages(
    question: str,
    results: list[dict],
    found: dict | None,
    history: list[dict[str, str]] | None,
    with_lessons: bool = True,
    picked: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Prior turns, then the current question with its retrieval block.

    Only the current turn carries retrieval results. Replaying every past turn's
    hit list would both dominate the context window and invite the model to
    answer from a stale one.

    Only COMPLETE (user, assistant) pairs are replayed. A turn whose answer is
    empty -- the gateway was down when it was asked -- is dropped entirely. Just
    filtering empty content instead leaves two consecutive user messages, which
    breaks the API's strict role alternation, so one failed turn would poison
    every follow-up in that conversation.
    """
    msgs: list[dict[str, str]] = []
    pending_user: str | None = None
    for m in history or []:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role == "user":
            pending_user = content or None
        elif role == "assistant" and pending_user and content:
            msgs.append({"role": "user", "content": pending_user})
            msgs.append({"role": "assistant", "content": content})
            pending_user = None
    msgs.append({
        "role": "user",
        "content": build_prompt(question, results, found,
                                with_lessons=with_lessons, picked=picked),
    })
    return msgs


def _complete(msgs: list[dict[str, str]], system: str = SYSTEM,
              max_tokens: int | None = None, *, temperature: float | None = None) -> tuple[str, str | None]:
    """一次呼叫 -> (文字, 錯誤)。答題、重寫、裁判三條路共用同一個出入口，
    所以 thinking block 的過濾與 max_tokens 耗盡的判斷只有一份。"""
    try:
        from .request_budget import checkpoint, client_for_request
        checkpoint()
        model = _effective_model()
        checkpoint()
        msg = client_for_request(_client()).messages.create(
            model=model,
            max_tokens=max_tokens or MAX_TOKENS,
            system=system,
            messages=msgs,
            **({"temperature": temperature} if temperature is not None else {}),
        )
        checkpoint()
        # Skip `thinking` blocks; concatenate text blocks.
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        if not text.strip() and msg.stop_reason == "max_tokens":
            return "", "模型的思考過程用盡 max_tokens，請調高 LLM_MAX_TOKENS"
        return text.strip(), None
    except Exception as e:  # noqa: BLE001
        return "", f"{type(e).__name__}: {e}"


def answer(
    question: str,
    results: list[dict],
    found: dict | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Non-streaming. Never raises -- degrades to retrieval-only."""
    text, err = _complete(_messages(question, results, found, history))
    if err:
        return {"ok": False, "text": "", "error": err}
    return {"ok": True, "text": text, "error": None}


# --------------------------------------------------------------------------
# 事實核對 + 重新生成
# --------------------------------------------------------------------------
def answer_verified(
    question: str,
    results: list[dict],
    found: dict | None = None,
    history: list[dict[str, str]] | None = None,
    with_lessons: bool = True,
    learn: bool | None = None,
) -> dict[str, Any]:
    """答一次 -> 逐字核對 -> 有幻覺就把違規項回饋給模型、重寫一次 -> 再核對。

    回傳
      ok           gateway 有沒有給出東西（核對不過**不算** not ok —— 那是內容問題）
      text         最終要送給使用者的文字（仍有 block 違規時只回資料核對失敗提示）
      verified     最終回答是否通過字面事實核對；不代表語意裁判或人工驗收通過
      violations   最終仍未通過的項目（結構化，給 API payload 與跑分用）
      first_pass   第一次就通過核對嗎（這個數字才是模型真正的品質指標）
      attempts     總共呼叫了幾次 gateway
      lessons      這一題掛了哪幾條教訓（跑分時用來判斷教訓有沒有效）

    ★ 為什麼只重寫一次
      實測第二次還過不了的，第三次也過不了 —— 因為那通常代表資料本身不足以
      回答（模型硬要補洞），再問一次只是多花十秒得到同一段話。過不了就據實
      告訴使用者回答未通過核對，不送出仍有錯誤的文字。
    """
    learn = LEARN_RUNTIME if learn is None else learn
    route = route_of(found)
    # 撈一次，同時用於「寫進 prompt」與「回報掛了哪幾條」（見 pick_lessons）。
    picked = pick_lessons(question, found) if with_lessons else []

    msgs = _messages(question, results, found, history,
                     with_lessons=with_lessons, picked=picked)
    context = msgs[-1]["content"]

    text, err = _complete(msgs)
    if err or not text:
        return {"ok": False, "text": "", "error": err or "模型沒有產生文字",
                "violations": [], "first_pass": False, "attempts": 1,
                "lessons": [r["key"] for r in picked], "route": route}

    v1 = grounding.check(text, context, found, question)
    final_text, final_v, attempts = text, v1, 1

    if grounding.blocking(v1):
        retry = [*msgs,
                 {"role": "assistant", "content": text},
                 {"role": "user", "content": grounding.feedback(v1)}]
        text2, err2 = _complete(retry)
        attempts = 2
        if not err2 and text2:
            v2 = grounding.check(text2, context, found, question)
            # ★ 只有**真的變好**才採用重寫版。模型偶爾會為了避開一個違規而生出
            #   兩個新的（最常見：刪掉料號之後改用「類似的線材」這種更難查證的
            #   說法）。比的是 block 級的數量，warn 級不參與決策。
            if len(grounding.blocking(v2)) < len(grounding.blocking(v1)):
                final_text, final_v = text2, v2

    # 第一次就沒過 -> 這是模型真的犯過的錯，寫進教訓庫。即使重寫救回來了也要記：
    # 記住的目的是讓**下一題的第一次**就對，省掉那次重寫。
    if learn and grounding.blocking(v1):
        for v in grounding.blocking(v1):
            rule = lessons.rule_for(v.kind)
            if rule:
                lessons.record(route, rule, question=question, wrong=v.detail,
                               source="runtime")

    blocked = grounding.blocking(final_v)
    return {
        "ok": True,
        "text": grounding.UNVERIFIED_NOTICE if blocked else final_text,
        "verified": not blocked,
        "error": None,
        "violations": grounding.as_payload(final_v),
        "first_pass": not grounding.blocking(v1),
        "attempts": attempts,
        "lessons": [r["key"] for r in picked],
        "route": route,
    }


# LLM 裁判的角色設定。
#
# ★ 為什麼要求它預設判「有問題」而不是「沒問題」
#   裁判模型的預設偏誤是配合 —— 問它「這個答案對嗎」，它傾向說對。把舉證責任
#   反過來（「找不到根據就是不合格」）之後，它才會真的去比對每一句話。
JUDGE_SYSTEM = """你是嚴格的事實查核員，不是助理。

你會拿到三段東西：使用者的問題、系統提供給助理的**全部資料**、助理的回答。
你的工作只有一件：檢查回答裡的每一句陳述，是不是都能在「提供的資料」裡找到根據。

判準：
- 找不到根據的陳述一律算不合格，即使它聽起來很合理、即使你知道它在現實中是對的。
- 把某個欄位的值講成另一個欄位（例如拿品名當工程圖 TITLE、拿 A 料號的長度講成 B 料號的）算不合格。
- 資料裡明講「查無」「尚未包含」，回答卻給出一個值，算不合格。
- 回答只是沒有講到某些資料，不算不合格（那是取捨，不是錯誤）。
- 語氣、排版、詳略一律不評。
- 等值寫法與欄位別名不是錯誤：622mm 與 622 mm、GEN5 與 GEN 5、AWG 與線規
  可以互換；SHEET 1 OF 2 表示第 1 頁／共 2 頁。摘要省略描述中已另列的規格也可以。
  A/R 與 A/R（依需求）是同一個數量值，中文解釋不是擅改數量。Q'TY 只有 PCS
  而沒有數字時，不能宣稱是固定件數或補成 1 PCS。
  「ITEM 5 的 Q'TY 只有 PCS，無法確認件數」對原值 PCS 是正確說法，應 PASS。
- 材料名稱的概括分類與型號縮寫不同：只說「線材」可以，但聲稱某 ITEM 的
  DESCRIPTION／型號為某字串時，不能把原文完整型號縮短（UL1006430AWG 不能
  改成 UL1006），也不能刪除中間字詞、補字或修正 OCR 的 O/0。
- 數量與筆數必須逐項核對。Q'TY 統計明列 A/R 為 N 列，回答說成其他筆數就
  FAIL；不能以全表列數、未標示數值的列數或其他表格的列數取代 N。
- 提問引導、輸出格式要求與助理任務指令不是事實資料，不用檢查是否照做。
  判定「資料沒有」前，先逐列讀完檢索結果與材料表，不要只讀區塊標題。
- FAIL 的每一項都必須是一項真正不符的事實，引用回答中的主張及資料中的原文
  作為依據。正確的主張不要列入問題；不要把缺少額外細節當成錯誤。
- 先在心中完成核對，再給最後判定。若核對後發現某項主張與資料一致，刪除該問題；
  沒有真正錯誤就輸出 PASS。不要先寫 FAIL 再猜測為什麼失敗，不輸出推理、反問、
  「等等」「可能是」或承認問題其實正確的段落。

輸出格式（不要有其他任何文字）：
第一行只寫 PASS 或 FAIL。
FAIL 時，接下來每一行寫一個問題，格式為「- 問題：<哪一句> ｜ 依據：<資料裡是怎麼寫的>」。"""


def judge(question: str, context: str, answer_text: str) -> dict[str, Any]:
    """語意層核對。字面核對（grounding）抓不到的那一類：答非所問、欄位張冠李戴。

    預設只在跑分時開（LLM_JUDGE=1 或 evalrun --judge）—— 它每題多一次呼叫，
    而且它自己也是一顆會犯錯的模型，所以它的判定進的是報表，不會拿去攔線上答案。
    """
    msg = (f"【使用者的問題】\n{question}\n\n"
           f"【提供給助理的全部資料】\n{context}\n\n"
           f"【助理的回答】\n{answer_text}")
    text, err = _complete([{"role": "user", "content": msg}], system=JUDGE_SYSTEM,
                          max_tokens=800, temperature=0)
    if err:
        return {"ok": None, "issues": [], "error": err}
    head = text.strip().splitlines()
    verdict = (head[0].strip().upper() if head else "")
    issues = [ln.strip(" -") for ln in head[1:] if ln.strip().startswith("-")]
    if verdict == "PASS" and not any(line.strip() for line in head[1:]):
        return {"ok": True, "issues": [], "error": None}
    if verdict == "FAIL" and issues:
        return {"ok": False, "issues": issues, "error": None}
    # 格式沒照講 —— 不猜，回 None 讓報表標成「裁判無效」而不是誤判成失敗。
    return {"ok": None, "issues": issues, "error": f"裁判輸出格式不符：{text[:80]}"}


def stream_answer(
    question: str,
    results: list[dict],
    found: dict | None = None,
    history: list[dict[str, str]] | None = None,
    status: dict[str, Any] | None = None,
    verify: bool | None = None,
) -> Iterator[str]:
    """Yield text deltas. Falls back to a single non-streamed chunk, then to
    an error string -- the caller always gets something.

    `status` is an optional dict owned by the caller, filled in with
    {"ok": bool, "error": str | None} once the generator is exhausted.
    核對開啟時 status 還會多帶 violations / first_pass / attempts，呼叫端
    （src/a2a.py）據此決定要不要把核對結果一起送到前端。

    ★ 核對模式下**不能真的串流**，這是刻意的取捨
      要核對就得先看到完整的答案；而一旦有一個字已經送到前端，就再也收不回來了
      —— 前端會把「已經印出來的幻覺」與「後來補上的更正」並排顯示，那比不核對
      更糟。所以 verify=True 時整段答案先在伺服器端跑完核對，通過（或重寫過）
      才一次送出。使用者感受到的是「想比較久」，不是「先講錯再改口」。
      verify=False 保留原本逐字串流的行為，兩條路的失敗處理完全一致。

    It exists because the last-resort yield below is an ERROR MESSAGE dressed as
    an answer, and a caller watching only the yielded chunks cannot tell the two
    apart. /api/chat reported `llm_available: true` for a dead gateway on exactly
    that basis, and stored "[LLM 無法使用：…]" as the assistant's turn -- which
    _messages() then replayed as a real answer, because its "drop the whole
    failed turn" rule keys on the content being EMPTY.
    """
    def _done(ok: bool, error: str | None = None) -> None:
        if status is not None:
            status["ok"], status["error"] = ok, error

    _done(False, "串流未開始")

    if VERIFY if verify is None else verify:
        res = answer_verified(question, results, found, history)
        if status is not None:
            status.update({
                "violations": res.get("violations") or [],
                "first_pass": res.get("first_pass"),
                "attempts": res.get("attempts"),
                "route": res.get("route"),
                "verified": res.get("verified"),
            })
        if res["ok"]:
            _done(True)
            yield res["text"]
        else:
            err = str(res["error"] or "LLM 無回應")
            _done(False, err)
            yield f"[LLM 無法使用：{err}]"
        return

    try:
        from .request_budget import checkpoint, client_for_request
        checkpoint()
        model = _effective_model()
        checkpoint()
        with client_for_request(_client()).messages.stream(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            messages=_messages(question, results, found, history),
        ) as s:
            got = False
            for event in s:
                checkpoint()
                if (
                    event.type == "content_block_delta"
                    and getattr(event.delta, "type", None) == "text_delta"
                ):
                    got = True
                    yield event.delta.text
        if got:
            _done(True)
            return
        # Streamed cleanly but produced no text -- almost always the gateway
        # spending the whole max_tokens budget on `thinking`. Falling through to
        # answer() surfaces that as a message instead of an empty reply, which
        # is what a silent `return` here used to do.
        stream_err = RuntimeError("串流未產生文字")
    except Exception as e:
        stream_err = e

    res = answer(question, results, found, history)
    if res["ok"]:
        _done(True)
        yield res["text"]
    else:
        err = str(res["error"] or stream_err)
        _done(False, err)
        yield f"[LLM 無法使用：{err}]"


if __name__ == "__main__":  # python -m src.llm
    # Answers "is the gateway usable, and if not what do I change" without
    # booting the service.
    import sys

    for _s in (sys.stdout, sys.stderr):
        if getattr(_s, "encoding", "").lower() != "utf-8":
            _s.reconfigure(encoding="utf-8", errors="replace")

    print(f"LLM_BASE_URL = {BASE_URL}")
    print(f"LLM_MODEL    = {MODEL}"
          + (f"（實際使用 {_effective_model()}，主模型不在 /v1/models 清單）"
             if _effective_model() != MODEL else ""))
    print(f"LLM_MODEL_FALLBACK = {MODEL_FALLBACK or '(未設定)'}")
    print(f"LLM_API_KEY  = {'(未設定)' if not API_KEY else f'(已設定，長度 {len(API_KEY)})'}")
    print(f"gateway 可用模型 = {models() or '(空清單)'}")

    d = diagnose()
    print(f"\n{'✅' if d['ok'] else '❌'} {d['title']}")
    if not d["ok"]:
        print(f"   發生什麼：{d['detail']}")
        print(f"   該做什麼：{d['fix']}")
        print(f"   誰能修　：{d['owner']}")
        # 「要不要去要一把新 key」是使用者唯一會問的下一個問題，直接回答掉。
        print(f"   換新 key 有沒有用：{'有' if d['needs_new_key'] else '沒有'}")
        if d["raw"]:
            print(f"   原始錯誤：{d['raw'][:300]}")
    sys.exit(0 if d["ok"] else 1)
