"""公司本地 LLM gateway 客戶端（含 tool use）。

Gateway 是 Anthropic Messages API 相容的，所以直接用官方 anthropic SDK 指過去。
以下每一點都是實測過的，不要憑印象改：

  1. 認證走 `Authorization: Bearer`，SDK 要用 `auth_token=`，
     用 `api_key=` 會送成 x-api-key，gateway 收不到。

  2. ★ 一定會產生 thinking 區塊，而且算進 max_tokens。
     送 `thinking: {"type":"disabled"}` 會被忽略。連問 1+1 都燒 600~900 個
     思考 token，RAG 形狀的 prompt 實測 636。所以 max_tokens 是「思考+答案」
     的總預算，抓 4000 起跳。預算不夠的症狀很特別：
     stop_reason='max_tokens' 而 text 是空字串，不是被截一半的句子。
     ★ 委派迴圈每多一步就多一次思考，所以這裡把預算再放寬（見 MAX_TOKENS）。

  3. 回應有三個 content block：index 0 是空 text、index 1 是 thinking、
     index 2 才是答案。空的 text block 不能原樣送回去當歷史（Messages API
     不收空字串的 text block），_assistant_content() 會濾掉。

  4. Gateway 對「完全相同的請求」會回快取結果（連 usage 都一樣）。
     測試時要改內容，否則會以為參數沒有作用。

  5. tool use 可用，但 ★ **stop_reason 不可信**。
     回應可能含四個 block
     （空 text / thinking / text / tool_use），而 stop_reason 是 **'end_turn'**，
     不是 Anthropic 規格說的 'tool_use'。
     所以「要不要執行工具」必須看 content 裡有沒有 tool_use block
     —— 見 delegate.py 的判斷。照規格看 stop_reason 的症狀極度誤導：
     模型講了「我來呼叫材料建議 Agent」然後停住，軌跡一片空白，
     看起來像模型不肯用工具，其實工具呼叫已經生成、只是被丟掉了。

  6. ★ 工具呼叫有時候會變成**純文字**跑出來，而且是 OpenAI 的格式。
     gateway 後面接的是 OpenAI 相容引擎，這顆模型偶爾不把工具呼叫轉成
     tool_use block，而是把 {"id":"call_…","type":"function","function":…}
     整段寫進 text。使用者看到的是一段 JSON 亂碼當答案，而那次呼叫根本
     沒被執行。extract_text_tool_calls() 會把它解析出來合成正規的
     tool_use block，並從要播放的文字裡拿掉。
"""
from __future__ import annotations

import html
import json
import os
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ★ 一定要用明確路徑，不能裸呼叫 load_dotenv()。
#   python-dotenv 的 find_dotenv() 在「模組被 sys.path 插進去、從別的專案
#   import」這種情境下並不會可靠地從**這個檔案自己的位置**往上找，
#   它會照呼叫端的 cwd 找到那個專案自己的 .env（RFQ-RAG/Part-RAG
#   用 run.bat 啟動時 cwd 就是專案根目錄）——四個 agent 各自 cwd 不同，
#   裸呼叫等於「隨機」找到哪一份 .env，而不是這裡真正要的 Agent_Hub_RAG_Frontend/.env。
#   固定用 __file__ 往上兩層算路徑，才能不管是誰、從哪個目錄 import 這個
#   模組，都保證讀到同一份。
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:4000")
MODEL = os.environ.get("LLM_MODEL", "local-llm")
# 只有 MODEL 真的不在 gateway 的模型表裡時才切這顆，見 _effective_model()。
# 沒設就不會退——舊模型下架後這裡空著也沒關係，不會假裝有安全網。
MODEL_FALLBACK = os.environ.get("LLM_MODEL_FALLBACK", "")
# 委派迴圈的每一步都要重新思考一次，4000 只夠單步；這裡預設 6000。
MAX_TOKENS = int(os.environ.get("A2A_MAX_TOKENS", os.environ.get("LLM_MAX_TOKENS", "9000")))
TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "240"))
# 一次對話最多幾輪工具呼叫。實測兩步鏈就能回答，5 是給「查不到 -> 換工具」
# 留的餘裕；再高只會讓一題跑上好幾分鐘。
MAX_STEPS = int(os.environ.get("A2A_MAX_STEPS", "5"))


# --------------------------------------------------------------------------
# 故障分類
#
# ★ 為什麼 available() 要真的打一次 gateway
#   它以前只檢查「LLM_API_KEY 有沒有設」＋「anthropic 匯不匯得進來」，
#   所以 gateway 已經把這把 key 的模型授權收掉、每一題都 400 的時候，
#   Hub 左上角照樣寫「LLM 可用」。使用者看到的是「可用」但問什麼都失敗，
#   然後去懷疑檢索、懷疑 agent 沒開 —— 唯一沒被懷疑的是真正壞掉的那一層。
#   狀態列說「可用」就必須代表剛剛真的問到了。
#
# ★ 分類依據是對 gateway 的實際觀察（同一把 key、同一時刻）：
#     GET  /v1/models                 200 {"object":"list","data":[]}
#     POST /v1/messages 亂打的模型名    404 {"detail":"unknown model: …"}
#     POST /v1/messages 設定的模型名    400 Invalid model name … for your key
#     POST /v1/messages 亂打的 key      401 {"detail":"invalid API key"}
#   「設定的模型名拿到 400 而不是 404」證明模型還在，是 key 沒被授權；
#   /v1/models 回空清單證明這把 key 名下一顆模型都沒有。
#   兩者合起來 -> no_model_access，該做的是換 key，不是改 LLM_MODEL。
#
# ★ quota 與 rate_limit 分成兩類，因為處置相反：429 等一下就好，
#   額度用盡才要換新 key。混在一起講，使用者會為了限流去要一把新 key。
# --------------------------------------------------------------------------
OWNER_ADMIN = "gateway 管理者"
OWNER_LOCAL = "Agent_Hub_RAG_Frontend 的 .env"
NEEDS_NEW_KEY = {"auth", "quota", "no_model_access"}

# 探測結果快取秒數。/hub/agents 每次載入頁面、每次切 agent 都會呼叫，
# 沒有快取的話等於每個動作都對 gateway 多打一次 ping。
_PROBE_TTL = float(os.environ.get("LLM_PROBE_TTL", "60"))
# ★ 探測要用**短**逾時，不能共用 client() 的 TIMEOUT（240 秒）。
#   /hub/agents 是整個前端載入時的第一個請求；gateway 只是塞住（不是拒絕）
#   的時候，共用 240 秒等於整頁空白四分鐘，比直接說「LLM 不可用」還糟。
_PROBE_TIMEOUT = float(os.environ.get("LLM_PROBE_TIMEOUT", "10"))
_probe_cache: tuple[float, dict[str, Any]] | None = None


def _diag(kind: str, title: str, detail: str, fix: str, owner: str,
          raw: str = "", models: list[str] | None = None) -> dict[str, Any]:
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
        "models": models or [],
        "raw": raw,
    }


def models_for_key() -> list[str]:
    """這把 key 實際能用的模型。空清單是一個**真的答案**，不是查詢失敗：
    gateway 的 400 就是叫你來問這支，而它對授權被收走的 key 回的正是空清單。"""
    key = os.environ.get("LLM_API_KEY")
    if not key:
        return []
    try:
        import httpx

        r = httpx.get(BASE_URL.rstrip("/") + "/v1/models",
                      headers={"Authorization": f"Bearer {key}"}, timeout=10.0)
        r.raise_for_status()
        return [m["id"] for m in (r.json().get("data") or []) if m.get("id")]
    except Exception:
        return []


_MODEL_CACHE_TTL = 300.0
_model_cache: tuple[float, str] | None = None


def _effective_model() -> str:
    """實際要打的模型 id。

    預設就是 MODEL；只有在 gateway 的 /v1/models 清單裡**真的找不到** MODEL、
    但找得到 MODEL_FALLBACK 時才切過去 —— 這是「換模型當天新模型還沒上線」
    或「新模型被暫時下架」這類情況的安全網，不是每次呼叫都在兩顆之間選。
    查不到清單（空清單，通常是這把 key 沒有 /v1/models 權限或探測本身失敗）
    時**不猜**，照樣用 MODEL，讓真正的呼叫把錯誤原封不動回報出來。

    結果快取 5 分鐘，換模型/換 key 當下想立刻生效就重啟服務。
    """
    global _model_cache
    now = time.monotonic()
    if _model_cache and now - _model_cache[0] < _MODEL_CACHE_TTL:
        return _model_cache[1]
    chosen = MODEL
    if MODEL_FALLBACK and MODEL_FALLBACK != MODEL:
        avail = models_for_key()
        if avail and MODEL not in avail and MODEL_FALLBACK in avail:
            chosen = MODEL_FALLBACK
            print(f"[gateway] {MODEL} 不在 /v1/models 清單，改用 fallback {MODEL_FALLBACK}")
    _model_cache = (now, chosen)
    return chosen


def classify(err: BaseException | str) -> dict[str, Any]:
    """把一個錯誤翻成「發生什麼 / 該做什麼 / 誰能修 / 要不要換 key」。
    只做分類，不打網路（模型名那一類例外，它得問 /v1/models 才答得完整）。"""
    raw = err if isinstance(err, str) else f"{type(err).__name__}: {err}"
    low = raw.lower()
    code = getattr(err, "status_code", None) if isinstance(err, BaseException) else None
    key = os.environ.get("LLM_API_KEY") or ""

    if code == 401 or "invalid api key" in low or "authenticationerror" in low:
        return _diag("auth", "API key 無效或已被撤銷",
                     "gateway 回 401，目前設定的這把 key 連認證都過不了。",
                     "跟 gateway 管理者換一把新的 API key，填進 .env 的 "
                     "LLM_API_KEY 後重啟 Hub 與各 agent。", OWNER_ADMIN, raw)

    if code == 429 or "rate limit" in low or "too many requests" in low:
        return _diag("rate_limit", "呼叫太密集，被 gateway 限流",
                     "gateway 回 429。這是每分鐘的頻率上限，不是總量用完。",
                     "等幾十秒再送一次。★ 這種情況換新 key 沒有用。",
                     OWNER_ADMIN, raw)

    if any(w in low for w in ("budget", "quota", "exceeded", "insufficient", "credit")):
        return _diag("quota", "這把 key 的額度用完了",
                     "gateway 的訊息裡提到額度／預算已用盡 —— "
                     "這才是「用到一定次數就要換一把 key」的那種情況。",
                     "跟 gateway 管理者要一把新的 API key（或請他把額度加回來），"
                     "換進 .env 的 LLM_API_KEY 後重啟 Hub 與各 agent。",
                     OWNER_ADMIN, raw)

    if code == 404 or "unknown model" in low:
        avail = models_for_key()
        return _diag("unknown_model", f"gateway 不認識模型名稱 {_effective_model()}",
                     "gateway 回 404 unknown model —— 這個名字在它的模型表裡"
                     "根本不存在，多半是打錯字。",
                     ("先請 gateway 管理者給正確的模型名稱。" if not avail
                      else "把 Agent_Hub_RAG_Frontend/.env 的 LLM_MODEL 改成 " + "、".join(avail[:10])
                           + " 其中之一，換過的專案都會一起生效。"),
                     OWNER_LOCAL if avail else OWNER_ADMIN, raw, avail)

    if "invalid model name" in low or (code == 400 and "model" in low):
        avail = models_for_key()
        if avail:
            return _diag("wrong_model", f"這把 key 不能用 {_effective_model()}，但還有別的可以用",
                         "gateway 回 400 說這個模型名不被接受，"
                         "而 /v1/models 仍列得出其他模型。",
                         "把 Agent_Hub_RAG_Frontend/.env 的 LLM_MODEL 改成 " + "、".join(avail[:10])
                         + " 其中之一後重啟 Hub 與各 agent。", OWNER_LOCAL, raw, avail)
        return _diag("no_model_access", "這把 API key 名下沒有任何可用模型",
                     f"gateway 認得 {_effective_model()}（亂打的模型名會回 404，這裡回的是 400），"
                     "但 /v1/models 對這把 key 回的是空清單 —— key 還活著、"
                     "模型授權卻被收走了（額度到期，或管理者改了這把 key 的模型綁定）。",
                     "跟 gateway 管理者要一把新的 API key，或請他把這顆模型重新綁"
                     "回這把 key。★ 改 Agent_Hub_RAG_Frontend/.env 的 LLM_MODEL 沒有用，"
                     "這把 key 一顆模型都沒有。",
                     OWNER_ADMIN, raw)

    if any(w in low for w in ("connect", "timeout", "timed out", "unreachable")):
        return _diag("offline", "連不上 LLM gateway 或逾時",
                     f"打 {BASE_URL} 的時候連線失敗或逾時。",
                     "確認 gateway 服務有起來、且這台機器連得到那個網段。",
                     OWNER_ADMIN, raw)

    if "max_tokens" in low:
        return _diag("token_budget", "模型的思考佔滿了 token 預算",
                     f"這顆模型的 thinking 區塊算進 max_tokens，目前是 {MAX_TOKENS}。",
                     "調高 .env 的 A2A_MAX_TOKENS 後重啟服務。", OWNER_LOCAL, raw)

    return _diag("other", "LLM gateway 回了未分類的錯誤", raw or "（沒有錯誤訊息）",
                 "把下面的原始訊息貼給 gateway 管理者。", OWNER_ADMIN, raw)


def diagnose(*, fresh: bool = False) -> dict[str, Any]:
    """真的問一次 gateway，回結構化診斷。結果快取 _PROBE_TTL 秒。"""
    global _probe_cache

    if not os.environ.get("LLM_API_KEY"):
        return _diag("not_configured", "LLM 尚未設定", "LLM_API_KEY 未設定",
                     "在 Agent_Hub_RAG_Frontend 的 .env 補上 LLM_API_KEY 後重啟 Hub。", OWNER_LOCAL)
    try:
        import anthropic  # noqa: F401
    except Exception as e:
        return _diag("not_configured", "anthropic SDK 不可用", str(e),
                     "在 Hub 的 .venv 裝上 anthropic 套件。", OWNER_LOCAL)

    now = time.monotonic()
    if not fresh and _probe_cache and now - _probe_cache[0] < _PROBE_TTL:
        return _probe_cache[1]

    try:
        import anthropic

        # 16 個 token 的 ping。這顆模型一定會先產 thinking，所以 ping 也會
        # 撞到 max_tokens —— 那不算故障，能回應就代表 gateway 通了。
        # 用獨立的短逾時 client，不重試（見 _PROBE_TIMEOUT）。
        model = _effective_model()
        anthropic.Anthropic(
            base_url=BASE_URL, auth_token=os.environ["LLM_API_KEY"],
            timeout=_PROBE_TIMEOUT, max_retries=0,
        ).messages.create(
            model=model, max_tokens=16,
            messages=[{"role": "user", "content": "ping"}],
        )
        out = _diag("ok", "LLM 可用", f"{model} @ {BASE_URL}", "", "")
    except BaseException as e:  # noqa: BLE001 -- 狀態檢查不該把服務弄掛
        out = classify(e)
    _probe_cache = (now, out)
    return out


def available() -> tuple[bool, str]:
    """一行摘要版的 diagnose()，維持原本 (ok, 說明) 的回傳形狀。"""
    d = diagnose()
    if d["ok"]:
        return True, f"{_effective_model()} @ {BASE_URL}"
    return False, f"{d['title']}：{d['detail']}"


_client: Any = None


def client() -> Any:
    global _client
    if _client is None:
        import anthropic

        token = os.environ.get("LLM_API_KEY")
        if not token:
            raise RuntimeError("LLM_API_KEY 未設定")
        _client = anthropic.Anthropic(
            base_url=BASE_URL, auth_token=token, timeout=TIMEOUT, max_retries=1
        )
    return _client


def _assistant_content(blocks: list[Any]) -> list[dict[str, Any]]:
    """把回應的 content block 轉成可以再送回去的歷史。

    濾掉空的 text block（見模組 docstring 第 3 點）；thinking 保留，
    因為 tool use 續談時把 thinking 拿掉，模型會失去它剛剛的推理脈絡。
    """
    out: list[dict[str, Any]] = []
    for b in blocks:
        d = b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else dict(b)
        if d.get("type") == "text" and not (d.get("text") or "").strip():
            continue
        out.append(d)
    return out


def _strip_thinking(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把歷史裡的 thinking block 拿掉。

    只在 gateway 因為 thinking 的 signature 而回 400 時當備援用 —— 本地
    gateway 不見得會簽章 thinking，簽章不合就會被拒。寧可少了推理脈絡，
    也不要整條委派鏈掛掉。
    """
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = [b for b in c if b.get("type") != "thinking"] or [
                {"type": "text", "text": "（略）"}
            ]
        out.append({**m, "content": c})
    return out


# --------------------------------------------------------------------------
# 從「純文字」裡把工具呼叫救回來
#
# ★ 症狀（問「<料號> 的工程圖標題和材料表」這類問句時出現過）
#   使用者看到的答案整段是這個：
#     {"id": "call_…", "type": "function",
#      "function": {"name": "search_drawings", "arguments": {"query": "<料號>"}}}
#   注意這是 **OpenAI 的 function-call 格式**，不是 Anthropic 的 tool_use
#   —— gateway 後面接的是 OpenAI 相容的推論引擎，這顆模型偶爾會把工具呼叫
#   直接寫進 text block，沒有被轉成 tool_use content block。
#   改用非串流 create() 已經讓大部分情況正常（見上面第 6 點），但不是 100%。
#
#   不處理的話有兩層傷害：使用者看到一段 JSON 亂碼當答案，而那次工具呼叫
#   **根本沒有被執行**，等於白白燒掉一步。所以這裡把它解析出來、合成成正規
#   的 tool_use block 交回委派迴圈去執行，同時把那段 JSON 從要播放的文字裡
#   拿掉。救得回來就是一次正常的工具呼叫，救不回來至少畫面上不是亂碼。
# --------------------------------------------------------------------------

# 有些模型會用這些標記把工具呼叫包起來，清乾淨才不會在畫面上留殘骸。
_CALL_WRAPPERS = ("<tool_call>", "</tool_call>", "<tool_use>", "</tool_use>",
                  "```json", "```tool_code", "```")


def _iter_json_objects(text: str) -> Iterator[tuple[int, int, Any]]:
    """掃出文字裡每一段成對大括號的 JSON 物件，yield (起, 迄, 物件)。

    不能用正規表示式：參數本身就是巢狀 JSON，`\\{.*\\}` 會貪心到整段、
    非貪心又會在第一個內層 `}` 就斷掉。這裡照大括號深度掃，並且要略過
    字串內容（料號、標題裡出現 `{` `}` 不算層數）。
    """
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        j = i
        in_str = esc = False
        while j < n:
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if j < n and depth == 0:
            try:
                obj = json.loads(text[i:j + 1])
            except Exception:  # noqa: BLE001 -- 不是 JSON 就當普通文字
                obj = None
            if isinstance(obj, dict):
                yield i, j + 1, obj
                i = j + 1
                continue
        i += 1


def _as_tool_call(obj: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """把一個 dict 判讀成 (工具名, 參數)。不像工具呼叫就回 None。

    判準要嚴：答案裡本來就可能出現 JSON（例如模型引用工具回傳的內容），
    誤判會把使用者真正要看的東西吃掉。所以只認兩種形狀 ——
    明確標成 function 的（OpenAI 格式），或是鍵只有 name/arguments 那幾個的。
    """
    fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
    name = fn.get("name")
    if not isinstance(name, str) or not name.strip():
        return None

    args = fn.get("arguments")
    if args is None:
        args = fn.get("input")
    if args is None:
        args = fn.get("parameters")
    # OpenAI 慣例是把 arguments 再包成一層 JSON 字串
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:  # noqa: BLE001
            return None
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None

    if obj.get("type") == "function" or isinstance(obj.get("function"), dict):
        return name.strip(), args
    if set(obj) <= {"name", "arguments", "input", "parameters", "id", "type"}:
        return name.strip(), args
    return None


def extract_text_tool_calls(text: str) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    """回 (把工具呼叫 JSON 拿掉之後的文字, [(工具名, 參數), ...])。"""
    calls: list[tuple[str, dict[str, Any]]] = []
    spans: list[tuple[int, int]] = []
    for s, e, obj in _iter_json_objects(text):
        c = _as_tool_call(obj)
        if c:
            calls.append(c)
            spans.append((s, e))
    if not calls:
        markup_call = _as_markup_tool_call(text)
        return ("", [markup_call]) if markup_call else (text, [])

    keep, prev = [], 0
    for s, e in spans:
        keep.append(text[prev:s])
        prev = e
    keep.append(text[prev:])
    clean = "".join(keep)
    for w in _CALL_WRAPPERS:
        clean = clean.replace(w, "")
    return clean.strip(), calls


def _as_markup_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    r"""解析模型偶爾輸出的 Markdown/YAML 風格工具呼叫。

    例如 ``tool name：partrag\_\_search\_drawings`` 後面接 ``tool call``、
    ``type：function`` 與縮排的 ``arguments``。只有整段文字都符合這個結構
    才接受，避免把正常回答裡提到的工具名稱誤當成呼叫。
    """
    rows: list[str] = []
    for raw in html.unescape(text).splitlines():
        row = re.sub(r"\\([_\\])", r"\1", raw).rstrip().rstrip("\\").strip()
        if row:
            rows.append(row)
    if not rows:
        return None

    saw_call = saw_function = in_args = False
    outer_name = function_name = ""
    args: dict[str, Any] = {}
    for row in rows:
        bare = re.sub(r"[\s_-]+", " ", row.lower()).strip()
        if bare == "tool call":
            saw_call = True
            continue
        if bare in {"function", "arguments"}:
            in_args = bare == "arguments"
            continue
        if "：" in row:
            key, value = (part.strip() for part in row.split("：", 1))
        elif ":" in row:
            key, value = (part.strip() for part in row.split(":", 1))
        else:
            return None
        norm_key = re.sub(r"[\s-]+", "_", key.lower())
        if in_args:
            try:
                parsed = json.loads(value)
            except Exception:  # noqa: BLE001 -- 一般字串參數
                parsed = value
            args[norm_key] = parsed
        elif norm_key in {"tool_name", "tool"}:
            outer_name = value
        elif norm_key == "name":
            function_name = value
        elif norm_key == "type":
            saw_function = value.lower() == "function"
        elif norm_key == "id":
            continue
        else:
            return None

    name = function_name or outer_name
    if not (saw_call and saw_function and name and in_args):
        return None
    if outer_name and function_name and outer_name != function_name:
        return None
    return name, args


# --------------------------------------------------------------------------
# 把「被包成 JSON 信封的答案」拆開
#
# ★ 症狀（同一類工程圖問句）
#   工具呼叫的問題修掉之後，模型改成把**答案本身**包成 JSON 吐出來：
#     {"summary": "這是客戶…", "source": "<圖檔名>.pdf"}
#   內容其實是對的，但使用者看到的是一整片 \uXXXX 逃逸序列 —— 比沒有答案
#   更糟，因為它看起來像程式壞掉。
#
#   這跟工具呼叫那段是兩回事，不能共用 extract_text_tool_calls()：那邊要找的是
#   「有 name/arguments 的呼叫」，這裡是「有 summary/answer 的答案」。硬合成
#   一個判斷會讓兩邊的誤判互相污染（把答案當呼叫執行，或把呼叫當答案印出來）。
#
#   system 已經明講「不要輸出 JSON」，但這顆模型不是每次都聽。提示是預防，
#   這裡是保險：真的包了，至少拆得開。
# --------------------------------------------------------------------------

# 答案本文可能被放在哪個鍵。順序就是優先序。
_ANSWER_KEYS = ("summary", "answer", "text", "content", "response", "reply", "message", "result")
# 附註類的鍵，接在本文後面（不是答案主體，但丟掉可惜 —— source 是可稽核性）
_EXTRA_KEYS = (("source", "來源"), ("sources", "來源"), ("note", "註"),
               ("caveat", "註"), ("warning", "注意"))

_UNI_ESC = re.compile(r"\\u([0-9a-fA-F]{4})")


def _decode_unicode_escapes(s: str) -> str:
    """把殘留的 \\uXXXX 還原成字。

    只動格式完全正確的那種，其他一律不碰 —— 不能用
    s.encode().decode('unicode_escape')，那會把已經是中文的字元照 latin-1
    重新解讀，好好的字會變成亂碼。
    """
    return _UNI_ESC.sub(lambda m: chr(int(m.group(1), 16)), s)


def unwrap_json_answer(text: str) -> str:
    """答案被包成 JSON 信封時把本文拆出來；不是信封就原樣回。"""
    s = text.strip()
    if s.startswith("```"):
        s = s[3:].lstrip()
        if s[:4].lower() == "json":
            s = s[4:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    if not (s.startswith("{") and s.endswith("}")):
        # 沒包成 JSON，但可能還是有逃逸序列漏出來
        return _decode_unicode_escapes(text) if "\\u" in text else text

    try:
        obj = json.loads(s)
    except Exception:  # noqa: BLE001 -- 不是合法 JSON 就當普通文字
        return _decode_unicode_escapes(text) if "\\u" in text else text
    if not isinstance(obj, dict):
        return text

    body = next((obj[k].strip() for k in _ANSWER_KEYS
                 if isinstance(obj.get(k), str) and obj[k].strip()), None)
    if body is None:
        # 是 JSON，但找不到像答案本文的鍵 —— 不要亂猜，原樣留著讓人看得出
        # 模型回了什麼，總比我們挑錯一個欄位當答案好。
        #
        # ★ 但「原樣」不包含「連 \uXXXX 都不還原」。上面兩條 return 都有
        #   decode，只有這一條漏了，所以走到這裡的字會以逃逸序列見人。
        #   例如模型把工具結果整包貼出來當答案（"summary" 裝的是 list，
        #   型別不符被跳過），畫面上一整片 \uXXXX，看起來像亂碼但其實只是
        #   沒還原。別的 agent 隨時會用別的欄位名再撞一次，所以這裡一定要還原。
        return _decode_unicode_escapes(text) if "\\u" in text else text

    parts = [body]
    for key, label in _EXTRA_KEYS:
        v = obj.get(key)
        if isinstance(v, list):
            v = "、".join(str(x) for x in v if x)
        if isinstance(v, str) and v.strip() and v.strip() not in body:
            parts.append(f"{label}：{v.strip()}")
    return "\n".join(parts)


class _RecoveredToolUse:
    """從文字救回來的工具呼叫。介面對齊 anthropic 的 ToolUseBlock，
    這樣 delegate 的迴圈與 _assistant_content() 都不必知道它的來歷。"""

    type = "tool_use"

    def __init__(self, id: str, name: str, input: dict[str, Any]) -> None:  # noqa: A002
        self.id, self.name, self.input = id, name, input

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.input}


def _tool_use_block(block_id: str, name: str, args: dict[str, Any]) -> Any:
    try:
        from anthropic.types import ToolUseBlock

        return ToolUseBlock(id=block_id, name=name, input=args, type="tool_use")
    except Exception:  # noqa: BLE001 -- SDK 版本不合就用自己的替身
        return _RecoveredToolUse(block_id, name, args)


# 純為使用者體感的逐字播放效果切段；不影響正確性或工具判斷。
_PLAYBACK_CHUNK = 24


def stream_step(
    system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
) -> Iterator[tuple[str, Any]]:
    """跑一步 LLM。yield ("text", str) 逐字，最後 yield ("final", message)。

    任何錯誤都轉成 ("error", classify() 的診斷 dict) 而不往上拋：呼叫端
    （委派迴圈）需要把已經拿到的檢索結果留在畫面上，不能因為 gateway 掛掉
    整頁壞掉。

    ★ 這裡送出去的是 dict 不是字串。原本送的是 "LLM gateway 回應 400" ——
      前端照著印，使用者只知道有個 400，不知道是模型名錯、key 沒授權、
      還是額度用完，這三件事的處置完全不同。

    ★ 不用 SDK 的 messages.stream()：這顆本地模型在 gateway 的**串流**
      通道不會把工具呼叫組成 tool_use content block，而是把它當一段文字
      逐字吐出來——同一個請求用 messages.create()（非
      串流）正確回 tool_use block，換成 messages.stream() 卻在 text_stream
      裡看到完整的 `{"name": "...", "arguments": {...}}` 被當成答案播出去，
      使用者看到的是沒被執行的工具呼叫 JSON，答案是空的。
      改用非串流 create() 才能拿到真正的 content block（tool_use 判斷因此
      永遠正確），拿到的純文字答案再自己切段模擬「逐字」效果，使用者觀感
      不變，只是打字動畫從「真的即時」變成「生成完之後快速播放」——
      這條路本來就要等模型想完整輪才知道要不要呼叫工具，等待時間沒有變長。
    """
    import anthropic

    kwargs: dict[str, Any] = {
        "model": _effective_model(),
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools

    def _run(kw: dict[str, Any]) -> Iterator[tuple[str, Any]]:
        msg = client().messages.create(**kw)

        # ★ 先把「被寫成純文字的工具呼叫」救出來，再決定要播什麼。
        #   非串流拿的是完整內容，所以來得及在播放前處理 —— 這正是不能用
        #   messages.stream() 的另一個理由：串流時第一個字吐出去就收不回來了。
        texts: list[str] = []
        kept: list[Any] = []
        recovered: list[tuple[str, dict[str, Any]]] = []
        for b in msg.content:
            if getattr(b, "type", "") != "text":
                kept.append(b)
                continue
            clean, calls = extract_text_tool_calls(b.text or "")
            recovered.extend(calls)
            if not clean.strip():
                continue          # 整段都是工具呼叫 JSON，這個 block 直接丟掉
            # 剩下的才可能是答案 —— 被包成 JSON 信封的話拆開再播
            clean = unwrap_json_answer(clean)
            texts.append(clean)
            try:
                kept.append(b.model_copy(update={"text": clean}))
            except Exception:  # noqa: BLE001 -- 改不動就留原樣，別讓串流掛掉
                kept.append(b)

        if recovered:
            for i, (name, args) in enumerate(recovered):
                kept.append(_tool_use_block(f"txt_{getattr(msg, 'id', 'x')}_{i}", name, args))
            print(f"[gateway] 從文字救回 {len(recovered)} 個工具呼叫："
                  f"{', '.join(n for n, _ in recovered)}")
            try:
                msg.content = kept
            except Exception as e:  # noqa: BLE001
                print(f"[gateway] 救回的工具呼叫寫不回 msg.content：{type(e).__name__}: {e}")

        for text in texts:
            for i in range(0, len(text), _PLAYBACK_CHUNK):
                yield "text", text[i:i + _PLAYBACK_CHUNK]
        yield "final", msg

    try:
        yield from _run(kwargs)
    except anthropic.APIStatusError as e:
        if e.status_code == 400 and any(
            isinstance(m.get("content"), list) for m in messages
        ):
            # 見 _strip_thinking 的說明：多半是 thinking 送回去被拒
            try:
                yield from _run({**kwargs, "messages": _strip_thinking(messages)})
                return
            except Exception as e2:
                yield "error", classify(e2)
                return
        yield "error", classify(e)
    except anthropic.APIConnectionError as e:
        yield "error", _diag(
            "offline", "連不上 LLM gateway", f"打不到 {BASE_URL}。",
            "確認 gateway 服務有起來、且這台機器連得到那個網段。",
            OWNER_ADMIN, f"{type(e).__name__}: {e}",
        )
    except Exception as e:
        yield "error", classify(e)


def empty_answer_note(final: Any) -> str | None:
    """模型一個字都沒吐時，判斷是不是 thinking 把預算吃光了。

    這個症狀太容易被誤判成「服務壞了」，所以要明確講出來並給解法。
    """
    if getattr(final, "stop_reason", None) == "max_tokens":
        return (
            f"（模型的思考佔滿了 {MAX_TOKENS} 個 token 預算，沒能產出回答。"
            "資料仍在上方；調高 A2A_MAX_TOKENS 後重試。）"
        )
    return None
