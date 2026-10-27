"""五個 Agent 的單一事實來源：id、埠、上下游關係。

★ 為什麼要有這個檔
  在此之前「誰跑在哪個埠」散落在四份 CLAUDE.md、兩支 run.bat 與前端的
  hard-coded fetch('/api/...') 裡。要讓 agent 互相呼叫，位址必須集中一處，
  否則每加一條串聯就要改四個地方。

埠是刻意錯開的，與各專案既有的 pgvector 容器埠一一對應，不要重新分配：

    agent_id   API 埠   DB 埠   專案（repo）
    partrag    8000     5434    Part-RAG
    rfq        8300     5435    RFQ-RAG
    material   8400     5436    material-agent
    bom        8500     —       Bom-RAG            （不落地，容器已作廢）
    (hub)      8600     —       Agent_Hub_RAG_Frontend
    car8d      8700     —       8D-Report-RAG      （不落地，見下）

★ car8d 沒有 DB 埠，這是刻意的，不是漏填。
  它的語料量很小，全表 LIKE 掃描就只要幾毫秒。
  落地成 pgvector 換不到速度，只會換到過期資料。與 bom 同一個決策。

★ 一律用 127.0.0.1，不要用 localhost。
  Windows 的 localhost 先解析到 IPv6 ::1，而 uvicorn 只綁 IPv4，每個請求
  要等 IPv6 逾時再回退，每筆多出約兩秒。
  agent 互打是機器對機器、一次對話可能打十幾次，這 2 秒會直接乘上去。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentRef:
    agent_id: str
    name: str
    role: str
    port: int
    project: str
    # 架構圖上的位置。upstream 會呼叫我，downstream 是我可以呼叫的人。
    # 委派迴圈只把 downstream 的工具掛給模型，避免 A->B->A 繞回來。
    downstream: tuple[str, ...] = ()
    upstream: tuple[str, ...] = ()
    # 本專案原有的前端路徑（統一 UI 之外，仍然可以單獨開）。None = 沒有原生介面，
    # 純 A2A 服務 —— 四個 agent 的原生介面都已移除，統一走 Agent Hub 的前端。
    native_ui: str | None = None
    # ★ tags 不只是畫面上的標籤，它是**有功能的欄位**。
    #   delegate._prefetch_peers() 用它判斷「這個問句是不是落在某個下游的主題上」，
    #   命中就先替模型去打那個下游的 /a2a/search，把資料塞進問句裡
    #   （理由見 Agent_Hub_RAG_Frontend/README.md 的坑第 11 條：模型在「自己碰巧有這個料號」
    #   的時候一個工具都不會呼叫，而那件事靠提示詞解不掉）。
    #   所以 tags 要寫**使用者真的會打出來的詞**，不是內部術語 ——
    #   寫錯的後果是 prefetch 不會觸發，而且不會有任何錯誤訊息。
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def base_url(self) -> str:
        # 允許用環境變數改寫，方便日後搬到別台機器而不用改程式
        env = os.environ.get(f"A2A_{self.agent_id.upper()}_URL")
        return env or f"http://127.0.0.1:{self.port}"


AGENTS: tuple[AgentRef, ...] = (
    AgentRef(
        agent_id="bom",
        name="Bom Agent",
        role="依規格建構新產品 BOM：決定要哪些槽位、各多少用量",
        port=8500,
        project="Bom-RAG",
        # car8d：換料/改版之前要知道這顆料有沒有客訴史（責任在誰、根本原因是什麼）
        downstream=("material", "partrag", "car8d"),
        tags=("BOM", "槽位", "用量"),
    ),
    AgentRef(
        agent_id="material",
        name="材料建議 Agent",
        role="每個 BOM 槽位放哪一個具體料號（元件層規格比對）",
        port=8400,
        project="material-agent",
        upstream=("bom",),
        tags=("選料", "元件", "替代料"),
    ),
    AgentRef(
        agent_id="partrag",
        name="相似搜尋建議 Agent",
        role="成品線材語料的相似料號檢索",
        port=8000,
        project="Part-RAG",
        # rfq 也直接掛上來，不再只經 bom 轉問（理由見 rfq 的 downstream 註解）。
        # 這個 agent 是葉節點（downstream 空、can_delegate=False），
        # 多一個上游不會產生任何環。
        upstream=("bom", "rfq"),
        # ★ tags 不只是給人看的分類，它決定 delegate._prefetch_peers() 會不會
        #   確定性地先替上游把資料撈回來（命中就打 /a2a/search，不經 LLM）。
        #   所以這裡放什麼字要照「上游會怎麼問」來挑，不是照自我介紹來挑。
        #
        #   「工程圖」是補上的：工程圖只有這個 agent 有（search_drawings），
        #   但問句寫「還有工程圖嗎？」時，partrag 與 bom 的 tags 一個都不中，
        #   於是完全沒有 prefetch，能不能拿到全看模型有沒有想到要呼叫工具
        #   （同一句話可能一次問到、一次沒問到）。/a2a/search 不經 LLM、
        #   不到一秒就回結果，連「工程圖收錄範圍有限」的限制都自己帶著。
        #
        #   「線材」是**刻意拿掉**的：非 ASCII 的 tag 走純子字串比對
        #   （見 _topic_hit），而上游 rfq 是**線材**估價 agent，幾乎每一句
        #   問話都含這兩個字 —— 留著等於每題都多打一次 search 並往問句塞
        #   最多 3000 字，把 thinking 的預算擠掉。相似料號的情境已經由
        #   「相似料號」這個 tag 蓋住了。
        tags=("相似料號", "工程圖", "PLM 欄位"),
    ),
    AgentRef(
        agent_id="rfq",
        name="產品估價 Agent",
        role="RFQ/S/D 歷史紀錄查詢與成本區間參考",
        port=8300,
        project="RFQ-RAG",
        # car8d：報價前的品質風險 —— 這顆料/這個客戶過去出過幾次事、成立了幾次
        # partrag：工程圖與成品線材相似料號。**刻意不只靠 bom 轉問**——
        #   兩段式問句（「…的 bom 有什麼？還有工程圖嗎？」）經 bom 轉問一趟要
        #   跑兩層 LLM 迴圈，慢一個數量級，而且模型可能把步數全花在第一段、
        #   根本沒問工程圖，最後寫出沒查證過的否定句（guard 抓不到，
        #   否定句裡沒有可比對的編號或數字）。
        #   直接掛上來之後 partrag 進了 downstream，「工程圖」那個 tag 才會觸發
        #   prefetch —— 資料在模型開口之前就已經在問句裡，它不需要「想到」
        #   （見 delegate._prefetch_peers）。
        downstream=("bom", "car8d", "partrag"),
        tags=("RFQ", "歷史成本", "合理性檢核"),
    ),
    AgentRef(
        agent_id="car8d",
        name="8D 客訴單 Agent",
        role="客訴抱怨單(8D) 查詢：單號/料號/客戶反查、客訴是否成立、責任歸屬、D3~D8 對策",
        port=8700,
        project="8D-Report-RAG",
        # ★ downstream 刻意留空 —— 這是葉節點。
        #   delegate.py 的防環做法就是「圖是單向的」：car8d 沒有下游，
        #   所以不可能出現 rfq -> car8d -> rfq 或 bom -> car8d -> bom。
        #   它的資料全部來自 PLM，不需要問任何人。
        downstream=(),
        upstream=("bom", "rfq"),
        tags=("客訴", "8D", "CAR", "品質異常", "根本原因"),
    ),
)

BY_ID: dict[str, AgentRef] = {a.agent_id: a for a in AGENTS}

HUB_PORT = int(os.environ.get("A2A_HUB_PORT", "8600"))


def get(agent_id: str) -> AgentRef:
    if agent_id not in BY_ID:
        raise KeyError(f"未知的 agent_id：{agent_id}（可用：{', '.join(BY_ID)}）")
    return BY_ID[agent_id]


def peers_of(agent_id: str) -> tuple[AgentRef, ...]:
    """某個 agent 可以呼叫的下游。"""
    return tuple(BY_ID[p] for p in get(agent_id).downstream if p in BY_ID)
