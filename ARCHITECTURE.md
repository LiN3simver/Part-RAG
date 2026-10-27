# Part-RAG 架構

資料來源 → 入庫 → 儲存 → 檢索 → LLM → API。十一路檢索的權重與設計理由見 [README.md](README.md)。

```mermaid
flowchart TB

subgraph SRC["資料來源"]
    CSV["料號 CSV\nitem_number, name\n(一次性全量入庫)"]
    PLMCSV["PLM 料件主檔 CSV\n建立人員、成本、分類…\n每日自動匯出"]
    PLMDB[("PLM\nSQL Server")]
    PDFS["工程圖 PDF\n+ download_log.csv"]
end

subgraph PIPE["入庫與更新管線"]
    EXTRACT["extract.py\n正規化 + 屬性抽取\ningest / enrich / retrieval 共用"]
    TAXO["taxonomy.py\n產品分類 + 同義字\nPLM 欄位單一事實來源"]
    INGEST["ingest.py\n全量入庫 + BGE-M3\nbuild-then-swap 原子換表"]
    ENRICH["enrich.py\n匯入主檔（冪等）\n改品名 → 屬性與向量一起重算"]
    PLMEX["主檔匯出\nCSV 原子替換 + 筆數安全閥"]
    FETCH["圖檔 API\n候選料號 → 下載工程圖"]
    OCRPY["pdf_ocr.py\nRapidOCR + PDF 向量框線定欄"]
    DWG["drawing.py\nPDF → 標題欄 + 材料表\n文字層與 OCR 取聯集"]
    FRAMES["drawing_frames.py\n多種圖框版式的版面詞彙\n依圖上的字判斷，不靠檔名"]
    INGDWG["ingest_drawing.py\n批次入庫（可續跑、md5 去重）\n人工修正回寫"]

    CSV --> INGEST
    PLMDB --> PLMEX --> PLMCSV --> ENRICH
    PLMDB --> FETCH --> PDFS --> DWG
    OCRPY --- DWG
    FRAMES --- DWG
    DWG --> INGDWG
    EXTRACT --- INGEST
    EXTRACT --- ENRICH
    TAXO --- INGEST
    TAXO --- ENRICH
end

subgraph SCHED["每日排程"]
    SCHEDPY["schedule.py\n主檔線：匯出 → 匯入\n工程圖線：候選 → 下載 → OCR 入庫\n兩條線互不連累、advisory lock 防重疊"]
    SYNCST[("sync_state\n各 stage 的 last_run / last_success")]
    SCHEDPY --> SYNCST
    SCHEDPY -.-> PLMEX
    SCHEDPY -.-> ENRICH
    SCHEDPY -.-> FETCH
    SCHEDPY -.-> INGDWG
end

subgraph MANUAL["人工維運（刻意不進排程）"]
    OBS["obsolete.py\n作廢標記（soft delete）\n預設 dry-run、可還原"]
end

subgraph DB["PostgreSQL + pgvector"]
    PARTS[("parts\n名稱 / 屬性 / PLM 欄位\n+ embedding vector(1024)\n+ tsvector")]
    DRAW[("drawings / drawing_items\n標題欄、材料表、出處、可信度")]
    CONV[("conversations / messages")]
    EVAL[("lessons / eval_runs / eval_cases")]
end

INGEST --> PARTS
ENRICH --> PARTS
INGDWG --> DRAW
OBS --> PARTS

subgraph RETR["檢索層"]
    RETRIEVAL["retrieval.py\n十一路檢索 → RRF(k=60)\n→ 屬性加權 → top-k"]
    AGG["aggregate.py\n清單／統計旁路\n確定性 SQL + CSV 匯出"]
    VALUES["values.py\n值索引：DB 現有的人名、廠別…"]
    TIME["timerange.py\n自然語言時間 → 日期區間"]
    RETRIEVAL --> VALUES
    RETRIEVAL --> TIME
    AGG --> VALUES
    AGG --> TIME
end

PARTS --> RETRIEVAL
PARTS --> AGG
DRAW --> RETRIEVAL

subgraph APP["API 層"]
    MAIN["main.py\n/api/*（search / chat / export / part / drawing …）"]
    A2A["a2a.py\n/a2a/*，五支工具\n給 Agent Hub 與其他 Agent"]
    LLMPY["llm.py\nprompt 組裝 + gateway 客戶端\n+ 故障分類"]
    WRITE["write.py\n唯一的寫入路徑\n欄位白名單 + X-Write-Token"]
    CHAT["chat.py\n對話保存 + 追問主詞合併"]
    MAIN --> RETRIEVAL
    MAIN --> AGG
    MAIN --> LLMPY
    MAIN --> WRITE
    MAIN --> CHAT
    A2A --> RETRIEVAL
    A2A --> AGG
    A2A --> LLMPY
end

WRITE --> PARTS
CHAT --> CONV

subgraph GUARD["幻覺防治（回饋迴路）"]
    GROUND["grounding.py\n答案中的料號 / 檔名 / 規格 / 連結\n必須出現在送出的 prompt\n純字串比對"]
    TGUARD["toolguard.py\n工具模式的核對\n最多 3 次嘗試"]
    LESSON["lessons.py\n失敗 → 通則 → 下次的 prompt\n連續通過 3 輪退役"]
    EVALRUN["evalrun.py\n各路徑問法跑分"]
    LLMPY --> GROUND
    GROUND -->|違規| LLMPY
    A2A --> TGUARD --> GROUND
    GROUND --> LESSON
    LESSON -->|教訓放在 prompt 末段| LLMPY
    EVALRUN --> A2A
    EVALRUN --> LESSON
    LESSON --> EVAL
end

subgraph EXT["外部服務"]
    GATEWAY["LLM gateway\n(Anthropic Messages API 相容)"]
    BGE["BGE-M3 embedding\n本機 CPU（ONNX）"]
end

LLMPY <--> GATEWAY
RETRIEVAL <--> BGE
INGEST <--> BGE
ENRICH <--> BGE

FE["Agent Hub 統一前端"] <--> A2A
```

## 怎麼看這張圖

- **`parts` 是唯一的事實來源**：同一張表同時扛字面檢索、向量檢索與結構化篩選 —— 這不是一個「純向量資料庫」。
- **兩個資料來源走不同管線**：原始料號 CSV 走 `ingest.py`（全量、build-then-swap 不停機）；
  PLM 主檔走每日匯出 → `enrich.py`（增量、冪等）。
- **工程圖刻意不寫進 `parts`**：`drawings` / `drawing_items` 是獨立的表，工程圖通道查完後 join 回料號才進 RRF，
  既有通道的排名與向量都不受影響；`_wants_drawing()` 閘門讓沒提到工程圖的問題不啟動這兩路。
- **`aggregate.py` 是旁路**：「列出所有…／共有幾筆」完全繞過 RRF 與向量，直接下確定性 SQL。
- **寫入只有一條路**：`write.py` 被 API 與 CLI 共用，所以 API 不可能少做「改品名要重算向量」這一步；
  沒設 `WRITE_TOKEN` 時整組回 503（預設關閉，而不是預設開放）。
- **作廢刻意不進排程**：排程只負責把資料帶進來，不負責決定什麼該消失 ——
  「匯出檔漏了一半」與「一半真的作廢」在資料上長得一模一樣。
- **`GUARD` 是回饋迴路，不是旁路**：核對的基準是**送出去的 prompt 字面**而不是資料庫，
  兩條答題路徑（純摘要、工具迴圈）都要經過它；違規回饋給模型重寫，歸納出的通則由教訓庫帶到下一次。
- **LLM 只出現在最後一步**：`/api/chat` 先把檢索結果送給前端才呼叫 gateway，LLM 失效時檢索結果照常可用。
- **`main.py` 與 `a2a.py` 並列**：`/api/*` 是原生介面、`/a2a/*` 是多 Agent 契約，形狀不同但共用同一條檢索路徑。
