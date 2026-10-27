"""Hybrid retrieval over partrag.parts.

Eleven candidate channels (seven over partrag.parts, four over the drawing
tables), fused with Reciprocal Rank Fusion, then re-scored
against attributes and PLM vocabulary parsed out of the query.

Why not pure vector search: a large share of rows live in groups whose names
are identical once digits are masked. Embeddings place those on top of each
other; only lexical + numeric matching separates them.

Why no cross-encoder reranker: a cross-encoder on a CPU-only box costs seconds
per query. Deterministic attribute scoring is free and is a better
discriminator for the actual failure mode (L=270mm vs L=760mm).

Why a `plm` generator: the words engineers type come off the PLM 零組件 form
(產品小類 "Wire Harness cable", 環保標準 GC, 料件類型 製造件) and appear nowhere in
the CSV's free-text description. src/taxonomy.py resolves those to columns and
this generator searches by them. Without it, "Wire Harness cable" fell through
to trigram similarity and returned unrelated rows.

Why a `neighbor` generator: a well-formed part number the corpus does not hold
used to produce four channels of noise. It now falls back to the number's real
neighbours inside its own block.

Why a `drawing` generator: the engineering drawing's title block and BOM say
things the CSV never does -- the customer's own product name (<客戶品名>) and
the VENDOR part numbers of the terminals and housings inside the assembly
("<廠商> P/N:ABC-0001-TPE"). "which of our parts use the ABC-0001 terminal" is
unanswerable from partrag.parts alone. It is a separate channel over separate
tables: parts.search_text and parts.embedding are deliberately untouched, so
the seven parts-table channels (of the 11 fused) rank exactly as they did
before.
"""
from __future__ import annotations

import os
import re
import time
from functools import lru_cache
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from psycopg.rows import dict_row

from . import drawing_frames
from . import taxonomy as T
from . import timerange
from . import values as V
from .db import PG_DSN, pool  # noqa: F401  -- re-exported; chat.py imports pool from here
from .extract import CJK, cjk_bigrams, parse_query, space_cjk_ascii

load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
MAX_SEQ = int(os.getenv("EMBED_MAX_SEQ", "256"))

# ★ 為什麼要釘 revision（與 MAX_SEQ 是同一條規則的另一面）
#   不指定的話每次都拉 HuggingFace 的 main。上游一改，新權重會算出**不同的
#   向量**，而庫裡的向量是舊版算的 —— 兩邊於是在不同的向量空間裡比對。
#   症狀與 MAX_SEQ 不一致時完全相同：不會報錯，只會讓排名慢慢變差，而且沒有
#   任何一個畫面看得出來。
#
#   第二個症狀在磁碟上：HF 快取不刪舊版，每多一個 revision 就多一份完整權重
#   （數 GB），刪掉之後只要跑一次未鎖版的載入就又會被抓回來。
#
#   下面這個值就是庫裡向量對應的 commit。換 embedding 模型、或者
#   重跑一次全量 `python -m src.ingest` 之後才需要動它，平常不該變。
#   三個模組（retrieval / enrich / ingest）必須用同一個值，理由同 MAX_SEQ。
EMBED_REVISION = os.getenv(
    "EMBED_REVISION", "5617a9f61b028005a4858fdac845db406aefb181"
)

# ONNX 推論的 intra-op 執行緒數。預設維持「全部邏輯核心」，也就是這個變數
# 出現之前的行為 —— 不設它，跑起來跟以前一模一樣。
#
# ★ 為什麼需要這個旋鈕
#   一次推論預設就吃滿全部核心。單一請求最快，但兩個請求同時進來就是兩份
#   工作各自要全部核心去搶同一批核心，兩邊都變慢 —— 壓測時高併發下吞吐反而
#   **下降**，主因在此。
#   調小會犧牲單一請求的延遲換併發吞吐，是取捨不是純贏，所以預設不動，
#   量過幾種設定之後再決定要不要在 compose 裡設值。
#   （RFQ-RAG 的 src/embed.py 用同一個變數名，兩個專案的旋鈕面一致。）
EMBED_THREADS = int(os.getenv("EMBED_THREADS", "0")) or (os.cpu_count() or 4)

RRF_K = 60
CAND = 60  # candidates per generator


class _OnnxEmbedder:
    """BGE-M3 的 ONNX 版本，介面與 SentenceTransformer 相容的那一小塊。

    只實作四個呼叫端真的用到的東西：`max_seq_length` 與 `encode()` 的
    normalize_embeddings / convert_to_numpy / batch_size / show_progress_bar。
    刻意不做成通用包裝 —— 這裡要的是「換掉底層、上面四支不用改」，多做的
    每一個參數都是一份沒有人驗證過的行為。
    """

    def __init__(self, session, tokenizer, max_seq_length: int):
        self._sess = session
        self._tok = tokenizer
        self._outputs = {o.name for o in session.get_outputs()}
        self.max_seq_length = max_seq_length

    @property
    def max_seq_length(self) -> int:
        return self._max_seq

    @max_seq_length.setter
    def max_seq_length(self, n: int) -> None:
        # 截斷長度是正確性的一部分（見 EMBED_MAX_SEQ 的說明），所以設定它就
        # 要真的傳到 tokenizer，不能只是存一個沒有人讀的屬性。
        self._max_seq = int(n)
        self._tok.enable_truncation(max_length=self._max_seq)

    def encode(self, sentences, *, batch_size: int = 32,
               normalize_embeddings: bool = True, convert_to_numpy: bool = True,
               show_progress_bar: bool = False, **_ignored):
        import numpy as np

        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)

        rng = range(0, len(texts), batch_size)
        if show_progress_bar and len(texts) > batch_size:
            try:
                from tqdm import tqdm
                rng = tqdm(rng, desc="embed", unit="batch")
            except ImportError:      # tqdm 是批次工具才有的相依，服務端沒有
                pass

        out = []
        for i in rng:
            encs = self._tok.encode_batch(texts[i:i + batch_size])
            feed = {
                "input_ids": np.array([e.ids for e in encs], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encs],
                                           dtype=np.int64),
            }
            if normalize_embeddings and "sentence_embedding" in self._outputs:
                # 這個輸出已經做完 CLS pooling 與 L2 normalize（實測 norm=1.0），
                # 與 modules.json 的 Transformer -> Pooling -> Normalize 一致。
                out.append(self._sess.run(["sentence_embedding"], feed)[0])
            else:
                # normalize_embeddings=False 這條路目前沒有呼叫端，但留著才不會
                # 讓「傳了 False 卻拿到 normalize 過的向量」變成一個沉默的謊。
                tok_emb = self._sess.run(["token_embeddings"], feed)[0]
                v = tok_emb[:, 0]                      # CLS pooling
                if normalize_embeddings:
                    v = v / np.linalg.norm(v, axis=1, keepdims=True)
                out.append(v)

        vecs = np.vstack(out).astype(np.float32)
        if single:
            return vecs[0]
        return vecs if convert_to_numpy else list(vecs)


def embed_device() -> str:
    """這台機器該用哪個 execution provider（給 ingest.py 決定 device 用）。"""
    import onnxruntime as ort

    return "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"


def load_embedder(device: str = "cpu"):
    """載入 embedding 模型，鎖定 EMBED_REVISION。

    ★ 為什麼是 ONNX 而不是 sentence-transformers + torch
      兩者算出來的向量逐元素誤差在 1e-7 量級，評估集的 top-5 排名與 torch 版
      一致（極少數差異屬於這個系統本來就有的排序抖動，不是 ONNX 造成的）。
      換來的是：api image 體積、常駐記憶體與模型載入時間都大幅下降。載入那項
      對這個專案特別有意義，main.py 那整套 WARMUP 背景執行緒與 Dockerfile 的
      start-period=60s 都是為了它而存在。
      推論本身沒有變快：檢索時間主要花在十一條通道的 SQL，embedding 只佔一小塊。

    ★ 光是傳 revision= 擋不住 PR 權重（torch 那條路的教訓）
      transformers 5.x 偏好 safetensors，而 EMBED_REVISION 那個 commit 只有
      pytorch_model.bin。找不到時它會**自己去抓 HuggingFace 上的自動轉檔 PR**
      （`resolve/refs%2Fpr%2F<n>/model.safetensors`），另外下載一份完整權重
      並改用那一份；傳 use_safetensors=False 也擋不住。
      ONNX 這條路不經過 transformers，所以那個行為從根本上不存在 —— 下面
      `allow_patterns=["onnx/*"]` 明確指定要哪幾個檔案，不會有別的東西被拉進來。

    ★ 「先試本機、失敗才下載」
      寫死 local_files_only=True 會讓第一次部署直接炸（容器裡還沒有模型），
      做成環境變數則要靠人記得設，而忘了設的症狀是安靜地多下載一份。
      先試鎖定、真的沒有才連線：常態走第一條，第一次部署走第二條，兩種
      情況都不必有人記得做什麼。
    """
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    kw = {"repo_id": EMBED_MODEL, "revision": EMBED_REVISION,
          "allow_patterns": ["onnx/*"]}
    try:
        root = snapshot_download(local_files_only=True, **kw)
    except Exception as e:  # noqa: BLE001 -- 本機沒有這個 revision，只能連線抓
        print(f"[embed] 本機快取沒有 {EMBED_MODEL}@{EMBED_REVISION[:8]} 的 ONNX"
              f"（{type(e).__name__}），改為從 HuggingFace 下載")
        root = snapshot_download(**kw)

    d = Path(root) / "onnx"
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    # pad token 從 tokenizer 自己查，不寫死 id：換模型時寫死的那個會安靜地
    # 把某個真實 token 當成 padding，而 attention_mask 仍然是對的，所以只會
    # 表現成「向量有點怪」。
    pad = "<pad>"
    tok.enable_padding(pad_id=tok.token_to_id(pad), pad_token=pad)

    opts = ort.SessionOptions()
    if device == "cpu":
        opts.intra_op_num_threads = EMBED_THREADS
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if device == "cuda" else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(str(d / "model.onnx"), sess_options=opts,
                                providers=providers)
    return _OnnxEmbedder(sess, tok, MAX_SEQ)


_model_lock = threading.Lock()
_model_loading = threading.Event()


@lru_cache(maxsize=1)
def _cached_model():
    m = load_embedder("cpu")
    m.max_seq_length = MAX_SEQ
    return m


def _model():
    # functools caches results, but does not prevent concurrent initial loads.
    with _model_lock:
        _model_loading.set()
        try:
            return _cached_model()
        finally:
            _model_loading.clear()


def embedding_ready() -> bool:
    return _cached_model.cache_info().currsize > 0


# ★ ingest.py 與 enrich.py 也從這裡拿模型（from .retrieval import load_embedder）。
#   三處各寫一份載入邏輯就會重演 EMBED_MAX_SEQ 那個「這裡寫 96、另兩處寫 256」
#   的分岔，而載入邏輯分岔的症狀是「產生語料向量的那一端與查詢端用了不同的權重」，
#   比截斷長度不一致更難看出來。


@lru_cache(maxsize=512)
def embed_query(q: str):
    return _model().encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]


SELECT_COLS = """
    item_number, name, name_norm, base_pn, variant, prefix4, category,
    is_placeholder, is_obsolete, obsoleted_on, obsolete_reason,
    length_mm, pin_counts, pin_layout, pins_total, pins_row,
    pitch_series, conn_types, awg, gen, pitch_mm, speed_g, color,
    customer_pn, drawing_no, conn_a, conn_b, has_connector, is_coaxial,
    rated_voltage, prod_l1, prod_l2, prod_l3, tax_source, item_type, eco_std,
    plant, announce_cat, lot_control, item_class, item_subclass, unit,
    shelf_days, lot_size, std_seconds, mat_cost_usd,
    outsource_cost_usd, labor_cost_usd, mfg_cost_usd, lifecycle, plm_synced,
    created_by, created_on, modified_by, modified_on, locked_by,
    major_rev, generation, state, release_date, effective_date
"""

_NUM_COLS = (
    "length_mm", "lot_size", "std_seconds", "pitch_mm",
    "mat_cost_usd", "outsource_cost_usd", "labor_cost_usd", "mfg_cost_usd",
)


def _jsonable(row: dict) -> dict:
    """psycopg returns Decimal for numeric / numeric[]; json.dumps rejects it."""
    out = {k: v for k, v in row.items() if k != "name_norm"}
    for c in _NUM_COLS:
        v = out.get(c)
        if isinstance(v, list):
            out[c] = [float(x) for x in v]
        elif v is not None:
            out[c] = float(v)
    return out


# --------------------------------------------------------------------------
# candidate generators
# --------------------------------------------------------------------------
def _filter_sql(f: dict) -> tuple[str, list]:
    """Build a WHERE fragment for UI filters.

    These MUST be pushed into SQL rather than applied to the fused result set.
    Post-filtering asks "which of the top-60 for this query happen to be
    AWG 30?", which is usually none; pre-filtering asks the intended question,
    "of the AWG 30 parts, which best match this query?".
    """
    clauses: list[str] = []
    params: list = []
    if not f.get("include_placeholder"):
        clauses.append("NOT is_placeholder")
    # 作廢列預設不進「靠相似度找」的通道，理由與佔號列相同：問「600mm 的 MCIO 線
    # 有哪些」時，一顆 PLM 已經作廢的料號不是答案。但**料號精確比對與鄰域回退
    # 刻意不套這段**（見 _by_item_number / _by_neighborhood）—— 打了完整料號就是
    # 要那一筆，回「查無此料號」會讓人以為打錯字，而正確答案是
    # 「這顆存在，但已作廢」。那兩條路回傳的列帶著 is_obsolete，由 llm.py 講出來。
    if not f.get("include_obsolete"):
        clauses.append("NOT is_obsolete")
    if f.get("category"):
        clauses.append("category = ANY(%s)")
        params.append(list(f["category"]))
    # The PLM form's own fields, as first-class filters.
    for key in ("prod_l1", "prod_l2", "prod_l3"):
        if f.get(key):
            clauses.append(f"{key} = ANY(%s)")
            params.append(list(f[key]))
    for key in ("awg", "gen", "item_type", "eco_std", "plant", "announce_cat", "state"):
        if f.get(key):
            clauses.append(f"{key} = %s")
            params.append(f[key])
    # 建立人員/修改人員 are matched by substring, not equality: the stored value is
    # "<帳號> <姓名>" and the UI (or a query) supplies only a fragment. parts_*_trgm
    # keeps that indexed.
    for key in ("created_by", "modified_by"):
        if f.get(key):
            clauses.append(f"{key} ILIKE %s")
            params.append(f"%{f[key]}%")
    for key, op in (("created_after", ">="), ("created_before", "<")):
        if f.get(key):
            clauses.append(f"created_on {op} %s")
            params.append(f[key])
    if f.get("pins"):
        clauses.append("%s = ANY(pin_counts)")
        params.append(f["pins"])
    if f.get("pitch"):
        clauses.append("%s = ANY(pitch_series)")
        params.append(f["pitch"])
    if f.get("conn_type"):
        clauses.append("%s = ANY(conn_types)")
        params.append(f["conn_type"])
    lo, hi = f.get("len_min"), f.get("len_max")
    if lo is not None or hi is not None:
        sub = ["true"]
        if lo is not None:
            sub.append("v >= %s")
            params.append(lo)
        if hi is not None:
            sub.append("v <= %s")
            params.append(hi)
        clauses.append(
            "EXISTS (SELECT 1 FROM unnest(length_mm) v WHERE " + " AND ".join(sub) + ")"
        )
    return (" AND " + " AND ".join(clauses) if clauses else ""), params


def _by_item_number(cur, codes: list[str]) -> tuple[list[dict], set[str]]:
    """Exact + prefix match on item_number. -> (rows, codes that actually hit)

    Deliberately NOT filtered: an explicit part number is an explicit request,
    and placeholder stubs are a legitimate answer to "what is X?".

    The hit set must be derived from the SAME three-way condition as the query,
    not from `item_number` alone. A base number matches several real rows through
    `base_pn`, yet equals none of their item_numbers -- computing "missing" by
    item_number then reported 查無料號 for a number whose variants were sitting
    in the result list.
    """
    if not codes:
        return [], set()
    cur.execute(
        f"""SELECT {SELECT_COLS} FROM partrag.parts
            WHERE item_number = ANY(%s)
               OR base_pn     = ANY(%s)
               OR item_number LIKE ANY(%s)
            LIMIT %s""",
        (codes, codes, [c + "%" for c in codes], CAND),
    )
    rows = cur.fetchall()
    hit = {
        c
        for c in codes
        for r in rows
        if r["item_number"] == c or r["base_pn"] == c or r["item_number"].startswith(c)
    }
    return rows, hit


def _by_exact_code(cur, codes: list[str]) -> list[dict]:
    """codes 裡**完全等於**某個 item_number 的那幾筆。

    ★ 為什麼 _by_item_number 不夠
      extract 的料號 pattern 認的是廠內的編碼規則（字母區 + 數字區 + 變體碼）。
      語料裡還有客戶自己格式的料號（<客戶料號>，例如 `XX-ABC-0001-HF`）。
      它們打完整料號進來也不會被放進 q["item_numbers"]，所以整條精確通道
      不啟動，只剩字面與向量去撈：實測兩者都撈不到（tsvector 把這種純編碼字串
      切成一堆無意義 token，向量對它也沒有語意可言）。結果是**打了完整料號、
      料號就在庫裡、卻查不到**。

    ★ 為什麼只做完全相等，不做前綴也不做 base_pn
      _by_item_number 的三路比對（相等／base_pn／前綴）是為認證過的料號設計的。
      codes 撈進來的字串雜得多，一個短前綴會被放大成 60 筆候選，把既有題目的
      排名整個推開。完全相等不會有這個問題：不在庫裡的 code 就是命中 0 筆。

    ★ 刻意不套 filters
      與 _by_item_number 同一個理由，而且這正是「作廢料號點名仍查得到」的依據
      —— 作廢列裡也有客戶格式的料號。

    ★ 加一條「連字號邊界的後綴」比對，理由與 drawing_code_hits() 相同
      客戶料號進到廠內會被包上一層前綴：使用者打的 `ABC-0001-01` 在 parts 裡
      叫 `AA-BB-ABC-0001-01`。完全相等對不上、前綴也對不上（前綴反了），
      於是打了一個**貼在圖上、也在語料裡**的編號卻查無此料。
      用 `right(x, n+1) = '-' || code` 而不是 `LIKE '%' || code`：
      後者會讓 `ABC-0001-01` 一併命中 `AA-BB-ABC-0001-01-HF`（那是另一顆
      料號的另一份圖），也不必處理 code 裡的 `_`/`%` 轉義。
      長度下限 6 是為了不讓短 code（`ABC1`、`2026-05`）去撞後綴。
    """
    if not codes:
        return []
    long = [c for c in codes if len(c) >= 6]
    cur.execute(
        f"""SELECT {SELECT_COLS} FROM partrag.parts
            WHERE item_number = ANY(%s)
               OR EXISTS (SELECT 1 FROM unnest(%s::text[]) c
                          WHERE right(upper(item_number), length(c) + 1)
                                = '-' || upper(c))
            LIMIT %s""",
        (codes, long, CAND),
    )
    return cur.fetchall()


# A part number is a block plus a 3-digit sequence inside it:
# Z0000XX00|123 (from Z0000XX00123-1). Splitting there is what lets a miss fall back to
# its numeric neighbours instead of to unrelated trigram noise.
_SEQ_RE = re.compile(r"^(.*?)(\d{3})$")


def _by_neighborhood(cur, missing: list[str], limit: int = 12) -> list[dict]:
    """Nearest existing part numbers for a well-formed number we do not hold.

    A number created in PLM after the last CSV dump is absent from the corpus.
    Such a query used to fuse four channels of semantic noise (an unrelated top
    hit) and the answer layer could only say "not found". It now returns the
    block's real neighbours, which are the same product family by construction.
    """
    if not missing:
        return []
    out: list[dict] = []
    for pn in missing:
        base = pn.split("-")[0]
        m = _SEQ_RE.match(base)
        if not m:
            continue
        stem, seq = m.group(1), int(m.group(2))
        pos = len(stem) + 1
        cur.execute(
            f"""SELECT {SELECT_COLS},
                       abs(substring(base_pn from %s for 3)::int - %s) AS dist
                FROM partrag.parts
                WHERE base_pn LIKE %s
                  AND length(base_pn) = %s
                  AND substring(base_pn from %s for 3) ~ '^[0-9]{{3}}$'
                  AND NOT is_placeholder
                ORDER BY dist, item_number
                LIMIT %s""",
            (pos, seq, stem + "%", len(base), pos, limit),
        )
        out.extend(cur.fetchall())
    return out


def _by_customer_code(cur, codes: list[str], fsql: str, fp: list) -> list[dict]:
    """Reverse lookup on customer P/N and drawing number."""
    if not codes:
        return []
    cur.execute(
        f"""SELECT {SELECT_COLS} FROM partrag.parts
            WHERE (customer_pn && %s OR drawing_no && %s) {fsql}
            LIMIT %s""",
        (codes, codes, *fp, CAND),
    )
    return cur.fetchall()


def _by_vector(cur, qvec, fsql: str, fp: list) -> list[dict]:
    # ★ `, item_number` 是正確性的一部分，不是排版
    #   分數打平時 PostgreSQL 不保證回傳順序（掃描方式、平行度、頁面快取狀態
    #   都會影響），所以少了這個決勝欄位，**同一個問題連問兩次會拿到不同的
    #   前 5 名**。這條通道每次只取 CAND 筆，打平的那幾筆誰進得了這 60 名
    #   於是也跟著飄，一路飄到融合結果上。
    #   實測程式碼原封不動連跑數次，少數題目的 top-5 集合或順序會變。
    #   使用者看到的是「昨天問明明是這幾筆」。
    #   _by_neighborhood 與 _by_plm 一直都有補這個欄位，這裡、_by_lexical 與
    #   _by_drawing_confident 三處是漏掉的。
    cur.execute(
        f"""SELECT {SELECT_COLS}, 1 - (embedding <=> %s) AS vscore
            FROM partrag.parts
            WHERE embedding IS NOT NULL {fsql}
            ORDER BY embedding <=> %s, item_number
            LIMIT %s""",
        (qvec, *fp, qvec, CAND),
    )
    return cur.fetchall()


def _split_terms(text: str) -> list[str]:
    """一段文字切成 tsquery 用的詞。

    ★ 中文一定要切成 **bigram**，因為語料端存的就是 bigram
      `extract.build_search_text()` 把每一段中文拆成字元二元組寫進 search_text
      （PostgreSQL 沒有中文斷詞器，pgvector 這顆 image 也沒裝 zhparser）。
      查詢端沒有做同一件事，於是兩邊永遠對不上 —— 而且**失敗是安靜的**：
      切割用的字元類是「不是 \\w 也不是中文」，但 Python 的 \\w 含中文，
      所以一句純中文問話切出來是**一個 token：整句話**。

      例如「找有香蕉头端子的工程圖」：tsq 是 `找有香蕉头端子的工程图 or CONN
      or CONNECTOR or ...`，前面那個 token 一列都打不到，於是整條字面通道
      實際上只在拿「連接器」的同義詞去撈 —— 回來的是與香蕉頭端子毫無關係的
      連接器圖，而 `香蕉头端子` 明明就寫在某些圖的材料表上。

      兩字詞（線束）以前是**碰巧**會過的：長度 2 的中文串，bigram 就是它自己。
      三個字以上一律無聲失敗，而料號品名以英數為主，所以這個洞一直沒被看見。
    """
    # ★ 先補中英之間的邊界再切
    #   切割用的字元類「不是 \w 也不是中文」對混寫的 token 完全無效，因為
    #   Python 的 \w 含中文 ——「有LABEL WHITE的工程图」切出來是
    #   `有LABEL` 與 `WHITE的工程图` 兩個混合 token，兩個都打不中任何一列，
    #   於是 LABEL 與 WHITE **從來沒有變成獨立的查詢詞**。
    #   於是「幫我找某月有LABEL WHITE的工程圖」這種問法會回 0 份。
    out: list[str] = []
    for tok in re.split(r"[^\w一-鿿.*/+-]+", space_cjk_ascii(text)):
        if not tok:
            continue
        # 中英混寫的 token（`28AWG`、`PH4.2`）照原樣留著；純中文的切 bigram。
        if re.fullmatch(f"[{CJK}]+", tok):
            out.extend(cjk_bigrams(tok) if len(tok) > 2 else [tok])
        else:
            out.append(tok)
            # 「铜箔MYLAR」這種黏在一起的：中文那半也要能單獨命中。
            for run in re.findall(f"[{CJK}]{{2,}}", tok):
                out.extend(cjk_bigrams(run) if len(run) > 2 else [run])
    return out


def _tsquery_terms(qnorm: str, expand: list[str]) -> str:
    """OR-query over the query's own tokens plus the synonyms of any PLM class it
    named. The expansion is what lets a query written in the form's English
    ("Wire Harness cable") reach rows written in Chinese (線束) and vice versa."""
    terms = _split_terms(qnorm)
    for phrase in expand:
        terms.extend(_split_terms(phrase))
    seen: list[str] = []
    for t in terms:
        if t not in seen:
            seen.append(t)
    return " or ".join(seen) if seen else qnorm


def _by_lexical(cur, qnorm: str, tsq: str, fsql: str, fp: list) -> list[dict]:
    """tsvector (latin tokens + CJK bigrams + PLM wording) OR trigram similarity.

    search_text now carries plm_text, so the taxonomy words are real tsvector
    terms rather than something only the taxonomy channel can see.
    """
    cur.execute(
        f"""SELECT {SELECT_COLS},
                   ts_rank(tsv, websearch_to_tsquery('simple', %s)) AS trank,
                   similarity(name_norm, %s) AS sim
            FROM partrag.parts
            WHERE (tsv @@ websearch_to_tsquery('simple', %s)
                   OR name_norm %% %s) {fsql}
            ORDER BY (ts_rank(tsv, websearch_to_tsquery('simple', %s)) * 2
                      + similarity(name_norm, %s)) DESC, item_number
            LIMIT %s""",
        (tsq, qnorm, tsq, qnorm, *fp, tsq, qnorm, CAND),
    )
    return cur.fetchall()


def _by_plm(cur, v: T.QueryVocab, tsq: str, fsql: str, fp: list) -> list[dict]:
    """Candidates from the PLM classification the query named.

    Pushed into SQL as equality on prod_l2 / eco_std / item_type: the
    wire-harness class holds thousands of rows, so asking "which of the top-60 for this
    query are Wire Harness?" would usually answer none, while asking "of the
    Wire Harness rows, which match this query best?" answers what was asked.
    """
    where: list[str] = []
    params: list = []
    if v.prod_l2:
        where.append("prod_l2 = ANY(%s)")
        params.append(v.prod_l2)
    if v.eco_std:
        where.append(T.ECO_MATCH_SQL)
        params.append(v.eco_std)
    if v.item_type:
        where.append("item_type = ANY(%s)")
        params.append(v.item_type)
    if not where:
        return []
    cur.execute(
        f"""SELECT {SELECT_COLS},
                   ts_rank(tsv, websearch_to_tsquery('simple', %s)) AS trank
            FROM partrag.parts
            WHERE ({" OR ".join(where)}) {fsql}
            ORDER BY ts_rank(tsv, websearch_to_tsquery('simple', %s)) DESC,
                     -- tie-break on completeness: a row carrying pitch, pins and
                     -- a length is a more useful example of its class than a stub
                     ((pitch_series <> '{{}}')::int
                      + (pins_total <> '{{}}')::int
                      + (length_mm  <> '{{}}')::int) DESC,
                     item_number
            LIMIT %s""",
        (tsq, *params, *fp, tsq, CAND),
    )
    return cur.fetchall()


def _by_drawing(cur, qnorm: str, tsq: str, fsql: str, fp: list) -> list[dict]:
    """Candidates whose ENGINEERING DRAWING matches -- title block, BOM line, or
    a vendor part number inside the BOM.

    Returns rows of `parts`, not of `drawings`. Joining back here is what lets
    this drop straight into _rrf() and _attr_bonus() without either of them
    learning about a second table: to the rest of the pipeline a drawing hit is
    just another reason a part number is a candidate.

    The vendor_pn arm is matched by prefix rather than equality. Users type the
    series ("ABC-0001"), the BOM carries the full ordering code ("ABC-0001H-06P",
    "ABC-0001-TPE"), and OCR sometimes eats the hyphen -- equality would answer
    nothing for the query this channel exists to serve.

    Fuzzy text only -- the vendor-code arm is its own generator, see
    _by_drawing_pn() for why they cannot share one.

    ★ 第二條通道用 word_similarity()／`<%`，不是 similarity()／`%`
      `d.search_text` 是**整份圖**的標題欄＋材料表全部列串起來，可能是好幾百字；
      查詢通常只是抄下來的其中一列（例如「WIRE:UL1571 28AWG(7/0.127TS)
      OD:0.6MM VW-1 80°C 30V JACKET:PVC COLOR:SEE WIRE LIST」這種一列線材規格）。
      `similarity()` 是整串對整串的 trigram Jaccard，查詢越短、目標串越長，
      分數被稀釋得越兇 —— 這一列**真正的出處**分數會低於預設門檻 0.3，
      兩條通道都掛零，於是這張圖完全沒有進入候選池，跑出來的反而是分數同樣
      不到門檻、只是材料表也含同款線材的其他圖。
      `word_similarity()` 比的是查詢對目標**任一段等長子串**的相似度，
      同一組文字分數接近 0.9，且把出處排在第一名、其他圖分數明顯拉開。
      這正是「抄一段材料表文字回來問」這個用法需要的比對方式。
    """
    cur.execute(
        f"""SELECT {SELECT_COLS} FROM partrag.parts p
            JOIN (
                -- 欄位改名成 hit_item：SELECT_COLS 是不帶表別名的欄位清單，
                -- join 進來的子查詢再叫 item_number 就會 ambiguous。
                SELECT item_number AS hit_item, max(w) AS w FROM (
                    SELECT d.item_number,
                           1.0 + ts_rank(d.tsv, websearch_to_tsquery('simple', %s)) AS w
                    FROM partrag.drawings d
                    WHERE d.tsv @@ websearch_to_tsquery('simple', %s)
                    UNION ALL
                    SELECT d.item_number, word_similarity(%s, d.search_text) AS w
                    FROM partrag.drawings d
                    WHERE %s <%% d.search_text
                ) x GROUP BY item_number
            ) h ON h.hit_item = p.item_number
            WHERE true {fsql}
            ORDER BY h.w DESC, p.item_number
            LIMIT %s""",
        (tsq, tsq, qnorm, qnorm, *fp, CAND),
    )
    return cur.fetchall()


# 材料表英文自由文字比對信心夠高時，視同 _by_drawing_desc()／_by_drawing_pn()
# 那兩條「精準通道」，search() 撈 drawings 時會跳過融合排序的雜訊 —— 見那裡
# 用到 _by_drawing_confident() 的地方。
#
# ★ 門檻 0.7 是量出來的，不是猜的，而且只看 word_similarity 一個數字
#   同一句線材規格（上面 _by_drawing() 那個例子）：真正抄自的那份圖、以及另一份
#   材料表上逐字相同的圖都落在 0.7 以上 —— 兩者都是對的答案。下一群只到 0.6
#   左右，材料表寫的是別種線材，只是字面上剛好沾到「28AWG」「80C」「VW-1」
#   這幾個詞。0.7 卡在這兩群中間：撈進真正同一款線材的圖，濾掉只是字面相近的雜訊。
#
#   ★ 刻意不採 `_by_drawing()` 的 tsv 分支
#     那一支的分數是 `1.0 + ts_rank(...)`——只要 tsv 有任何一點命中就 >= 1.0，
#     必過任何合理門檻。而這裡的 tsq 是 `_tsquery_terms()` 灌過詞彙擴展的
#     OR 查詢，「有命中」跟「這一題答得對」是兩件事：實測拿它當信心指標，
#     大多數圖都 >= 1.0，等於完全沒有篩選效果。word_similarity 沒有
#     這個問題，分數本身就是 0~1 的真實相似度。
_DRAWING_TEXT_CONFIDENT = 0.7

# 剪掉問句鷹架之後，短到這個長度以下就不走這條捷徑（字元數，剪完的字串）。
#
# ★ 為什麼光有 0.7 還不夠，一定要再加長度下限
#   `word_similarity(查詢, 目標)` 比的是查詢對目標**任一等長視窗**的相似度，
#   所以查詢越短，越容易在幾百字的 search_text 裡找到一個幾乎完全吻合的視窗
#   —— 這是這個函式的數學性質，不是資料的巧合。而上面 0.7 那條門檻是拿
#   數十字元的**整段材料描述**標定出來的，把它直接套到一兩個詞上等於
#   換了一個意思。
#   實測（剪完鷹架後）：`LABEL`、`MCIO` 這種單一短詞會讓大部分的圖都過 0.7。
#   而這條捷徑的作用是**跳過融合排序、直接當答案**，一大批「confident」
#   等於把這一題的排序整個關掉。單一物件名本來就是 `_by_drawing_desc()`（中文字面）與
#   `_by_drawing()`（模糊排序）的守備範圍，不該由這裡接手。
#
# ★ 24 這個數字
#   這條通道要服務的是「抄一整段材料表／規格文字回來問」，而那種字串的下限
#   就在這附近：`WIRE:UL1571 28AWG(7/0.127TS)` 本身是 28 字元。反過來，會
#   失控的那幾種（LABEL / MCIO / 树脂胶 / 香蕉头端子 / ABC-0001-TPE）都在 12
#   字元以內，中間這一段空白很寬，不是卡著某一題硬調出來的。
_DRAWING_TEXT_MIN = 24


def _drawing_content_query(qnorm: str) -> str:
    """問句剪掉「在講圖這件事」的鷹架之後，剩下的那一段。

    ★ 這一步漏掉的後果
      `有RJ45 Jack Housing ABS+PC, UL 94V-0的工程圖?` —— 使用者照著圖上
      `Specification` 那一段抄了材料規格回來問。整句直接拿去比對會差一點過不了
      0.7；剪掉「有…的工程图?」之後明顯過門檻，而且與第二名分得非常開。
      差別不在資料，在**查詢裡混了不描述任何材料的字**：
      word_similarity 要求查詢的每一個 trigram 都在同一個視窗裡找到對應，
      鷹架字每多一個就往下拉一次分數。
      落榜的後果不是排名掉一點 —— 三條精準通道同時空手時 `search()` 會退回
      用 `results[:top_k]`（parts 的一般相關度）當名單，於是使用者問材料、
      畫面上第一張卡卻是另一顆料號的圖。答案文字是對的，卡片是錯的。

    `_cjk_phrases()` 與 `_latin_content()` 一直都有做這一步，這裡是把同一件事
    補給第三條通道；沒有共用同一支函式是因為那兩支各自還要再抽詞，這裡要的是
    剪完的**整段字串**。
    """
    s = timerange.strip(_RECENT_N_RE.sub(" ", qnorm))
    for w in _DRAWING_SCAFFOLD:
        s = s.replace(w, " ")
    # 標點在 trigram 上跟一般字元同權重，留著等於留著鷹架的殘骸。
    s = re.sub(r"[?？!！。，,、]+", " ", s)
    return re.sub(r"\s+", " ", s).strip().strip(_CJK_PARTICLES + " ")


def _by_drawing_confident(cur, qnorm: str) -> list[str]:
    """材料表描述跟這句話的 word_similarity 夠高的那幾份圖的料號，由高到低排序。

    只回料號字串，不回 `parts` 欄位：分數是拿來判斷「這一題該不該跳過融合
    排序」用的，`_by_drawing()` 回的列要塞進 _rrf()，欄位形狀不能夾帶這個
    分數往下傳、又汙染最後給使用者看的資料。

    ★ 比對的是**剪掉鷹架**的查詢，不是原句 —— 見 _drawing_content_query()
      與 _DRAWING_TEXT_MIN 上面那兩段說明。太短就直接不走這條路，讓
      `_by_drawing_desc()` / `_by_drawing()` 照原本的方式處理。
    """
    content = _drawing_content_query(qnorm)
    if len(content) < _DRAWING_TEXT_MIN:
        return []
    cur.execute(
        """SELECT item_number, word_similarity(%s, search_text) AS w
           FROM partrag.drawings
           WHERE %s <%% search_text AND word_similarity(%s, search_text) >= %s
           ORDER BY w DESC, item_number""",
        (content, content, content, _DRAWING_TEXT_CONFIDENT),
    )
    return [r["item_number"] for r in cur.fetchall()]


# 查工程圖時屬於「講到圖這件事」而不是「圖上有什麼」的詞。這些詞每一份圖都
# 沾得上邊，拿去比對材料表只會把所有圖全部撈回來。
#
# ★ 一定要由長到短剪，否則短的會先把長的咬掉一半
#   「香蕉头端子在哪份工程图上」用宣告順序剪的話，`哪份` 排在 `在哪` 前面，
#   剪完剩下「香蕉头端子在」—— 多帶一個「在」就 ILIKE 不到任何一列，
#   而症狀跟「這個詞不在語料裡」一模一樣。
_DRAWING_SCAFFOLD = tuple(sorted((
    "工程图", "图面", "图纸", "材料表", "标题栏", "零件表", "用料", "图档",
    "这份图", "那份图", "张图", "份图", "图上", "图里", "图中", "料号",
    "哪些", "哪份", "哪一份", "哪一张", "在哪份", "在哪张", "在哪", "更新",
    "有没有", "找有", "查有", "给我", "是哪", "请问", "帮我",
    "用了", "用到", "使用", "有用到", "包含", "含有", "里有", "上有",
    # 時間詞也是鷹架：「最近三筆工程圖」問的是排序，不是某個材料叫「最近」。
    "最近", "最新", "最后", "最後", "近期",
), key=len, reverse=True))

# 只從**頭尾**剝掉的虛詞。刻意不含「上下中裡里」——「上下壳」是材料表上真的
# 有的東西，把它們當虛詞剝掉就查不到了。
_CJK_PARTICLES = "的了是和與与及跟或在要有嗎吗呢喔啊呀过過着著个個"


# 剪掉時間與問句鷹架之後還留著的英數詞，就是使用者真正在找的東西。
# 這些字自己出現時不算條件（它們是在講「圖」這件事，不是圖上的某個零件）。
_LATIN_STOP = frozenset({
    "PDF", "REV", "NO", "PN", "P/N", "ITEM", "BOM", "TITLE", "DWG", "SHEET",
    "AND", "THE", "FOR", "ALL", "ANY", "NEW", "OLD", "PLM",
})


def _latin_content(qnorm: str) -> list[str]:
    """「2026 三月有 LABEL WHITE 的工程圖」-> ['LABEL', 'WHITE']。

    ★ 為什麼不能只看中文
      `_cjk_phrases()` 只抽中文，於是上面那句話被判成「只有日期、沒有條件」，
      系統回了三月**全部**的圖，其中混著材料表上根本沒有 LABEL WHITE 的。
      多給一份不會被使用者發現 —— 它看起來就像答案的一部分。
      材料表上的東西一半是英文（LABEL / TAPE / HSG / WIRE），漏掉這半邊等於
      這個判斷只做了一半。
    """
    s = timerange.strip(qnorm)
    for w in _DRAWING_SCAFFOLD:
        s = s.replace(w, " ")
    return [t for t in re.findall(r"[A-Z][A-Z0-9./+-]{2,}", s)
            if t not in _LATIN_STOP and not t.replace(".", "").isdigit()]


def _by_drawing_desc(cur, qnorm: str, fsql: str, fp: list) -> list[dict]:
    """材料表描述（或標題欄）**字面包含**問句裡那個中文詞的圖。

    ★ 為什麼模糊比對不夠，非要一條字面通道不可
      例如「找有香蕉头端子的工程圖」：`香蕉` 只出現在極少數圖裡，是這句話
      鑑別力最高的詞。但 PostgreSQL 的 `ts_rank` **沒有 IDF** —— 它不知道
      「香蕉」比「端子」稀有，於是 `香蕉 or 蕉头 or 头端 or 端子 or 工程 or 程图 …`
      這個 OR 查詢裡，那兩份真正該回答的圖被一堆只是碰巧有「端子」的
      連接器圖壓下去，前幾名一份都排不進。`_by_drawing` 是模糊比對，
      它的排序永遠會有這個問題，補權重只是搬動同一個病灶。

      所以照 `_by_drawing_pn` 的先例把它拆成自己的通道：那一支是「廠商料號
      完全對上」，這一支是「材料表描述字面對上」，兩者都是**證據等級遠高於
      模糊比對**的東西，用同一個 5.0 權重。

    ★ 為什麼用 ILIKE 而不是 tsvector 或 trigram
      中文在這兩者上都沒有鑑別力：`simple` parser 把整串中文當一個 token
      （所以 bigram 才要人工塞），而 pg_trgm 對兩三個中文字的相似度低於門檻。
      實測 `香蕉`：ILIKE 命中、tsv 與 trigram 都命中 0。
      工程圖與材料表列數都很小，全表掃是微不足道的成本。
    """
    phrases = _cjk_phrases(qnorm)
    if not phrases:
        return []
    pats = [f"%{p}%" for p in phrases]
    cur.execute(
        f"""SELECT {SELECT_COLS} FROM partrag.parts p
            JOIN (
                SELECT DISTINCT d.item_number AS hit_item
                FROM partrag.drawings d
                LEFT JOIN partrag.drawing_items i ON i.drawing_id = d.id
                WHERE i.description ILIKE ANY(%s) OR d.dwg_title ILIKE ANY(%s)
            ) h ON h.hit_item = p.item_number
            WHERE true {fsql}
            ORDER BY p.item_number
            LIMIT %s""",
        (pats, pats, *fp, CAND),
    )
    return cur.fetchall()


# 中文詞的長度下限。兩個字（端子、膠帶、標籤）就有意義，一個字沒有。
_CJK_MIN = 2


def _cjk_phrases(qnorm: str) -> list[str]:
    """問句裡值得拿去字面比對的中文詞。

    做法是「把講到圖的那些詞剪掉，剩下的中文串就是使用者真正在找的東西」——
    不需要斷詞器：使用者打的是「香蕉头端子」這種完整的物件名，剪完剩下的
    連續中文串本身就是最好的查詢詞。太短或剪剩下的碎片（一個字）一律丟掉。
    """
    # 數量詞與**時間**都不是材料名。少了 timerange.strip()，
    # 「給我修改日期為2026三月跟五月的工程圖」會抽出「修改日期为」當成材料表
    # 要比對的中文詞，然後 has_content 變成真、日期分支被跳過 —— 答案回 0。
    s = timerange.strip(_RECENT_N_RE.sub(" ", qnorm))
    for w in _DRAWING_SCAFFOLD:
        s = s.replace(w, " ")
    out: list[str] = []
    for run in re.findall(f"[{CJK}]{{{_CJK_MIN},}}", s):
        # 「的」「在」這類虛詞黏在頭尾時剝掉，剝完仍夠長才收。
        run = run.strip(_CJK_PARTICLES)
        if len(run) >= _CJK_MIN and run not in out:
            out.append(run)
    return out[:4]


# 標題欄裡「這份圖是誰」的那幾格。刻意只收這兩個 key，不掃整個 title_fields
# —— `description` / `material` 那幾格是自由文字，拿使用者打的編號去比對它們
# 會在錯的地方命中（例如 MATERIAL 欄寫著另一顆料號）。
_DWG_ID_KEYS = ("part_no", "customer_pn")


def drawing_code_hits(cur, codes: list[str]) -> tuple[list[str], set[str]]:
    """使用者打的編號**就是某一份圖自己的編號**時，回那幾份圖的歸檔料號。

    -> (drawings.item_number 依命中順序, codes 裡真的命中的那幾個)

    ★ 為什麼非要這一條精準通道不可（「找ABC-0001-01工程圖」這種問法）
      `ABC-0001-01` 是圖上印的**客戶料號**，這個字串在 `parts` 裡不存在
      （那顆料號叫 `AA-BB-ABC-0001-01`），在 `drawings.dwg_part_no` 裡也不存在
      （圖上印的是 `BB-ABC-0001-01`）—— 它只活在 `title_fields->>'customer_pn'`。
      於是三條精確通道（`_by_item_number` / `_by_exact_code` /
      `_by_customer_code`）全部落空，只剩 `_by_drawing` 的模糊比對在跑。
      而模糊比對在這一題必然答錯：`simple` 分詞把 `abc-0001-01` 切成
      `abc` / `0001` / `01`，`ts_rank` 沒有 IDF，同一個客戶同系列的圖
      這些 token 一模一樣 —— 系統回的是**相鄰編號 ABC-0002-01 那份圖**，而模型
      照著卡片寫出「這是 ABC-0001-01 的工程圖」。兩份圖的材料表恰好同款，
      使用者完全無法從答案看出拿錯了圖。

    ★ 四種比對都要有，因為同一個編號在四個地方各長一個樣
        d.item_number                  AA-BB-ABC-0001-01   （下載紀錄歸檔用）
        d.dwg_part_no                  BB-ABC-0001-01      （圖上 PART NUMBER）
        title_fields->>'customer_pn'   ABC-0001-01         （圖上 CUSTOMER P/N）
        使用者打的                       ABC-0001-01
      前兩個要**連字號邊界的後綴比對**（`AA-BB-` / `BB-` 是廠內加的前綴），
      不是 LIKE '%code'：後者會讓 `ABC-0001-01` 命中
      `AA-BB-ABC-0001-01-HF`（`-HF` 是另一顆料號的另一份圖）。用
      `right(x, n+1) = '-' || code` 同時解掉這件事與 LIKE 的 `_`/`%` 轉義問題。
    """
    codes = [c for c in dict.fromkeys(codes) if c and len(c) >= 4]
    if not codes:
        return [], set()
    keys = list(_DWG_ID_KEYS)
    cur.execute(
        """SELECT d.item_number, t.c AS code
           FROM partrag.drawings d, unnest(%s::text[]) AS t(c)
           WHERE upper(d.item_number) = upper(t.c)
              OR upper(d.dwg_part_no) = upper(t.c)
              OR right(upper(d.item_number), length(t.c) + 1)
                   = '-' || upper(t.c)
              OR right(upper(d.dwg_part_no), length(t.c) + 1)
                   = '-' || upper(t.c)
              OR EXISTS (
                   SELECT 1 FROM unnest(%s::text[]) k
                   WHERE upper(d.title_fields->>k) = upper(t.c)
                 )
           ORDER BY d.item_number, d.id""",
        (codes, keys),
    )
    rows = cur.fetchall()
    hits = list(dict.fromkeys(r["item_number"] for r in rows))
    return hits, {r["code"] for r in rows}


def drawing_content_codes(cur, codes: list[str]) -> set[str]:
    """codes 裡，哪幾個是**圖上材料表的一列**（廠商料號或描述字面），
    而不是某一份圖自己的編號。

    ★ 這一支要解掉的案例
      「哪些料號的工程圖材料表用了 ABC-0001-TPE？」`ABC-0001-TPE` 是端子的廠商料號，
      它當然不會是任何一份圖自己的編號 —— 於是 `drawing_code_hits()` 一個都沒
      命中，`drawing_absent` 就把它標成「語料裡沒有這份圖」。接著 grounding 的
      `wrong_drawing_subject` 看到答案裡有這個編號、又沒有否定句，就判定模型
      拿別份圖冒充它，把一個**完全正確**的答案（那兩份圖的材料表裡真的有這顆
      端子，連來源檔名都寫對了）貼上「事實核對未通過」。

      檢索端其實早就分得開這兩件事：`search()` 裡的 `precise_content_hit` 就是
      拿來判斷「這編號是靠材料表命中的、圖不要清空」。問題只是那個判斷沒有
      傳給 `drawing_absent` —— 同一次檢索裡一邊說「有命中」、一邊說「不存在」。
      這一支把它補成**逐個編號**的答案：一次問兩個編號、只有一個在材料表裡時，
      另一個仍然要被標成 absent（整批放行會把那道防護一起關掉）。

    ★ 比對方式與那兩條精準通道對齊
      廠商料號用**前綴**（使用者打 `ABC-0001`、材料表寫 `ABC-0001-TPE`，見
      `_by_drawing_pn`），描述用字面包含（見 `_by_drawing_desc`）。
      兩邊用不同規則的話，會出現「通道撈得到、但這裡說沒著落」的矛盾。

    ★ 第三條 EXISTS：沒有材料表的圖框，內容在 `title_raw` 而不是 drawing_items
      `drawing_frames.NO_BOM_FRAMES` 那幾種圖（廠商規格書、客戶自由文字
      規格書）的 `drawing_items` 是**零列** —— 圖上的每一個字都只存在
      `title_raw`。前兩條 EXISTS 對它們因此恆為假，於是圖上明明印著的編號
      一律被判成「沒有著落」。
      例如使用者把一份規格書上的線材規格整段抄回來問，裡面的 DWG No. 與幾個
      線材編號**全部**印在那份圖上，卻同時觸發 `drawing_absent`。畫面上
      因此一邊是正確的工程圖卡、一邊是「該料號沒有工程圖」——
      而模型被教過看到 NOT_COVERED 就照講，所以它會把矛盾照抄給使用者。

      刻意只查這幾種圖框、也刻意只查 `title_raw` 而不是 `search_text`：
      後者還含標題欄與 `title_fields`，編號出現在那裡代表「這份圖就叫這個
      名字」，那是 `drawing_code_hits()` 的職責，混進來會讓兩支對同一件事
      給出不同答案。
    """
    codes = [c for c in dict.fromkeys(codes) if c and len(c) >= 4]
    if not codes:
        return set()
    cur.execute(
        """SELECT DISTINCT t AS code
           FROM unnest(%s::text[]) t
           WHERE EXISTS (
                   SELECT 1 FROM partrag.drawing_items i, unnest(i.vendor_pn) v
                   WHERE v ILIKE t || '%%'
                 )
              OR EXISTS (
                   SELECT 1 FROM partrag.drawing_items i
                   WHERE i.description ILIKE '%%' || t || '%%'
                 )
              OR EXISTS (
                   SELECT 1 FROM partrag.drawings d
                   WHERE d.frame = ANY(%s) AND d.title_raw ILIKE '%%' || t || '%%'
                 )""",
        (codes, sorted(drawing_frames.NO_BOM_FRAMES)),
    )
    return {r["code"] for r in cur.fetchall()}


def _by_drawing_pn(cur, qnorm: str, fsql: str, fp: list) -> list[dict]:
    """Parts whose drawing BOM carries a VENDOR part number the query named.

    Kept out of _by_drawing for the same reason _by_master is kept out of
    _by_plm, and it is the same query shape that forced that split: an exact code
    plus a broad class word in one sentence.

    Example: "哪些料號的工程圖用了 ABC-0001 端子？". Only a couple of parts carry
    that terminal. But 端子 resolves to 產品中類 Connector, `_by_plm` returns a
    full candidate budget of it, and every one collects a +0.25 classification
    bonus in _attr_bonus() -- so the class floor sits far above what a rank-1 hit
    in a weight-1.6 channel is worth (1.6/61 = 0.026). The real answers were
    pushed off the page by rows whose only merit was being connectors.

    So this is its own generator at `code` strength (a vendor ordering code is an
    exact code match, exactly like a customer P/N), AND its hits take an attribute
    bonus in search() -- because in this pipeline the bonus, not the RRF weight,
    is what actually clears a class flood.

    Prefix, not equality: users type the series ("ABC-0001"), the BOM carries the full
    ordering code ("ABC-0001H-06P", "ABC-0001-TPE"), and OCR sometimes eats the hyphen.
    """
    # 4 個字元以上才算 —— 短 token（`PN`、`OR`）會 prefix 到整批廠商料號。
    #
    # ★ 先補中英之間的空白再切，而且丟掉帶中文的 token。
    #   切割用的字元類是「不是 \w 也不是中文」，而 Python 的 \w **含中文** ——
    #   所以「工程圖材料表裡有用到ABC-0001-TPE的料號」整句沒有任何一個分隔字元，
    #   切出來是一個 token：整句話。接著拿整句去 `v ILIKE '整句%'`，
    #   當然一份圖都對不到。同一句話只要在料號兩側各打一個空格就會正常，
    #   而使用者不會這樣打。
    #   廠商料號永遠是英數，所以帶中文的 token 一律不是它。
    toks = [
        t.upper()
        for t in re.split(r"[^\w一-鿿.*/+-]+", space_cjk_ascii(qnorm))
        if len(t) >= 4 and not re.search(f"[{CJK}]", t)
    ]
    if not toks:
        return []
    cur.execute(
        f"""SELECT {SELECT_COLS} FROM partrag.parts p
            JOIN (
                SELECT DISTINCT d.item_number AS hit_item
                FROM partrag.drawings d
                JOIN partrag.drawing_items i ON i.drawing_id = d.id
                WHERE EXISTS (
                    SELECT 1 FROM unnest(i.vendor_pn) v, unnest(%s::text[]) t
                    WHERE v ILIKE t || '%%'
                )
            ) h ON h.hit_item = p.item_number
            WHERE true {fsql}
            ORDER BY p.item_number
            LIMIT %s""",
        (toks, *fp, CAND),
    )
    return cur.fetchall()


# 「最近三筆工程圖」這類問題。刻意要求同時出現**時間詞**與**數量詞**：
# 只有「最近」的話，「最近有沒有 MCIO 的工程圖」會被誤判成要一份排行榜。
_RECENT_RE = re.compile(r"最近|最新|最后|最後|近期")
# ★ 後面接時間單位就不算數量詞
#   「最近三个月」裡的「三个」原本會被讀成「三筆」，於是「最近三個月更新的
#   工程圖」回的是**最新的 3 份圖**，而不是三個月內的全部。兩個答案都
#   長得像正經答案，而且都附了正確的檔名與時間 —— 使用者沒有辦法分辨。
_RECENT_N_RE = re.compile(
    r"(\d{1,2}|一|两|兩|二|三|四|五|六|七|八|九|十)\s*(?:笔|筆|份|张|張|个|個)"
    r"(?!\s*(?:月|年|周|週|天|日|季))")
_CN_NUM = {"一": 1, "两": 2, "兩": 2, "二": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
RECENT_DEFAULT = 5
RECENT_MAX = 20


def recent_n(qnorm: str) -> int | None:
    """「最近三筆工程圖」-> 3。不是這種問法回 None。

    ★ 為什麼要專門支援它
      這句話以前走的是一般工程圖檢索：沒有任何料號、沒有任何材料詞可以比對，
      十一路全部空手而回，`drawings` 是空的。而模型手上有「這個 agent 會查工程圖」
      這個設定，於是它**自己編了一份排行榜**（實測回了三個料號、三個時間、
      三個連結，其中一筆的時間還寫「資料中未記錄」）。
      這不是模型不老實，是我們給了一個它答不出來、卻又看起來該答得出來的問題。
      能答就答，答不了就要讓它拿到「查不到」這個明確的事實。
    """
    if not _RECENT_RE.search(qnorm):
        return None
    m = _RECENT_N_RE.search(qnorm)
    if not m:
        return RECENT_DEFAULT
    raw = m.group(1)
    n = _CN_NUM.get(raw) or (int(raw) if raw.isdigit() else RECENT_DEFAULT)
    return max(1, min(n, RECENT_MAX))


def recent_drawings(cur, n: int) -> list[dict]:
    """最近更新的 n 份工程圖（依 PLM 的檔案更新時間）。

    `doc_modified_on` 是 NULL 的排最後 —— 「沒有時間戳」不等於「很舊」，
    但它一定不該混在「最近的」前面充數。
    """
    cur.execute(
        """SELECT item_number FROM partrag.drawings
           ORDER BY doc_modified_on DESC NULLS LAST, id DESC
           LIMIT %s""",
        (n,),
    )
    return [r["item_number"] for r in cur.fetchall()]


def drawings_in_range(cur, tq: "timerange.TimeQuery", limit: int = 200) -> list[str]:
    """檔案更新時間落在這段區間內的工程圖，對應到的料號（新到舊）。

    ★ 這是「2026 三月的工程圖」唯一正確的查法
      實測系統原本拿「2026 三月」去比對**標題欄與材料表的文字**，然後回報
      「這些圖裡均無含該字串」—— 完全正確的一句話，答的卻不是使用者的問題。
      日期不在圖上的字裡，它在這一欄。
    """
    sql_, params = timerange.sql(tq, timerange.DRAWING_COLUMN)
    cur.execute(
        f"""SELECT item_number FROM partrag.drawings
            WHERE {sql_}
            ORDER BY {timerange.DRAWING_COLUMN} DESC NULLS LAST, id DESC
            LIMIT %s""",
        params + [limit],
    )
    return list(dict.fromkeys(r["item_number"] for r in cur.fetchall()))


def drawing_buckets(tq: "timerange.TimeQuery | None",
                    drawings: list[dict]) -> list[dict]:
    """每一段時間區間裡**實際**有幾份圖，以及那幾份是誰。

    ★ 這一支要解掉的案例
      「給我修改日期為某年三月跟五月的工程圖」檢索到幾十份（五月佔大多數），
      答案層只展開得下前幾份 —— 而排序是新到舊，所以展開的全是五月。模型於是
      寫出「語料中沒有修改日期為三月的工程圖」：**照它看到的東西推論完全
      正確**，而使用者的畫面上就有一張三月的卡片。
      根因是「被截掉的那些份沒有留下任何痕跡」。這一支把每段區間的份數算成
      確定性的事實跟著 found 走，答案層拿它寫總結、grounding 拿它擋「說某段
      沒有」的幻覺（見 grounding 的 denied_present）。

    ★ 為什麼份數不從 SQL 再查一次
      這裡要回答的是「**這一題端出去的這批圖**分佈在哪幾段」，不是「資料庫裡
      那幾段各有幾份」。有內容條件時（「三月的 MCIO 工程圖」）兩個數字不一樣，
      而模型手上只有前者 —— 拿後者去核對它，會把一個正確的答案判成漏講。
    """
    if tq is None or not drawings:
        return []
    out: list[dict] = []
    for s in tq.spans:
        hit = [d for d in drawings if _in_span(d.get("doc_modified_on"), s)]
        out.append({
            "label": s.label,
            "start": str(s.start),
            "end": str(s.end),
            "n": len(hit),
            # 料號與檔名都留：grounding 用它們判斷「答案有沒有引用這一段裡的圖」，
            # 兩種寫法模型都可能用。
            "items": [d["item_number"] for d in hit],
            "files": [d.get("local_file") or "" for d in hit if d.get("local_file")],
        })
    return out


def _in_span(value: Any, span: "timerange.Span") -> bool:
    """`doc_modified_on`（datetime 或 date，也可能是 None）落在這段區間裡嗎。"""
    if value is None:
        return False
    d = value.date() if hasattr(value, "date") else value
    try:
        return span.start <= d < span.end
    except TypeError:          # 型別出乎預料時算「不在」，不要讓統計拋例外
        return False


def drawings_for(cur, item_numbers: list[str],
                 tq: "timerange.TimeQuery | None" = None) -> list[dict]:
    """Every drawing held for these part numbers, title block + BOM + provenance.

    This is what the answer layer quotes from, so it carries the three things a
    citation needs -- local_file, doc_modified_on, pdf_url -- plus the three that
    say how much to trust it (extract_method / ocr_conf / part_no_ok). A drawing
    value with no provenance attached is worse than no value: most of these
    drawings were read by OCR, and O/0 confusion is a measured, not hypothetical,
    error mode.

    ★ Matched on `dwg_part_no` as well as `item_number`, because those two
      genuinely disagree and each is right about a different thing.
      `item_number` is who the download log filed the drawing under (the part
      whose PLM record served the file); `dwg_part_no` is the number printed on
      the drawing itself. A drawing can be filed under `Z0000XX00000-1HT` while
      its title block -- and the filename -- say `Z0000XX00000-1`, which is a
      separate row in `parts`. Keying on the filing number alone answered
      "這個料號沒有工程圖" for the exact number printed on the drawing being
      returned in the same response.
    """
    if not item_numbers:
        return []
    # ★ 日期條件要 AND 在整個 OR 外面
    #   料號比對本來就是 `item_number = ANY OR dwg_part_no = ANY`，少了括號的話
    #   日期只會綁到 OR 的右半邊，左半邊照樣把範圍外的圖撈進來 —— 而且回來的
    #   結果看起來完全正常，只是多了幾份不該在的。
    tsql, tparams = ("", [])
    if tq is not None:
        s, p = timerange.sql(tq, "d." + timerange.DRAWING_COLUMN)
        tsql, tparams = " AND " + s, p
    # ★ 排序要照呼叫端給的 item_numbers 順序，不能改成按料號字母排
    #   `item_numbers` 進來的時候已經是排好的名單 —— 可能是相關度融合排序
    #   （search() 的 lookup），也可能是時間排序（recent_drawings() 的
    #   「最近三筆」）。這裡如果自己重排成 `ORDER BY d.item_number`，兩種
    #   排序都會被字母序蓋掉：「最近三筆工程圖」餵進來的是新到舊的名單，
    #   舊寫法回來的卻是字母序 —— 前端顯示看起來完全正常（都是有效的工程圖卡片），
    #   使用者卻拿到「字母序前三名」而不是「最新三筆」，沒有任何錯誤訊息。
    #   COALESCE 兩個 array_position 是因為命中可能來自 item_number 或
    #   dwg_part_no 任一邊（見上面的類別註解），只在其中一邊的名單裡也要排得到。
    cur.execute(
        f"""SELECT d.id, d.item_number, d.local_file, d.plm_filename,
                  d.doc_modified_on, d.dwg_title, d.dwg_part_no, d.part_no_ok,
                  d.dwg_rev, d.dwg_size, d.dwg_scale, d.sheet_no, d.sheet_of,
                  d.pd_file_no, d.page_no, d.extract_method, d.ocr_conf,
                  d.frame, d.title_fields, d.title_raw
           FROM partrag.drawings d
           WHERE (d.item_number = ANY(%s) OR d.dwg_part_no = ANY(%s)){tsql}
           ORDER BY COALESCE(array_position(%s::text[], d.item_number),
                             array_position(%s::text[], d.dwg_part_no)),
                    d.item_number, d.id""",
        (item_numbers, item_numbers, *tparams, item_numbers, item_numbers),
    )
    out = [dict(r) for r in cur.fetchall()]
    if not out:
        return []
    cur.execute(
        # table_tag 排在 item_no 前面：客戶圖框的一份圖上有兩張表（材料表 +
        # 料號長度對照表），照 item_no 排會把兩張表的 1、2、3… 交錯在一起。
        # 'BOM' < 'PN_LENGTH'，材料表自然排在前面。
        """SELECT drawing_id, item_no, description, qty_raw, qty_num,
                  vendor_pn, kind, ocr_conf,
                  table_tag, unit, remark, color, length_raw
           FROM partrag.drawing_items
           WHERE drawing_id = ANY(%s)
           ORDER BY drawing_id, table_tag, item_no NULLS LAST, id""",
        ([d["id"] for d in out],),
    )
    by_dwg: dict[int, list[dict]] = {}
    for r in cur.fetchall():
        row = dict(r)
        if row.get("qty_num") is not None:
            row["qty_num"] = float(row["qty_num"])
        if row.get("ocr_conf") is not None:
            row["ocr_conf"] = float(row["ocr_conf"])
        by_dwg.setdefault(row.pop("drawing_id"), []).append(row)
    for d in out:
        if d.get("ocr_conf") is not None:
            d["ocr_conf"] = float(d["ocr_conf"])
        d["items"] = by_dwg.get(d["id"], [])
        d["pdf_url"] = f"/api/drawing/file/{d['id']}"
    return out


NO_DRAWING = "該料號沒有工程圖"


_OWNER_COLS = ("created_by", "modified_by", "locked_by")


def _by_master(
    cur, matches: list[V.FieldMatch], named_fields: list[str], fsql: str, fp: list
) -> list[dict]:
    """Candidates matching an imported PLM master value exactly.

    Kept out of _by_plm rather than OR-ed into it: a query like
    "公告大類 DNT 的線束" names both a master value (a handful of rows) and a class
    (thousands of rows), and inside one OR-ed generator the class floods the 60-row candidate
    budget and the exact match never surfaces. As its own generator with a high
    RRF weight, the exact match leads and the class still contributes.

    `matches` now comes from two places -- taxonomy's hand-written VALUE_TERMS
    and values.resolve()'s DB-built index -- and each carries a value LIST,
    because a bare surname legitimately names more than one colleague.
    """
    if not (matches or named_fields):
        return []
    where = [f"{m.column} = ANY(%s)" for m in matches]
    # A field named without a value ("保存天數") means "which parts have one".
    where += [f"{col} IS NOT NULL" for col in named_fields]

    # "<人名> 建立了什麼" means the recent things, not the alphabetically first.
    # Ordering by item_number there would answer with the oldest part numbers.
    owner_query = any(m.column in _OWNER_COLS for m in matches) or any(
        c in _OWNER_COLS for c in named_fields
    )
    order = "created_on DESC NULLS LAST, item_number" if owner_query else "item_number"

    cur.execute(
        f"""SELECT {SELECT_COLS} FROM partrag.parts
            WHERE {" AND ".join(where)} {fsql}
            ORDER BY {order}
            LIMIT %s""",
        (*[m.values for m in matches], *fp, CAND),
    )
    return cur.fetchall()


# --------------------------------------------------------------------------
# fusion + rescoring
# --------------------------------------------------------------------------
# Source weights reflect measured reliability: an exact part-number hit is
# near-certain; lexical beats dense here because of the near-duplicate problem.
#
# `neighbor` sits between lexical and an exact hit. It was first set to 0.5,
# which put the nearest neighbour (0.5/61 = 0.008) BELOW rank-1 vector noise
# (1.0/61 = 0.016), so the fallback existed but never showed. 3.0 puts it
# clearly above noise and still well under an `item` hit (6.0/61 = 0.098) --
# and neighbours are only generated for a number the corpus does not hold, so
# there is no real match for them to displace.
#
# `drawing` sits just above lexical. A drawing hit is evidence from a document
# that was signed off, so it beats a trigram brush; but it is read by OCR and
# covers only a small fraction of parts, so it must not outrank an exact number
# or an imported master value. Note this weight can only ever ADD a candidate:
# queries that never touch a drawing must rank exactly as before with it on.
#
# ★ 模組級常數，不是 _rrf() 裡的區域變數：README 的檢索機制章節要把
#   這份權重原樣抄出來給人看，改這裡的話記得順手同步那邊的說明。
CHANNEL_WEIGHTS: dict[str, float] = {
    "item": 6.0,
    # 使用者打的編號就是某一份圖自己的編號（圖上的 PART NUMBER / CUSTOMER P/N，
    # 或該圖歸檔的料號）。與 `item` 同級而不是與 `code` 同級：這是**完全相符**的
    # 編號比對，只是那個編號印在圖上而不在 parts 的欄位裡（見 drawing_code_hits）。
    "drawing_code": 6.0,
    "code": 5.0,
    "drawing_pn": 5.0,
    # 材料表描述字面命中。與 drawing_pn 同級，理由見 _by_drawing_desc()：
    # 兩者都是「完全對上」，而 drawing 那 1.6 是模糊比對。
    "drawing_desc": 5.0,
    "master": 4.0,
    "neighbor": 3.0,
    "plm": 2.0,
    "drawing": 1.6,
    "lexical": 1.4,
    "vector": 1.0,
}


# ★ 從 src/aggregate.py 搬過來的（原本住在那邊）。
#   搬家的理由是「同一組條件、兩個呼叫端」：清單路徑要用它組 WHERE，
#   而排序檢索也要用它回答「符合這些條件的**全庫**共幾筆」。留在 aggregate
#   的話 retrieval 拿不到（aggregate import retrieval，反向會變成循環），
#   而各寫一份的下場是同一句話在兩條路上得到兩個不同的筆數。
def _attr_filters(
    q: dict, vocab: T.QueryVocab, matches: list[V.FieldMatch]
) -> tuple[list[str], list[Any], list[str]]:
    """Hard attribute predicates. Only the discriminators that are safe as an
    exact filter -- pins_total and pins_row are deliberately absent, they are
    noisy enough that ANDing them empties the result set."""
    sql: list[str] = []
    params: list[Any] = []
    labels: list[str] = []
    pinned = {m.column for m in matches}

    if q["length_mm"]:
        sql.append("length_mm && %s::numeric[]")
        params.append(q["length_mm"])
        labels.append("長度 " + "/".join(f"{v:g}mm" for v in q["length_mm"]))
    if q["pitch_series"]:
        sql.append("pitch_series && %s")
        params.append(q["pitch_series"])
        labels.append("/".join(q["pitch_series"]))
    if q["pin_layout"]:
        sql.append("pin_layout && %s")
        params.append(q["pin_layout"])
        labels.append("PIN 排列 " + "/".join(q["pin_layout"]))
    if q["conn_types"]:
        sql.append("conn_types && %s")
        params.append(q["conn_types"])
        labels.append("接頭 " + "/".join(q["conn_types"]))
    for f, label in (("awg", "AWG"), ("gen", "GEN")):
        if q.get(f):
            sql.append(f"{f} = %s")
            params.append(q[f])
            labels.append(f"{label} {q[f]}")
    for col, vals, label in (
        ("prod_l2", vocab.prod_l2, "產品中類"),
        ("prod_l1", vocab.prod_l1, "產品大類"),
        ("eco_std", vocab.eco_std, "環保標準"),
        ("item_type", vocab.item_type, "料件類型"),
    ):
        if vals and col not in pinned:
            # 環保標準是「含不含這個代碼」，不是等號 —— 不少列的值是 `GC HF`，
            # 兩個代碼都成立。條件與 taxonomy.ECO_MATCH_SQL 必須一致，不然同一個問題
            # 走清單查詢與走排序檢索會得到不同的筆數。
            sql.append(T.ECO_MATCH_SQL if col == "eco_std" else f"{col} = ANY(%s)")
            params.append(list(vals))
            labels.append(f"{label} {'/'.join(vals)}")
            pinned.add(col)
    # 產品中類 implies 產品大類 (taxonomy.read_query adds it), and ANDing both is
    # redundant -- drop the coarser one so the label list stays readable.
    if any(s.startswith("prod_l2") for s in sql):
        for i, s in enumerate(sql):
            if s.startswith("prod_l1"):
                sql.pop(i)
                params.pop(i)
                labels = [x for x in labels if not x.startswith("產品大類")]
                break
    return sql, params, labels



def _rrf(ranked: dict[str, list[dict]]) -> dict[str, dict]:
    """Reciprocal Rank Fusion over CHANNEL_WEIGHTS. See that constant's
    comment for why each weight sits where it does."""
    merged: dict[str, dict] = {}
    for src, rows in ranked.items():
        for rank, row in enumerate(rows):
            key = row["item_number"]
            e = merged.setdefault(key, {"row": row, "score": 0.0, "sources": []})
            e["score"] += CHANNEL_WEIGHTS[src] / (RRF_K + rank + 1)
            if src not in e["sources"]:
                e["sources"].append(src)
    return merged


def merge_matches(
    vocab: T.QueryVocab, dynamic: list[V.FieldMatch]
) -> list[V.FieldMatch]:
    """Hand-written form values (taxonomy.VALUE_TERMS) + DB-built ones (values.py).

    taxonomy wins on a column both resolve: its table is curated, so
    "示例廠" maps to the canonical plant string even if the corpus happens
    to hold a variant spelling.
    """
    out = [
        V.FieldMatch(col, [val], T.field_label(col), val) for col, val in vocab.master
    ]
    taken = {m.column for m in out}
    out += [m for m in dynamic if m.column not in taken]
    return out


def _attr_bonus(
    row: dict, q: dict, v: T.QueryVocab, matches: list[V.FieldMatch]
) -> tuple[float, list[str]]:
    """Reward exact agreement on parsed attributes and PLM classification.

    This is what separates rows inside a near-duplicate group -- they differ
    only in these values, so an exact hit here is very strong evidence.
    """
    bonus, why = 0.0, []

    if q["length_mm"] and row["length_mm"]:
        rl = {float(x) for x in row["length_mm"]}
        for want in q["length_mm"]:
            if want in rl:
                bonus += 0.35
                why.append(f"長度 {want:g}mm")
                break
            near = min(rl, key=lambda x: abs(x - want))
            if abs(near - want) / max(want, 1) < 0.02:
                bonus += 0.15
                why.append(f"長度≈{near:g}mm")
                break

    # Pitch series is the strongest single discriminator in the 線束 family:
    # "PH4.2 2*3P HSG TO ..." and "PH3.0 2*3P HSG TO ..." are the same string
    # otherwise, and a large share of rows carry a pitch.
    pit = set(q["pitch_series"]) & set(row["pitch_series"] or [])
    if pit:
        bonus += 0.40
        why.append(sorted(pit)[0])

    # Layout beats total beats per-row count. A query for "3P" must not score a
    # 2*3P part (6 pins) as highly as a single-row 3P part -- that was the
    # observed failure on "PH4.2 3P HSG".
    lay = set(q["pin_layout"]) & set(row["pin_layout"] or [])
    tot = set(q["pins_total"]) & set(row["pins_total"] or [])
    if lay:
        bonus += 0.40
        why.append(f"排列 {sorted(lay)[0]}")
    elif tot:
        bonus += 0.30
        why.append(f"{sorted(tot)[0]}P")
    elif q["pins_total"] and set(q["pins_total"]) & set(row["pins_row"] or []):
        bonus += 0.10
        why.append("單排 PIN 數相符")

    if q["conn_types"] and row["conn_types"]:
        c = set(q["conn_types"]) & set(row["conn_types"])
        if c:
            bonus += 0.15
            why.append(sorted(c)[0])

    for f, label, w in (("awg", "AWG", 0.20), ("gen", "GEN", 0.20), ("speed_g", "速率", 0.15)):
        if q.get(f) and row.get(f) == q[f]:
            bonus += w
            why.append(f"{label} {q[f]}")

    # PLM classification agreement. 產品中類 is specific enough to be evidence;
    # 產品大類 is only four values wide, so it barely moves the score.
    if v.prod_l2 and row.get("prod_l2") in v.prod_l2:
        bonus += 0.25
        why.append(f"產品中類 {row['prod_l2']}")
    elif v.prod_l1 and row.get("prod_l1") in v.prod_l1:
        bonus += 0.08
        why.append(f"產品大類 {row['prod_l1']}")
    if v.eco_std and T.eco_matches(row.get("eco_std"), v.eco_std):
        bonus += 0.15
        why.append(f"環保標準 {row['eco_std']}")
    if v.item_type and row.get("item_type") in v.item_type:
        bonus += 0.10
        why.append(f"料件類型 {row['item_type']}")
    # An imported master value is an exact hit on an authoritative field, so it
    # outweighs the derived classification above it.
    for m in matches:
        if row.get(m.column) in m.values:
            bonus += 0.30
            why.append(f"{m.label} {row[m.column]}")

    return bonus, why


_SOURCE_LABEL = {
    "item": "料號精確比對",
    "code": "客戶料號/圖號比對",
    "master": "PLM 主檔欄位比對",
    "neighbor": "同系列鄰近料號",
    "plm": "PLM 分類比對",
    "lexical": "字面比對",
    "vector": "語意相近",
    "drawing": "工程圖內容比對",
    "drawing_code": "工程圖標題欄料號完全相符",
    "drawing_pn": "工程圖材料表廠商料號",
    "drawing_desc": "工程圖材料表描述比對",
}


def search(query: str, filters: dict | None = None, top_k: int = 10) -> dict[str, Any]:
    filters = filters or {}
    q = parse_query(query)
    qnorm = q["norm"]
    vocab = T.read_query(qnorm)
    # Values the corpus actually holds -- people, plants, states. This is what
    # turns "<人名> 建立的線束" into created_by = '<帳號> <姓名>' instead of a
    # trigram search for the letters of that name inside product descriptions.
    matches = merge_matches(vocab, V.resolve(qnorm))
    tsq = _tsquery_terms(qnorm, vocab.expand_terms)
    fsql, fp = _filter_sql(filters)

    # 問句裡的日期 -> 硬條件，而且**每一條通道都要吃到**。
    #
    # ★ 為什麼是接進 fsql，不是事後濾 results
    #   事後濾的話，RRF 先挑出全庫最相關的 10 筆、再把不在日期內的丟掉，
    #   剩下可能是 0 筆 —— 但語料裡明明有幾百筆同時符合。日期是使用者
    #   講出來的**篩選條件**，不是排序偏好，所以它必須在候選階段就成立。
    #   工程圖那條路不走這裡：圖的日期在 drawings.doc_modified_on，
    #   跟 parts 是兩張表（見下面 drawings 那一段）。
    tq = timerange.parse(qnorm)
    if tq and not _wants_drawing(qnorm):
        tsql, tparams = timerange.sql(tq, timerange.parts_column(tq))
        fsql += " AND " + tsql
        fp = list(fp) + tparams

    # Load/encode before reserving the query's DB connection.
    qvec = None
    vector_reason = "unavailable"
    # Explicit part lookup uses exact/base/neighbor channels. Semantic neighbors
    # only add work unless the user actually requested similar alternatives.
    needs_vector = not q["item_numbers"] or wants_similar(qnorm)
    try:
        # During cold background loading, serve lexical/structured results now.
        if not needs_vector:
            vector_reason = "explicit_part_number"
        elif embedding_ready() or not _model_loading.is_set():
            qvec = embed_query(qnorm)
            vector_reason = "used"
        else:
            vector_reason = "warming"
    except Exception as e:
        print(f"[retrieval] vector channel unavailable: {e}")

    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        ranked: dict[str, list[dict]] = {}
        ranked["item"], hit_codes = _by_item_number(cur, q["item_numbers"])

        # 客戶格式的完整料號（<客戶料號>）不會進 q["item_numbers"]，補這一路才不會
        # 「打了完整料號、料號就在庫裡、卻查不到」。併進 item 這一路而不是新增一路：
        # 它就是精確料號，本來就該拿同一個 6.0 權重，也省掉改 _rrf 的 weights。
        seen_exact = {r["item_number"] for r in ranked["item"]}
        ranked["item"] += [
            r for r in _by_exact_code(cur, [c for c in q["codes"] if c not in hit_codes])
            if r["item_number"] not in seen_exact
        ]

        # Which explicitly-named part numbers does the corpus simply not hold?
        # 基準刻意仍是 q["item_numbers"]：codes 裡有客戶料號、圖號、規格片段，
        # 它們沒命中 parts.item_number 是正常的，不是「查無此料號」。
        missing = [pn for pn in q["item_numbers"] if pn not in hit_codes]
        ranked["neighbor"] = _by_neighborhood(cur, missing)

        ranked["code"] = _by_customer_code(cur, q["codes"], fsql, fp)
        # "Z0000XX00000-1 的批號管控" names a part AND a field, and it is asking for
        # THAT row's value. The bare-field fallback (col IS NOT NULL) answers a
        # different question -- "which parts have a 批號管控", i.e. nearly all --
        # and filled most of the rendered rows with alphabetically-first junk.
        named_fields = [] if q["item_numbers"] else vocab.master_fields
        ranked["master"] = _by_master(cur, matches, named_fields, fsql, fp)
        ranked["plm"] = _by_plm(cur, vocab, tsq, fsql, fp)
        ranked["lexical"] = _by_lexical(cur, qnorm, tsq, fsql, fp)
        try:
            # An optional channel SQL timeout must not poison the transaction.
            with conn.transaction():
                ranked["vector"] = _by_vector(cur, qvec, fsql, fp) if qvec is not None else []
        except Exception as e:  # embedding model unavailable -> lexical still works
            ranked["vector"] = []
            print(f"[retrieval] vector channel unavailable: {e}")
        # Gated on the query actually naming a drawing concept (工程圖/材料表/BOM/...).
        # Without this, drawing_items.description -- free text like "LABEL:W=12mm
        # COLOR:WHITE" -- matches lexically/trigram against ordinary spec questions
        # that never asked about a drawing, and a coincidental hit outranks the
        # channels that were actually answering the question. Measured: "LABEL: W =
        # 12mm COLOR: WHITE在哪" (no mention of a drawing) pulled in unrelated
        # drawings and let a length_mm coincidence (12mm) outscore everything
        # else. Same degradation rule as the vector channel otherwise: before the
        # drawing tables exist, or with no drawings ingested, this goes quiet and
        # the other channels answer exactly as they did before it existed.
        wants_drawing = _wants_drawing(qnorm)
        dwg_confident: list[str] = []
        # 問句點名的編號（料號、客戶料號、圖號）。時間講法先剪掉：`2026-05` 長得
        # 完全像一個客戶料號，parse_query 會把它收進 codes——理由與下面
        # `codes_left` 那段完全相同，所以兩邊共用同一份計算。
        named_codes = [c for c in dict.fromkeys(
            list(q["item_numbers"]) + list(q["codes"]))
            if c in timerange.strip(qnorm)]
        # 這幾個編號各自「有沒有一份自己的圖」。dwg_code_hits 是那幾份圖的歸檔
        # 料號，dwg_code_seen 是真的命中的編號 —— 後者是 `_drawing_note` 判斷
        # 「使用者點名的圖到底在不在」的依據。
        dwg_code_hits: list[str] = []
        dwg_code_seen: set[str] = set()
        # 這幾個編號裡，哪幾個是**圖上材料表的一列**（端子、膠帶的廠商料號）而
        # 不是一份圖的編號。「沒有一份圖叫這個名字」對它們來說是正常的，不是
        # 「語料裡沒有」——見 drawing_content_codes 的說明。
        dwg_content_seen: set[str] = set()
        if wants_drawing:
            try:
                dwg_code_hits, dwg_code_seen = drawing_code_hits(cur, named_codes)
                dwg_content_seen = drawing_content_codes(cur, named_codes)
                ranked["drawing_code"] = _by_exact_code(cur, dwg_code_hits)
                ranked["drawing"] = _by_drawing(cur, qnorm, tsq, fsql, fp)
                ranked["drawing_pn"] = _by_drawing_pn(cur, qnorm, fsql, fp)
                ranked["drawing_desc"] = _by_drawing_desc(cur, qnorm, fsql, fp)
                dwg_confident = _by_drawing_confident(cur, qnorm)
            except Exception as e:  # noqa: BLE001
                ranked["drawing"] = ranked["drawing_pn"] = ranked["drawing_desc"] = []
                ranked["drawing_code"] = []
                print(f"[retrieval] drawing channel unavailable: {e}")
        else:
            ranked["drawing"] = ranked["drawing_pn"] = ranked["drawing_desc"] = []
            ranked["drawing_code"] = []

        # 符合這句話**硬性規格條件**的全庫筆數。
        #
        # ★ 為什麼一定要有這個數字
        #   例如「找有 PH4.2 2*6P HSG 的成品」：全庫可能有**上百筆**同樣符合，
        #   系統回前 10 筆，而使用者手上那顆排在幾十名外。
        #   使用者看到的是「它沒找到我說的東西」—— 他合理地把這當成幻覺。
        #   但檢索沒有錯，錯的是**沒有講母體有多大**：一旦答案寫成
        #   「共 N 筆符合，以下是相關度最高的 10 筆」，同一個結果就從
        #   「錯的答案」變成「對的答案 + 一個可以繼續操作的下一步」。
        #   排序檢索天生只給一個視窗，不講視窗大小就是在暗示視窗就是全部。
        #
        #   只在問句真的有**規格判別條件**時才算（長度/pitch/PIN 排列/接頭/AWG/GEN）。
        #   只有分類詞（Cable）時這個數字是幾萬，講出來沒有意義也白花一次 COUNT。
        attr_sql, attr_params, attr_labels = _attr_filters(q, vocab, matches)
        match_total = None
        if attr_sql and not q["item_numbers"] and any(
            q.get(k) for k in ("length_mm", "pitch_series", "pin_layout",
                               "conn_types", "awg", "gen")
        ):
            try:
                # 排除條件與 aggregate.run() 一致，否則同一句話問「有哪些」與
                # 「列出所有」會拿到兩個不同的總數。
                cur.execute(
                    "SELECT count(*) AS n FROM partrag.parts "
                    "WHERE NOT is_placeholder AND NOT is_obsolete AND "
                    + " AND ".join(attr_sql),
                    attr_params,
                )
                match_total = {"total": int(cur.fetchone()["n"]),
                               "conditions": attr_labels}
            except Exception as e:  # noqa: BLE001  少一個數字不該讓整題掛掉
                print(f"[retrieval] match_total unavailable: {e}")

        merged = _rrf(ranked)

        # An exact vendor code off a signed drawing is evidence of the same
        # calibre as an imported master value, so it takes a bonus in the same
        # band (master is 0.30). Without it the RRF weight alone loses to any
        # broad class the same sentence happens to name -- see _by_drawing_pn.
        dwg_pn_hits = {r["item_number"] for r in ranked.get("drawing_pn") or []}
        # 材料表描述字面命中同理，而且更需要這一份加權：這條通道命中的通常
        # 只有兩三筆，而同一句話裡的「端子」「工程圖」會讓分類洪水灌進來
        # 幾十筆 —— 光靠 RRF 權重壓不過（見 _by_drawing_desc 的實測）。
        dwg_desc_hits = {r["item_number"] for r in ranked.get("drawing_desc") or []}

        # 使用者打的字串**完全等於**這顆料號。這是整個管線裡證據等級最高的一種命中，
        # 但 RRF 權重擋不住類別洪水 —— 例如「XX-ABC-0001-HF 是什麼規格」：
        # 料號字串尾巴那個 `HF` 被讀成環保標準 HF，於是全庫所有 GC HF 的列每一筆
        # 都拿 +0.15，而精確命中只值 6.0/61 = 0.098，**被自己料號裡的兩個字母
        # 擠出前 10**。0.5 是刻意訂在所有單項加權之上（最高的 pitch/layout 是 0.40）：
        # 和「工程圖廠商料號 +0.35」同一個道理，能壓過類別洪水的是加權不是 RRF 權重。
        exact_pn = {c for c in (*q["codes"], *q["item_numbers"])}

        results = []
        for e in merged.values():
            row = e["row"]
            bonus, abonus_why = _attr_bonus(row, q, vocab, matches)
            if row["item_number"] in exact_pn:
                bonus += 0.5
                abonus_why.append("料號完全相符")
            if row["item_number"] in dwg_pn_hits:
                bonus += 0.35
                abonus_why.append("工程圖材料表用到該廠商料號")
            if row["item_number"] in dwg_desc_hits:
                bonus += 0.35
                abonus_why.append("工程圖材料表描述含問句中的關鍵詞")
            why = [_SOURCE_LABEL[s] for s in e["sources"]]
            results.append(
                {
                    **_jsonable(row),
                    "score": round(e["score"] + bonus, 5),
                    "match_reasons": why + abonus_why,
                }
            )

        results.sort(key=lambda r: -r["score"])

        # 使用者沒有說「相關 / 類似 / 替代」時，**不符合他點名條件的料號一律不端出去**。
        #
        # ★ 為什麼要這麼硬
        #   RRF 保證前 k 名一定被填滿 —— 就算全庫只有 2 筆真的符合，第 3 到第 10 名
        #   也會被相近的東西補齊，而那 8 筆在畫面上與前 2 筆長得一模一樣。
        #   實測「找有 PH4.2 2*6P HSG 的成品」回來的前幾名裡就有
        #   PIN 排列是 2*4P、pitch 是 PH3.0/PH4.2 混的料號，跟問句要的不是同一個東西。
        #   使用者要「類似的」時他會講；沒講就代表那些是雜訊，
        #   而「查不到」是一個完整的答案。
        #
        # ★ 濾完是空的就讓它空著（與工程圖那條路同一個原則）。
        #   `strict_dropped` 讓上層知道「不是沒查到，是查到的都不完全符合」，
        #   模型才講得出「沒有完全符合的，要不要放寬條件」而不是「查無此料」。
        strict = not wants_similar(qnorm) and not q["item_numbers"]
        strict_dropped = 0
        if strict and any(q.get(k) for k, _, _ in _STRICT_FIELDS):
            kept = [r for r in results if _satisfies(r, q)]
            strict_dropped = len(results) - len(kept)
            results = kept

        # Fetched AFTER the final sort, for the rows actually being returned --
        # a drawing is context for an answer, not a second ranking signal. Reading
        # it off the pre-bonus RRF order would attach drawings to rows the caller
        # never sees, and miss the ones it does.
        #
        # Gated the same as the ranking channels above: a query that never named a
        # drawing concept does not get a "工程圖來源" card attached just because its
        # top hit happens to have one on file. "Z0000XX00000-1 是什麼規格？" answers
        # from parts alone; "Z0000XX00000-1 的工程圖材料表" gets the card.
        # ★ 查圖的名單不只是融合後的前 k 名，還要**補上工程圖兩條通道自己命中的
        #   那幾筆**。這兩件事會分岔，而分岔的後果是「明明有圖卻說沒有」：
        #   實測「MCIO 線的工程圖有哪些」的 drawing 通道找得到圖，但同一句話裡的
        #   「線」「工程圖」把字面與向量通道灌滿了泛用 MCIO 料號，融合後前 10 名
        #   一份圖都沒有 —— drawings 是空的，回答變成「查不到任何工程圖」。
        #   通道命中的那幾筆放在名單後面（不改排序，只是不要漏查）。
        # ★ 但「補上」與「淹沒」只差一步，而這一步實測會出事
        #   高精度的兩條通道（材料表描述、廠商料號）命中時，它們**就是答案**：
        #   「找有香蕉頭端子的工程圖」只有極少數圖對得上，而同一句話裡
        #   的「端子」會把分類與字面通道灌成一整排連接器圖。兩者混在同一個
        #   lookup 裡，回來的是一整排圖、正確的那幾份排在後面 ——
        #   模型照著前幾份回答，講的是 PCIe/MCIO 線束，跟香蕉頭端子毫無關係。
        #   使用者沒有問「相似的圖」，就不該拿相似的圖填版面。
        #   所以精準通道有命中時，**只查它們**；沒有才退回融合結果那份名單。
        drawings: list[dict] = []
        # 這句話除了時間之外，還有沒有別的條件（料號、廠商碼、中文材料詞、
        # 分類、接頭、pitch、長度、AWG）。「有沒有其他條件」決定兩件事：
        #   1. 「最近」是排序需求還是語氣詞
        #   2. 日期是「整段時間都給我」還是「在這段時間裡找這個東西」
        # ★ 日期本身不算「內容條件」
        #   `2026-05` 長得完全像一個客戶料號（七個字元、有數字、有連字號），
        #   parse_query 會把它收進 q["codes"]。於是「2026-05 的工程圖」被判成
        #   「有內容條件」，走內容通道與日期取交集 —— 而內容通道拿 `2026-05`
        #   去比對客戶料號，一筆都沒有，答案變成 0 份（實際上該有整個月的圖）。
        #   判準：把時間講法整段剪掉之後，這個 code 還在不在句子裡。
        stripped = timerange.strip(qnorm)
        codes_left = [c for c in q["codes"] if c in stripped]
        has_content = bool(
            q["item_numbers"] or codes_left or _cjk_phrases(qnorm)
            or vocab.prod_l2 or q["conn_types"] or q["pitch_series"]
            or q["length_mm"] or q["awg"]
            or _latin_content(qnorm)
        )
        # 「最近」只有在**問句沒有其他條件**時才代表「給我最新的幾份」。
        # 「最近有沒有 MCIO 的工程圖」問的是 MCIO，時間只是語氣詞 —— 照排行榜
        # 回答的話，回來的五份可能一份 MCIO 都沒有，而且看起來像正經答案。
        # ★ 解析得出**時間區間**時，區間優先於「最近 N 筆」
        #   「最近三個月更新的工程圖」兩邊都會有意見：recent_n 讀到「最近」
        #   （沒有數量詞就給預設 5），timerange 讀到「三個月」的區間。
        #   前者回最新的 5 份，後者回那三個月的全部 —— 使用者要的顯然是後者，
        #   而且兩個答案都附著正確的檔名與日期，錯了也看不出來。
        #   「最近三筆工程圖」不會走到這裡：那句話 timerange 解不出區間（回 None），
        #   因為「筆」不是時間單位。
        recent = recent_n(qnorm) if (wants_drawing and tq is None) else None
        if recent and has_content:
            recent = None
        if recent:
            # 「最近三筆工程圖」：名單直接由更新時間決定，不經過融合排序 ——
            # 這一題問的是 drawings 這張表的順序，parts 的相關度與它無關。
            try:
                drawings = drawings_for(cur, recent_drawings(cur, recent))
            except Exception as e:  # noqa: BLE001
                print(f"[retrieval] recent drawings unavailable: {e}")
        elif wants_drawing:
            # ★ dwg_confident 跟 drawing_desc / drawing_pn 同一類：都是「這句話
            #   本身就是材料表描述的字面」，該跳過融合排序，不該跟一般候選混。
            #   少了它，抄一段材料表英文回來查（例如整段 WIRE:... 規格）只會走
            #   `_by_drawing` 的模糊排序，前面幾名就算分數天差地遠，也一律跟
            #   results[:top_k] 的雜訊混在同一份 lookup 裡 —— 使用者沒問「相似」
            #   卻拿到好幾張材料完全對不上的圖。
            # ★ 點名了某一份圖自己的編號時，那一份**就是**答案，而且要排在最前面
            #   `drawing_code` 比另外三條精準通道更精準（那三條是「材料表裡有這個
            #   東西」，這一條是「這份圖就叫這個編號」）。放在 exact 的最前面，
            #   drawings_for() 會照這個順序回，卡片第一張就是使用者點名的那份圖。
            exact = list(dwg_code_hits)
            exact += [r["item_number"] for ch in ("drawing_desc", "drawing_pn")
                      for r in (ranked.get(ch) or [])]
            exact += dwg_confident
            if exact:
                lookup = exact
            else:
                lookup = [r["item_number"] for r in results[:top_k]]
                lookup += [r["item_number"] for r in (ranked.get("drawing") or [])[:top_k]]
            # ★ 有日期時，交集要在**截斷之前**做
            #   `lookup` 上面取的是 results[:top_k] 與通道的前 top_k 名。材料表寫著
            #   LABEL WHITE 的圖很多，落在某一個月的只是其中幾份 ——
            #   先截成 10 名、再用日期去濾，那幾份幾乎一定不在裡面，答案變成 0。
            #   這跟料號那邊「日期要進 fsql 而不是事後濾 results」是同一個道理：
            #   日期是篩選條件，必須在候選階段成立。
            if tq:
                try:
                    dated_order = drawings_in_range(cur, tq, limit=1000)
                except Exception as e:  # noqa: BLE001
                    dated_order = []
                    print(f"[retrieval] drawings_in_range unavailable: {e}")
                dated = set(dated_order)
                if has_content:
                    # 內容通道的**全部**命中（不截斷）與日期取交集
                    cand = [r["item_number"]
                            for ch in ("drawing_desc", "drawing_pn", "drawing")
                            for r in (ranked.get(ch) or [])]
                    cand += [r["item_number"] for r in results]
                    lookup = [x for x in dict.fromkeys(cand) if x in dated]
                else:
                    # 只有日期、沒有別的條件：名單直接由日期決定。融合排序在這一題
                    # 沒有意義 —— 要的是「那段時間更新過的圖」，不是「最像的圖」。
                    #
                    # ★ 用 drawings_in_range() 回來的**那個順序**，不要繞過 set
                    #   舊寫法是 `lookup = [x for x in set(...)]`，而 set 的迭代順序
                    #   是 hash 順序 —— drawings_in_range 的 `ORDER BY
                    #   doc_modified_on DESC` 在這一行被整個丟掉。實測「三月跟
                    #   五月的工程圖」回來的第一份不是最新的那份，而
                    #   答案層只展開得下前幾份：使用者以為看到的是最新的幾張圖，
                    #   拿到的其實是 hash 序前幾名，畫面上完全看不出來。
                    lookup = dated_order
            try:
                # tq 傳下去：有日期時連同一顆料號的多份圖也要照日期篩，
                # 否則「2026 三月的 X 工程圖」會把 X 的六月版一起端出來。
                drawings = drawings_for(cur, list(dict.fromkeys(lookup)), tq)
            except Exception as e:  # noqa: BLE001
                print(f"[retrieval] drawings payload unavailable: {e}")

            # ★ 點名了編號、而那個編號沒有任何一份圖 -> 一份圖都不端出去
            #
            #   這是實測抓到的最危險的一種錯答，而且它不是模型的錯：
            #   「找ABC-0001-01工程圖」在 `drawing_code` 這條通道上線之前撈到的是
            #   **相鄰編號 ABC-0002-01 那份圖**（同一個客戶同系列的圖，token
            #   幾乎完全一樣，而 ts_rank 沒有 IDF）。系統把那張卡片送上去，模型照著
            #   卡片寫出「這是 ABC-0001-01 的工程圖」——
            #   材料表一字不差地照抄了另一份圖的內容。兩份圖的材料恰好同款，
            #   使用者沒有任何辦法從答案看出拿錯了圖。
            #
            #   所以判準放在檢索端，不是靠提示詞求模型自律：使用者點名的編號
            #   一個都沒有對應的圖時，這一題的正確答案是「沒有這份圖」，
            #   而不是「這裡有一份長得很像的圖」。條件刻意收得很緊——
            #   問句除了那些編號之外還有別的條件（中文材料詞、英文材料詞、
            #   分類、接頭…）時不套用，那時候相近的圖是在回答另一個問題
            #   （「有沒有用到香蕉頭端子的圖」）。
            #
            #   ★ 「別的條件」要把編號自己剪掉才算
            #     `_latin_content()` 抓的是英數詞，而 `ABC-0001-01` 本身就是一個
            #     英數詞 —— 不剪的話這個判斷永遠成立，整段防護等於沒寫（而且是
            #     安靜地沒寫）。
            other = [t for t in _latin_content(qnorm)
                     if not any(t in c or c in t for c in named_codes)]
            # ★ 這條編號如果是靠材料表廠商料號／描述字面命中的，不算「沒有對應的圖」
            #   `dwg_code_seen` 只認「這份圖自己的編號」（標題欄 PART NO. /
            #   CUSTOMER P/N），不包含「這個編號出現在材料表裡」——例如
            #   「哪份工程圖上有 ABC-0001-07 這個料號？」：`ABC-0001-07`
            #   是 `ABC-0001-XX` 那份圖 `P/N AND LENGTH` 表裡的多個廠商料號
            #   之一，不是那份圖自己的編號，`dwg_code_seen` 因此是空的，
            #   而這一題除了那個編號沒有別的內容詞——會被上面那組例外
            #   （`other`／中文詞／分類…）全部判定成「沒有」，於是這條防護
            #   把 `_by_drawing_pn`／`_by_drawing_desc` 這兩條**精準**通道剛好
            #   找到的正確答案原地清空。這兩條通道本來就是「這個編號/描述
            #   字面出現在材料表裡」的精確證據，跟 drawing_code 是同一等級
            #   的精準（RRF 權重同樣是 5.0），不該被這裡當成「沒找到」。
            precise_content_hit = bool(ranked.get("drawing_pn")
                                        or ranked.get("drawing_desc"))
            if named_codes and not dwg_code_seen and drawings and not (
                other or _cjk_phrases(qnorm) or precise_content_hit
                or vocab.prod_l2 or q["conn_types"] or q["pitch_series"]
            ):
                print(f"[retrieval] {named_codes} 沒有對應的工程圖，"
                      f"不端出另外 {len(drawings)} 份相近的圖")
                drawings = []

        # 問工程圖，就只回工程圖那幾筆。
        #
        # ★ 這一步同時解掉三個症狀，它們其實是同一個原因：
        #   1. 「Z0000XX00000-1的工程圖」會連它的變體與其他相近料號一起列出來，
        #      使用者要的那一筆明明已經精確命中了，旁邊還擺著一排無關料號。
        #   2. 那些列會一路進 llm.build_prompt()，prompt 膨脹好幾倍，
        #      而 gateway 是會 thinking 的模型 —— 整題要跑好幾分鐘，
        #      檢索本身不到一秒，時間全部花在讓模型讀它用不到的東西。
        #   3. 模型讀了那些列，就會拿它們的品名來補圖上沒有的欄位。
        #
        #   找不到圖、但問題有點名料號時，收斂成**那顆料號本身**：使用者要的是
        #   「這顆有沒有圖」，答案是沒有，旁邊擺一排鄰居只會讓模型改去比較
        #   它們的規格（回答有一半篇幅在列同系列的相近料號）。留著那顆本身，
        #   _drawing_note 才不會變成一句沒有主詞的話。
        #
        # ★ 收斂之後是空的就讓它空著，**不要退回原本那串排序結果**。
        #   舊版寫的是 `if narrowed: results = narrowed`，於是最該擋住的兩種
        #   情況剛好都漏掉了：問句點名的料號根本不在語料裡（narrowed 為空）、
        #   或問句沒點名任何料號（keep 為空）。這兩種都會原封不動地回 10 筆
        #   相似料號，前端與模型都會把它們當成「查到的工程圖」。
        #   「查不到」是一個完整的答案，相似料號要等使用者開口才給。
        if wants_drawing:
            keep: set[str] = set()
            if drawings:
                keep = {d["item_number"] for d in drawings}
                keep |= {d["dwg_part_no"] for d in drawings if d.get("dwg_part_no")}
            elif named_codes:
                # 點名了編號但沒有圖：留下那顆料號本身，_drawing_note 才有主詞。
                # ★ 名單要含**精確通道實際命中的料號**，不只是使用者打的字串
                #   使用者打 `ABC-0001-01`，語料裡那顆叫 `AA-BB-ABC-0001-01`
                #   （見 _by_exact_code 的後綴比對）。只留 q["item_numbers"] 的話
                #   keep 是空的，results 被清成 0 筆 —— 畫面上「這顆料號存在、
                #   只是沒有圖」變成「什麼都查不到」，兩件事的下一步不一樣。
                keep = set(q["item_numbers"])
                keep |= {r["item_number"] for r in ranked.get("item") or []}
            results = [r for r in results if r["item_number"] in keep]

    return {
        "query": query,
        "retrieval_modes": {"vector": vector_reason},
        "parsed": {k: v for k, v in q.items() if k != "norm" and v},
        "plm_vocab": {
            "prod_l1": vocab.prod_l1,
            "prod_l2": vocab.prod_l2,
            "eco_std": vocab.eco_std,
            "item_type": vocab.item_type,
            "expanded_terms": vocab.expand_terms,
        },
        "plm_notes": _plm_notes(vocab, matches, qnorm),
        "field_matches": [
            {"column": m.column, "label": m.label, "values": m.values, "via": m.via}
            for m in matches
        ],
        "not_found": missing,
        # {"total": N, "conditions": ["PH4.2", "PIN 排列 2*6P", "接頭 HSG"]}
        # 或 None（問句沒有規格判別條件時）。見上面計算它的那一段。
        "match_total": match_total,
        # 問句裡解析出來的時間條件（給回答與 hint 用）。None＝這句話沒問時間。
        "time_query": None if tq is None else {
            "field": tq.field,
            "explicit_field": tq.explicit_field,
            "spans": [{"start": str(s_.start), "end": str(s_.end), "label": s_.label}
                      for s_ in tq.spans],
            "describe": timerange.describe(tq, for_drawing=_wants_drawing(qnorm)),
        },
        # 嚴格模式濾掉了幾筆「查得到但不完全符合」的料號。>0 時上層要講
        # 「沒有完全符合的」而不是「查無此料」—— 兩句話的下一步不一樣。
        "strict_dropped": strict_dropped,
        "drawings": drawings,
        # 這批圖在問句的每一段時間區間裡各有幾份。答案層用它寫總結、
        # grounding 用它擋「說某一段沒有」的幻覺（見 drawing_buckets）。
        "drawing_buckets": drawing_buckets(tq, drawings),
        "drawing_note": _drawing_note(qnorm, named_codes, dwg_code_seen, drawings,
                                      dwg_content_seen),
        # 問句點名了、但語料裡沒有這一份圖的那幾個編號。grounding.check() 用它
        # 抓「照著別的圖講這個編號」——那是這條線上最像真的一種幻覺，光看
        # drawing_note 有沒有值抓不到（有兩顆料號、只有一顆有圖時也會有值）。
        # ★ 「圖上有這顆零件」不算「沒有這份圖」
        #   `ABC-0001-TPE` 是端子的廠商料號，它本來就不會是任何一份圖自己的編號。
        #   只看 dwg_code_seen 的話，「哪些工程圖用了 ABC-0001-TPE」這種問法每次都會
        #   被標成 absent，然後 grounding 的 wrong_drawing_subject 把正確答案
        #   判成張冠李戴。見 drawing_content_codes。
        "drawing_absent": [c for c in named_codes
                           if c not in dwg_code_seen and c not in dwg_content_seen]
                          if wants_drawing else [],
        "total_candidates": len(merged),
        "results": results[:top_k],
    }


NOT_COVERED = "目前系統尚未包含該資料"

# 使用者**主動要**近似的東西。有這些詞才可以把不完全符合條件的料號端出去。
_SIMILAR_RE = re.compile(
    r"相似|類似|类似|相關|相关|相近|接近|替代|代替|差不多|類同|类同|"
    r"同系列|其他|其它|還有|还有|推薦|推荐|建議|建议|SIMILAR|RELATED|ALTERNATIV"
)


def wants_similar(qnorm: str) -> bool:
    return bool(_SIMILAR_RE.search(qnorm))


# 嚴格模式下拿來過濾的欄位：(問句解析出來的鍵, parts 的欄位, 是不是陣列欄)。
#
# ★ 只收「講出來就一定是硬條件」的那幾個
#   pitch / PIN 排列 / 長度 / AWG / GEN / 接頭 —— 使用者打出這些數字時，
#   它們**就是**篩選條件，不是傾向。產品分類（Cable / Wire Harness）刻意不收：
#   那是分類不是規格，拿它過濾等於把「線束」兩個字變成硬條件。
_STRICT_FIELDS: tuple[tuple[str, str, bool], ...] = (
    ("pitch_series", "pitch_series", True),
    ("pin_layout", "pin_layout", True),
    ("length_mm", "length_mm", True),
    ("conn_types", "conn_types", True),
    ("awg", "awg", False),
    ("gen", "gen", False),
)


def _eq(a: Any, b: Any) -> bool:
    """數字用數值比、其他用字串比。長度在 DB 是 Decimal('600')、問句解析出來是
    600.0 —— 用字串比會永遠不相等，而且是安靜地不相等。"""
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def _satisfies(row: dict, q: dict) -> bool:
    """這一列是不是**每一個**問句點名的硬條件都符合。"""
    for key, col, is_array in _STRICT_FIELDS:
        want = q.get(key)
        if not want:
            continue
        got = row.get(col)
        if got is None or got == []:
            return False
        wants = want if isinstance(want, list) else [want]
        if is_array:
            if not any(_eq(g, w) for g in got for w in wants):
                return False
        elif not any(_eq(got, w) for w in wants):
            return False
    return True

# 問題裡出現這些字，才算是在問圖上的東西。既是「有沒有必要跑 _by_drawing /
# _by_drawing_pn / drawings_for」的閘門，也是「沒有圖要不要講清楚」的判斷 ——
# 兩件事共用同一個訊號，行為才會一致：問「材料表」沒圖會被告知，問規格
# 沒提到圖則整段工程圖邏輯都不會啟動。
#
# ★ 一律寫簡體，不是打錯字。這裡比對的是 qnorm，而 qnorm 一律經過
#   extract.normalize() 的 t2s 轉換（README「繁簡混用」那節的同一個原則）。
#   最初寫的是繁體「工程圖」，結果「哪些料號的工程圖用了 ABC-0001 端子？」
#   （這題明明打繁體）比對永遠落空 —— qnorm 裡的「圖」早就變成「图」，
#   閘門對著一個永遠不會出現的字比對，靜靜地把整條通道關掉。
_DRAWING_WORDS = ("工程图", "图面", "图纸", "材料表", "BOM", "标题栏", "TITLE BLOCK",
                  "零件表", "用料", "DRAWING",
                  # 講到第二句就不會再說「工程圖」三個字了 —— 使用者說的是
                  # 「這份圖的 REV」「圖上的 PART NO.」「圖是第幾張」。這些以前
                  # 全部掉回十一路檢索，回的是規格而不是圖。
                  # 每一個都含「图」但都不是「图号」，所以與 _DRAWING_FIELD_WORDS
                  # 的剪除規則不衝突：那一步先把「图号/工程图号」整個拿掉才比對。
                  "这份图", "那份图", "张图", "份图", "图上", "图里", "图中", "图档",
                  # 「圖是第幾張」「圖是哪一份」—— 圖當主詞的句子一定在講那份圖。
                  # 與「工程圖號是…」不衝突：图号/工程图号 在比對前已經被剪掉。
                  "图是")

# 先剪掉才比對的詞。**「工程圖號」是 parts.drawing_no 這個欄位，不是那份 PDF**
# —— 它字面上包含「工程圖」，於是整條工程圖路徑會被一句在問欄位的話啟動。
# 以前這只是多跑兩條沒有結果的通道，看不出症狀；「沒有圖就不給相似答案」上線
# 之後就會把一句只是在問「工程圖號」欄位的話的 results 整個清空。
_DRAWING_FIELD_WORDS = ("工程图号", "图号")


def _wants_drawing(qnorm: str) -> bool:
    """問題裡有沒有提到工程圖概念。

    ★ 這個閘門存在的理由是量出來的：`drawing_items.description` 是自由文字
      （`LABEL:W=12mm COLOR:WHITE...`），沒有這道閘門時，任何字面上湊巧像
      材料表描述的一般規格問句都會被 `_by_drawing` 撈進候選，而且湊巧命中
      `length_mm`（`12mm` 被裸長度規則誤判成線長）的料號還會拿到 +0.35 加權，
      把真正該回答這題的通道全部蓋掉。實測「LABEL: W = 12mm COLOR: WHITE在哪」
      （完全沒提到工程圖）就是這樣把無關的圖拉進結果、還讓一顆不相干的
      料號排到第一名。要求明確提到工程圖，才讓這整條路徑啟動。
    """
    up = qnorm.upper()
    for w in _DRAWING_FIELD_WORDS:
        up = up.replace(w, "")
    return any(w in up for w in _DRAWING_WORDS)


def wants_drawing(query: str) -> bool:
    """_wants_drawing() 的公開版：吃原始問句，自己做 normalize。

    給外部的路由自我檢查用（「這句話實際上走哪條路」）。
    包一層而不是讓呼叫端自己 parse_query 再叫私有函式，是為了讓「怎麼算
    normalize」只有一個地方說了算 —— 檢查用的正規化跟檢索時不一樣的話，
    自我檢查就會開始報假訊息。
    """
    return _wants_drawing(parse_query(query)["norm"])


def _drawing_note(qnorm: str, named: list[str], code_seen: set[str],
                  drawings: list[dict],
                  content_seen: set[str] | None = None) -> str | None:
    """問了工程圖、但該料號沒有圖時要講清楚的那句話。

    工程圖只覆蓋一小部分料號，所以「沒有工程圖」會是最常見的回答 ——
    這句話比另外兩種情況都重要。用既有的 NOT_COVERED 字串起頭，是因為 llm.py
    的 SYSTEM 已經教過模型看到它就照講、不要拿相近的資料代答；再寫一句新的
    等於要模型學兩套規則。

    ★ `named` 收的是**問句點名的所有編號**，不只 `q["item_numbers"]`
      這是「找ABC-0001-01工程圖」錯答的第二個成因（第一個見
      `drawing_code_hits`）。`ABC-0001-01` 不符合廠內的料號規則，
      `parse_query` 只會把它放進 `codes`、不會放進 `item_numbers` ——
      於是舊版的 `absent` 恆為空，`drawings` 又非空（撈到的是相鄰編號那份），
      這句提示整個不會產生。系統於是把一份**別的圖**交給模型，而且沒有附上
      任何一句「這不是你要的那份」。模型沒有辦法知道，使用者也沒有辦法知道。

    ★ `code_seen` 是「這個編號真的有一份自己的圖」的唯一依據
      不能改用「drawings 裡有沒有哪一份的欄位長得像這個編號」去猜：圖上印的是
      `BB-ABC-0001-01`、歸檔的是 `AA-BB-ABC-0001-01`、使用者打的是
      `ABC-0001-01`，三個字串沒有一個相等。判斷已經由
      `drawing_code_hits()` 在 SQL 端用同一套規則做過了，這裡只讀結果。
    """
    if not _wants_drawing(qnorm):
        return None
    # 「有沒有圖」要同時看歸檔料號與圖上印的料號，理由同 drawings_for()。
    # 只看 item_number 的話，查 Z0000XX00000-1 會一邊回傳一份 PART NO. 明明
    # 就是 Z0000XX00000-1 的圖、一邊說「該料號沒有工程圖」—— 同一則回答自相矛盾。
    covered = {d["item_number"] for d in drawings}
    covered |= {d["dwg_part_no"] for d in drawings if d.get("dwg_part_no")}
    covered |= set(code_seen)
    # ★ 「這個編號是圖上的一顆零件」也算有著落
    #   少了這一行，「哪些工程圖用了 ABC-0001-TPE」會在答對的同時附上
    #   「ABC-0001-TPE 該料號沒有工程圖」—— 同一則回答自相矛盾，而模型會照講。
    #   端子、膠帶這些材料表上的東西本來就不會有自己的一份圖。
    covered |= set(content_seen or ())
    absent = [pn for pn in named if pn not in covered]
    if drawings and not absent:
        return None
    # 問句沒點名料號時（「MCIO 線的工程圖有哪些」）沒有主詞可寫，就講這一句 ——
    # 寫成「查詢到的料號沒有工程圖」會變成在講某幾筆不存在的東西。
    who = "、".join(absent) if absent else None
    head = f"{who} {NO_DRAWING}" if who else "這個問題查不到任何工程圖"
    return (f"{NOT_COVERED}：{head}"
            f"（data/pdf/ 目前只收錄一小部分料號的工程圖）。"
            "不要拿其他料號的工程圖、相似料號或 parts 的品名代答。")


def _plm_notes(
    vocab: T.QueryVocab, matches: list[V.FieldMatch], qnorm: str
) -> list[dict[str, Any]]:
    """Explain PLM fields the query named.

    A term like DNT resolves to 公告大類, a field that cannot be derived from a
    part number and a description. Saying so beats returning the trigram noise
    the old pipeline produced for it.

    Three cases end in NOT_COVERED, and the distinction matters because the model
    is told to repeat that phrase verbatim rather than answer from a near miss:
      - the field exists in PLM but was never imported here (count 0)
      - the query named a person the corpus has never seen
      - the query named a value of a loaded field that holds no such value
    """
    loaded = V.column_counts()
    notes: list[dict[str, Any]] = []
    for p in vocab.plm_terms:
        note = dict(p)
        col = p["column"]
        if p["origin"] == "constant":
            note["answer"] = T.constant_answer(p["field"])
        elif p["origin"] == "import":
            n = loaded.get(col or "", 0)
            note["answer"] = (
                f"{p['field']} 已從 PLM 匯入 {n} 筆，可直接檢索"
                if n
                else f"{NOT_COVERED}：{p['field']} 尚未從 PLM 匯入語料庫。"
                "匯入方式：python -m src.enrich --skip-backfill "
                "--plm-csv <匯出.csv>"
            )
        else:
            note["answer"] = None
        notes.append(note)

    gap = V.creator_gap(qnorm, matches)
    if gap:
        notes.append(
            {
                "field": "建立人員",
                "value": "、".join(gap.names) or None,
                "origin": "import",
                "column": "created_by",
                "note": gap.reason,
                "answer": f"{NOT_COVERED}：{gap.message}",
            }
        )
    return notes


def get_part(item_number: str) -> dict | None:
    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"SELECT {SELECT_COLS} FROM partrag.parts WHERE item_number = %s",
            (item_number,),
        )
        row = cur.fetchone()
        if not row:
            return None
        cur.execute(
            f"""SELECT {SELECT_COLS} FROM partrag.parts
                WHERE base_pn = %s AND item_number <> %s ORDER BY item_number""",
            (row["base_pn"], item_number),
        )
        out = _jsonable(row)
        # Siblings share a base but ARE DIFFERENT PRODUCTS -- surfaced for
        # context, never merged.
        out["variants"] = [_jsonable(s) for s in cur.fetchall()]
        return out


def find_similar(item_number: str, top_k: int = 10) -> list[dict]:
    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT embedding FROM partrag.parts WHERE item_number = %s", (item_number,)
        )
        r = cur.fetchone()
        if not r or r["embedding"] is None:
            return []
        cur.execute(
            f"""SELECT {SELECT_COLS}, 1 - (embedding <=> %s) AS score
                FROM partrag.parts
                WHERE embedding IS NOT NULL AND item_number <> %s
                  AND NOT is_placeholder AND NOT is_obsolete
                ORDER BY embedding <=> %s, item_number LIMIT %s""",
            (r["embedding"], item_number, r["embedding"], top_k),
        )
        return [
            {**_jsonable(x), "score": round(float(x["score"]), 4)} for x in cur.fetchall()
        ]


def _build_facets() -> dict[str, Any]:
    """Filter options, including the PLM form's own fields so the UI can filter
    by 產品大類/中類/小類 / 料件類型 / 環保標準 the way the form is laid out."""
    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        # ★ 每個 ORDER BY n DESC 後面都要接一個唯一欄位（理由同 _by_vector）。
        #   統計這一組的後果跟檢索那邊不完全一樣：下面幾支帶 LIMIT，所以筆數
        #   打平時飄的不只是順序，而是**哪些值出現在清單裡**。`created_by` 的
        #   人數多於只取的前 40，第 40 與第 41 名同筆數時每次重算可能換人 ——
        #   而 /api/values/{column} 正是使用者用來判斷「查不到是因為沒這個人、
        #   還是這個人根本沒被匯入」的地方，清單本身會飄就答不了那個問題。
        cur.execute(
            """SELECT category, count(*) n FROM partrag.parts
               WHERE NOT is_placeholder GROUP BY 1 ORDER BY n DESC, category"""
        )
        cats = cur.fetchall()
        cur.execute(
            """SELECT prod_l1, prod_l2, prod_l3, count(*) n FROM partrag.parts
               WHERE NOT is_placeholder AND prod_l2 IS NOT NULL
               GROUP BY 1,2,3 ORDER BY n DESC, prod_l1, prod_l2, prod_l3"""
        )
        taxonomy = cur.fetchall()
        simple: dict[str, list] = {}
        for col in ("item_type", "eco_std", "plant", "announce_cat",
                    "created_by", "modified_by", "state"):
            cur.execute(
                f"""SELECT {col} AS v, count(*) n FROM partrag.parts
                    WHERE {col} IS NOT NULL GROUP BY 1 ORDER BY n DESC, v LIMIT 40"""
            )
            simple[col] = cur.fetchall()
        cur.execute(
            """SELECT v, count(*) n FROM partrag.parts, unnest(pitch_series) v
               GROUP BY 1 ORDER BY n DESC, v LIMIT 30"""
        )
        pitches = cur.fetchall()
        cur.execute(
            """SELECT v, count(*) n FROM partrag.parts, unnest(conn_types) v
               GROUP BY 1 ORDER BY n DESC, v LIMIT 40"""
        )
        conns = cur.fetchall()
        cur.execute(
            "SELECT DISTINCT awg FROM partrag.parts WHERE awg IS NOT NULL ORDER BY awg"
        )
        awgs = [r["awg"] for r in cur.fetchall()]
        cur.execute(
            "SELECT DISTINCT gen FROM partrag.parts WHERE gen IS NOT NULL ORDER BY gen"
        )
        gens = [r["gen"] for r in cur.fetchall()]
        cur.execute(
            """SELECT count(*) total,
                      count(*) FILTER (WHERE is_placeholder) ph,
                      count(*) FILTER (WHERE is_obsolete) obsolete,
                      count(embedding) emb,
                      count(prod_l2) classified,
                      count(*) FILTER (WHERE plm_synced) plm_synced,
                      count(created_by) has_creator,
                      min(created_on) first_created,
                      max(created_on) last_created
               FROM partrag.parts"""
        )
        stats = cur.fetchone()
    return {
        "categories": cats,
        "taxonomy": taxonomy,
        "pitch_series": pitches,
        "conn_types": conns,
        "item_type": simple["item_type"],
        "eco_std": simple["eco_std"],
        "plant": simple["plant"],
        "announce_cat": simple["announce_cat"],
        "created_by": simple["created_by"],
        "modified_by": simple["modified_by"],
        "state": simple["state"],
        "awg": awgs,
        "gen": gens,
        "corpus_constants": {
            T.field_label(col): val for col, (val, _why) in T.CORPUS_CONSTANTS.items()
        },
        "stats": stats,
    }


# 快取這一份的理由，與 src/values.py 的值索引一字不差：這些數字只有在
# ingest / enrich / write 動過語料時才會變，而那不會重啟伺服器 —— 所以短 TTL
# 勝過永久快取，也勝過不快取。
#
# ★ 為什麼非快取不可（量測，不是預防性最佳化）
#   _build_facets() 是十多個對整張 parts 表的 GROUP BY，暖/冷各要數百毫秒。
#   而它有兩個呼叫端是**狀態查詢**，不是使用者按下篩選：
#   main.health() 與 a2a._status()。任何看板輪詢那兩支，等於每輪把整個 parts
#   表數 13 遍，只為了回答「資料庫還活著嗎」。
#
# ★ TTL 而不是「寫入時失效」為主
#   ingest.py 的 build-then-swap 是在資料庫端換表，Python 這邊沒有任何掛鉤能知道
#   它發生了；排程跑 enrich 也是另一個行程。寫入端能主動失效的只有 write.py
#   那一條（見該模組），其餘一律靠 TTL 收斂。300 秒的代價是「剛匯入完，篩選選單
#   最多晚 5 分鐘出現新值」，而語料一天只變一次。
FACETS_TTL_SECONDS = float(os.getenv("FACETS_TTL_SECONDS", "300"))

_facets_cache: tuple[float, dict[str, Any]] | None = None


def facets() -> dict[str, Any]:
    """Filter options + corpus stats. Cached for FACETS_TTL_SECONDS.

    回傳的形狀與內容和 _build_facets() 完全相同 —— 這層只決定要不要重算。
    """
    global _facets_cache
    if _facets_cache is None or time.time() - _facets_cache[0] > FACETS_TTL_SECONDS:
        _facets_cache = (time.time(), _build_facets())
    return _facets_cache[1]


def invalidate_facets() -> None:
    """下一次 facets() 重算。write.py 改完語料會呼叫。"""
    global _facets_cache
    _facets_cache = None
