"""Load Part-rag.csv -> partrag.parts, with BGE-M3 embeddings.

Run:  .venv/Scripts/python.exe -m src.ingest
      .venv/Scripts/python.exe -m src.ingest --resume     (skip existing rows)
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from tqdm import tqdm

from . import taxonomy as T
from .extract import (
    build_search_text,
    categorize,
    extract_attrs,
    is_placeholder,
    normalize,
    split_item,
)

load_dotenv()
PG_DSN = os.environ["PG_DSN"]
CSV = os.getenv("CSV_PATH", "data/Part-rag.csv")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
BATCH = int(os.getenv("EMBED_BATCH", "32"))
MAX_SEQ = int(os.getenv("EMBED_MAX_SEQ", "256"))
# 模型的 revision 不在這裡定義：載入一律走 retrieval.load_embedder，設定跟著它走。
# 這支是**產生**語料向量的那一端，跟查詢端用同一份載入邏輯正是重點所在。

COLS = (
    "item_number name name_norm search_text plm_text base_pn variant prefix4 "
    "category prod_l1 prod_l2 prod_l3 tax_source item_type eco_std "
    "item_class item_subclass unit "
    "is_placeholder length_mm pin_counts pin_layout pins_total pins_row "
    "pitch_series conn_types has_connector is_coaxial rated_voltage "
    "awg gen pitch_mm speed_g color "
    "customer_pn drawing_no conn_a conn_b has_to_pattern embedding"
).split()


def build_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for item, name in tqdm(
        zip(df["item_number"], df["name"]), total=len(df), desc="extract", unit="row"
    ):
        nn = normalize(name)
        base, variant = split_item(item)
        p4, cat = categorize(item)
        a = extract_attrs(name)
        # PLM 產品大類/中類/小類 plus the form's wording, folded into search_text
        # so a query typed off the form reaches rows whose description never
        # uses those words. See src/taxonomy.py.
        c = T.classify(item, nn)
        plm_text = T.build_plm_text(c, a)
        rows.append(
            {
                "item_number": item,
                "name": name,
                "name_norm": nn,
                "search_text": build_search_text(nn, plm_text),
                "plm_text": plm_text,
                "base_pn": base,
                "variant": variant,
                "prefix4": p4,
                "category": cat,
                "prod_l1": c.l1 if c.l1 != "其他" else None,
                "prod_l2": c.l2 if c.l2 != "其他" else None,
                "prod_l3": c.l3 if c.l3 != "其他" else None,
                "tax_source": c.source,
                "item_type": a["item_type"],
                "eco_std": a["eco_std"],
                "is_placeholder": is_placeholder(name),
                "length_mm": a["length_mm"],
                "pin_counts": a["pin_counts"],
                "pin_layout": a["pin_layout"],
                "pins_total": a["pins_total"],
                "pins_row": a["pins_row"],
                "pitch_series": a["pitch_series"],
                "conn_types": a["conn_types"],
                "has_connector": a["has_connector"],
                "is_coaxial": a["is_coaxial"],
                "rated_voltage": a["rated_voltage"],
                "awg": a["awg"],
                "gen": a["gen"],
                "pitch_mm": a["pitch_mm"],
                "speed_g": a["speed_g"],
                "color": a["color"],
                "customer_pn": a["customer_pn"],
                "drawing_no": a["drawing_no"],
                "conn_a": a["conn_a"],
                "conn_b": a["conn_b"],
                "has_to_pattern": a["has_to_pattern"],
                "embedding": None,
                # asserted from the PLM form, not read from it -- see
                # taxonomy.CORPUS_CONSTANTS
                **{col: val for col, (val, _) in T.CORPUS_CONSTANTS.items()},
            }
        )
    return rows


def embed(rows: list[dict]) -> None:
    """Embed non-placeholder rows only.

    Placeholder rows (領號用 reservation stubs, a sizeable share of the corpus) are
    reachable by exact item_number lookup and are excluded from semantic search
    anyway, so embedding them would burn compute for nothing.
    """
    from .retrieval import EMBED_REVISION, embed_device, load_embedder

    # 鎖 revision + 先試本機快取，與查詢端同一份邏輯（見 retrieval.load_embedder）。
    # 執行緒數由那支設在 onnxruntime 的 SessionOptions 上。
    device = embed_device()
    model = load_embedder(device)
    print(f"[embed] model={EMBED_MODEL}@{EMBED_REVISION[:8]} device={device} "
          f"threads={os.cpu_count() or 4}")
    # BGE-M3 defaults to an 8192-token window. Part names are only a few hundred
    # chars at most, and attention cost is quadratic in sequence length -- capping this is
    # the single biggest speedup available on CPU, with zero quality loss.
    model.max_seq_length = MAX_SEQ

    targets = [r for r in rows if not r["is_placeholder"]]
    print(f"[embed] {len(targets)} rows (skipping {len(rows)-len(targets)} placeholders)")
    texts = [r["name_norm"] for r in targets]

    vecs = model.encode(
        texts,
        batch_size=BATCH,
        normalize_embeddings=True,  # cosine via inner product
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    for r, v in zip(targets, vecs):
        r["embedding"] = v


# --------------------------------------------------------------------------
# 全量重灌：先建好一張新表，再原子換過去
#
# ★ 舊做法為什麼不能在線上跑
#   TRUNCATE partrag.parts + DROP INDEX ... hnsw + 插全量 + 重建 HNSW。
#   TRUNCATE 一 commit，線上使用者就看到一個**空庫**；接著幾分鐘沒有向量索引，
#   dense 通道全表掃描。整段過程對使用者是「系統壞了」，而不是「正在更新」。
#
# ★ 新做法
#   所有工作都在 partrag.parts__build 上做（插入、建索引、ANALYZE），正式表
#   完全沒被碰過，查詢照舊。最後在一個 transaction 內把兩張表對調 —— DDL 在
#   PostgreSQL 是 transactional，所以使用者看到的是「上一秒舊資料、下一秒新資料」，
#   沒有中間狀態。切換本身要等現有查詢放掉 lock，實測是毫秒級。
#
# ★ 索引定義從資料庫讀，不在這裡重寫一份
#   正式表有十幾個索引（HNSW / GIN trgm / GIN 陣列 / btree / 部分索引）。
#   在這裡抄一份的話，schema 加一個索引、重灌一次就少一個，而症狀只是
#   「某類查詢忽然變慢」。所以是 pg_indexes 讀正式表的 indexdef，把表名換掉再跑。
BUILD = "parts__build"
OLD = "parts__old"


def _index_defs(cur, table: str) -> list[tuple[str, str]]:
    """(indexname, indexdef)，不含 PRIMARY KEY —— 它由 LIKE INCLUDING ALL 帶過去。"""
    cur.execute(
        """SELECT i.indexname, i.indexdef
           FROM pg_indexes i
           WHERE i.schemaname = 'partrag' AND i.tablename = %s
             AND NOT EXISTS (
                 SELECT 1 FROM pg_constraint c
                 WHERE c.conname = i.indexname AND c.contype IN ('p', 'u'))
           ORDER BY i.indexname""",
        (table,),
    )
    return [(r[0], r[1]) for r in cur.fetchall()]


def _all_index_names(cur, table: str) -> list[str]:
    """該表的**所有**索引名，含 PRIMARY KEY。

    _index_defs 刻意排除 PK（重建時不需要，LIKE INCLUDING ALL 已經帶了一個）；
    但換表時 PK 也得讓位 —— 舊表 rename 之後它的 PK 索引仍然叫 parts_pkey，
    佔著那個名字，新表的 parts__build_pkey 就改不回來（實測：
    relation "parts_pkey" already exists，整個 swap 在最後一步炸掉）。
    索引不會隨著表 rename 而改名，這是兩件獨立的事。
    """
    cur.execute(
        "SELECT indexname FROM pg_indexes "
        "WHERE schemaname = 'partrag' AND tablename = %s ORDER BY indexname",
        (table,),
    )
    return [r[0] for r in cur.fetchall()]


def _prepare_build(cur) -> list[tuple[str, str]]:
    """建空的 build 表（結構、預設值、GENERATED 欄位、PK 全帶過來），回傳待建索引。

    索引先不建：HNSW 與 GIN 在**已經填好**的表上一次建立，比邊插邊更新快得多，
    這也是舊做法要先 DROP INDEX 的原因。差別只在於這裡拿掉的是新表的索引，
    而不是線上那張表的。
    """
    live_idx = _index_defs(cur, "parts")
    cur.execute(f"DROP TABLE IF EXISTS partrag.{BUILD}")
    cur.execute(
        f"CREATE TABLE partrag.{BUILD} (LIKE partrag.parts INCLUDING ALL)"
    )
    # LIKE INCLUDING ALL 把索引也複製了（名字自動產生）。全部拿掉，插完再用
    # 正式表的定義重建 —— 這樣建出來的索引名稱、參數、WHERE 子句都與正式表一致。
    for name, _ in _index_defs(cur, BUILD):
        cur.execute(f"DROP INDEX partrag.{name}")
    return live_idx


def _finish_build(cur, live_idx: list[tuple[str, str]]) -> None:
    for name, ddl in live_idx:
        # indexdef 形如 CREATE INDEX parts_real ON partrag.parts USING ...
        # 換成 build 表與暫時的索引名（swap 時再改回正式名）。
        sql = ddl.replace(
            f" ON partrag.parts ", f" ON partrag.{BUILD} "
        ).replace(f"INDEX {name} ", f"INDEX {name}__b ", 1)
        print(f"[build] index {name}")
        cur.execute(sql)
    cur.execute(f"ANALYZE partrag.{BUILD}")


def _swap(cur, expect_rows: int, min_ratio: float = 0.90) -> None:
    """把 build 表換成正式表。整段在呼叫方的 transaction 裡，要嘛全成要嘛全不動。

    ★ 先擋掉「換上一張殘缺的表」
      重灌途中 CSV 讀壞、embedding 中斷、--limit 忘了拿掉，build 表都會是
      「看起來正常、只是少了很多列」。那種表換上去之後，症狀是使用者查不到
      東西而系統一切正常 —— 比整個掛掉難查得多。所以低於正式表列數的
      min_ratio 就中止（第一次建庫時正式表是空的，這個檢查自然通過）。
    """
    cur.execute(f"SELECT count(*) FROM partrag.{BUILD}")
    n_build = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM partrag.parts")
    n_live = cur.fetchone()[0]
    if n_build != expect_rows:
        raise RuntimeError(
            f"build 表有 {n_build} 列，但這一輪應該寫入 {expect_rows} 列 —— 中止換表"
        )
    if n_live and n_build < n_live * min_ratio:
        raise RuntimeError(
            f"build 表只有 {n_build} 列，正式表有 {n_live} 列"
            f"（不到 {min_ratio:.0%}）—— 中止換表。"
            f"確認過是刻意縮減的話用 --min-ratio 放寬。"
        )

    # 舊表與新表的索引撞名，而索引**不會**隨著表 rename 改名。所以順序是：
    # 舊的全部讓位（含 PK）-> 新的全部改成正式名 -> 丟掉舊表（連帶它的索引）。
    old_names = _all_index_names(cur, "parts")
    build_names = _all_index_names(cur, BUILD)

    cur.execute(f"ALTER TABLE partrag.parts RENAME TO {OLD}")
    for name in old_names:
        cur.execute(f"ALTER INDEX partrag.{name} RENAME TO {name}__o")

    cur.execute(f"ALTER TABLE partrag.{BUILD} RENAME TO parts")
    for name in build_names:
        # _finish_build 建的是 <正式名>__b；PK 是 LIKE 自動命名的 parts__build_pkey。
        target = name[:-3] if name.endswith("__b") else name.replace(BUILD, "parts")
        if target != name:
            cur.execute(f"ALTER INDEX partrag.{name} RENAME TO {target}")

    cur.execute(f"DROP TABLE partrag.{OLD}")
    after = _all_index_names(cur, "parts")
    missing = set(old_names) - set(after)
    if missing:
        # 索引少一個不會報錯，只會讓某類查詢忽然變慢 —— 在這裡擋下來，
        # 整個 swap 一起 rollback，比事後從查詢延遲反推容易得多。
        raise RuntimeError(f"換表後少了索引：{sorted(missing)} —— 已回復，不換表")
    print(f"[swap] 已換表：{n_live} 列 -> {n_build} 列，{len(after)} 個索引名稱一致")


def write(rows: list[dict], resume: bool, swap: bool = True,
          min_ratio: float = 0.90) -> None:
    with psycopg.connect(PG_DSN, autocommit=False) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            if resume:
                # 增量：目標就是「只補沒有的列」，沒有中斷服務的問題，也不該換表。
                cur.execute("SELECT item_number FROM partrag.parts")
                have = {r[0] for r in cur.fetchall()}
                rows = [r for r in rows if r["item_number"] not in have]
                print(f"[write] resume: {len(rows)} new rows")
                if not rows:
                    return
                _insert_into(cur, "parts", rows)
                conn.commit()
                cur.execute("ANALYZE partrag.parts")
                conn.commit()
                return

            if not swap:
                # 舊行為，只在明確要求時走：會讓線上先看到空庫。
                print("[write] --in-place：TRUNCATE 現有資料（線上服務會中斷）")
                cur.execute("TRUNCATE partrag.parts")
                cur.execute("DROP INDEX IF EXISTS partrag.parts_embedding_hnsw")
                _insert_into(cur, "parts", rows)
                conn.commit()
                print("[write] rebuilding HNSW index (this takes a few minutes)")
                cur.execute(
                    "CREATE INDEX parts_embedding_hnsw ON partrag.parts "
                    "USING hnsw (embedding vector_cosine_ops) "
                    "WITH (m=16, ef_construction=64)"
                )
                conn.commit()
                cur.execute("ANALYZE partrag.parts")
                conn.commit()
                return

            print(f"[write] build-then-swap：先寫進 partrag.{BUILD}，正式表不受影響")
            live_idx = _prepare_build(cur)
            conn.commit()
            _insert_into(cur, BUILD, rows)
            conn.commit()
            _finish_build(cur, live_idx)
            conn.commit()
            _swap(cur, expect_rows=len(rows), min_ratio=min_ratio)
            conn.commit()


def _insert_into(cur, table: str, rows: list[dict]) -> None:
    ph = ",".join(["%s"] * len(COLS))
    sql = (
        f"INSERT INTO partrag.{table} ({','.join(COLS)}) VALUES ({ph}) "
        "ON CONFLICT (item_number) DO NOTHING"
    )
    CH = 2000
    for i in tqdm(range(0, len(rows), CH), desc=f"insert->{table}", unit="chunk"):
        cur.executemany(sql, [[r[c] for c in COLS] for r in rows[i : i + CH]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: only N rows")
    ap.add_argument("--in-place", action="store_true",
                    help="舊行為：TRUNCATE 現有表再插入（線上服務會中斷）")
    ap.add_argument("--min-ratio", type=float, default=0.90,
                    help="換表前的列數下限（相對於現有正式表）")
    args = ap.parse_args()

    # --limit 是 debug 用的，換上去等於把整個庫縮成幾百列。要它就得自己講明白。
    if args.limit and not args.resume and not args.in_place:
        raise SystemExit(
            "[ingest] --limit 會讓 build 表只有幾百列，換表後整個庫就只剩那幾列。\n"
            "  要 debug 請加 --resume（增量插入，不換表），\n"
            "  真的想用少量資料重建整個庫就明確加 --in-place --limit N。"
        )

    df = pd.read_csv(CSV, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    df["item_number"] = df["item_number"].str.strip()
    df["name"] = df["name"].str.strip()
    df = df[df["item_number"] != ""]
    if args.limit:
        df = df.head(args.limit)
    print(f"[load] {len(df)} rows from {CSV}")

    rows = build_rows(df)
    embed(rows)
    write(rows, args.resume, swap=not args.in_place, min_ratio=args.min_ratio)

    with psycopg.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*), count(embedding), count(*) FILTER (WHERE is_placeholder) "
            "FROM partrag.parts"
        )
        total, emb, ph = cur.fetchone()
        print(f"[done] rows={total} embedded={emb} placeholders={ph}")


if __name__ == "__main__":
    main()
