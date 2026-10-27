"""pandas structure report for Part-rag.csv.

Produces reports/structure_report.md + supporting CSVs. Re-runnable.
Run:  .venv/Scripts/python.exe analyze.py
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))
from extract import (  # noqa: E402
    categorize,
    extract_attrs,
    is_placeholder,
    normalize,
    split_item,
)

load_dotenv()
CSV = os.getenv("CSV_PATH", r"data/Part-rag.csv")
OUT = Path(__file__).parent / "reports"
OUT.mkdir(exist_ok=True)

TRAD = set("圖號標準製程線術傳導經構檢驗銀鍍錫銅顏繞護層規範產團體貼")
SIMP = set("图号标准制程线术传导经构检验银镀锡铜颜绕护层规范产团体贴")


def main() -> None:
    df = pd.read_csv(CSV, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    df["item_number"] = df["item_number"].str.strip()
    df["name"] = df["name"].str.strip()

    L: list[str] = []
    add = L.append
    add("# Part-rag.csv 結構分析報告\n")
    add(f"來源：`{CSV}`\n")

    # -- 1. basic integrity -------------------------------------------------
    add("## 1. 基本完整性\n")
    dup = int(df["item_number"].duplicated().sum())
    integrity = pd.DataFrame(
        {
            "指標": [
                "總列數",
                "欄位數",
                "item_number 唯一數",
                "item_number 重複數",
                "item_number 空值",
                "name 空值",
                "name 唯一數",
            ],
            "值": [
                len(df),
                df.shape[1],
                df["item_number"].nunique(),
                dup,
                int((df["item_number"] == "").sum()),
                int((df["name"] == "").sum()),
                df["name"].nunique(),
            ],
        }
    )
    add(integrity.to_markdown(index=False) + "\n")
    add(
        f"> `item_number` {'可' if dup == 0 else '**不可**'}作為主鍵"
        f"（重複 {dup} 筆）。\n"
    )

    # -- 2. name length -----------------------------------------------------
    ln = df["name"].str.len()
    add("\n## 2. name 長度分布（決定是否需要 chunking）\n")
    desc = ln.describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]).round(1)
    add(desc.to_frame("字元數").to_markdown() + "\n")
    add(
        f"\n> 中位數 {int(ln.median())} 字元、最長 {int(ln.max())}。"
        " **1 row = 1 chunk，不需切塊**；這是結構化紀錄檢索，不是文件 RAG。\n"
    )

    # -- 3. placeholder rows -------------------------------------------------
    df["is_placeholder"] = df["name"].map(is_placeholder)
    n_ph = int(df["is_placeholder"].sum())
    add("\n## 3. 佔號/垃圾列\n")
    add(
        pd.DataFrame(
            {
                "類別": ["佔號列(領號用等)", "實際可用列"],
                "列數": [n_ph, len(df) - n_ph],
                "占比": [f"{n_ph/len(df)*100:.1f}%", f"{(len(df)-n_ph)/len(df)*100:.1f}%"],
            }
        ).to_markdown(index=False)
        + "\n"
    )
    add("\n> 佔號列仍入庫並標記 `is_placeholder`，預設排除於語意檢索，仍可用料號精確查得。\n")

    real = df[~df["is_placeholder"]].copy()

    # -- 4. item_number structure -------------------------------------------
    add("\n## 4. item_number 結構\n")
    parts = df["item_number"].map(split_item)
    df["base_pn"] = [p[0] for p in parts]
    df["variant"] = [p[1] for p in parts]
    vc = df.groupby("base_pn").size().value_counts().sort_index()
    add(f"- 字元長度分布：`{df['item_number'].str.len().value_counts().sort_index().to_dict()}`\n")
    add(f"- base 料號數：{df['base_pn'].nunique()}，其中有多變體者：{int((vc.index > 1).sum() and (df.groupby('base_pn').size() > 1).sum())}\n")
    add(f"- 單一 base 最多變體數：{int(df.groupby('base_pn').size().max())}\n")
    add("\n**變體數分布**\n")
    add(vc.head(12).to_frame("base 數").rename_axis("變體數").to_markdown() + "\n")

    multi = df.groupby("base_pn").filter(lambda g: len(g) > 1)
    diff_names = multi.groupby("base_pn")["name"].nunique()
    add(
        f"\n> **關鍵：變體碼不是版本號。** 有多變體的 base 中，"
        f"{int((diff_names > 1).sum())}/{len(diff_names)} 個 base 底下的名稱彼此不同 —"
        " 同 base 的不同變體是**不同產品**，不可去重或合併。\n"
    )
    ex = multi[multi["base_pn"] == diff_names[diff_names > 1].index[0]]
    add("\n範例：\n\n```\n" + ex[["item_number", "name"]].head(5).to_string(index=False) + "\n```\n")

    # -- 5. prefix -> category ----------------------------------------------
    add("\n## 5. 前綴 × 產品類別關聯\n")
    real["prefix4"] = real["item_number"].str[:4]
    real["category"] = real["item_number"].map(lambda x: categorize(x)[1])
    cat = (
        real.groupby(["prefix4", "category"])
        .size()
        .reset_index(name="列數")
        .sort_values("列數", ascending=False)
        .head(20)
    )
    add(cat.to_markdown(index=False) + "\n")
    cat.to_csv(OUT / "prefix_category.csv", index=False, encoding="utf-8-sig")

    # -- 6. language mix -----------------------------------------------------
    add("\n## 6. 語言與繁簡\n")
    def lang(s: str) -> str:
        han = bool(re.search(r"[\u4e00-\u9fff]", s))
        lat = bool(re.search(r"[A-Za-z]", s))
        return "中英混合" if han and lat else "純中文" if han else "純英數" if lat else "其他"

    add(real["name"].map(lang).value_counts().to_frame("列數").to_markdown() + "\n")

    both = int(real["name"].map(lambda s: any(c in TRAD for c in s) and any(c in SIMP for c in s)).sum())
    add(f"\n- **同列同時含繁體與簡體：{both} 筆** → 入庫與查詢都必須做繁簡正規化\n")
    add(
        "\n"
        + pd.DataFrame(
            {
                "詞": ["圖號/图号", "標準/标准", "製程/制程", "線/线"],
                "繁體": [real["name"].str.contains(w).sum() for w in ["圖號", "標準", "製程", "線"]],
                "簡體": [real["name"].str.contains(w).sum() for w in ["图号", "标准", "制程", "线"]],
            }
        ).to_markdown(index=False)
        + "\n"
    )

    # -- 7. attribute coverage ----------------------------------------------
    add("\n## 7. 可抽取屬性覆蓋率（extract.py）\n")
    attrs = real["name"].map(extract_attrs).tolist()
    adf = pd.DataFrame(attrs)
    def _has_value(v: object) -> bool:
        # NB: pandas turns None into NaN for scalar columns, and bool(NaN) is
        # True -- checking truthiness alone reports 100% coverage for every
        # scalar attribute. Test for null explicitly first.
        if isinstance(v, (list, tuple)):
            return len(v) > 0
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return False
        if isinstance(v, str):
            return v != ""
        return bool(v)

    rows = []
    for c in adf.columns:
        n = int(adf[c].map(_has_value).sum())
        rows.append({"屬性": c, "有值列數": n, "覆蓋率": f"{n/len(real)*100:.1f}%"})
    covdf = pd.DataFrame(rows).sort_values("有值列數", ascending=False)
    add(covdf.to_markdown(index=False) + "\n")
    covdf.to_csv(OUT / "attribute_coverage.csv", index=False, encoding="utf-8-sig")

    # -- 8. near-duplicates: THE architectural driver ------------------------
    add("\n## 8. 近重複分析（決定必須用混合檢索的關鍵證據）\n")
    real["name_masked"] = real["name"].map(lambda s: re.sub(r"\d+", "#", normalize(s)))
    grp = real.groupby("name_masked").size().sort_values(ascending=False)
    ndup = int((grp > 1).sum())
    add(f"- **除數字外完全相同的名稱群組：{ndup} 組**\n")
    add(f"- 最大群組：{int(grp.max())} 筆\n")
    add(f"- 落在這些群組中的列數：{int(grp[grp > 1].sum())}（{grp[grp>1].sum()/len(real)*100:.1f}%）\n")
    add(
        "\n> 這些列在向量空間中幾乎重合，純語意檢索**無法區分**。"
        " 必須靠字面比對 + 數值屬性（長度/PIN/AWG）才能選出正確那一筆 —"
        " 這就是採用混合檢索 + reranker 的直接理由。\n"
    )
    top = grp[grp > 1].head(10).rename_axis("遮蔽數字後的名稱").to_frame("群組大小")
    add("\n" + top.to_markdown() + "\n")
    grp[grp > 1].to_csv(OUT / "near_duplicate_groups.csv", encoding="utf-8-sig")

    ex2 = real[real["name_masked"] == grp.index[1]][["item_number", "name"]].head(4)
    add("\n範例（同一群組內）：\n\n```\n" + ex2.to_string(index=False) + "\n```\n")

    # -- 9. acronyms ---------------------------------------------------------
    acr = Counter()
    for s in real["name"]:
        acr.update(re.findall(r"\b[A-Z]{2,10}\b", s))
    add("\n## 9. 領域縮寫\n")
    add(f"- 不重複大寫縮寫：**{len(acr)}** 個\n")
    add(f"- Top 20：`{[w for w, _ in acr.most_common(20)]}`\n")
    add("\n> MCIO / SlimSAS / ThinLine / HSG / FFC 等對通用 embedding 模型是 OOV，需 lexical 通道補足。\n")

    # -- 10. conclusion ------------------------------------------------------
    add("\n## 10. 架構結論\n")
    add(
        "| 觀察 | 架構決策 |\n|---|---|\n"
        "| 每列極短（中位數 %d 字元） | 不做 chunking，1 row = 1 chunk |\n"
        "| %d 組僅差數字的近重複 | 必須混合檢索 + cross-encoder 重排 |\n"
        "| %d 個領域縮寫為 OOV | 必須有 tsvector / pg_trgm 字面通道 |\n"
        "| %d 筆繁簡混用 | 入庫與查詢共用 OpenCC 正規化 |\n"
        "| item_number 全唯一且不透明 | 精確比對短路優先 |\n"
        "| %.1f%% 為佔號列 | 標記後預設排除語意檢索 |\n"
        % (int(ln.median()), ndup, len(acr), both, n_ph / len(df) * 100)
    )

    (OUT / "structure_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[ok] {OUT/'structure_report.md'}")
    print(f"[ok] {OUT/'prefix_category.csv'}")
    print(f"[ok] {OUT/'attribute_coverage.csv'}")
    print(f"[ok] {OUT/'near_duplicate_groups.csv'}")


if __name__ == "__main__":
    main()
