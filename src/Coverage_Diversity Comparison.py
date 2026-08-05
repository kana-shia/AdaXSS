import os
import re
import itertools
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from datetime import datetime


# =====================================================
# Config
# =====================================================
INPUT_CSVS = [

    r"res/stats/stage3_filter_payload_Claude.csv",
    r"res/stats/stage3_filter_payload_CRS_Audit_Log.csv",
    r"res/stats/stage3_filter_payload_CRS_Reference.csv",

    r"res/stats/stage3_filter_payload_LLM_Driven.csv",
    #r"res/stats/stage3_filter_payload_DDQN.csv",
    
    #r"res/stats/stage3_filter_payload_Loop_1.csv",
    #r"res/stats/stage3_filter_payload_Loop_2.csv",
    #r"res/stats/stage3_filter_payload_Loop_3.csv",
    

]

OUTPUT_DIR = r"res/stats/cluster_compare_all_sources"

BYPASS_LABELS = {"pass", "bypass"}
BLOCK_LABELS = {"block"}
FILTER_MODE = "all" # 使用來源全部的樣本(pl+ml)  

SAMPLE_SIZE =  366 #指定抽樣數量；若任一來源不足就中止

NGRAM_RANGE = (3, 5) 

MIN_DF = 1
NORM = "l2"

LOWERCASE = False
ENABLE_SAFE_NORMALIZATION = False
ENABLE_AGGRESSIVE_NORMALIZATION = False

DEDUP_WITHIN_SOURCE_BY_NORMALIZED_PAYLOAD = False

RANDOM_SEEDS = list(range(1, 2))   # 預設抽 100 個隨機種子
THRESHOLDS = [round(i * 0.05, 2) for i in range(21)]


# =====================================================
# Utils
# =====================================================

def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def source_style(source_file: str):
    """回傳 (legend_order, legend_label, color)。"""
    base = os.path.basename(str(source_file)).lower()

    # 固定顯示順序與顏色：紅、金、綠、藍
    rules = [
        # 基準：Claude + CRS Reference
        ("stage3_filter_payload_crs_reference", 0, "Claude + CRS Reference", "red"),
        ("stage3_all_final_combined_1896_now", 0, "Claude + CRS Reference", "red"),
        ("stage3_ref_think", 0, "Claude + CRS Reference", "red"),
        ("stage3_offical_no_think", 0, "Claude + CRS Reference", "red"),

        # 多輪生成：R1、R2、R3
        ("stage3_filter_payload_loop_1", 1, "Claude + CRS Reference (R1)", "gold"),
        ("stage3_condition_results_20260601_134846_reloop1", 1, "Claude + CRS Reference (R1)", "gold"),
        ("stage3_filter_payload_loop_2", 2, "Claude + CRS Reference (R2)", "green"),
        ("stage3_condition_results_20260601_215323_reloop2", 2, "Claude + CRS Reference (R2)", "green"),
        ("stage3_filter_payload_loop_3", 3, "Claude + CRS Reference (R3)", "blue"),
        ("stage3_condition_results_20260603_195710_reloop3", 3, "Claude + CRS Reference (R3)", "blue"),

        # 不同生成／參考策略比較
        ("stage3_filter_payload_crs_audit_log", 1, "Claude + CRS Audit Log", "gold"),
        ("stage3_all_final_combined_2428_now", 1, "Claude + CRS Audit Log", "gold"),
        ("stage3_filter_payload_claude", 2, "Claude", "green"),
        ("stage3_no_ref_think", 2, "Claude", "green"),
        ("stage3_filter_payload_llm_driven", 3, "LLM-Driven", "blue"),
        ("stage3_all_final_combined_llmdriven_now", 3, "LLM-Driven", "blue"),
        ("205144_llm_driven", 3, "LLM-Driven", "blue"),

        # 其他方法
        ("stage3_filter_payload_ddqn", 4, "DDQN", "purple"),
        ("stage3_all_final_combined_ddqn_now", 4, "DDQN", "purple"),
    ]

    for token, order, label, color in rules:
        if token in base:
            return order, label, color

    fallback = re.sub(r"\.csv$", "", os.path.basename(str(source_file)), flags=re.IGNORECASE)
    return 999, fallback, "black"


def pretty_source_name(source_file: str) -> str:
    return source_style(source_file)[1]


def source_color(source_file: str) -> str:
    return source_style(source_file)[2]

def normalize_control_chars(s: str) -> str:
    """
    將控制字元統一為單一空白。
    保留一般可見字元，不做 decode。
    """
    s = re.sub(r'[\x00-\x1f\x7f]+', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def normalize_tag_name_case(s: str) -> str:
    """
    只統一 HTML tag 名稱大小寫，不影響其他內容。
    例如: <IMG ...> -> <img ...>、</ScRiPt> -> </script>
    """
    def repl(match):
        full = match.group(0)
        tag = match.group(1)
        return full.replace(tag, tag.lower(), 1)

    return re.sub(r'<\s*/?\s*([a-zA-Z][\w:-]*)', repl, s)


def normalize_event_handler_case(s: str) -> str:
    """
    只統一事件屬性名稱大小寫。
    例如: ONERROR / OnLoad -> onerror / onload
    不改引號內 JavaScript 內容。
    """
    return re.sub(
        r'\bon([a-zA-Z]+)\b',
        lambda m: 'on' + m.group(1).lower(),
        s
    )

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)


def slugify_source_name(name: str) -> str:
    name = os.path.basename(str(name))
    name = re.sub(r"\.csv$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")
    return name.lower()


def get_crs_status(row) -> str:
    if "crs" in row and pd.notna(row["crs"]) and str(row["crs"]).strip() != "":
        return str(row["crs"]).strip().lower()

    if "crs_bypass" in row and pd.notna(row["crs_bypass"]):
        val = row["crs_bypass"]

        if isinstance(val, bool):
            return "bypass" if val else "block"

        if isinstance(val, (int, np.integer)):
            return "bypass" if val == 1 else "block"

        if isinstance(val, str):
            v = val.strip().lower()
            if v in {"true", "1", "yes", "bypass", "pass"}:
                return "bypass"
            if v in {"false", "0", "no", "block"}:
                return "block"

    return ""


def strip_redundant_outer_parens(s: str, max_rounds: int = 5) -> str:
    s = s.strip()

    def is_fully_wrapped_by_parens(text: str) -> bool:
        if len(text) < 2 or text[0] != "(" or text[-1] != ")":
            return False

        depth = 0
        for i, ch in enumerate(text):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(text) - 1:
                    return False
            if depth < 0:
                return False
        return depth == 0

    rounds = 0
    while rounds < max_rounds and is_fully_wrapped_by_parens(s):
        s = s[1:-1].strip()
        rounds += 1

    return s


def normalize_html_entity_case(s: str) -> str:
    s = re.sub(r'&#X([0-9A-Fa-f]+);?', lambda m: f'&#x{m.group(1).lower()};', s)
    s = re.sub(r'&#([0-9]+);?', lambda m: f'&#{m.group(1)};', s)
    s = re.sub(r'&([A-Za-z][A-Za-z0-9]+);?', lambda m: f'&{m.group(1).lower()};', s)
    return s


def normalize_percent_encoding_case(s: str) -> str:
    return re.sub(r'%([0-9A-Fa-f]{2})', lambda m: f'%{m.group(1).upper()}', s)


def normalize_self_closing_tags(s: str) -> str:
    return re.sub(r'<\s*([a-zA-Z][\w:-]*)([^<>]*?)\s*/\s*>', r'<\1\2>', s)


def normalize_tag_spacing(s: str) -> str:
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'<\s+', '<', s)
    s = re.sub(r'</\s+', '</', s)
    s = re.sub(r'\s*=\s*', '=', s)
    s = re.sub(r'\s+>', '>', s)
    return s


def normalize_quote_style(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("‘", "'").replace("’", "'")

    if len(s) >= 2 and (s[0] == s[-1]) and s[0] in {"'", '"'}:
        s = s[1:-1].strip()

    return s


def normalize_trailing_semicolons(s: str) -> str:
    s = s.strip()
    s = re.sub(r'^[;\s]+', '', s)
    s = re.sub(r'[;\s]+$', '', s)
    return s


def normalize_payload(
    payload: str,
    lowercase: bool = False,
    enable_safe_normalization: bool = True,
    enable_aggressive_normalization: bool = False,
) -> str:
    s = safe_str(payload)
    s = s.strip()

    # -----------------------------
    # Safe normalization
    # 幾乎不改攻擊語意，只清理雜訊
    # -----------------------------
    if enable_safe_normalization:
        s = normalize_control_chars(s)
        s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        s = re.sub(r"\s+", " ", s)
        s = s.strip()

    # -----------------------------
    # Aggressive normalization
    # 可能改變 WAF / ML / PL 所看到的差異
    # -----------------------------
    if enable_aggressive_normalization:
        s = normalize_quote_style(s)
        s = normalize_html_entity_case(s)
        s = normalize_percent_encoding_case(s)
        s = normalize_tag_name_case(s)
        s = normalize_event_handler_case(s)
        s = normalize_tag_spacing(s)
        s = normalize_self_closing_tags(s)
        s = re.sub(r"\s+", " ", s).strip()
        s = normalize_trailing_semicolons(s)
        s = strip_redundant_outer_parens(s)
        s = s.strip()

    if lowercase:
        s = s.lower()

    return s


def pick_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["crs_status"] = out.apply(get_crs_status, axis=1)

    if FILTER_MODE == "all":
        return out.copy()

    if FILTER_MODE == "bypass_only":
        if ("crs" not in df.columns) and ("crs_bypass" not in df.columns):
            raise ValueError("CSV 缺少 'crs' 或 'crs_bypass' 欄位，無法判斷 CRS bypass。")
        return out[out["crs_status"].isin(BYPASS_LABELS)].copy()

    if FILTER_MODE == "block_only":
        if ("crs" not in df.columns) and ("crs_bypass" not in df.columns):
            raise ValueError("CSV 缺少 'crs' 或 'crs_bypass' 欄位，無法判斷 CRS block。")
        return out[out["crs_status"].isin(BLOCK_LABELS)].copy()

    raise ValueError(f"不支援的 FILTER_MODE: {FILTER_MODE}")


def build_cluster_model(distance_threshold: float):
    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=None,
            affinity="cosine",
            linkage="average",
            distance_threshold=distance_threshold
        )
    return model

def source_order_key(source_file: str) -> int:
    return source_style(source_file)[0]

# =====================================================
# Main
# =====================================================
def main():
    ensure_dir(OUTPUT_DIR)
    run_ts = make_timestamp()

    # -------------------------------------------------
    # Load & merge
    # -------------------------------------------------
    all_dfs = []

    for path in INPUT_CSVS:
        print(f"[LOAD] {path}")
        df = pd.read_csv(path, encoding="utf-8-sig")

        if "payload" not in df.columns:
            raise ValueError(f"{path} 缺少必要欄位: payload")

        if "index" not in df.columns:
            df["index"] = range(len(df))

        df["source_file"] = os.path.basename(path)
        df["source_path"] = path
        df["source_slug"] = slugify_source_name(path)
        all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)

    print(f"[INFO] total merged rows = {len(df)}")
    print("[INFO] rows by source:")
    print(df["source_file"].value_counts())

    # -------------------------------------------------
    # Filter rows
    # -------------------------------------------------
    work_df = pick_rows(df).reset_index(drop=True)

    print(f"[INFO] filter mode = {FILTER_MODE}")
    print(f"[INFO] selected rows = {len(work_df)}")

    if len(work_df) == 0:
        print("[WARN] 沒有可分群 payload。")
        return

    # -------------------------------------------------
    # Normalize payload
    # -------------------------------------------------
    work_df["payload_raw"] = work_df["payload"].astype(str)
    work_df["payload_norm"] = work_df["payload_raw"].apply(
        lambda x: normalize_payload(
            x,
            lowercase=LOWERCASE,
            enable_safe_normalization=ENABLE_SAFE_NORMALIZATION,
            enable_aggressive_normalization=ENABLE_AGGRESSIVE_NORMALIZATION,
        )
    )

    work_df = work_df[work_df["payload_norm"].str.len() > 0].copy()
    work_df = work_df.reset_index(drop=True)

    print(f"[INFO] selected rows after cleaning = {len(work_df)}")

    if len(work_df) == 0:
        print("[WARN] 選取的 payload 經清理後為空。")
        return

    print(f"[INFO] exact unique payloads before any dedup = {work_df['payload_raw'].nunique()}")
    print(f"[INFO] normalized unique payloads before any dedup = {work_df['payload_norm'].nunique()}")

    if DEDUP_WITHIN_SOURCE_BY_NORMALIZED_PAYLOAD:
        before = len(work_df)
        work_df = (
            work_df
            .drop_duplicates(subset=["source_file", "payload_norm"])
            .reset_index(drop=True)
        )
        after = len(work_df)
        print(f"[INFO] dedup within source by (source_file, payload_norm): {before} -> {after}")

    print("[INFO] selected rows by source (after optional within-source dedup):")
    print(work_df["source_file"].value_counts())

    # -------------------------------------------------
    # Build unique payload_norm table for clustering
    # -------------------------------------------------
    unique_norm_df = (
        work_df[["payload_norm"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .copy()
    )

    print(f"[INFO] unique normalized payloads for clustering = {len(unique_norm_df)}")

    # -------------------------------------------------
    # TF-IDF on unique normalized payloads only
    # -------------------------------------------------
    print("[TF-IDF] vectorizing unique normalized payloads...")
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        norm=NORM
    )

    X_unique = vectorizer.fit_transform(unique_norm_df["payload_norm"].tolist())
    print(f"[TF-IDF] shape = {X_unique.shape}")

    # -------------------------------------------------
    # 單次整體分群（保留）
    # -------------------------------------------------
    print("[CLUSTER] clustering unique normalized payloads...")
    cluster_model = build_cluster_model(0.5)
    unique_labels = cluster_model.fit_predict(X_unique.toarray())
    unique_norm_df["cluster_id"] = unique_labels

    work_df = work_df.merge(unique_norm_df, on="payload_norm", how="left")

    if work_df["cluster_id"].isna().any():
        raise ValueError("存在 payload_norm 無法對應 cluster_id，請檢查 merge 流程。")

    work_df["cluster_id"] = work_df["cluster_id"].astype(int)

    cluster_count = int(unique_norm_df["cluster_id"].nunique())
    selected_row_count = len(work_df)

    print(f"[RESULT] selected row count (all kept rows) = {selected_row_count}")
    print(f"[RESULT] total cluster count = {cluster_count}")

    source_files = sorted(
    work_df["source_file"].unique().tolist(),
    key=source_order_key
)
    grouped = work_df.groupby("cluster_id", dropna=False)

    cluster_rows = []
    for cluster_id, g in grouped:
        g = g.copy().sort_values(by=["source_file", "index"], ascending=True)
        g_unique_norm = g.drop_duplicates(subset=["payload_norm"]).copy()
        payload_list = g_unique_norm["payload_raw"].tolist()
        cluster_size_rows = len(g)
        cluster_size_unique_norm = g["payload_norm"].nunique()

        norms_in_cluster = g_unique_norm["payload_norm"].tolist()
        idx_map = unique_norm_df.reset_index().set_index("payload_norm")["index"].to_dict()
        unique_idx = [idx_map[pn] for pn in norms_in_cluster if pn in idx_map]

        if len(unique_idx) >= 2:
            X_sub = X_unique[unique_idx]
            sim_mat = cosine_similarity(X_sub)
            upper_vals = sim_mat[np.triu_indices_from(sim_mat, k=1)]
            avg_intra_sim = float(upper_vals.mean()) if len(upper_vals) > 0 else 1.0
        else:
            avg_intra_sim = 1.0

        source_counts = g["source_file"].value_counts().to_dict()

        row = {
            "cluster_id": int(cluster_id),
            "cluster_size_rows": int(cluster_size_rows),
            "cluster_size_unique_norm": int(cluster_size_unique_norm),
            "avg_intra_cosine_similarity": avg_intra_sim,
            "source_count": int(g["source_file"].nunique()),
            "sources": " | ".join(sorted(g["source_file"].unique().tolist())),
            "example_1": payload_list[0] if len(payload_list) > 0 else "",
            "example_2": payload_list[1] if len(payload_list) > 1 else "",
            "example_3": payload_list[2] if len(payload_list) > 2 else "",
        }

        for src in source_files:
            row[f"count__{slugify_source_name(src)}"] = int(source_counts.get(src, 0))

        cluster_rows.append(row)

    cluster_summary_df = pd.DataFrame(cluster_rows).sort_values(
        by=["cluster_size_rows", "cluster_id"], ascending=[False, True]
    ).reset_index(drop=True)

    source_cluster_stats = []
    for source_name, g in work_df.groupby("source_file"):
        covered_clusters = set(g["cluster_id"].unique().tolist())
        source_cluster_stats.append({
            "source_file": source_name,
            "selected_row_count": int(len(g)),
            "exact_unique_count": int(g["payload_raw"].nunique()),
            "normalized_unique_count": int(g["payload_norm"].nunique()),
            "covered_cluster_count": int(len(covered_clusters)),
        })

    source_cluster_df = pd.DataFrame(source_cluster_stats).sort_values(
        by="covered_cluster_count", ascending=False
    ).reset_index(drop=True)

    source_to_clusters = {
        src: set(g["cluster_id"].unique().tolist())
        for src, g in work_df.groupby("source_file")
    }

    pairwise_rows = []
    for src_a, src_b in itertools.combinations(source_files, 2):
        clusters_a = source_to_clusters.get(src_a, set())
        clusters_b = source_to_clusters.get(src_b, set())
        shared = clusters_a & clusters_b
        exclusive_a = clusters_a - clusters_b
        exclusive_b = clusters_b - clusters_a
        union_ab = clusters_a | clusters_b

        pairwise_rows.append({
            "source_a": src_a,
            "source_b": src_b,
            "cluster_count_a": int(len(clusters_a)),
            "cluster_count_b": int(len(clusters_b)),
            "shared_cluster_count": int(len(shared)),
            "exclusive_cluster_count_a": int(len(exclusive_a)),
            "exclusive_cluster_count_b": int(len(exclusive_b)),
            "union_cluster_count": int(len(union_ab)),
            "jaccard_similarity": round(len(shared) / len(union_ab), 6) if len(union_ab) > 0 else 0.0,
        })

    pairwise_cluster_df = pd.DataFrame(pairwise_rows).sort_values(
        by=["shared_cluster_count", "jaccard_similarity"],
        ascending=[False, False]
    ).reset_index(drop=True)

    presence_rows = []
    for _, row in cluster_summary_df.iterrows():
        base = {
            "cluster_id": int(row["cluster_id"]),
            "source_count": int(row["source_count"]),
            "sources": row["sources"],
        }
        for src in source_files:
            base[f"present__{slugify_source_name(src)}"] = int(row[f"count__{slugify_source_name(src)}"] > 0)
        presence_rows.append(base)

    cluster_presence_df = pd.DataFrame(presence_rows)

    # -------------------------------------------------
    # Save with timestamp
    # -------------------------------------------------
    merged_clustered_csv = os.path.join(OUTPUT_DIR, f"merged_selected_payloads_clustered_{run_ts}.csv")
    cluster_summary_csv = os.path.join(OUTPUT_DIR, f"cluster_summary_{run_ts}.csv")
    source_compare_csv = os.path.join(OUTPUT_DIR, f"source_cluster_comparison_{run_ts}.csv")
    pairwise_cluster_csv = os.path.join(OUTPUT_DIR, f"pairwise_cluster_overlap_{run_ts}.csv")
    cluster_presence_csv = os.path.join(OUTPUT_DIR, f"cluster_presence_matrix_{run_ts}.csv")
    summary_txt = os.path.join(OUTPUT_DIR, f"summary_{run_ts}.txt")

    work_df.to_csv(merged_clustered_csv, index=False, encoding="utf-8-sig")
    cluster_summary_df.to_csv(cluster_summary_csv, index=False, encoding="utf-8-sig")
    source_cluster_df.to_csv(source_compare_csv, index=False, encoding="utf-8-sig")
    pairwise_cluster_df.to_csv(pairwise_cluster_csv, index=False, encoding="utf-8-sig")
    cluster_presence_df.to_csv(cluster_presence_csv, index=False, encoding="utf-8-sig")

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("=== COMBINED CLUSTER SUMMARY ===\n")
        f.write(f"Filter mode: {FILTER_MODE}\n")
        f.write(f"LOWERCASE: {LOWERCASE}\n")
        f.write(f"ENABLE_SAFE_NORMALIZATION: {ENABLE_SAFE_NORMALIZATION}\n")
        f.write(f"ENABLE_AGGRESSIVE_NORMALIZATION: {ENABLE_AGGRESSIVE_NORMALIZATION}\n")
        f.write(f"Dedup within source by normalized payload: {DEDUP_WITHIN_SOURCE_BY_NORMALIZED_PAYLOAD}\n")
        f.write(f"TF-IDF ngram_range: {NGRAM_RANGE}\n")
        f.write(f"Selected row count (all kept rows): {selected_row_count}\n")
        f.write(f"Unique normalized payload count: {len(unique_norm_df)}\n")
        f.write(f"Total cluster count (single-run threshold=0.5): {cluster_count}\n")
        f.write(f"Sample size mode: {SAMPLE_SIZE}\n")
        f.write(f"Random seeds: {RANDOM_SEEDS}\n")
        f.write(f"Threshold sweep: {THRESHOLDS}\n\n")

        f.write("=== SOURCE CLUSTER COMPARISON ===\n")
        for _, row in source_cluster_df.iterrows():
            f.write(f"Source: {row['source_file']}\n")
            f.write(f"  Selected row count      : {row['selected_row_count']}\n")
            f.write(f"  Exact unique count      : {row['exact_unique_count']}\n")
            f.write(f"  Normalized unique count : {row['normalized_unique_count']}\n")
            f.write(f"  Covered cluster count   : {row['covered_cluster_count']}\n\n")

        f.write("=== PAIRWISE CLUSTER OVERLAP ===\n")
        for _, row in pairwise_cluster_df.iterrows():
            f.write(f"{row['source_a']}  <->  {row['source_b']}\n")
            f.write(f"  shared_cluster_count      : {row['shared_cluster_count']}\n")
            f.write(f"  exclusive_cluster_count_a : {row['exclusive_cluster_count_a']}\n")
            f.write(f"  exclusive_cluster_count_b : {row['exclusive_cluster_count_b']}\n")
            f.write(f"  jaccard_similarity        : {row['jaccard_similarity']}\n\n")

    plot_threshold_sensitivity_random_avg(
        work_df,
        OUTPUT_DIR,
        run_ts=run_ts,
        sample_size=SAMPLE_SIZE
    )

    print("\n[COMPARE RESULT]")
    print(source_cluster_df.to_string(index=False))

    print("\n[PAIRWISE CLUSTER OVERLAP]")
    if len(pairwise_cluster_df) > 0:
        print(pairwise_cluster_df.to_string(index=False))
    else:
        print("No pairwise comparison available.")

    print("\n[OUTPUT]")
    print(f"  merged clustered csv  : {merged_clustered_csv}")
    print(f"  cluster summary csv   : {cluster_summary_csv}")
    print(f"  source compare csv    : {source_compare_csv}")
    print(f"  pairwise overlap csv  : {pairwise_cluster_csv}")
    print(f"  presence matrix csv   : {cluster_presence_csv}")
    print(f"  summary txt           : {summary_txt}")

def plot_threshold_sensitivity_random_avg(work_df, output_dir, run_ts, sample_size=None):
    thresholds = THRESHOLDS
    source_files = sorted(
    work_df["source_file"].unique().tolist(),
    key=source_order_key
)

    # 顏色由 source_color() 統一控制

    # =================================================
    # 模式 A：全量模式（SAMPLE_SIZE = None）
    # =================================================
    if sample_size is None:
        print("[INFO] full-data mode: using all samples under current FILTER_MODE")
        print("[INFO] no random sampling, no repeated seeds")

        unique_norm_df = (
            work_df[["payload_norm"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .copy()
        )

        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=NGRAM_RANGE,
            min_df=MIN_DF,
            norm=NORM
        )

        X_unique = vectorizer.fit_transform(unique_norm_df["payload_norm"].tolist())
        base_df = work_df[["source_file", "payload_norm"]].copy()

        summary_rows = []

        for th in thresholds:
            print(f"[SWEEP] threshold = {th}")

            cluster_model = build_cluster_model(th)
            labels = cluster_model.fit_predict(X_unique.toarray())

            tmp_unique = unique_norm_df.copy()
            tmp_unique["cluster_id"] = labels

            tmp_df = base_df.merge(
                tmp_unique[["payload_norm", "cluster_id"]],
                on="payload_norm",
                how="left"
            )

            for src in source_files:
                clusters = set(tmp_df.loc[tmp_df["source_file"] == src, "cluster_id"])
                summary_rows.append({
                    "source_file": src,
                    "pretty_source_name": pretty_source_name(src),
                    "threshold": th,
                    "mean_cluster_count": float(len(clusters)),
                    "std_cluster_count": 0.0,
                    "min_cluster_count": int(len(clusters)),
                    "max_cluster_count": int(len(clusters)),
                    "sample_size": int((work_df["source_file"] == src).sum()),
                    "seed_count": 1,
                    "mode": "full_data",
                })

        summary_df = pd.DataFrame(summary_rows)

        summary_csv = os.path.join(output_dir, f"threshold_full_data_{run_ts}.csv")
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

        plt.figure(figsize=(8, 5))

        for src in source_files:
            pretty_name = pretty_source_name(src)
            color = source_color(src)

            src_df = summary_df[summary_df["source_file"] == src].sort_values("threshold")

            plt.plot(
                src_df["threshold"],
                src_df["mean_cluster_count"],
                marker="o",
                label=pretty_name,
                color=color
            )

        plt.xlabel("Distance Threshold")
        plt.ylabel("Covered Cluster Count")
        plt.title(f"Cluster Coverage vs Distance Threshold (char n-gram {NGRAM_RANGE[0]}–{NGRAM_RANGE[1]})")
        plt.legend()
        plt.grid(True)

        out_path = os.path.join(output_dir, f"threshold_full_data_{run_ts}.pdf")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        print(f"[CSV SAVED] {summary_csv}")
        print(f"[PLOT SAVED] {out_path}")
        return

    # =================================================
    # 模式 B：抽樣模式（SAMPLE_SIZE = 整數）
    # =================================================
    random_seeds = RANDOM_SEEDS
    source_counts = work_df["source_file"].value_counts().to_dict()

    for src, cnt in source_counts.items():
        if cnt < sample_size:
            print(f"[ERROR] {src} 樣本數不足：{cnt} < {sample_size}")
            print("[ABORT] 請降低 sample_size 或更換 FILTER_MODE。")
            raise SystemExit(1)

    print(f"[INFO] sampled mode: sample_size = {sample_size}")
    print(f"[INFO] random seeds = {random_seeds}")

    results = {
        src: {th: [] for th in thresholds}
        for src in source_files
    }

    for seed in random_seeds:
        print(f"[SEED] {seed}")

        sampled_parts = []
        for src in source_files:
            g = work_df[work_df["source_file"] == src].copy()
            sampled_g = g.sample(n=sample_size, random_state=seed, replace=False)
            sampled_parts.append(sampled_g)

        sampled_df = pd.concat(sampled_parts, ignore_index=True)

        unique_norm_df = (
            sampled_df[["payload_norm"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .copy()
        )

        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=NGRAM_RANGE,
            min_df=MIN_DF,
            norm=NORM
        )

        X_unique = vectorizer.fit_transform(unique_norm_df["payload_norm"].tolist())
        base_df = sampled_df[["source_file", "payload_norm"]].copy()

        for th in thresholds:
            print(f"  [SWEEP] threshold = {th}")

            cluster_model = build_cluster_model(th)
            labels = cluster_model.fit_predict(X_unique.toarray())

            tmp_unique = unique_norm_df.copy()
            tmp_unique["cluster_id"] = labels

            tmp_df = base_df.merge(
                tmp_unique[["payload_norm", "cluster_id"]],
                on="payload_norm",
                how="left"
            )

            for src in source_files:
                clusters = set(tmp_df.loc[tmp_df["source_file"] == src, "cluster_id"])
                results[src][th].append(len(clusters))

    summary_rows = []
    for src in source_files:
        for th in thresholds:
            vals = results[src][th]
            summary_rows.append({
                "source_file": src,
                "pretty_source_name": pretty_source_name(src),
                "threshold": th,
                "mean_cluster_count": float(np.mean(vals)),
                "std_cluster_count": float(np.std(vals, ddof=0)),
                "min_cluster_count": int(np.min(vals)),
                "max_cluster_count": int(np.max(vals)),
                "sample_size": sample_size,
                "seed_count": len(random_seeds),
                "mode": "random_sample",
            })

    summary_df = pd.DataFrame(summary_rows)

    summary_csv = os.path.join(output_dir, f"threshold_random_seed_avg_{run_ts}.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))

    for src in source_files:
        pretty_name = pretty_source_name(src)
        color = source_color(src)

        src_df = summary_df[summary_df["source_file"] == src].sort_values("threshold")

        plt.plot(
            src_df["threshold"],
            src_df["mean_cluster_count"],
            marker="o",
            label=pretty_name,
            color=color
        )
    plt.xlim(0.0, 1.0)
    plt.xticks(np.arange(0.0, 1.01, 0.1))
    plt.xlabel("Distance Threshold")
    plt.ylabel("Average Covered Cluster Count")
    plt.title(f"Average Cluster Coverage vs Distance Threshold (Seeds 1–{len(random_seeds)})")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(output_dir, f"threshold_random_seed_avg_{run_ts}.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"[CSV SAVED] {summary_csv}")
    print(f"[PLOT SAVED] {out_path}")


if __name__ == "__main__":
    main()