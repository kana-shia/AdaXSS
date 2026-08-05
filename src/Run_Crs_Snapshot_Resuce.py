#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
import random

from BERT_detector import XSSDetector_BERT

# ==================================================
# 固定 seed
# ==================================================
os.environ["PYTHONHASHSEED"] = "0"
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# ==================================================
# 三種實驗設定
# ==================================================
EXPERIMENTS = [
   
    {
        "name"            : "Exp1_Baseline",
        "snapshot_csv"    : "res/crs_snapshots/crs_snapshot_test0.csv",
        "label_source_csv": "res/train_data/xss_dataset.csv",
        "label_mode"      : "baseline",
        "bert_model_path" : "res/BestModel_TinyBERT_1.keras",
        "out_filename"    : "summary_exp1_baseline.csv",
        "desc"            : "CRS=baseline / Rescue=baseline",
    },
    {
        "name"            : "Exp2_Baseline_Claude80",
        "snapshot_csv"    : "res/crs_snapshots/crs_snapshot_test800.csv",
        "label_source_csv": "res/train_data/xss_dataset.csv",
        "label_mode"      : "baseline_plus_attack",
        "bert_model_path" : "res/BestModel_TinyBERT_1.keras",
        "out_filename"    : "summary_exp2_baseline_claude80.csv",
        "desc"            : "CRS=baseline+80% / Rescue=baseline",
    },
    {
        "name"            : "Exp3_FT_Claude80",
        "snapshot_csv"    : "res/crs_snapshots/crs_snapshot_test800.csv",
        "label_source_csv": "res/train_data/xss_dataset.csv",
        "label_mode"      : "baseline_plus_attack",
        "bert_model_path" : "res/BestModel_TinyBERT_1_FT_distant.keras",
        "out_filename"    : "summary_exp3_ft_claude80.csv",
        "desc"            : "CRS=baseline+80% / Rescue=baseline+20%",
    },
]


THRESHOLDS = list(range(1, 11))
EXCLUDE_RULES = {"949110", "980130"}

OUT_DIR = "res/false_positive_summary"
os.makedirs(OUT_DIR, exist_ok=True)

BERT_MODEL_TYPE = "tinyBERT"
BERT_MAX_LENGTH = 256

# ==================================================
# Utils
# ==================================================
def normalize_payload(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\r", "").replace("\n", "").strip()
    return s


def recount_rules(rule_ids_str):
    if pd.isna(rule_ids_str) or str(rule_ids_str).strip() == "":
        return 0
    rules = [r.strip() for r in str(rule_ids_str).split(",")]
    return len([r for r in rules if r not in EXCLUDE_RULES])


def compute_confusion_and_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    total = len(y_true)

    acc  = (TP + TN) / total if total else 0.0
    tpr  = TP / (TP + FN)   if (TP + FN) else 0.0
    tnr  = TN / (TN + FP)   if (TN + FP) else 0.0
    prec = TP / (TP + FP)   if (TP + FP) else 0.0
    fpr  = FP / (FP + TN)   if (FP + TN) else 0.0
    fnr  = FN / (FN + TP)   if (FN + TP) else 0.0

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "Accuracy": acc, "TPR": tpr, "TNR": tnr,
        "Precision": prec, "FPR": fpr, "FNR": fnr,
    }


def attach_label_from_source(snapshot_df: pd.DataFrame,
                             label_source_csv: str,
                             label_mode: str = "baseline",
                             attack_len: int = 0) -> pd.DataFrame:
    src = pd.read_csv(
        label_source_csv,
        header=None,
        names=["payload", "label"],
        encoding="utf-8-sig"
    )
    src["payload_norm"] = src["payload"].apply(normalize_payload)
    src["label"] = src["label"].astype(int)

    out = snapshot_df.copy()

    baseline_len = len(src)

    

    if "index" in out.columns:
        def get_label_by_index(idx):
            idx = int(idx)

            if 1 <= idx <= baseline_len:
                return int(src.loc[idx - 1, "label"])

            if label_mode == "baseline_plus_attack":
                return 1

            return np.nan

        out["label"] = out["index"].apply(get_label_by_index)

        if out["label"].isna().sum() == 0:
            return out

    out["payload_norm"] = out["payload"].apply(normalize_payload)
    src_map = (
        src.drop_duplicates("payload_norm")
           .set_index("payload_norm")["label"]
           .to_dict()
    )
    out["label"] = out["payload_norm"].map(src_map)

    if label_mode == "baseline_plus_attack":
        out["label"] = out["label"].fillna(1)

    return out


# ==================================================
# 單次實驗執行
# ==================================================
def run_experiment(exp: dict):
    print(f"\n{'='*60}")
    print(f"[EXP] {exp['name']}")
    print(f"      {exp['desc']}")
    print(f"{'='*60}")

    # 讀取 snapshot
    df = pd.read_csv(exp["snapshot_csv"], encoding="utf-8-sig")

    # 必要欄位檢查
    for c in ["payload", "rule_ids"]:
        if c not in df.columns:
            raise ValueError(f"[{exp['name']}] Missing column: {c}")

    # 重新計算 rule_count（排除 949110/980130）
    df["rule_count"] = df["rule_ids"].apply(recount_rules)
    print(f"[INFO] rule_count recomputed (excluding {EXCLUDE_RULES})")

    # 補齊 label
    if "label" not in df.columns:
        print(f"[INFO] Attaching label from: {exp['label_source_csv']}")
        df = attach_label_from_source(
            df,
            exp["label_source_csv"],
            exp.get("label_mode", "baseline"),
            exp.get("attack_len", 0)
        )

    if "label" not in df.columns:
        raise ValueError(f"[{exp['name']}] Failed to attach label.")

    missing = int(df["label"].isna().sum())
    if missing > 0:
        raise ValueError(f"[{exp['name']}] Still missing {missing} labels.")

    df["label"]   = df["label"].astype(int)
    df["payload"] = df["payload"].astype(str)

    print(f"[INFO] Snapshot rows  : {len(df)}")
    print(f"[INFO] Label dist     : {df['label'].value_counts().to_dict()}")

    # 載入 BERT 模型
    print(f"[INFO] Loading model  : {exp['bert_model_path']}")
    det = XSSDetector_BERT(
        exp["bert_model_path"], BERT_MODEL_TYPE, BERT_MAX_LENGTH)
    det.model.trainable = False
    try:
        det.model.compile()
    except Exception:
        pass

    # ML 推論（一次性）
    print("[INFO] Running ML inference...")
    model_pred = []
    for p in tqdm(df["payload"].tolist(),
                  desc=f"ML Inference [{exp['name']}]",
                  total=len(df)):
        model_pred.append(int(det.is_xss(p)))
    df["model_pred"] = model_pred

    # Threshold sweep
    summary_rows = []
    for thr in tqdm(THRESHOLDS, desc=f"Threshold Sweep [{exp['name']}]"):
        crs_block  = (df["rule_count"].astype(int) >= thr).astype(int)
        final_block = crs_block.copy()
        bypass_mask = (crs_block == 0)
        final_block[bypass_mask] = df.loc[bypass_mask, "model_pred"].astype(int)

        y_true        = df["label"].values
        metrics_crs   = compute_confusion_and_metrics(y_true, crs_block.values)
        metrics_final = compute_confusion_and_metrics(y_true, final_block.values)

        summary_rows.append({
            "experiment" : exp["name"],
            "policy"     : f"Threshold_{thr}_block",
            "threshold"  : thr,

            "CRS_TP"       : metrics_crs["TP"],
            "CRS_TN"       : metrics_crs["TN"],
            "CRS_FP"       : metrics_crs["FP"],
            "CRS_FN"       : metrics_crs["FN"],
            "CRS_Accuracy" : metrics_crs["Accuracy"],
            "CRS_TPR"      : metrics_crs["TPR"],
            "CRS_TNR"      : metrics_crs["TNR"],
            "CRS_Precision": metrics_crs["Precision"],
            "CRS_FPR"      : metrics_crs["FPR"],
            "CRS_FNR"      : metrics_crs["FNR"],

            "FINAL_TP"       : metrics_final["TP"],
            "FINAL_TN"       : metrics_final["TN"],
            "FINAL_FP"       : metrics_final["FP"],
            "FINAL_FN"       : metrics_final["FN"],
            "FINAL_Accuracy" : metrics_final["Accuracy"],
            "FINAL_TPR"      : metrics_final["TPR"],
            "FINAL_TNR"      : metrics_final["TNR"],
            "FINAL_Precision": metrics_final["Precision"],
            "FINAL_FPR"      : metrics_final["FPR"],
            "FINAL_FNR"      : metrics_final["FNR"],
        })

    summary_df = pd.DataFrame(summary_rows)
    out_path   = os.path.join(OUT_DIR, exp["out_filename"])
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ Saved: {out_path}")
    print(summary_df.to_string(index=False))

    return summary_df


# ==================================================
# Main
# ==================================================
def main():
    all_results = []
    for exp in EXPERIMENTS:
        result = run_experiment(exp)
        all_results.append(result)

    # 合併三份結果成一個總表
    combined = pd.concat(all_results, ignore_index=True)
    combined_path = os.path.join(OUT_DIR, "summary_all_experiments.csv")
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Combined results saved: {combined_path}")


if __name__ == "__main__":
    main()
