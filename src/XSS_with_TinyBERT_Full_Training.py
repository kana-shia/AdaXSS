# =====================================================
# Environment
# =====================================================
import os
os.environ["TF_KERAS"] = "1"
os.environ["TRANSFORMERS_NO_KERAS3_WARNING"] = "1"

import tensorflow as tf
import transformers as ppb
from transformers import TFBertModel

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# =====================================================
# Paths & config
# =====================================================
BASELINE_PATH = "res/train_data/xss_dataset.csv"
SP4000_PATH   = "res/train_data/xss_dataset_20260509_020826_800.csv"

PRETRAINED_MODEL_PATH = "res/BestModel_TinyBERT_1.keras"

EPOCHS     = 10
BATCH_SIZE = 16
MAX_LEN    = 256
TRAIN_RATIO = 0.20
MODEL_NAME  = "prajjwal1/bert-tiny"

# =====================================================
# Load datasets
# =====================================================
df_base = pd.read_csv(BASELINE_PATH, header=None, names=["payload", "label"])
df_sp   = pd.read_csv(SP4000_PATH,   header=None, names=["payload", "label"])

print("============================================================")
print("[Dataset]")
print(f"Baseline samples   : {len(df_base)}")
print(f"Claude XSS samples : {len(df_sp)}")
print("============================================================")

# =====================================================
# TF-IDF char n-gram vectorizer (shared)
# =====================================================
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=1,
    norm="l2"
)
X_char = vectorizer.fit_transform(df_sp["payload"].astype(str))
centroid = X_char.mean(axis=0).A
dist = 1.0 - cosine_similarity(X_char, centroid).ravel()
df_sp["dist"] = dist

# =====================================================
# Tokenizer (shared)
# =====================================================
tokenizer = ppb.BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize(texts):
    tokens = texts.apply(
        lambda x: tokenizer.encode(
            x,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LEN
        )
    )
    padded = []
    for t in tokens:
        t = t[:MAX_LEN]
        t += [0] * (MAX_LEN - len(t))
        padded.append(t)
    return np.array(padded, dtype=np.int32)

# =====================================================
# Three sampling strategies
# =====================================================
strategies = {
    "distant":  "結構距離最遠 20%（本研究方法）",
    "random":   "隨機取樣 20%",
    "nearest":  "結構距離最近 20%（反向對照）",
}

results = {}

for strategy_key, strategy_name in strategies.items():
    print(f"\n{'='*60}")
    print(f"[Strategy] {strategy_name}")
    print(f"{'='*60}")

    # --- Split ---
    df_sp_copy = df_sp.copy()
    df_sp_copy["split"] = "test"

    if strategy_key == "distant":
        threshold = np.quantile(dist, 1.0 - TRAIN_RATIO)
        df_sp_copy.loc[df_sp_copy["dist"] >= threshold, "split"] = "train"

    elif strategy_key == "random":
        train_idx = df_sp_copy.sample(frac=TRAIN_RATIO, random_state=42).index
        df_sp_copy.loc[train_idx, "split"] = "train"

    elif strategy_key == "nearest":
        threshold_low = np.quantile(dist, TRAIN_RATIO)
        df_sp_copy.loc[df_sp_copy["dist"] <= threshold_low, "split"] = "train"

    df_sp_train = df_sp_copy[df_sp_copy["split"] == "train"].drop(
        columns=["dist", "split"])
    df_sp_test  = df_sp_copy[df_sp_copy["split"] == "test"].drop(
        columns=["dist", "split"])

    print(f"Train : {len(df_sp_train)} | Test : {len(df_sp_test)}")

    # --- Export baseline + remaining 80% Claude XSS ---
    export_dir = "res/train_data/exported_remaining80"
    os.makedirs(export_dir, exist_ok=True)

    df_base_export = df_base.copy()
    df_base_export["label"] = df_base_export["label"].astype(int)

    df_sp_test_export = df_sp_test.copy()
    df_sp_test_export["label"] = df_sp_test_export["label"].astype(int)

    df_export = pd.concat(
        [df_base_export, df_sp_test_export],
        ignore_index=True
    )

    export_path = os.path.join(
        export_dir,
        f"xss_dataset_baseline_plus_remaining80_{strategy_key}.csv"
    )

    df_export.to_csv(
        export_path,
        index=False,
        header=False,
        encoding="utf-8"
    )

    print(f"[INFO] Exported baseline + remaining 80% Claude XSS: {export_path}")
    print(f"  - Baseline samples      : {len(df_base_export)}")
    print(f"  - Remaining Claude XSS  : {len(df_sp_test_export)}")
    print(f"  - Total                 : {len(df_export)}")

    # --- Build train/test sets ---
    df_train = pd.concat([df_base, df_sp_train], ignore_index=True)
    y_train  = df_train["label"].values

    df_base_benign = df_base[df_base["label"] == 0]
    df_benign_test = df_base_benign.sample(n=len(df_sp_test), random_state=42)
    df_test  = pd.concat([df_sp_test, df_benign_test], ignore_index=True)
    y_test   = df_test["label"].values

    # --- Tokenize ---
    X_train = tokenize(df_train["payload"])
    X_test  = tokenize(df_test["payload"])
    mask_train = (X_train != 0).astype(np.int32)
    mask_test  = (X_test  != 0).astype(np.int32)

    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_test_cat  = tf.keras.utils.to_categorical(y_test,  num_classes=2)

    # --- Load & fine-tune ---
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(
        PRETRAINED_MODEL_PATH,
        custom_objects={"TFBertModel": TFBertModel}
    )
    for layer in model.layers:
        if isinstance(layer, TFBertModel):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(
        [X_train, mask_train],
        y_train_cat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # --- Save fine-tuned model ---
    save_path = f"res/BestModel_TinyBERT_1_FT_{strategy_key}.keras"
    model.save(save_path)
    print(f"[INFO] Model saved: {save_path}")

    # --- Evaluate ---
    pred   = model.predict([X_test, mask_test])
    y_prob = pred[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr    = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr    = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)

    results[strategy_key] = {
        "strategy": strategy_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "tnr": tnr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
    }

    print(f"\n[Result – {strategy_name}]")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")
    print(f"FPR       : {fpr:.4f}")
    print(f"TNR       : {tnr:.4f}")
    print(f"Confusion Matrix:\n{cm}")

# =====================================================
# Summary comparison table
# =====================================================
print("\n")
print("=" * 60)
print("[Ablation Study Summary]")
print("=" * 60)
print(f"{'取樣方式':<30} {'Acc':>6} {'Prec':>6} {'Rec':>6} "
      f"{'F1':>6} {'ROC':>6} {'FPR':>6}")
print("-" * 60)
for key, r in results.items():
    print(f"{r['strategy']:<30} "
          f"{r['accuracy']:>6.4f} "
          f"{r['precision']:>6.4f} "
          f"{r['recall']:>6.4f} "
          f"{r['f1']:>6.4f} "
          f"{r['roc_auc']:>6.4f} "
          f"{r['fpr']:>6.4f}")
print("=" * 60)

# =====================================================
# Save summary to CSV
# =====================================================
df_summary = pd.DataFrame(results).T
df_summary.to_csv("res/ablation_sampling_results.csv", index=True)
print("\n[INFO] Summary saved to res/ablation_sampling_results.csv")