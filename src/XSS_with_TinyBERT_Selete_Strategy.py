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
BASELINE_PATH         = "res/train_data/xss_dataset.csv"
SP4000_PATH           = "res/train_data/stage3_filter_payload_CRS_Reference.csv"
PRETRAINED_MODEL_PATH = "res/BestModel_TinyBERT_1.keras"

EPOCHS     = 10
BATCH_SIZE = 16       # 改小避免 batch 不足
MAX_LEN    = 256
MODEL_NAME = "prajjwal1/bert-tiny"

# 從 pool 裡選多少比例當訓練資料
TRAIN_RATIOS  = [0.25, 0.50, 1.00]

# 用哪個比例切出固定測試集
FIXED_TRAIN_RATIO = 0.20

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
# TF-IDF char n-gram（對全部 Claude 樣本）
# =====================================================
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=1,
    norm="l2"
)
X_char   = vectorizer.fit_transform(df_sp["payload"].astype(str))
centroid = X_char.mean(axis=0).A
dist     = 1.0 - cosine_similarity(X_char, centroid).ravel()
df_sp["dist"] = dist

# =====================================================
# 固定測試集（在迴圈外先切好）
# =====================================================
threshold_fixed = np.quantile(dist, 1.0 - FIXED_TRAIN_RATIO)

df_sp["split_fixed"] = "test"
df_sp.loc[df_sp["dist"] >= threshold_fixed, "split_fixed"] = "train_pool"

df_sp_test_fixed = df_sp[df_sp["split_fixed"] == "test"].copy()
df_sp_pool       = df_sp[df_sp["split_fixed"] == "train_pool"].copy()

print(f"\n[Fixed Split (distant {int(FIXED_TRAIN_RATIO*100)}%)]")
print(f"Training pool : {len(df_sp_pool)}")
print(f"Fixed test    : {len(df_sp_test_fixed)}")

# 固定測試集加入等量 benign
df_base_benign = df_base[df_base["label"] == 0]
df_benign_test = df_base_benign.sample(
    n=len(df_sp_test_fixed), random_state=42, replace=True
)
df_test_fixed = pd.concat(
    [
        df_sp_test_fixed.drop(columns=["dist", "split_fixed"]),
        df_benign_test
    ],
    ignore_index=True
)
y_test_fixed = df_test_fixed["label"].values

print(f"Fixed test total : {len(df_test_fixed)}"
      f"  (XSS={len(df_sp_test_fixed)}, benign={len(df_benign_test)})")

# =====================================================
# pool 內部重新計算距離
# =====================================================
X_pool        = vectorizer.transform(df_sp_pool["payload"].astype(str))
centroid_pool = X_pool.mean(axis=0).A
dist_pool     = 1.0 - cosine_similarity(X_pool, centroid_pool).ravel()
df_sp_pool    = df_sp_pool.copy()
df_sp_pool["dist_pool"] = dist_pool

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

# 測試集 tokenize（固定，只做一次）
X_test_fixed    = tokenize(df_test_fixed["payload"])
mask_test_fixed = (X_test_fixed != 0).astype(np.int32)

# =====================================================
# 三種取樣策略
# =====================================================
strategies = {
    "distant" : "結構距離最遠（本研究方法）",
    "random"  : "隨機取樣",
    "nearest" : "結構距離最近（反向對照）",
}

all_results = {}

for TRAIN_RATIO in TRAIN_RATIOS:
    ratio_label = f"{int(TRAIN_RATIO * 100)}pct"

    for strategy_key, strategy_name in strategies.items():

        exp_name  = f"{strategy_key}_{ratio_label}"
        full_name = f"{strategy_name}  pool {int(TRAIN_RATIO*100)}%"

        print(f"\n{'='*60}")
        print(f"[Experiment] {full_name}")
        print(f"{'='*60}")

        # --- 從 pool 選訓練資料 ---
        pool_copy = df_sp_pool.copy()
        n_select  = max(1, int(len(pool_copy) * TRAIN_RATIO))

        if strategy_key == "distant":
            idx = pool_copy["dist_pool"].nlargest(n_select).index
        elif strategy_key == "random":
            idx = pool_copy.sample(n=n_select, random_state=42).index
        elif strategy_key == "nearest":
            idx = pool_copy["dist_pool"].nsmallest(n_select).index

        df_sp_train = pool_copy.loc[idx].drop(
            columns=["dist", "split_fixed", "dist_pool"]
        )

        print(f"Train : {len(df_sp_train)}"
              f"  |  Test (fixed) : {len(df_test_fixed)}")

        # --- Tokenize 訓練集 ---
        y_train     = df_sp_train["label"].values
        X_train     = tokenize(df_sp_train["payload"])
        mask_train  = (X_train != 0).astype(np.int32)
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)

        # --- Load pretrained & fine-tune ---
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

        # --- 儲存權重 ---
        save_path = (
            f"res/ablation_fixed_"
            f"{strategy_key}_"
            f"{ratio_label}.keras"
        )
        model.save(save_path)
        print(f"[INFO] Model saved: {save_path}")

        # --- Evaluate on fixed test set ---
        pred   = model.predict([X_test_fixed, mask_test_fixed])
        y_prob = pred[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        acc     = accuracy_score(y_test_fixed, y_pred)
        prec    = precision_score(y_test_fixed, y_pred, zero_division=0)
        rec     = recall_score(y_test_fixed, y_pred, zero_division=0)
        f1      = f1_score(y_test_fixed, y_pred, zero_division=0)
        cm      = confusion_matrix(y_test_fixed, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr     = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tnr     = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        roc_auc = roc_auc_score(y_test_fixed, y_prob)
        pr_auc  = average_precision_score(y_test_fixed, y_prob)

        all_results[exp_name] = {
            "strategy"   : strategy_name,
            "train_ratio": f"{int(TRAIN_RATIO*100)}%",
            "train_n"    : len(df_sp_train),
            "accuracy"   : acc,
            "precision"  : prec,
            "recall"     : rec,
            "f1"         : f1,
            "fpr"        : fpr,
            "tnr"        : tnr,
            "roc_auc"    : roc_auc,
            "pr_auc"     : pr_auc,
            "tp": int(tp), "tn": int(tn),
            "fp": int(fp), "fn": int(fn),
        }

        print(f"\n[Result – {full_name}]")
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
# Summary table
# =====================================================
print("\n")
print("=" * 75)
print("[Ablation Study – Fixed Test Set]")
print(f"Fixed test : {len(df_test_fixed)} samples"
      f"  (XSS={len(df_sp_test_fixed)}, benign={len(df_benign_test)})")
print("=" * 75)
print(f"{'策略':<22} {'比例':>6} {'N':>5} {'Acc':>7}"
      f" {'Rec':>7} {'F1':>7} {'FPR':>7}")
print("-" * 75)

for ratio in TRAIN_RATIOS:
    ratio_label = f"{int(ratio*100)}pct"
    for strategy_key in strategies:
        exp_name = f"{strategy_key}_{ratio_label}"
        r = all_results[exp_name]
        print(
            f"{r['strategy']:<22} "
            f"{r['train_ratio']:>6} "
            f"{r['train_n']:>5} "
            f"{r['accuracy']:>7.4f} "
            f"{r['recall']:>7.4f} "
            f"{r['f1']:>7.4f} "
            f"{r['fpr']:>7.4f}"
        )
    print("-" * 75)

# =====================================================
# Save to CSV
# =====================================================
df_summary = pd.DataFrame(all_results).T
df_summary.to_csv(
    "res/ablation_fixed_test11791_results.csv", index=True
)
print("\n[INFO] Saved to res/ablation_fixed_test_results.csv")