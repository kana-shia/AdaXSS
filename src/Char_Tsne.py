import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

# =====================================================
# Paths
# =====================================================
BASELINE_PATH = "res/train_data/xss_dataset.csv"
SP4000_PATH   = "res/stats/stage3_filter_payload_CRS_Reference.csv"

RANDOM_STATE = 42
SAMPLE_N     = 800 #等量取樣 800 筆

# =====================================================
# Load datasets
# =====================================================
df_base = pd.read_csv(BASELINE_PATH, header=None, names=["payload", "label"])

df_sp = pd.read_csv(SP4000_PATH)
df_sp["label"] = 1
df_sp = df_sp[["payload", "label"]].copy()
df_sp = df_sp.dropna(subset=["payload"]).reset_index(drop=True)

print(f"Baseline samples : {len(df_base)}")
print(f"claude_1896 samples   : {len(df_sp)}")


df_base_xss    = df_base[df_base["label"] == 1].sample(
    n=SAMPLE_N, random_state=RANDOM_STATE)
df_base_normal = df_base[df_base["label"] == 0].sample(
    n=SAMPLE_N, random_state=RANDOM_STATE)

df_base_eq = pd.concat(
    [df_base_xss, df_base_normal], ignore_index=True)

print(f"\n[Sampled Baseline]")
print(f"  XSS    : {len(df_base_xss)}")
print(f"  Normal : {len(df_base_normal)}")
print(f"  Total  : {len(df_base_eq)}")
print(f"[Claude with official_reference] : {len(df_sp)}")

# =====================================================
# Char-level TF-IDF
# =====================================================
print("\nTF-IDF Vectorization...")
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=1,
    norm="l2"
)

all_payloads = pd.concat([
    df_base_eq["payload"],
    df_sp["payload"]
]).astype(str)

X_all_sparse = vectorizer.fit_transform(all_payloads)

X_base_eq = X_all_sparse[:len(df_base_eq)]
X_sp      = X_all_sparse[len(df_base_eq):]

print(f"Feature dimensions: {X_all_sparse.shape[1]}")

print("\nt-SNE Dimensionality Reduction (this may take a while)...")
tsne = TSNE(
    n_components=2,
    perplexity=30,
    n_iter=1000,
    random_state=RANDOM_STATE,
    init="random",   # TF-IDF 為稀疏矩陣，無法使用 'pca' init，需改用 'random'
    verbose=1
)
X_2d = tsne.fit_transform(X_all_sparse)

X_base_eq_2d = X_2d[:len(df_base_eq)]
X_sp_2d      = X_2d[len(df_base_eq):]

# =====================================================
# Plot
# =====================================================
plt.figure(figsize=(10, 5))

for label, color, name in [
    (1, "red",   "Baseline (XSS)"),
    (0, "green", "Baseline (Normal)"),
]:
    idx = df_base_eq["label"].values == label
    plt.scatter(
        X_base_eq_2d[idx, 0],
        X_base_eq_2d[idx, 1],
        c=color,
        alpha=0.4,
        s=15,
        label=name
    )

for label, color, name in [
    (1, "blue", "Claude + CRS Reference"),
]:
    idx = df_sp["label"].values == label
    plt.scatter(
        X_sp_2d[idx, 0],
        X_sp_2d[idx, 1],
        c=color,
        alpha=0.4,
        s=15,
        label=name
    )

plt.title("XSS Payload Distribution (Char-TFIDF + t-SNE)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.xticks([])
plt.yticks([])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("res/figures/tsne_payload_distribution.pdf",
            dpi=300, bbox_inches="tight")
plt.show()