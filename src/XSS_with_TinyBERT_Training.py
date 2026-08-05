"""
比較兩種黑資料來源(XSS payload)的偵測模型效能
==============================================
- 測試集：固定 200黑 + 200白
    黑 = claude_200.txt 全部 200 筆
    白 = baseline (xss_dataset.csv) 隨機抽樣 200 筆 (只抽一次，全程共用)
- 訓練集：黑資料分別來自兩個來源檔案，白資料來自 baseline 剩餘白樣本(排除測試集白)
  每個來源各跑 3 次 (data_seed = 1, 2, 3)，每次重新抽樣 300黑 + 300白
- 模型初始化 / 訓練隨機性：固定 MODEL_SEED = 3 (每次重新建模時都重設，確保起始權重一致)
- 輸出：
  1) 每個 (來源, data_seed) 訓練出的 .keras 模型檔
  2) 每個 (來源, data_seed) 在固定測試集上的 TP/TN/FP/FN、FPR、FNR、Recall、F1
  3) 兩個來源的 3 次平均 ± 標準差 總表 (CSV)

使用前請先確認：
  - BASELINE_PATH 指向的 xss_dataset.csv 格式為 header=None，欄位0=文字，欄位1=標籤(0白/1黑)
  - BLACK_SOURCES 兩個檔案格式，若非「單一文字欄位(payload)」請調整 load_black_only() 的 text_col
  - CLAUDE200_PATH 假設是「純文字檔，一行一個 payload」；若格式不同請調整 load_claude200()
"""

import os
os.environ["TF_KERAS"] = "1"

import random
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers as ppb
from transformers import TFBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from tensorflow.keras.utils import to_categorical

keras = tf.keras

# ========================= 全域設定 =========================
MODEL_SEED = 3            # 控制模型初始化與訓練隨機性，全程固定不變
DATA_SEEDS = [1, 2, 3]    # 控制訓練資料抽樣 (黑白各300)，每次重抽
TEST_SEED = 42            # 控制測試集白資料抽樣，只抽一次並固定

MAX_LEN = 256
EPOCHS = 10                # 第一階段找 best_epoch 用的上限
BATCH_SIZE = 128
MODEL_NAME = "prajjwal1/bert-tiny"

BASELINE_PATH = './res/train_data/xss_dataset.csv'
CLAUDE200_PATH = './res/train_data/successful_set_200.txt'

BLACK_SOURCES = {
    'combined_llmdriven': './res/stats/stage3_all_final_combined_llmdriven_now.csv',
    'offical_no_think':   './res/stats/stage3_offical_no_think.csv',
}

TEST_N = 200    # 測試集 黑/白 各抽多少 (黑=claude_200.txt全部, 白=baseline抽樣)
TRAIN_N = 366   # 訓練集 黑/白 各抽多少

OUTPUT_DIR = './outputs'
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


# ========================= 資料讀取 =========================
def load_baseline(path):
    """讀取 baseline xss_dataset.csv：header=None，欄位0=文字，欄位1=標籤(0/1)"""
    df = pd.read_csv(path, header=None)
    df = df.iloc[:, :2]
    df.columns = ['text', 'label']
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(int)
    return df


def load_black_only(path, text_col='payload'):
    """讀取純黑資料來源檔案 (欄位: index,payload,model,syntax,crs,...)，
       取 payload 欄位當文字，補上 label=1"""
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"{path} 找不到欄位 '{text_col}'，實際欄位為: {list(df.columns)}")
    out = pd.DataFrame()
    out['text'] = df[text_col].astype(str)
    out['label'] = 1
    return out


def load_claude200(path):
    """讀取 claude_200.txt：假設純文字檔，一行一個 payload，全部視為黑資料(label=1)
       若檔案其實是 csv 且有 payload 欄位，改用 load_black_only(path) 即可"""
    with open(path, encoding='utf-8') as f:
        lines = [line.rstrip('\n').rstrip('\r') for line in f]
    lines = [ln for ln in lines if ln.strip() != '']
    df = pd.DataFrame({'text': lines})
    df['label'] = 1
    return df


# ========================= 分詞工具 =========================
def tokenize_texts(tokenizer, texts, max_len=MAX_LEN):
    enc = tokenizer(
        list(texts),
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='np',
    )
    return enc['input_ids'], enc['attention_mask']


# ========================= 模型定義 =========================
class BertBiLSTMModel(tf.keras.Model):
    def __init__(self, model_name, num_classes=2):
        super().__init__()
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=False)
        )
        self.concat = tf.keras.layers.Concatenate()
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        input_ids = inputs[0]
        attention_mask = inputs[1]

        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, training=training)
        sequence_output = bert_outputs.last_hidden_state
        pooler_output = bert_outputs.pooler_output

        bilstm_output = self.bilstm(sequence_output)
        merged = self.concat([pooler_output, bilstm_output])
        return self.classifier(merged)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_and_compile_model():
    """重設 MODEL_SEED 後才建模，確保每次初始化權重相同"""
    set_all_seeds(MODEL_SEED)
    model = BertBiLSTMModel(model_name=MODEL_NAME, num_classes=2)

    dummy_ids = tf.zeros((2, MAX_LEN), dtype=tf.int32)
    dummy_mask = tf.ones((2, MAX_LEN), dtype=tf.int32)
    _ = model([dummy_ids, dummy_mask])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='BinaryCrossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall()],
    )
    return model


# ========================= 評估指標 =========================
def evaluate_on_test(model, test_input_ids, test_attention_mask, test_labels_int):
    preds = model.predict([test_input_ids, test_attention_mask], batch_size=BATCH_SIZE)
    pred_labels = preds.argmax(axis=1)

    # confusion_matrix(labels=[0,1]) -> 0=白(良性) 1=黑(惡意，視為正類)
    cm = confusion_matrix(test_labels_int, pred_labels, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float('nan')
    recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1 = f1_score(test_labels_int, pred_labels, pos_label=1, average='binary')

    return {
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'FPR': fpr, 'FNR': fnr, 'Recall': recall, 'F1': f1,
    }


# ========================= 主流程 =========================
def main():
    print("正在載入 tokenizer ...")
    tokenizer = ppb.BertTokenizer.from_pretrained("prajjwal1/bert-tiny")

    print("正在載入 baseline ...")
    baseline = load_baseline(BASELINE_PATH)
    baseline_white = baseline[baseline['label'] == 0].reset_index(drop=True)
    print(f"baseline 白: {len(baseline_white)} 筆")

    print("正在載入 claude_200.txt ...")
    claude200 = load_claude200(CLAUDE200_PATH)
    print(f"claude_200 黑: {len(claude200)} 筆")
    if len(claude200) != TEST_N:
        print(f"[提醒] claude_200.txt 實際筆數為 {len(claude200)}，跟 TEST_N={TEST_N} 不同，"
              f"將以實際筆數為準 (白資料也會抽同樣的數量)")

    test_black_n = len(claude200)

    # ---------- 固定測試集：黑=claude_200全部，白=baseline抽樣同樣數量，只抽一次 ----------
    test_white = baseline_white.sample(n=test_black_n, random_state=TEST_SEED)
    test_black = claude200.copy()
    test_df = pd.concat([test_white, test_black], ignore_index=True)
    test_df = test_df.sample(frac=1, random_state=TEST_SEED).reset_index(drop=True)  # 洗牌

    print(f"固定測試集: 白 {len(test_white)} + 黑 {len(test_black)} = {len(test_df)} 筆")

    print("正在對測試集做分詞...")
    test_input_ids, test_attention_mask = tokenize_texts(tokenizer, test_df['text'])
    test_labels_int = test_df['label'].values

    # ---------- 訓練用白資料池：baseline白 排除 測試集白 ----------
    white_train_pool = baseline_white.drop(index=test_white.index).reset_index(drop=True)
    print(f"可用於訓練抽樣的白資料池: {len(white_train_pool)} 筆 (已排除測試集重疊)")

    all_run_results = []

    for source_name, source_path in BLACK_SOURCES.items():
        print(f"\n===== 黑資料來源: {source_name} =====")
        black_pool = load_black_only(source_path)
        print(f"{source_name} 黑資料池大小: {len(black_pool)} 筆")

        for data_seed in DATA_SEEDS:
            print(f"\n--- 來源={source_name}, data_seed={data_seed} ---")

            # 1) 抽樣 300 黑 + 300 白 (data_seed 控制抽樣)
            train_black = black_pool.sample(n=TRAIN_N, random_state=data_seed)
            train_white = white_train_pool.sample(n=TRAIN_N, random_state=data_seed)
            train_df = pd.concat([train_black, train_white], ignore_index=True)
            train_df = train_df.sample(frac=1, random_state=data_seed).reset_index(drop=True)

            # 2) 分詞
            input_ids, attention_mask = tokenize_texts(tokenizer, train_df['text'])
            labels_cat = to_categorical(train_df['label'].values, num_classes=2)

            # 3) 切訓練/驗證 (用來找 best_epoch)，用 MODEL_SEED 控制這個切分
            (trainingData, validData,
             trainingMask, validMask,
             trainingLabel, validLabel) = train_test_split(
                input_ids, attention_mask, labels_cat,
                test_size=0.1, shuffle=True,
                stratify=labels_cat, random_state=MODEL_SEED,
            )

            # 4) 建模 (每次重設 MODEL_SEED，確保初始權重相同)
            model = build_and_compile_model()

            history = model.fit(
                [trainingData, trainingMask],
                trainingLabel,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=([validData, validMask], validLabel),
                verbose=1,
            )
            best_epoch = int(np.argmin(history.history['val_loss']) + 1)
            print(f"best_epoch = {best_epoch}")

            # 5) 用全部 600 筆訓練資料，以 best_epoch 重新訓練最終模型
            final_model = build_and_compile_model()
            final_model.fit(
                [input_ids, attention_mask],
                labels_cat,
                epochs=best_epoch,
                batch_size=BATCH_SIZE,
                verbose=1,
            )

            # 6) 儲存模型
            model_filename = f"{source_name}_seed{data_seed}.keras"
            model_path = os.path.join(MODEL_DIR, model_filename)
            final_model.save(model_path)
            print(f"模型已儲存: {model_path}")

            # 7) 在固定測試集上評估
            metrics = evaluate_on_test(final_model, test_input_ids, test_attention_mask, test_labels_int)
            metrics.update({
                'source': source_name,
                'data_seed': data_seed,
                'best_epoch': best_epoch,
            })
            print(f"測試結果: {metrics}")
            all_run_results.append(metrics)

            # 釋放記憶體
            del model, final_model
            tf.keras.backend.clear_session()

    # ========================= 結果彙整輸出 =========================
    results_df = pd.DataFrame(all_run_results)
    results_df = results_df[['source', 'data_seed', 'best_epoch',
                              'TP', 'TN', 'FP', 'FN', 'FPR', 'FNR', 'Recall', 'F1']]

    detail_path = os.path.join(OUTPUT_DIR, 'per_run_results.csv')
    results_df.to_csv(detail_path, index=False, encoding='utf-8-sig')
    print(f"\n每次執行的詳細結果已儲存: {detail_path}")
    print(results_df.to_string(index=False))

    # 每個來源的平均 ± 標準差
    summary = results_df.groupby('source')[['FPR', 'FNR', 'Recall', 'F1']].agg(['mean', 'std'])
    summary_path = os.path.join(OUTPUT_DIR, 'summary_mean_std.csv')
    summary.to_csv(summary_path, encoding='utf-8-sig')
    print(f"\n各來源 3 次平均 ± 標準差 已儲存: {summary_path}")
    print(summary)


if __name__ == '__main__':
    main()