#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, requests, pandas as pd, asyncio, time as t, subprocess, random, numpy as np, tensorflow as tf
from datetime import datetime
from playwright.async_api import async_playwright
from penetration_test_normal.test import test_payload
from BERT_detector import XSSDetector_BERT

# =============================
# 固定隨機種子
# =============================
os.environ["PYTHONHASHSEED"] = "0"
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# ---------------- Config ----------------
INPUT_FILE = "res/claude_test_payload/Claude_CRS_Referenc.txt" #claude_crs_think.txt #claude_no_crs_think.txt
DOCKER_CONTAINER = "waf"
MODSEC_LOG_PATH = "/tmp/modsec_audit.log"
POST_URL = "http://localhost:80/comment"
HTML_PATH = "http://127.0.0.1:5500/src/penetration_test_normal/test_innerHTML.html"
SLEEP_AFTER_POST, LOG_WRITE_WAIT = 0.15, 0.05

BERT_MODEL_PATH = "res/BestModel_TinyBERT_1.keras"
BERT_MODEL_TYPE = "tinyBERT"
BERT_MAX_LENGTH = 256

CHECKPOINT_STAGE1 = "res/checkpoint/stage1_checkpoint.csv"
CHECKPOINT_STAGE2 = "res/checkpoint/stage2_checkpoint.csv"

# 排除的 rule id
EXCLUDE_RULE_ID = "949110"

# ---------------- Regex ----------------
ID_REGEX = re.compile(r'\[id\s*["\']?(\d{3,9})["\']?.*?\]')
MSG_REGEX = re.compile(r'\[msg\s*["\']([^"\']+)["\']\]')
REF_REGEX = re.compile(r'\[ref\s*"([^"]*)"\]')
UNIQUE_ID_REGEX = re.compile(r'\[unique_id\s*"([^"]+)"\]')
TOTAL_SCORE_REGEX = re.compile(r'Total Score:\s*(\d+)', re.IGNORECASE)

# ---------------- 讀取檔案函數 ----------------
def load_payloads(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.txt':
        print(f"[INFO] 偵測到 TXT 格式，逐行讀取...")
        with open(filepath, encoding="utf-8") as f:
            payloads = [line.strip() for line in f if line.strip()]

    elif ext == '.csv':
        print(f"[INFO] 偵測到 CSV 格式，讀取 payload 欄...")
        df = pd.read_csv(filepath, encoding="utf-8-sig")

        if "payload" in df.columns:
            payloads = df["payload"].astype(str).str.strip().tolist()
        else:
            print("[WARN] 找不到 payload 欄，改讀第一欄")
            payloads = df.iloc[:, 0].astype(str).str.strip().tolist()

        payloads = [p for p in payloads if p]

    else:
        raise ValueError(f"不支援的檔案格式: {ext}，請使用 .txt 或 .csv")

    original_count = len(payloads)
    payloads = list(dict.fromkeys(payloads))
    duplicates = original_count - len(payloads)

    if duplicates > 0:
        print(f"[INFO] 移除了 {duplicates} 筆重複的 payload")

    print(f"✅ 已載入 {len(payloads)} 筆 payload")
    return payloads

# ---------------- CRS functions ----------------
def read_docker_audit_log(container_name: str, log_path: str):
    try:
        cmd = ["docker", "exec", container_name, "cat", log_path]
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',   # ← 加這行
            #errors='replace',   # ← 遇到無法解碼的字元用 ? 替代，不會直接崩潰
            check=True
        )

        return res.stdout
    except Exception as e:
        print(f"[read_docker_audit_log] Failed: {e}")
        return ""

def extract_rules_from_entry(entry_text: str):
    rule_ids = ID_REGEX.findall(entry_text)
    messages = MSG_REGEX.findall(entry_text)
    refs = REF_REGEX.findall(entry_text)
    total_score = int(TOTAL_SCORE_REGEX.search(entry_text).group(1)) if TOTAL_SCORE_REGEX.search(entry_text) else 0
    uid_match = UNIQUE_ID_REGEX.search(entry_text)
    unique_id = uid_match.group(1) if uid_match else ""
    return rule_ids, messages, refs, total_score, unique_id

def find_transaction_by_uid(uid: str, audit_text: str):
    for match in re.finditer(r'---[A-Za-z0-9]+---A--.*?---[A-Za-z0-9]+---Z--', audit_text, re.DOTALL):
        entry = match.group(0)
        if uid in entry:
            return extract_rules_from_entry(entry), entry
    return ([], [], [], 0, ""), None

def get_latest_uid(audit_text: str):
    matches = UNIQUE_ID_REGEX.findall(audit_text)
    return matches[-1] if matches else None

def post_and_check_modsecurity(payload: str):
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"message": payload}
    detected = 1
    try:
        resp = requests.post(POST_URL, headers=headers, data=data, timeout=3)
        detected = 0 if resp.status_code == 200 else 1
    except Exception:
        detected = 1
    t.sleep(SLEEP_AFTER_POST + LOG_WRITE_WAIT)
    audit_text = read_docker_audit_log(DOCKER_CONTAINER, MODSEC_LOG_PATH)
    uid = get_latest_uid(audit_text)
    rule_ids, messages, refs, total_score, unique_id = [], [], [], 0, ""
    if uid:
        (rule_ids, messages, refs, total_score, unique_id), _ = find_transaction_by_uid(uid, audit_text)
    return detected, rule_ids, messages, refs, total_score, unique_id

# ---------------- checkpoint 讀寫工具 ----------------
def load_checkpoint(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="utf-8-sig")
        done = set(df["index"].tolist())
        print(f"[CHECKPOINT] 讀取到 {len(done)} 筆已完成紀錄，將從中斷點續跑")
        return df, done
    return pd.DataFrame(), set()

def append_checkpoint(path, row: dict):
    df_new = pd.DataFrame([row])
    write_header = not os.path.exists(path)
    df_new.to_csv(path, mode="a", header=write_header, index=False, encoding="utf-8-sig")

# ==========================================================
# 🧪 Stage 1：模型 + Playwright（含 checkpoint）
# ==========================================================
async def stage1_model_playwright(detector, filepath):
    print("\n=== [Stage 1] 模型 + Playwright 測試 ===")
    payloads = load_payloads(filepath)

    checkpoint_df, done_indices = load_checkpoint(CHECKPOINT_STAGE1)
    rows = checkpoint_df.to_dict("records") if not checkpoint_df.empty else []

    async with async_playwright() as pw:
        for i, p in enumerate(payloads, 1):
            if i in done_indices:
                print(f"[{i}] ⏭ 已完成，跳過")
                continue
            det = detector.is_xss(p)
            try:
                syn = await test_payload(pw, HTML_PATH, p)
            except Exception as e:
                print(f"[{i}] syntax error: {e}")
                syn = False
            row = {
                "index": i,
                "payload": p,
                "model": det,
                "syntax": 1 if syn else 0
            }
            rows.append(row)
            append_checkpoint(CHECKPOINT_STAGE1, row)
            print(f"[{i}] model={det} syntax={syn}")

    df = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return df, timestamp

# ==========================================================
# 🧰 Stage 2：CRS 檢測（含 checkpoint）
# ==========================================================
async def stage2_crs_only(filepath, timestamp):
    print("\n=== [Stage 2] CRS (WAF) 測試 ===")
    payloads = load_payloads(filepath)

    checkpoint_df, done_indices = load_checkpoint(CHECKPOINT_STAGE2)
    rows = checkpoint_df.to_dict("records") if not checkpoint_df.empty else []

    for i, p in enumerate(payloads, 1):
        if i in done_indices:
            print(f"[{i}] ⏭ 已完成，跳過")
            continue
        detected, rule_ids, messages, refs, total_score, uid = await asyncio.to_thread(post_and_check_modsecurity, p)
        crs = "bypass" if detected == 0 else "block"
        row = {
            "index": i,
            "payload": p,
            "crs": crs,
            "rule_count": len(rule_ids),
            "rule_ids": ",".join(rule_ids),
            "rule_refs": " || ".join(refs),
            "total_score": total_score,
            "unique_id": uid
        }
        rows.append(row)
        append_checkpoint(CHECKPOINT_STAGE2, row)
        print(f"[{i}] crs={crs} rules={len(rule_ids)}")

    df = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
    return df

# ==========================================================
# 📊 統計工具：計算排除 949110 後的 rule_count
# ==========================================================
def calc_filtered_rule_count(rule_ids_str: str) -> int:
    """從逗號分隔的 rule_ids 字串中，排除 EXCLUDE_RULE_ID 後回傳數量"""
    if pd.isna(rule_ids_str) or rule_ids_str == "":
        return 0
    ids = [r.strip() for r in str(rule_ids_str).split(",") if r.strip()]
    filtered = [r for r in ids if r != EXCLUDE_RULE_ID]
    return len(filtered)

def compute_threshold_stats(merged: pd.DataFrame) -> pd.DataFrame:
    """
    對 rule_count 門檻 1~10 分別計算四項指標：
      - Playwright 觸發率  (syntax=1)
      - CRS 穿透率         (crs=bypass)
      - ML 穿透率          (model=0)
      - PW+CRS+ML 穿透率   (三者同時)

    CRS 穿透定義：filtered_rule_count < threshold（即觸發規則數不足門檻視為穿透）
    """
    total = len(merged)

    # 預先算好排除 949110 後的 rule count
    merged = merged.copy()
    merged["filtered_rule_count"] = merged["rule_ids"].apply(calc_filtered_rule_count)

    pw_triggered = merged["syntax"] == 1
    ml_bypass    = merged["model"] == 0

    records = []
    for threshold in range(1, 11):
        # 在此門檻下，filtered_rule_count < threshold 視為 CRS 穿透
        crs_bypass_mask = merged["filtered_rule_count"] < threshold

        pw_n   = pw_triggered.sum()
        crs_n  = crs_bypass_mask.sum()
        ml_n   = ml_bypass.sum()
        all_n  = (pw_triggered & crs_bypass_mask & ml_bypass).sum()

        records.append({
            "rule_count_threshold":  threshold,
            "total":                 total,
            "pw_triggered":          int(pw_n),
            "crs_bypassed":          int(crs_n),
            "ml_bypassed":           int(ml_n),
            "all_bypassed":          int(all_n),
            "pw_trigger_rate":       round(pw_n   / total, 4),
            "crs_bypass_rate":       round(crs_n  / total, 4),
            "ml_bypass_rate":        round(ml_n   / total, 4),
            "pw_crs_ml_bypass_rate": round(all_n  / total, 4),
        })

    return pd.DataFrame(records)

# ==========================================================
# 📊 Stage 3：整合結果
# ==========================================================
def stage3_merge_results(df1, df2, timestamp):
    print("\n=== [Stage 3] 整合結果 ===")
    merged = pd.merge(df1, df2, on=["index", "payload"], how="inner")

    # ── 輸出 1：完整結果，不做篩選 ────────────────────────
    os.makedirs("res/stats", exist_ok=True)

    all_df = merged.copy()
    out_all = f"res/stats/stage3_all_results_{timestamp}.csv"
    all_df.to_csv(out_all, index=False, encoding="utf-8-sig")
    print(f"[Stage3] 已儲存完整測試結果 {len(all_df)} 筆 -> {out_all}")

    # ── 輸出 2：條件下結果 ───────────────────────────────
    # 目前條件：Playwright 可觸發 + ML bypass
    # 如果你要加 CRS bypass，可以看下面補充版本
    cond = (
        (merged["syntax"] == 1) &
        (merged["model"] == 0)
    )

    final = merged[cond].copy()

    out_final = f"res/stats/stage3_condition_results_{timestamp}.csv"
    final.to_csv(out_final, index=False, encoding="utf-8-sig")
    print(f"[Stage3] 已儲存條件篩選結果 {len(final)} 筆 -> {out_final}")

    # ── 訓練資料輸出：通常用條件下結果 ───────────────────
    train_df = final[["payload", "syntax"]].copy()

    train_df  = final[["payload", "syntax"]].copy()
    os.makedirs("res/train_data", exist_ok=True)
    out_train = f"res/train_data/xss_dataset_{timestamp}.csv"
    train_df.to_csv(out_train, index=False, header=False, encoding="utf-8-sig")
    print(f"[Stage3.5] 已輸出訓練資料 {len(train_df)} 筆 -> {out_train}")

    # ── 各門檻統計 ────────────────────────────────────────
    stats_df = compute_threshold_stats(merged)

    # Console 輸出
    print("\n========== 📊 各 rule_count 門檻穿透率統計（排除 949110）==========")
    print(f"{'門檻':>4}  {'總數':>6}  {'PW觸發':>6}  {'CRS穿透':>7}  {'ML穿透':>6}  {'全穿透':>6}  "
          f"{'PW觸發率':>8}  {'CRS穿透率':>9}  {'ML穿透率':>8}  {'全穿透率':>8}")
    print("-" * 95)
    for _, row in stats_df.iterrows():
        print(
            f"  {int(row['rule_count_threshold']):>2}  "
            f"{int(row['total']):>6}  "
            f"{int(row['pw_triggered']):>6}  "
            f"{int(row['crs_bypassed']):>7}  "
            f"{int(row['ml_bypassed']):>6}  "
            f"{int(row['all_bypassed']):>6}  "
            f"{row['pw_trigger_rate']:>8.2%}  "
            f"{row['crs_bypass_rate']:>9.2%}  "
            f"{row['ml_bypass_rate']:>8.2%}  "
            f"{row['pw_crs_ml_bypass_rate']:>8.2%}"
        )
    print("=" * 95)

    # CSV 輸出
    stats_out = f"res/stats/stage3_threshold_stats_{timestamp}.csv"
    stats_df.to_csv(stats_out, index=False, encoding="utf-8-sig")
    print(f"[Stage3] 門檻統計已儲存 -> {stats_out}\n")

    # ── 清除 checkpoint ───────────────────────────────────
    for ckpt in [CHECKPOINT_STAGE1, CHECKPOINT_STAGE2]:
        if os.path.exists(ckpt):
            os.remove(ckpt)
            print(f"[CHECKPOINT] 已清除暫存檔 {ckpt}")

    return final

# ==========================================================
# 🚀 主執行點
# ==========================================================
async def main():
    print("[INFO] 初始化 TinyBERT 模型...")
    det = XSSDetector_BERT(BERT_MODEL_PATH, BERT_MODEL_TYPE, BERT_MAX_LENGTH)
    det.model.trainable = False
    try:
        det.model.compile()
    except:
        pass

    df1, timestamp = await stage1_model_playwright(det, INPUT_FILE)
    df2 = await stage2_crs_only(INPUT_FILE, timestamp)
    stage3_merge_results(df1, df2, timestamp)

if __name__ == "__main__":
    asyncio.run(main())