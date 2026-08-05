#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import subprocess

import pandas as pd
import requests
from tqdm import tqdm

# ===================== Config =====================
INPUT_FILE = "res/train_data/exported_remaining80/xss_dataset_baseline_plus_remaining80_distant.csv"
DOCKER_CONTAINER = "waf"
MODSEC_LOG_PATH = "/tmp/modsec_audit.log"
POST_URL = "http://localhost:80/comment"

SLEEP_AFTER_POST = 0.15
LOG_WRITE_WAIT = 0.05

OUT_DIR = "res/crs_snapshots"
os.makedirs(OUT_DIR, exist_ok=True)

# ===================== Regex =====================
ID_REGEX = re.compile(r'\[id\s*["\']?(\d{3,9})["\']?.*?\]')
REF_REGEX = re.compile(r'\[ref\s*"([^"]*)"\]')
UNIQUE_ID_REGEX = re.compile(r'\[unique_id\s*"([^"]+)"\]')
TOTAL_SCORE_REGEX = re.compile(r'Total Score:\s*(\d+)', re.IGNORECASE)
TX_PATTERN = re.compile(r'---[A-Za-z0-9]+---A--.*?---[A-Za-z0-9]+---Z--', re.DOTALL)

# ===================== Docker / CRS =====================

def read_docker_audit_log():
    try:
        res = subprocess.run(
            ["docker", "exec", DOCKER_CONTAINER, "cat", MODSEC_LOG_PATH],
            capture_output=True, text=True, check=True
        )
        return res.stdout
    except Exception:
        return ""

def clear_audit_log():
    try:
        subprocess.run(
            ["docker", "exec", DOCKER_CONTAINER, "truncate", "-s", "0", MODSEC_LOG_PATH],
            capture_output=True,
            text=True,
            check=True
        )
    except Exception:
        pass

def extract_rules_from_tx(tx_text):
    rule_ids = ID_REGEX.findall(tx_text)
    refs = REF_REGEX.findall(tx_text)

    total_score = 0
    m = TOTAL_SCORE_REGEX.search(tx_text)
    if m:
        try:
            total_score = int(m.group(1))
        except Exception:
            total_score = 0

    uid_match = UNIQUE_ID_REGEX.search(tx_text)
    uid = uid_match.group(1) if uid_match else ""

    return rule_ids, refs, total_score, uid

def extract_latest_transaction(audit_text):
    matches = list(TX_PATTERN.finditer(audit_text))
    if not matches:
        return None
    return matches[-1].group(0)

def post_and_check_modsecurity(payload):
    clear_audit_log()

    try:
        resp = requests.post(
            POST_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"message": payload},
            timeout=3
        )
        detected = 0 if resp.status_code == 200 else 1
    except Exception:
        detected = 1

    time.sleep(SLEEP_AFTER_POST + LOG_WRITE_WAIT)

    audit_text = read_docker_audit_log()
    tx = extract_latest_transaction(audit_text)

    if not tx:
        return detected, [], [], 0, ""

    rule_ids, refs, total_score, uid = extract_rules_from_tx(tx)
    return detected, rule_ids, refs, total_score, uid

# ===================== Main =====================

def main():
    print(f"[INFO] Loading dataset: {INPUT_FILE}")
    df = pd.read_csv(
        INPUT_FILE,
        header=None,
        names=["payload", "label"],
        encoding="utf-8-sig"
    )

    print(f"[INFO] Total samples: {len(df)}")

    snapshot_rows = []

    print("\n==== Running CRS snapshot (deterministic) ====")

    for i, (payload, label) in enumerate(
        tqdm(df.values, total=len(df), desc="CRS Snapshot"),
        start=1
    ):
        payload = str(payload)

        detected, rule_ids, refs, total_score, uid = post_and_check_modsecurity(payload)

        crs_status = "bypass" if detected == 0 else "block"

        snapshot_rows.append({
            "index": i,
            "payload": payload,
            "model": -1,             # 尚未跑 ML
            "syntax": -1,            # 尚未跑 Playwright
            "crs": crs_status,
            "rule_count": len(rule_ids),
            "rule_ids": ",".join(rule_ids),
            "rule_refs": " || ".join(refs),
            "total_score": total_score,
            "unique_id": uid
        })

    snapshot_df = pd.DataFrame(snapshot_rows)

    output_path = os.path.join(OUT_DIR, "crs_snapshot_test800.csv")
    snapshot_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ CRS snapshot saved to: {output_path}")
    print("This file will be used for ALL offline experiments.")

if __name__ == "__main__":
    main()