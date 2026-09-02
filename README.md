1. 訓練 Baseline ML-Rescue
步驟是 jupyter notebook XSS_with_TinyBERT_Training.ipynb(Baseline用kaggle的res/train_data/ xss_dataset.csv訓練完權重是BestModel_TinyBERT_1.keras)

2. 建立 200 筆 seed payload
python Gpt_XSS_Mutations_Filter.py(先篩選出可以通過BestModel_TinyBERT_1.keras以及dom的payload 蒐集完的結果為res/train_data/successful_set_200.txt)

3. 產生 Claude 變異樣本
python Claude_XSS_Mutations_No_Ref.py (Claude, res/claude_test_payload/Claude)
python Claude_XSS_Mutations_Local_Ref.py(Claude + CRS Audit Log, res/claude_test_payload/Claude_CRS_Audit_Log)
python Claude_XSS_Mutations_Official_Ref.py (Claude + CRS Reference, , res/claude_test_payload/Claude_CRS_Reference)

4. 統一執行 PL、ML 與 CRS 驗證
modsecurity-crs-docker 的 /etc/modsecurity.d/setup.conf 要設定 SecRuleEngine On, Include /etc/modsecurity.d/owasp-crs/crs-setup.conf, Include /etc/modsecurity.d/owasp-crs/rules/*.conf, SecAuditEngine On, SecAuditLog /tmp/modsec_audit.log, SecAuditLogParts ABIJDEFHZ, SecAuditLogType Serial
最後 python Group_All.py 對變異樣本過濾 PL、ML 後標註 CRS 的Audit Log資訊

5. 分析結構多樣性
python "Coverage_Diversity Comparison.py" (分析對手以及Claude生成的結構多樣性比較)
python "Char_Tsne.py" (用 tsne 畫二維圖)

6. 執行結構距離取樣消融實驗
python XSS_with_TinyBERT_Selete_Strategy.py 

7. 使用最遠(Distant) 20% 樣本微調模型
python XSS_with_TinyBERT_Full_Training.py (產出由Distant 訓練出的BestModel_TinyBERT_1_FT_distant.keras以及 Baseline 加上剩餘的80% 資料)

8. 建立 CRS Snapshot
python Run_Crs_Snapshot.py(對 Kaggle 的 xss_dataset.csv 以及 xss_dataset.csv + 剩餘的80% 做一次 CRS 標註) 

9. 評估 CRS + ML-Rescue
python Run_Crs_Snapshot_Resuce.py (評估 1 到 10 各個門檻)
