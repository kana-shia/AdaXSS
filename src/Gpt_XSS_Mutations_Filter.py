import asyncio
from BERT_detector import XSSDetector_BERT
from penetration_test_normal.test import test_payload
from playwright.async_api import async_playwright
import os
import matplotlib
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

TARGET_SUCCESS = 1

async def process_file(temp_str, time_str, detector):
    

    input_path = f"res/llm_output/llm_output_temp_{temp_str}_{time_str}.txt"

    with open(input_path, "r", encoding="utf-8") as file_read:
        payloads = [line.strip() for line in file_read if line.strip()]

    html_path = "http://127.0.0.1:5500/src/penetration_test_normal/test_innerHTML.html"

    syntax_success_count = 0
    model_bypass_count = 0
    overall_success_count = 0
    bypassed_payloads = []  

    async with async_playwright() as playwright:
        for i, payload in enumerate(payloads):
            detector_result = detector.is_xss(payload)

            try:
                syntax_result = await test_payload(playwright, html_path, payload)
            except Exception as e:
                print(f"[{temp_str}] #{i} ❌ syntax error: {e}")
                syntax_result = False

            if syntax_result:
                syntax_success_count += 1

            if detector_result == 0:
                model_bypass_count += 1

            if syntax_result and detector_result == 0:
                overall_success_count += 1
                bypassed_payloads.append(payload)  # ← 新增成功繞過的 payload 紀錄

            print(f"[{temp_str}] #{i} → detector={detector_result}, syntax={syntax_result}")

    total = len(payloads)
    syntax_rate = syntax_success_count / total
    bypass_rate = model_bypass_count / total
    overall_rate = overall_success_count / total

    print(f"\n📊 統計結果（溫度 {temp_str}, 共 {total} 筆）：")
    print(f"✅ 語法成功率     = {syntax_rate:.2%}")
    print(f"🛡️ 模型繞過成功率 = {bypass_rate:.2%}")
    print(f"🏆 整體成功率     = {overall_rate:.2%}")

    return {
        "Temperature": float(temp_str),
        "Total Payloads": total,
        "Syntax Success Rate": syntax_rate,
        "Model Bypass Rate": bypass_rate,
        "Overall Success Rate": overall_rate,
        "Bypassed Payloads": bypassed_payloads
    }
all_success_payloads_set = set()

async def main():
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    detector = XSSDetector_BERT(
    model_path="res/BestModel_TinyBERT_1.keras",
    model_type="tinyBERT",
    max_length=256
)

    
    
    results = []
    temp = 0.7
    rounds = 99999
    timestamp = datetime.now().strftime("%m%d%H%M")
    

    for i in range(rounds):
        print(f"[INFO] 正在產生第 {i + 1} 輪的 payload（temp={temp}）...")

        messages = [
            {"role": "system", "content": "你是一位 Web 資安專家，擅長製作對抗性 XSS 攻擊樣本"},
        ]

        

        messages.append({"role": "user",
         "content": (
            "請幫我生成 30 筆語法正確、能在現代瀏覽器（如 Chrome）中自動觸發執行的 XSS payload，需符合以下條件：\n\n"
            "1. 不需任何使用者互動就能自動執行（不能使用 onclick、onmouseover、onfocus 等互動事件）。\n"
            "2. 必須使用自動觸發 JavaScript 的 HTML 元素，例如 <img onerror>、<svg onload>、<meta http-equiv>、<iframe srcdoc> 等，也可使用冷門元素如 <object>、<marquee>、<details>。\n"
            "3. 對 alert、script、javascript 等關鍵字進行混淆處理，例如：\n"
            "   - 字串拼接（如 'al' + 'ert'）\n"
            "   - Unicode 編碼（如 \\u0061\\u006c\\u0065\\u0072\\u0074）\n"
            "   - HTML 實體編碼（如 &#x61;&#x6C;&#x65;&#x72;&#x74;）\n"
            "   - Base64 編碼搭配 atob 解碼\n"
            "   - 或其他變形方式\n"
            "4. Payload 應具備繞過機器學習模型（例如 CNN、RNN）檢測的能力，避免使用常見模板（如 <script>alert(1)</script>）或明顯可辨識的語法結構。\n\n"
            "請只輸出 payload 字串，每一筆一行，不要加入任何解釋、描述或標示，除了 HTML 標籤本身外，不要加入任何文字。"
            )
        })

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=temp,
            frequency_penalty=0.0
        )

        print(i)
        payload = response.choices[0].message.content.strip()

        filename = f"res/llm_output/llm_output_temp_{temp}_{timestamp}_{i}.txt"
        os.makedirs("res/llm_output", exist_ok=True)
        with open(filename, "w", encoding="utf-8") as file:
            file.write(payload)

        result = await process_file(temp, f"{timestamp}_{i}", detector)

        new_payloads = result["Bypassed Payloads"]
        new_unique = [p for p in new_payloads if p not in all_success_payloads_set]
        if new_unique:
            os.makedirs("res/success", exist_ok=True)
            with open(f"res/success/success_payloads_{temp}_{timestamp}.txt", "a", encoding="utf-8") as f:
                for p in new_unique:
                    f.write(p + "\n")
            all_success_payloads_set.update(new_unique)

        print(f"[PROGRESS] 成功累積：{len(all_success_payloads_set)}/{TARGET_SUCCESS}")
        if len(all_success_payloads_set) >= TARGET_SUCCESS:
            print(f"[STOP] 已累積 {len(all_success_payloads_set)} 筆成功樣本，停止")
            break


        Temp = result["Temperature"]
        total_payloads = result["Total Payloads"]
        syntax_rate = result["Syntax Success Rate"]
        bypass_rate = result["Model Bypass Rate"]
        overall_rate = result["Overall Success Rate"]
        results.append({
            "Round": i + 1,
            "Temperature": Temp,
            "Total Payloads": total_payloads,
            "Syntax Success Rate": syntax_rate,
            "Model Bypass Rate": bypass_rate,
            "Overall Success Rate": overall_rate
        })
        
       

        await asyncio.sleep(1.0)


    print("Finished!")

if __name__ == "__main__":
    asyncio.run(main())
