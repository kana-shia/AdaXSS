import json
import asyncio
import os
import sys
from playwright.async_api import async_playwright

def load_payloads(file_path: str) -> dict:
    payloads = {"innerHTML": []}
    if not os.path.exists(file_path):
        print(f"找不到檔案：{file_path}")
        return payloads
    with open(file_path, "rb") as f:
        for line in f:
            try:
                payload = line.decode("utf-8").strip()
                if payload:
                    payloads["innerHTML"].append(payload)
            except UnicodeDecodeError:
                continue
    return payloads

async def test_payload(playwright, html_path: str, payload: str) -> bool:
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context()
    context.alert_triggered = False

    async def on_dialog(dialog):
        context.alert_triggered = True
        await dialog.dismiss()

    page = await context.new_page()
    page.on("dialog", on_dialog)

    await page.goto(html_path)
    await page.evaluate(
    "payload => { window.name = payload; }",
    payload
)
    await page.reload()
    await asyncio.sleep(4)
    await browser.close()

    return context.alert_triggered

async def main(payload_file_path: str):
    all_payloads = load_payloads(payload_file_path)
    if not all_payloads["innerHTML"]:
        print("沒有載入到任何 payload")
        return

    async with async_playwright() as playwright:
        print("=== 開始測試 XSS Payloads ===")
        for payloads_typename, payloads in all_payloads.items():
            html_path = f"http://127.0.0.1:5500/src/penetration_test_normal/test_{payloads_typename}.html"
            print(f"Testing: {payloads_typename}")
            for i, payload in enumerate(payloads):
                try:
                    result = await test_payload(playwright, html_path, payload)
                    if result:
                        print(f"[{i}] Triggered | {payload}")
                except Exception as e:
                    print(f"[{i}] Error | {payload}")
                    print(f"Error: {e}")
            print(f"=== 完成測試: {html_path} ===\n\n")

if __name__ == "__main__":
    payload_file = sys.argv[1] if len(sys.argv) > 1 else "res/llm_output/default_input.txt"
    asyncio.run(main(payload_file))
