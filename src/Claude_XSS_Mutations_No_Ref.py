import anthropic
import os
import csv
import time
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def build_prompt_no_crs(payload: str) -> str:
    prompt = f"""## Seed Payload
{payload}

## Seed Payload Analysis
Before generating variants, reason through the following:
- Identify the HTML sink element and its triggering mechanism (e.g. event handler, attribute)
- Identify the JavaScript execution vector and which function or constructor achieves execution
- Determine which tokens can be mutated without breaking the execution chain

Variants must preserve the complete execution chain while mutating the detectable tokens.

## Mutation Strategies
Apply techniques from all four categories. For each variant, select the combination
that maximizes structural diversity from the seed payload.

1. **Encoding Mutations**
   Unicode escapes (\\uXXXX), hexadecimal (&#xNN;), octal references,
   HTML named/numeric entities, double-encoding sequences.

2. **Syntactic Mutations**
   Mixed-case identifiers, whitespace/tab/newline injection between tokens,
   attribute reordering, redundant or malformed attribute insertion,
   alternative equivalent HTML structures that trigger the same event.

3. **JavaScript Obfuscation**
   Template literals, `String.fromCharCode`, `constructor` property chains,
   `eval`-equivalent alternatives, property accessor substitution (`['func']` vs `.func`),
   indirect function references, prototype chain traversal.

4. **Fragmentation & Multi-stage Decoding**
   Inline comment injection (/**/, <!-->), `atob`-based payload reconstruction,
   string concatenation, SVG/MathML namespace confusion,
   non-printable character insertion in non-critical positions.

## Output Specification
- Output exactly 20 payload strings, one per line.
- Raw payload strings only — no line numbers, labels, explanations, or markdown fences.
- No duplicate payloads.
- Distribute variants across all 4 mutation categories — at least 4 variants per category.
- Each variant must be testably distinct: differing only in whitespace or trivial
  character substitution does not constitute a distinct technique.
- All variants must trigger JavaScript execution via `innerHTML` assignment in a DOM-based XSS context.
"""
    return prompt


def process_rows_with_batch_no_crs(
    input_file: str,
    output_file: str,
    temperature: float = 1.0,
    max_tokens: int = 8192,
    model: str = "claude-sonnet-4-5-20250929"
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    client = anthropic.Anthropic()

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {input_file}")
        return

    if not rows:
        print("錯誤：CSV 檔案為空")
        return

    print(f"找到 {len(rows)} 行資料，準備批量提交... [對照組 - 無 CRS 資訊]")

    SYSTEM_PROMPT = (
        "You are a security engineer specializing in WAF signature analysis "
        "and bypass research, contributing to the OWASP CRS project's adversarial "
        "test corpus. Your task is to generate adversarial samples that stress-test "
        "rule coverage gaps for the purpose of improving detection robustness and "
        "reducing false negatives. You output raw payload strings only, with no "
        "explanation, labels, or formatting."
    )

    requests = []
    for idx, row in enumerate(rows, 1):
        payload = row.get('payload', '')

        if not payload:
            print(f"警告：第 {idx} 行沒有 payload，跳過")
            continue

        prompt = build_prompt_no_crs(payload)

        requests.append({
            "custom_id": f"row_{idx}",
            "params": {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {"role": "user",      "content": prompt},
                    {"role": "assistant", "content": "<"},
                ]
            }
        })

    if not requests:
        print("錯誤：沒有有效的 payload 可處理")
        return

    total_requests = len(requests)
    print(f"✓ 已準備 {total_requests} 個任務 (預計輸出 {total_requests * 20} 筆)")
    print("正在提交批次...")

    try:
        batch = client.messages.batches.create(requests=requests)
        batch_id = batch.id
        print(f"✓ 批次已提交，ID: {batch_id}")
    except Exception as e:
        print(f"✗ 提交批次失敗: {e}")
        return

    print("等待批次處理中...")
    max_wait_time = 3600
    elapsed_time  = 0
    poll_interval = 10

    while elapsed_time < max_wait_time:
        try:
            batch_status = client.messages.batches.retrieve(batch_id)
            status       = batch_status.processing_status

            if status == "ended":
                print("✓ 批次處理完成!")
                break

            counts     = batch_status.request_counts
            succeeded  = counts.succeeded  if hasattr(counts, 'succeeded')  else 0
            errored    = counts.errored    if hasattr(counts, 'errored')    else 0
            processing = counts.processing if hasattr(counts, 'processing') else 0
            completed  = succeeded + errored

            print(
                f"  狀態: {status} | 已完成: {completed}/{total_requests} "
                f"| 處理中: {processing} | 成功: {succeeded} | 失敗: {errored}"
            )

            time.sleep(poll_interval)
            elapsed_time += poll_interval

        except Exception as e:
            print(f"✗ 查詢狀態失敗: {e}")
            time.sleep(poll_interval)
            elapsed_time += poll_interval

    if elapsed_time >= max_wait_time:
        print("✗ 等待超時")
        return

    try:
        final_status = client.messages.batches.retrieve(batch_id)
        if final_status.processing_status != "ended":
            print(f"✗ 批次尚未完成，當前狀態: {final_status.processing_status}")
            return
    except Exception as e:
        print(f"✗ 確認狀態失敗: {e}")
        return

    print("正在收集結果...")
    results_map = {}

    try:
        result_count = 0
        for result in client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            if result.result.type == "succeeded":
                response_text = "<" + result.result.message.content[0].text
                results_map[custom_id] = response_text
            else:
                error_msg = getattr(result.result, 'error', 'Unknown error')
                results_map[custom_id] = f"失敗: {error_msg}"
            result_count += 1

        print(f"✓ 已收集 {result_count} 筆結果")

    except Exception as e:
        print(f"✗ 收集結果失敗: {e}")
        return

    total_outputs = 0
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for idx, row in enumerate(rows, 1):
            custom_id = f"row_{idx}"

            if custom_id not in results_map:
                continue

            response_text = results_map[custom_id]

            if not response_text.startswith("失敗"):
                lines = [line.strip() for line in response_text.split('\n') if line.strip()]
                lines_written = 0
                for line in lines:
                    clean_line = line.lstrip('0123456789.-) ')
                    if clean_line.startswith("```"):
                        continue
                    clean_line = clean_line.replace("```html", "").replace("```", "").strip()
                    if clean_line:
                        out_f.write(f"{clean_line}\n")
                        lines_written += 1
                        total_outputs += 1

                if lines_written < 20:
                    print(f"  ⚠ row_{idx} 只輸出 {lines_written} 筆，預期 20 筆")
            else:
                out_f.write(f"# {response_text}\n")

    print(f"\n✓ 所有結果已保存至: {output_file}")
    print(f"  輸入行數  : {len(rows)}")
    print(f"  總輸出筆數: {total_outputs}")
    print(f"  成本節省  : 使用 Batch API 節省 50% 費用")


if __name__ == "__main__":
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_file  = "res/stats/stage3_seed_payload.csv"
    output_file = f"res/claude_test_payload/claude_no_crs_{timestamp}.txt"

    process_rows_with_batch_no_crs(
        input_file,
        output_file,
        temperature=1.0,
        max_tokens=8192,
        model="claude-sonnet-4-5-20250929"
    )