import anthropic
import os
import csv
import time
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


# ==========================================================
# CRS conf 設定
# ==========================================================

CRS_RULE_DIR = "rule"

EXPERIMENT_TARGET = "ARGS:message"

ONLY_KEEP_RULES_TARGETING_ARGS = True

EXCLUDE_RULE_IDS_FOR_LLM = {"949110", "980130"}

CRS_RULE_FILE_BY_PREFIX = {
    "920": "REQUEST-920-PROTOCOL-ENFORCEMENT.conf",
    "932": "REQUEST-932-APPLICATION-ATTACK-RCE.conf",
    "933": "REQUEST-933-APPLICATION-ATTACK-PHP.conf",
    "934": "REQUEST-934-APPLICATION-ATTACK-GENERIC.conf",
    "941": "REQUEST-941-APPLICATION-ATTACK-XSS.conf",
    "942": "REQUEST-942-APPLICATION-ATTACK-SQLI.conf",
}
RULE_ID_ALIAS = {
    "932100": ["932230"],
    "932105": ["932231"],
    "932150": ["932232"],
    "932115": ["932370", "932380"],  # 一對多
    "932110": ["932370", "932380"],  # 同上
}

# ==========================================================
# CRS conf parser
# ==========================================================

def get_rule_file_for_id(rule_id: str):
    """
    依 rule_id 前三碼分流到固定 CRS conf 檔。
    """
    prefix = str(rule_id).strip()[:3]
    fname = CRS_RULE_FILE_BY_PREFIX.get(prefix)

    if not fname:
        return None

    return os.path.join(CRS_RULE_DIR, fname)


def read_text_file(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"[WARN] Failed to read CRS rule file: {path}, error={e}")
        return ""


def unwrap_crs_lines(text: str) -> str:
    """
    將 CRS conf 中以反斜線續行的規則合併成單行。
    """
    lines = text.splitlines()
    merged_lines = []
    buffer = ""

    for line in lines:
        stripped = line.rstrip()

        if not stripped:
            if buffer:
                merged_lines.append(buffer)
                buffer = ""
            continue

        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
        else:
            buffer += stripped
            merged_lines.append(buffer)
            buffer = ""

    if buffer:
        merged_lines.append(buffer)

    return "\n".join(merged_lines)


def split_crs_actions(action_str: str) -> list:
    """
    切分 SecRule action list，避免把 msg 或 setvar 裡面的逗號切錯。
    """
    actions = []
    current = []
    quote = None
    escape = False

    for ch in action_str:
        if escape:
            current.append(ch)
            escape = False
            continue

        if ch == "\\":
            current.append(ch)
            escape = True
            continue

        if ch in ("'", '"'):
            current.append(ch)

            if quote is None:
                quote = ch
            elif quote == ch:
                quote = None

            continue

        if ch == "," and quote is None:
            item = "".join(current).strip()

            if item:
                actions.append(item)

            current = []
        else:
            current.append(ch)

    item = "".join(current).strip()

    if item:
        actions.append(item)

    return actions


def read_quoted(text: str, start: int):
    """
    從 text[start] 開始讀取一段雙引號字串，支援 escaped quote。
    回傳: (content, next_index)
    """
    if start >= len(text) or text[start] != '"':
        return None, start

    i = start + 1
    out = []
    escape = False

    while i < len(text):
        ch = text[i]

        if escape:
            out.append("\\" + ch)
            escape = False
            i += 1
            continue

        if ch == "\\":
            escape = True
            i += 1
            continue

        if ch == '"':
            return "".join(out), i + 1

        out.append(ch)
        i += 1

    return None, start


def parse_sec_rule_line(line: str):
    """
    解析單一 SecRule。
    """
    line = line.strip()

    if not line.startswith("SecRule "):
        return None

    rest = line[len("SecRule "):].strip()
    first_quote = rest.find('"')

    if first_quote == -1:
        return None

    targets = rest[:first_quote].strip()

    op_text, idx = read_quoted(rest, first_quote)

    if op_text is None:
        return None

    rest2 = rest[idx:].strip()

    actions_raw = ""

    if rest2.startswith('"'):
        actions_raw, _ = read_quoted(rest2, 0)

        if actions_raw is None:
            actions_raw = ""

    parts = op_text.split(None, 1)
    operator = parts[0].strip()
    pattern = parts[1].strip() if len(parts) > 1 else ""

    return {
        "targets": targets,
        "operator": operator,
        "pattern": pattern,
        "actions_raw": actions_raw,
    }


def parse_rule_actions(actions_raw: str) -> dict:
    """
    從 action list 抓 id, msg, transforms, chain, multiMatch。
    """
    actions = split_crs_actions(actions_raw)

    out = {
        "rule_id": None,
        "msg": "",
        "transforms": [],
        "has_chain": False,
        "multi_match": False,
    }

    for act in actions:
        act = act.strip()

        if act.startswith("id:"):
            out["rule_id"] = act.replace("id:", "", 1).strip().strip("'\"")

        elif act.startswith("msg:"):
            out["msg"] = act.replace("msg:", "", 1).strip().strip("'\"")

        elif act.startswith("t:"):
            out["transforms"].append(act.strip())

        elif act == "chain":
            out["has_chain"] = True

        elif act == "multiMatch":
            out["multi_match"] = True

    return out


def extract_conditions_from_rule_block(block_lines: list) -> list:
    """
    解析一條主規則與其 chain 後續 SecRule 條件。
    """
    conditions = []

    for line in block_lines:
        parsed = parse_sec_rule_line(line)

        if not parsed:
            continue

        conditions.append({
            "targets": parsed["targets"],
            "operator": parsed["operator"],
            "pattern": parsed["pattern"],
        })

    return conditions


def condition_targets_args(condition: dict) -> bool:
    targets = str(condition.get("targets", ""))
    return re.search(r'(^|\|)ARGS($|\||:)', targets) is not None


def rule_targets_args(conditions: list) -> bool:
    return any(condition_targets_args(cond) for cond in conditions)


def condition_to_text(condition: dict) -> str:
    targets = str(condition.get("targets", ""))
    operator = condition.get("operator", "")
    pattern = condition.get("pattern", "")

    display_target = EXPERIMENT_TARGET if condition_targets_args(condition) else targets

    if pattern:
        return f"{display_target} {operator} {pattern}"

    return f"{display_target} {operator}"

def get_aliased_ids(rule_id: str) -> list:
    """回傳 alias 後的 rule id list（無 alias 則回傳自身）"""
    rule_id = str(rule_id).strip()
    aliased = RULE_ID_ALIAS.get(rule_id)
    if aliased is None:
        return [rule_id]
    return aliased if isinstance(aliased, list) else [aliased]

def parse_crs_conf_by_rule_id(rule_id: str):
    """
    依 rule_id 前三碼找指定 conf，並解析該 rule。
    """
    rule_id = str(rule_id).strip()
    
    rule_file = get_rule_file_for_id(rule_id)

    if not rule_file:
        print(f"[WARN] No CRS file mapping for rule_id={rule_id}")
        return None

    if not os.path.exists(rule_file):
        print(f"[WARN] Mapped CRS file not found for rule_id={rule_id}: {rule_file}")
        return None

    text = unwrap_crs_lines(read_text_file(rule_file))
    lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip().startswith("SecRule ")
    ]

    for i, line in enumerate(lines):
        parsed = parse_sec_rule_line(line)

        if not parsed:
            continue

        actions = parse_rule_actions(parsed["actions_raw"])

        if actions["rule_id"] != rule_id:
            continue

        block_lines = [line]

        j = i + 1
        current_has_chain = actions["has_chain"]

        while current_has_chain and j < len(lines):
            next_line = lines[j]
            block_lines.append(next_line)

            next_parsed = parse_sec_rule_line(next_line)

            if not next_parsed:
                break

            next_actions = parse_rule_actions(next_parsed["actions_raw"])
            current_has_chain = next_actions["has_chain"]
            j += 1

        conditions = extract_conditions_from_rule_block(block_lines)

        return {
            "rule_id": rule_id,
            "msg": actions["msg"],
            "conditions": conditions,
            "transforms": actions["transforms"],
            "has_chain": actions["has_chain"],
            "multi_match": actions["multi_match"],
            "source_file": os.path.basename(rule_file),
        }

    print(f"[WARN] rule_id={rule_id} not found in mapped file: {rule_file}")
    return None


CRS_RULE_CACHE = {}


def get_official_rule_info(rule_id: str):
    """
    加 cache，避免每次 payload 都重讀 conf。
    """
    rule_id = str(rule_id).strip()

    if rule_id in CRS_RULE_CACHE:
        return CRS_RULE_CACHE[rule_id]

    info = parse_crs_conf_by_rule_id(rule_id)
    CRS_RULE_CACHE[rule_id] = info

    return info


def extract_audit_transforms(segment: str) -> list:
    """
    從 audit.log ref segment 裡抓 t:xxx。
    """
    transforms = re.findall(r't:([a-zA-Z]+)', str(segment))

    if not transforms:
        return ["raw"]

    return list(dict.fromkeys(transforms))


def parse_rule_refs(rule_ids_str: str, rule_refs_str: str) -> str:
    """
    將 stage3 CSV 中的 rule_ids / rule_refs 轉成給 LLM 的 CRS reference。

    新版流程：
    1. 先依 rule_id 讀取官方 CRS conf
    2. 抽出 Message / Detection Conditions / Official Transformation Chain
    3. 若官方 conf 找不到，才 fallback 到 audit.log ref 裡的 t:xxx
    """
    rule_ids = [
        rid.strip()
        for rid in str(rule_ids_str).split(",")
        if rid.strip()
    ]

    rule_segments = [
        seg.strip()
        for seg in str(rule_refs_str).split("||")
    ]

    raw_count = len(rule_ids)

    rule_order = []
    rule_to_audit_transforms = {}

    for idx, rule_id in enumerate(rule_ids):
        if rule_id in EXCLUDE_RULE_IDS_FOR_LLM:
            continue

        expanded_ids = get_aliased_ids(rule_id)  # 可能是 1 或多個
        segment = rule_segments[idx] if idx < len(rule_segments) else ""

        for expanded_id in expanded_ids:
            if expanded_id not in rule_to_audit_transforms:
                rule_to_audit_transforms[expanded_id] = []
                rule_order.append(expanded_id)

            audit_transforms = extract_audit_transforms(segment)
            for tr in audit_transforms:
                if tr not in rule_to_audit_transforms[expanded_id]:
                    rule_to_audit_transforms[expanded_id].append(tr)

    structured_rules = []

    for rule_id in rule_order:
        official = get_official_rule_info(rule_id)

        if official:
            if ONLY_KEEP_RULES_TARGETING_ARGS and not rule_targets_args(official.get("conditions", [])):
                print(f"[SKIP] rule_id={rule_id} does not target ARGS, not sent to LLM.")
                continue

            official_chain = (
                " → ".join(official["transforms"])
                if official["transforms"]
                else "t:none"
            )

            conditions = official.get("conditions", [])

            condition_lines = (
                "\n".join(
                    f"        {i + 1}. {condition_to_text(c)}"
                    for i, c in enumerate(conditions)
                )
                if conditions
                else "        N/A"
            )

            rule_block = (
                f"  [{len(structured_rules) + 1}] CRS Rule ID                  : {rule_id}\n"
                f"      Message                      : {official['msg'] or 'N/A'}\n"
                f"      Detection Conditions         :\n"
                f"{condition_lines}\n"
                f"      Official Transformation Chain: {official_chain}"
            )

            if official.get("has_chain"):
                rule_block += "\n      Chain Rule                   : yes"

            if official.get("multi_match"):
                rule_block += "\n      MultiMatch                   : yes"

        else:
            audit_chain = " → ".join(rule_to_audit_transforms.get(rule_id, []))

            print(
                f"[WARN] Official CRS rule definition not found for rule_id={rule_id}. "
                f"Fallback to audit.log transformation reference."
            )

            rule_block = (
                f"  [{len(structured_rules) + 1}] CRS Rule ID                  : {rule_id}\n"
                f"      Message                      : N/A\n"
                f"      Detection Conditions         : N/A\n"
                f"      Official Transformation Chain: N/A\n"
                f"      Fallback Audit Transform     : {audit_chain if audit_chain else 'raw'}"
            )

        structured_rules.append(rule_block)

    unique_count = len(structured_rules)

    if raw_count != unique_count:
        print(f"[DEDUP CRS] raw rules={raw_count}, unique rules sent to LLM={unique_count}")

    if len(rule_ids) != len(rule_segments):
        print(
            f"[WARN] rule_ids count != rule_refs count: "
            f"rule_ids={len(rule_ids)}, rule_refs={len(rule_segments)}"
        )

    if not structured_rules:
        return "No CRS rule reference is available."

    return "\n\n".join(structured_rules)


# ==========================================================
# Prompt：保留第一份原本內容
# ==========================================================

def build_prompt(payload: str, rule_ids: str, rule_refs: str) -> str:
    structured_refs = parse_rule_refs(rule_ids, rule_refs)

    prompt = f"""## Seed Payload
{payload}

## Seed Payload Analysis
Before generating variants, reason through the following:
- Identify the HTML sink element and its triggering mechanism (e.g. event handler, attribute)
- Identify the JavaScript execution vector and which function or constructor achieves execution
- Identify which specific tokens in the seed payload are likely matched by each CRS rule below
- Determine which tokens can be mutated without breaking the execution chain

Variants must preserve the complete execution chain while mutating only the detectable tokens.

## Active CRS Rule Constraints
The following ModSecurity CRS rules are triggered by the seed payload.
Each rule applies a deterministic normalization pipeline before regex matching.
All generated variants must evade pattern detection after every listed transformation is applied.

{structured_refs}

## Regex-aware Mutation Requirement
For each active CRS rule, read the Detection Condition as a regex-like pattern and infer what payload structure it is trying to detect. The exact match location may not be known.
Generated variants must avoid the regex-like pattern that appears most relevant to the seed payload. Do not only mutate unrelated parts of the payload.
If the pattern appears to target outer HTML syntax, change the HTML trigger structure itself, such as the tag name, event-handler name, attribute layout, or trigger mechanism.
If the pattern appears to target inner JavaScript syntax, change the JavaScript expression itself, such as the callable name, function invocation, property chain, string construction, or indirect execution method.
If the pattern appears to target URI schemes or attribute values, change the value, attribute layout, scheme representation, or trigger mechanism.
If the pattern appears to target encoded or escaped forms, avoid mutation techniques that will be decoded or normalized back into the same detectable regex pattern by the rule's Transformation Chain.
Use each active rule's Transformation Chain to avoid mutations that will be decoded or normalized back into the same detectable regex pattern. Prioritize mutations that remain structurally different after all listed transformations are applied.
For rules with no transformation or raw matching behavior, do not rely only on encoding or decoding tricks. Encoding may be used as a secondary obfuscation layer, but the raw token or structure that appears to match the regex pattern should also be structurally rewritten.

## Mutation Strategies
Apply techniques from all four categories. For each variant, select the combination
that is least likely to be normalized back to a detectable form by the active rule pipelines.

1. **Encoding Mutations**
   Unicode escapes (\\uXXXX), hexadecimal (&#xNN;), octal references,
   HTML named/numeric entities, double-encoding sequences.
   — Verify the chosen encoding survives the active transformation chain unchanged.

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
- Output exactly 4 payload strings, one per line. 
- Raw payload strings only — no line numbers, labels, explanations, or markdown fences.
- No duplicate payloads.
- Distribute variants across all 4 mutation categories — at least 1 variants per category.
- Each variant must be testably distinct: differing only in whitespace or trivial
  character substitution does not constitute a distinct technique.
- All variants must trigger JavaScript execution via `innerHTML` assignment in a DOM-based XSS context.
- Prefer variants that would evade multiple active rules simultaneously over single-rule evasions.
"""
    return prompt

#- Output exactly 20 payload strings, one per line.
#- Distribute variants across all 4 mutation categories — at least 4 variants per category. 4個變異都要有 餘4做其他變異
 
#- Output exactly 5 payload strings, one per line. 
#- Distribute variants across all 4 mutation categories — at least 1 variants per category. 多餘1做其他變異

#- Output exactly 4 payload strings, one per line. 
#- Distribute variants across all 4 mutation categories — at least 1 variants per category. 多餘0做其他變異

# ==========================================================
# Batch API 主流程：保留第一份原本內容
# ==========================================================
def extract_text_blocks(message) -> str:
    texts = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            texts.append(block.text)
    return "\n".join(texts).strip()

def process_rows_with_batch(
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

    print(f"找到 {len(rows)} 行資料，準備批量提交...")

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
        rule_refs = row.get('rule_refs', '')
        rule_ids = row.get('rule_ids', '')

        if not payload:
            print(f"警告：第 {idx} 行沒有 payload，跳過")
            continue

        prompt = build_prompt(payload, rule_ids, rule_refs)

        requests.append({
            "custom_id": f"row_{idx}",
            "params": {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": SYSTEM_PROMPT,
                # "thinking": {
                #      "type": "enabled",
                #      "budget_tokens": 2048
                #  },
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "<"},
                ]
            }
        })

    if not requests:
        print("錯誤：沒有有效的 payload 可處理")
        return

    total_requests = len(requests)

    print(f"✓ 已準備 {total_requests} 個任務 (預計輸出 {total_requests * 20} 筆)") # *5
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
    elapsed_time = 0
    poll_interval = 10

    while elapsed_time < max_wait_time:
        try:
            batch_status = client.messages.batches.retrieve(batch_id)
            status = batch_status.processing_status

            if status == "ended":
                print("✓ 批次處理完成!")
                break

            counts = batch_status.request_counts
            succeeded = counts.succeeded if hasattr(counts, 'succeeded') else 0
            errored = counts.errored if hasattr(counts, 'errored') else 0
            processing = counts.processing if hasattr(counts, 'processing') else 0
            completed = succeeded + errored

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
                response_text = "<" + extract_text_blocks(result.result.message)
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
                lines = [
                    line.strip()
                    for line in response_text.split('\n')
                    if line.strip()
                ]

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    input_file = "res/stats/stage3_seed_payload.csv"
    output_file = f"res/claude_test_payload/claude_{timestamp}.txt"

    process_rows_with_batch(
        input_file,
        output_file,
        temperature=1.0,
        max_tokens=8192,
        model="claude-sonnet-4-5-20250929"
    )