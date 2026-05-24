# -*- coding: utf-8 -*-
"""
Qwen3-8B 生成前10000条 Urdu 6类错误样本 + checkpoint断点续跑版。

每个原句生成6类错误：
1. spelling_error
2. postposition_substitution
3. word_insertion
4. postposition_deletion
5. word_deletion
6. gender_agreement

输出:
    /mnt/raid/hss/ahmad_data/llm_input_error_6types_qwen3_8b_first10000.csv

检查点:
    /mnt/raid/hss/ahmad_data/llm_input_error_6types_qwen3_8b_first10000.csv.done_ids.txt
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


ERROR_TYPES = [
    {
        "error_category": "grammar_error",
        "error_type": "spelling_error",
        "error_type_zh": "拼写错误",
        "instruction": "Introduce exactly one minor Urdu spelling error by changing, deleting, or inserting one character in one word. Do not change the main meaning.",
    },
    {
        "error_category": "grammar_error",
        "error_type": "postposition_substitution",
        "error_type_zh": "后置词替换错误",
        "instruction": "Replace exactly one Urdu postposition or case marker with an incorrect one, such as نے، کو، سے، میں، پر، کا، کی، کے.",
    },
    {
        "error_category": "grammar_error",
        "error_type": "word_insertion",
        "error_type_zh": "词语插入错误",
        "instruction": "Insert exactly one unnecessary Urdu word into the sentence, making it unnatural or grammatically wrong.",
    },
    {
        "error_category": "grammar_error",
        "error_type": "postposition_deletion",
        "error_type_zh": "后置词删除错误",
        "instruction": "Delete exactly one necessary Urdu postposition or case marker, such as نے، کو، سے، میں، پر، کا، کی، کے.",
    },
    {
        "error_category": "grammar_error",
        "error_type": "word_deletion",
        "error_type_zh": "词语删除错误",
        "instruction": "Delete exactly one ordinary Urdu word from the sentence. Do not delete only punctuation.",
    },
    {
        "error_category": "grammar_error",
        "error_type": "gender_agreement",
        "error_type_zh": "性别一致错误",
        "instruction": "Introduce exactly one Urdu gender agreement error by changing a masculine/feminine verb, adjective, participle, or possessive form.",
    },
]


def read_csv_auto(input_path: str, nrows: int) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1256", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            if nrows is None or nrows <= 0:
                df = pd.read_csv(input_path, dtype=str, encoding=enc)
            else:
                df = pd.read_csv(input_path, dtype=str, encoding=enc, nrows=nrows)

            df = df.fillna("")
            print(f"[INFO] CSV loaded: encoding={enc}, rows={len(df)}")
            return df
        except Exception as e:
            last_error = e

    raise RuntimeError(f"CSV读取失败，请检查编码。最后错误: {last_error}")


def build_error_type_text() -> str:
    lines = []
    for i, item in enumerate(ERROR_TYPES, start=1):
        lines.append(
            f"{i}. error_type: {item['error_type']}\n"
            f"   instruction: {item['instruction']}"
        )
    return "\n".join(lines)


def apply_chat_template(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def build_prompt(tokenizer, sentence: str) -> str:
    system_prompt = (
        "You are an Urdu grammar error generation expert. "
        "Generate controlled Urdu error sentences for NLP research. "
        "Keep Urdu script. Do not translate. Return valid JSON only."
    )

    user_prompt = f"""
Original Urdu sentence:
{sentence}

Generate exactly 6 error samples for the original sentence.

Target error types, in this exact order:
{build_error_type_text()}

Strict requirements:
1. Return JSON only. Do not output markdown or extra explanation.
2. The JSON must contain exactly 6 objects.
3. The order must match the 6 target error types.
4. Each object must contain:
   - error_type
   - error_sentence
   - changed_part
5. The error_sentence must still be Urdu.
6. Do not copy the original sentence as the error sentence.
7. Each error sentence should contain only the target error type.
8. Make the minimum necessary edit.

JSON format:
{{
  "errors": [
    {{"error_type": "spelling_error", "error_sentence": "...", "changed_part": "..."}},
    {{"error_type": "postposition_substitution", "error_sentence": "...", "changed_part": "..."}},
    {{"error_type": "word_insertion", "error_sentence": "...", "changed_part": "..."}},
    {{"error_type": "postposition_deletion", "error_sentence": "...", "changed_part": "..."}},
    {{"error_type": "word_deletion", "error_sentence": "...", "changed_part": "..."}},
    {{"error_type": "gender_agreement", "error_sentence": "...", "changed_part": "..."}}
  ]
}}

/no_think
""".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return apply_chat_template(tokenizer, messages)


def clean_model_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = text.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    return text


def extract_json_obj(text: str):
    text = clean_model_text(text)

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return {"errors": json.loads(text[start:end + 1])}
        except Exception:
            pass

    return None


def normalize_errors(obj) -> List[Dict[str, str]]:
    if obj is None:
        return []

    if isinstance(obj, dict):
        errors = obj.get("errors", [])
    elif isinstance(obj, list):
        errors = obj
    else:
        errors = []

    if not isinstance(errors, list):
        return []

    normalized = []
    for x in errors:
        if isinstance(x, dict):
            normalized.append({
                "error_type": str(x.get("error_type", "")).strip(),
                "error_sentence": str(x.get("error_sentence", "")).strip(),
                "changed_part": str(x.get("changed_part", "")).strip(),
            })
        elif isinstance(x, str):
            normalized.append({
                "error_type": "",
                "error_sentence": x.strip(),
                "changed_part": "",
            })

    return normalized


def align_errors(items: List[Dict[str, str]], original_sentence: str) -> List[Dict[str, str]]:
    expected_types = [x["error_type"] for x in ERROR_TYPES]
    output = []
    used = set()

    for etype in expected_types:
        found_idx = None
        for i, item in enumerate(items):
            if i in used:
                continue
            if item.get("error_type") == etype:
                found_idx = i
                break

        if found_idx is not None:
            used.add(found_idx)
            item = items[found_idx]
            error_sentence = item.get("error_sentence", "").strip()
            changed_part = item.get("changed_part", "").strip()

            if error_sentence and error_sentence != original_sentence:
                output.append({
                    "error_type": etype,
                    "error_sentence": error_sentence,
                    "changed_part": changed_part,
                    "parse_status": "success",
                })
            else:
                output.append({
                    "error_type": etype,
                    "error_sentence": "",
                    "changed_part": changed_part,
                    "parse_status": "failed_empty_or_same",
                })
        else:
            output.append(None)

    remaining = [item for i, item in enumerate(items) if i not in used]

    for i, val in enumerate(output):
        if val is not None:
            continue

        etype = expected_types[i]
        item = remaining.pop(0) if remaining else {}
        error_sentence = str(item.get("error_sentence", "")).strip()
        changed_part = str(item.get("changed_part", "")).strip()

        if error_sentence and error_sentence != original_sentence:
            output[i] = {
                "error_type": etype,
                "error_sentence": error_sentence,
                "changed_part": changed_part,
                "parse_status": "success_order_aligned",
            }
        else:
            output[i] = {
                "error_type": etype,
                "error_sentence": "",
                "changed_part": changed_part,
                "parse_status": "failed_missing",
            }

    return output


def load_done_ids(done_path: Path, output_path: Path, scan_output: bool = True) -> set:
    done_ids = set()

    if done_path.exists():
        with open(done_path, "r", encoding="utf-8") as f:
            for line in f:
                x = line.strip()
                if x:
                    done_ids.add(x)
        print(f"[INFO] Loaded checkpoint done ids: {len(done_ids)}")

    if scan_output and output_path.exists():
        try:
            old = pd.read_csv(output_path, usecols=["original_id"], dtype=str)
            ids = set(old["original_id"].astype(str).tolist())
            before = len(done_ids)
            done_ids.update(ids)
            print(f"[INFO] Loaded ids from output csv: {len(ids)}, merged {before}->{len(done_ids)}")
        except Exception as e:
            print(f"[WARN] 读取已有输出恢复失败: {e}")

    return done_ids


def append_done_ids(done_path: Path, ids: List[int]):
    done_path.parent.mkdir(parents=True, exist_ok=True)
    with open(done_path, "a", encoding="utf-8") as f:
        for x in ids:
            f.write(str(x) + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_rows(output_path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()

    df = pd.DataFrame(rows)
    df.to_csv(
        output_path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )


def append_failed_raw(raw_path: Path, records: List[Dict[str, Any]]):
    if not records:
        return

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def iter_batches(items: List[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def load_model(model_path: str):
    print(f"[INFO] Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[INFO] Loading model from: {model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="sdpa",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )

    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate_batch(tokenizer, model, prompts, max_new_tokens, temperature, top_p):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    results = []
    prompt_len = inputs["input_ids"].shape[-1]

    for i in range(len(prompts)):
        gen_ids = output_ids[i][prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        results.append(text)

    return results


def build_output_rows(original_id, row_dict, sentence_col, aligned_errors, keep_failed=False):
    original_sentence = row_dict.get(sentence_col, "").strip()
    rows = []
    type_info = {x["error_type"]: x for x in ERROR_TYPES}

    for item in aligned_errors:
        etype = item["error_type"]
        meta = type_info[etype]

        if not keep_failed and item["parse_status"] not in ["success", "success_order_aligned"]:
            continue

        out = {
            "original_id": original_id,
            "original_sentence": original_sentence,
            "error_sentence": item.get("error_sentence", ""),
            "error_category": meta["error_category"],
            "error_type": etype,
            "error_type_zh": meta["error_type_zh"],
            "changed_part": item.get("changed_part", ""),
            "parse_status": item.get("parse_status", ""),
            "generator": "qwen3_8b",
        }

        for col, val in row_dict.items():
            out[f"orig_{col}"] = val

        rows.append(out)

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, default="/mnt/raid/hss/ahmad_data/llm_input.csv")
    parser.add_argument("--output", type=str, default="/mnt/raid/hss/ahmad_data/llm_input_error_6types_qwen3_8b_first10000.csv")
    parser.add_argument("--model_path", type=str, default="/mnt/raid/zsb/llm_models/Qwen3-8B")
    parser.add_argument("--sentence_col", type=str, default="sentence")

    parser.add_argument("--nrows", type=int, default=10000)
    parser.add_argument("--start_row", type=int, default=0, help="起始行，从0开始，包含该行")
    parser.add_argument("--end_row", type=int, default=-1, help="结束行，不包含该行；-1表示不限制")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--top_p", type=float, default=0.90)

    parser.add_argument("--keep_failed", action="store_true")
    parser.add_argument("--no_scan_output_for_resume", action="store_true")
    parser.add_argument("--save_failed_raw", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    set_seed(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)
    done_path = Path(str(output_path) + ".done_ids.txt")
    raw_failed_path = Path(str(output_path) + ".failed_raw.jsonl")

    # 如果指定了 end_row，就至少读取到 end_row；
    # 否则使用 nrows。
    if args.end_row is not None and args.end_row > 0:
        read_nrows = args.end_row
    else:
        read_nrows = args.nrows

    df = read_csv_auto(str(input_path), read_nrows)

    # 按原始行号切片：[start_row, end_row)
    if args.start_row > 0 or args.end_row > 0:
        if args.end_row > 0:
            df = df.iloc[args.start_row:args.end_row]
        else:
            df = df.iloc[args.start_row:]

    print(f"[INFO] After range slicing, rows to consider: {len(df)}")

    if args.sentence_col in df.columns:
        sentence_col = args.sentence_col
    else:
        sentence_col = df.columns[0]
        print(f"[WARN] 未找到列 {args.sentence_col}，自动使用第一列: {sentence_col}")

    done_ids = load_done_ids(
        done_path=done_path,
        output_path=output_path,
        scan_output=not args.no_scan_output_for_resume,
    )

    pending = []
    for idx, row in df.iterrows():
        if str(idx) in done_ids:
            continue

        row_dict = {col: str(row[col]) for col in df.columns}
        sentence = row_dict.get(sentence_col, "").strip()

        if sentence:
            pending.append((idx, row_dict, sentence))
        else:
            append_done_ids(done_path, [idx])

    print(f"[INFO] Total loaded rows: {len(df)}")
    print(f"[INFO] Pending rows: {len(pending)}")
    print(f"[INFO] Expected max output samples: {len(pending) * len(ERROR_TYPES)}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Checkpoint: {done_path}")

    tokenizer, model = load_model(args.model_path)

    pbar = tqdm(total=len(pending), desc="Generating 6-type errors by Qwen3-8B")

    try:
        for batch in iter_batches(pending, args.batch_size):
            prompts = []
            metas = []

            for original_id, row_dict, sentence in batch:
                prompts.append(build_prompt(tokenizer, sentence))
                metas.append((original_id, row_dict, sentence))

            raw_outputs = generate_batch(
                tokenizer=tokenizer,
                model=model,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            output_rows = []
            done_this_batch = []
            failed_raw_records = []

            for (original_id, row_dict, sentence), raw in zip(metas, raw_outputs):
                obj = extract_json_obj(raw)
                items = normalize_errors(obj)
                aligned = align_errors(items, sentence)

                rows = build_output_rows(
                    original_id=original_id,
                    row_dict=row_dict,
                    sentence_col=sentence_col,
                    aligned_errors=aligned,
                    keep_failed=args.keep_failed,
                )

                output_rows.extend(rows)
                done_this_batch.append(original_id)

                success_count = sum(
                    1 for x in aligned
                    if x["parse_status"] in ["success", "success_order_aligned"]
                )

                if success_count < 6 and args.save_failed_raw:
                    failed_raw_records.append({
                        "original_id": original_id,
                        "original_sentence": sentence,
                        "success_count": success_count,
                        "raw_output": raw[:3000],
                    })

            # 核心检查点逻辑：先写结果，再写done_ids
            append_rows(output_path, output_rows)
            append_done_ids(done_path, done_this_batch)

            if args.save_failed_raw:
                append_failed_raw(raw_failed_path, failed_raw_records)

            pbar.update(len(batch))

    except KeyboardInterrupt:
        print("\n[STOP] 手动中断。已完成 batch 已保存，下次重新运行会自动续跑。")

    finally:
        pbar.close()

    print("\n[DONE] 当前运行结束")
    print(f"[OUTPUT] {output_path}")
    print(f"[CHECKPOINT] {done_path}")

    if args.save_failed_raw:
        print(f"[FAILED RAW] {raw_failed_path}")


if __name__ == "__main__":
    main()