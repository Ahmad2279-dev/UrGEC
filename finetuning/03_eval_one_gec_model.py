# -*- coding: utf-8 -*-
"""
Single-model Urdu GEC evaluation.

适配测试集格式：
{"messages": [
  {"role": "system", "content": "You are an expert in the Urdu language."},
  {"role": "user", "content": "Please correct all the spelling and grammatical errors in the following text:\n错误句"},
  {"role": "assistant", "content": "正确句"}
]}

支持：
1. Qwen / Llama / Gemma2 等模型
2. base model 直接评估
3. base model + LoRA adapter 评估
4. Gemma2 不支持 system role 的问题
5. Qwen3 flash_attn 报错问题
6. chrF++ / TER / GLEU / ExactMatch / 编辑距离指标
"""

import os
import re
import gc
import json
import math
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


SYSTEM_PROMPT = "You are an expert in the Urdu language."

USER_PROMPT_TEMPLATE = (
    "Please correct all the spelling and grammatical errors in the following text:\n"
    "{error_sentence}"
)


def patch_flash_attn_keyerror():
    """
    禁用 transformers 中的 flash_attn 检测，避免 Qwen3 导入时报错。
    """
    try:
        import transformers.utils.import_utils as iu

        for fn_name in [
            "is_flash_attn_2_available",
            "is_flash_attn_3_available",
            "is_flash_attn_4_available",
            "is_flash_attn_greater_or_equal",
        ]:
            if hasattr(iu, fn_name):
                setattr(iu, fn_name, lambda *args, **kwargs: False)

        if hasattr(iu, "_is_package_available"):
            old_func = iu._is_package_available

            def new_is_package_available(pkg_name, return_version=False):
                name = str(pkg_name).replace("-", "_")
                if name in {
                    "flash_attn",
                    "flash_attn_2",
                    "flash_attn_3",
                    "flash_attn_4",
                }:
                    if return_version:
                        return False, "N/A"
                    return False
                return old_func(pkg_name, return_version=return_version)

            iu._is_package_available = new_is_package_available

        for map_name in ["PACKAGE_DISTRIBUTION_MAPPING", "PACKAGES_DISTRIBUTION_MAPPING"]:
            if hasattr(iu, map_name):
                mapping = getattr(iu, map_name)
                if isinstance(mapping, dict):
                    mapping["flash_attn"] = ["flash-attn"]
                    mapping["flash-attn"] = ["flash-attn"]
                    mapping["flash_attn_2"] = ["flash-attn"]
                    mapping["flash_attn_3"] = ["flash-attn"]
                    mapping["flash_attn_4"] = ["flash-attn"]

    except Exception:
        pass


def setup_env(cache_dir: str, offline: bool = True):
    cache_dir = Path(cache_dir)

    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "huggingface/datasets"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    os.environ.setdefault("TRANSFORMERS_NO_FLASH_ATTENTION", "1")
    os.environ.setdefault("FLASH_ATTENTION_FORCE_DISABLE", "1")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def normalize_text(s: str) -> str:
    s = str(s)
    s = s.replace("\ufeff", "")
    s = s.replace("\u200c", "")
    s = s.replace("\u200d", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_error_from_user_content(content: str) -> str:
    content = str(content).strip()

    pattern = r"following text\s*:\s*(.*)$"
    m = re.search(pattern, content, flags=re.I | re.S)
    if m:
        return m.group(1).strip()

    lines = [x.strip() for x in content.splitlines() if x.strip()]
    if lines:
        return lines[-1]

    return content


def read_urgec_messages_jsonl(path: str, start: int = 0, limit: int = -1) -> pd.DataFrame:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id < start:
                continue

            if limit > 0 and len(rows) >= limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            error_sentence = ""
            correct_sentence = ""

            if "messages" in obj and isinstance(obj["messages"], list):
                for msg in obj["messages"]:
                    role = msg.get("role", "")
                    content = msg.get("content", "")

                    if role == "user":
                        error_sentence = extract_error_from_user_content(content)
                    elif role == "assistant":
                        correct_sentence = content.strip()
            else:
                error_sentence = (
                    obj.get("error")
                    or obj.get("error_sentence")
                    or obj.get("incorrect")
                    or obj.get("source")
                    or obj.get("src")
                    or obj.get("input")
                    or ""
                )
                correct_sentence = (
                    obj.get("correct")
                    or obj.get("correct_sentence")
                    or obj.get("target")
                    or obj.get("tgt")
                    or obj.get("output")
                    or obj.get("label")
                    or ""
                )

            error_sentence = normalize_text(error_sentence)
            correct_sentence = normalize_text(correct_sentence)

            if error_sentence and correct_sentence and error_sentence != correct_sentence:
                rows.append({
                    "sample_id": line_id,
                    "error_sentence": error_sentence,
                    "correct_sentence": correct_sentence,
                })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError("测试集为空，或未能从 messages 中解析出 user/assistant。")

    df = df.drop_duplicates(
        subset=["error_sentence", "correct_sentence"]
    ).reset_index(drop=True)

    return df


def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    """
    兼容 Qwen/Llama/Gemma2。

    Gemma2 的 chat_template 不支持 system role，会报：
    jinja2.exceptions.TemplateError: System role not supported

    处理方式：
    1. 先按原始 messages 尝试。
    2. 如果失败，把 system prompt 合并进 user prompt，只保留 user role。
    3. 如果还失败，手写普通 prompt。
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    except Exception:
        pass

    system_text = ""
    user_text = ""

    for msg in messages:
        role = msg.get("role", "")
        content = str(msg.get("content", "")).strip()

        if role == "system":
            system_text = content
        elif role == "user":
            user_text = content

    if system_text:
        merged_user_text = system_text + "\n\n" + user_text
    else:
        merged_user_text = user_text

    no_system_messages = [
        {
            "role": "user",
            "content": merged_user_text,
        }
    ]

    try:
        return tokenizer.apply_chat_template(
            no_system_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        pass

    return (
        f"{merged_user_text}\n\n"
        "Corrected sentence:"
    )


def build_prompt(tokenizer, error_sentence: str) -> str:
    user_prompt = USER_PROMPT_TEMPLATE.format(error_sentence=error_sentence)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    return apply_chat_template(tokenizer, messages)


def postprocess_pred(text: str) -> str:
    text = str(text).strip()

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = text.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for k in [
                "corrected_sentence",
                "correct_sentence",
                "prediction",
                "answer",
                "output",
                "text",
            ]:
                if k in obj and str(obj[k]).strip():
                    return normalize_text(obj[k])
    except Exception:
        pass

    prefixes = [
        "Corrected sentence:",
        "Corrected:",
        "Correction:",
        "Answer:",
        "Output:",
        "The corrected sentence is:",
    ]

    for p in prefixes:
        if text.lower().startswith(p.lower()):
            text = text[len(p):].strip()

    text = text.strip("\"'“”‘’ ")

    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if lines:
        urdu_lines = [x for x in lines if re.search(r"[\u0600-\u06FF]", x)]
        text = urdu_lines[0] if urdu_lines else lines[0]

    return normalize_text(text)


def get_tokenizer_path(base_model_path: str, adapter_path: str = "") -> str:
    if adapter_path:
        p = Path(adapter_path)
        if (p / "tokenizer_config.json").exists():
            return str(p)
    return base_model_path


def load_tokenizer(base_model_path: str, adapter_path: str = ""):
    tokenizer_path = get_tokenizer_path(base_model_path, adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def load_model(
    base_model_path: str,
    adapter_path: str = "",
    attn_implementation: str = "eager",
    load_in_4bit: bool = False,
):
    patch_flash_attn_keyerror()

    model_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
        "device_map": "auto",
        "attn_implementation": attn_implementation,
    }

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            **model_kwargs,
        )
    except TypeError:
        model_kwargs.pop("torch_dtype", None)
        model_kwargs["dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            **model_kwargs,
        )

    if adapter_path:
        from peft import PeftModel

        print(f"[INFO] Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            local_files_only=True,
        )

    model.eval()
    return model


@torch.no_grad()
def generate_batch(
    tokenizer,
    model,
    error_sentences: List[str],
    max_new_tokens: int,
    max_input_length: int,
) -> List[str]:
    prompts = [build_prompt(tokenizer, x) for x in error_sentences]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_len = inputs["input_ids"].shape[-1]

    preds = []
    for i in range(len(error_sentences)):
        gen_ids = output_ids[i][prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        preds.append(postprocess_pred(text))

    return preds


def levenshtein(a: str, b: str) -> int:
    a = list(a)
    b = list(b)

    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))

    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = cur

    return prev[m]


def ngram_counts(tokens: List[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def sentence_gleu(pred: str, ref: str, max_order: int = 4) -> float:
    pred_toks = normalize_text(pred).split()
    ref_toks = normalize_text(ref).split()

    if not pred_toks and not ref_toks:
        return 1.0
    if not pred_toks or not ref_toks:
        return 0.0

    scores = []

    for n in range(1, max_order + 1):
        pc = ngram_counts(pred_toks, n)
        rc = ngram_counts(ref_toks, n)

        if not pc and not rc:
            scores.append(1.0)
            continue

        if not pc or not rc:
            scores.append(0.0)
            continue

        overlap = sum((pc & rc).values())
        precision = overlap / max(sum(pc.values()), 1)
        recall = overlap / max(sum(rc.values()), 1)

        scores.append(min(precision, recall))

    return sum(scores) / len(scores)


def corpus_gleu(preds: List[str], refs: List[str]) -> float:
    if len(preds) == 0:
        return 0.0
    return sum(sentence_gleu(p, r) for p, r in zip(preds, refs)) / len(preds) * 100


def compute_metrics(srcs: List[str], preds: List[str], refs: List[str]) -> Dict[str, Any]:
    srcs = [normalize_text(x) for x in srcs]
    preds = [normalize_text(x) for x in preds]
    refs = [normalize_text(x) for x in refs]

    total = len(refs)

    exact = sum(p == r for p, r in zip(preds, refs)) / total * 100

    try:
        from sacrebleu.metrics import CHRF, TER

        chrf = CHRF(word_order=2).corpus_score(preds, [refs]).score
        ter = TER().corpus_score(preds, [refs]).score
    except Exception as e:
        print("[WARN] sacrebleu chrF++ / TER failed:", repr(e))
        chrf = float("nan")
        ter = float("nan")

    gleu = corpus_gleu(preds, refs)

    src_dist = []
    pred_dist = []
    closer = 0
    unchanged = 0
    reductions = []

    for src, pred, ref in zip(srcs, preds, refs):
        d_src = levenshtein(src, ref)
        d_pred = levenshtein(pred, ref)

        src_dist.append(d_src)
        pred_dist.append(d_pred)

        if pred == src:
            unchanged += 1

        if d_pred < d_src:
            closer += 1

        if d_src > 0:
            reductions.append((d_src - d_pred) / d_src)

    metrics = {
        "num_samples": total,
        "chrF++": round(float(chrf), 4) if not math.isnan(float(chrf)) else "nan",
        "TER": round(float(ter), 4) if not math.isnan(float(ter)) else "nan",
        "GLEU": round(float(gleu), 4),
        "ExactMatch": round(exact, 4),
        "SourceRefEditDistance": round(sum(src_dist) / total, 4),
        "PredRefEditDistance": round(sum(pred_dist) / total, 4),
        "EditDistanceReduction": round(sum(reductions) / len(reductions) * 100, 4) if reductions else 0.0,
        "CloserRate": round(closer / total * 100, 4),
        "UnchangedRate": round(unchanged / total * 100, 4),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default="")

    parser.add_argument("--test_path", type=str, default="/myfile/my/ahmad/data/urgec_test.jsonl")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="/myfile/my/ahmad/cache")

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_input_length", type=int, default=1024)

    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--offline", action="store_true", default=True)

    args = parser.parse_args()

    setup_env(args.cache_dir, offline=args.offline)
    patch_flash_attn_keyerror()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading test data...")
    df = read_urgec_messages_jsonl(
        args.test_path,
        start=args.start,
        limit=args.limit,
    )

    print(f"[INFO] Test samples: {len(df)}")
    df.to_csv(out_dir / f"{args.model_name}_eval_data_used.csv", index=False, encoding="utf-8-sig")

    print("\n========== Model Config ==========")
    print("model_name:", args.model_name)
    print("base_model_path:", args.base_model_path)
    print("adapter_path:", args.adapter_path)
    print("load_in_4bit:", args.load_in_4bit)
    print("attn_implementation:", args.attn_implementation)

    tokenizer = load_tokenizer(args.base_model_path, args.adapter_path)

    model = load_model(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        attn_implementation=args.attn_implementation,
        load_in_4bit=args.load_in_4bit,
    )

    srcs = df["error_sentence"].tolist()
    refs = df["correct_sentence"].tolist()
    preds = []

    for i in tqdm(range(0, len(srcs), args.batch_size), desc=f"Infer {args.model_name}"):
        batch_srcs = srcs[i:i + args.batch_size]
        batch_preds = generate_batch(
            tokenizer=tokenizer,
            model=model,
            error_sentences=batch_srcs,
            max_new_tokens=args.max_new_tokens,
            max_input_length=args.max_input_length,
        )
        preds.extend(batch_preds)

    metrics = compute_metrics(srcs, preds, refs)
    metrics["model"] = args.model_name
    metrics["base_model_path"] = args.base_model_path
    metrics["adapter_path"] = args.adapter_path

    rows = []

    for sample_id, src, ref, pred in zip(df["sample_id"].tolist(), srcs, refs, preds):
        src_ed = levenshtein(normalize_text(src), normalize_text(ref))
        pred_ed = levenshtein(normalize_text(pred), normalize_text(ref))

        rows.append({
            "model": args.model_name,
            "sample_id": sample_id,
            "error_sentence": src,
            "correct_sentence": ref,
            "prediction": pred,
            "exact_match": normalize_text(pred) == normalize_text(ref),
            "source_ref_edit_distance": src_ed,
            "pred_ref_edit_distance": pred_ed,
            "closer_than_source": pred_ed < src_ed,
            "unchanged_from_source": normalize_text(pred) == normalize_text(src),
        })

    pred_df = pd.DataFrame(rows)
    metrics_df = pd.DataFrame([metrics])
    metrics_df = metrics_df[["model"] + [c for c in metrics_df.columns if c != "model"]]

    pred_path = out_dir / f"{args.model_name}_predictions.csv"
    metrics_csv = out_dir / f"{args.model_name}_metrics.csv"
    metrics_json = out_dir / f"{args.model_name}_metrics.json"

    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n========== Metrics ==========")
    print(metrics_df.to_string(index=False))

    print("\n[DONE] predictions:", pred_path)
    print("[DONE] metrics csv:", metrics_csv)
    print("[DONE] metrics json:", metrics_json)

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()