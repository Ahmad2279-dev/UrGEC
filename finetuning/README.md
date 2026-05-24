# Urdu Grammatical Error Correction with LLaMA-Factory

### 1. Method Overview

This project addresses Urdu Grammatical Error Correction (Urdu GEC). The method follows an instruction-tuning paradigm, where each sample is converted into a chat-style supervised fine-tuning instance. The model is trained to generate a corrected Urdu sentence given an input sentence containing spelling or grammatical errors.

The fine-tuning framework is **LLaMA-Factory**. This experiment fine-tunes six base models: `gemma2_2b`, `llama3.2_3b`, `qwen3_4b`, `qwen3_8b`, `qwen3_14b`, and `qwen3_32b`. All models are fine-tuned with **LoRA**. During training, only the low-rank adapter parameters are updated, while the original base model parameters remain frozen. This reduces GPU memory usage and training cost. After training, the evaluation script evaluates both base and fine-tuned models on the 5,000-sample test set and reports chrF++, TER, GLEU, Exact Match, edit-distance reduction, and other metrics.

The project contains three main files:

| File | Purpose |
| --- | --- |
| `01_split.py` | Randomly splits the full JSONL dataset into train, development, and test sets |
| `02_urgec_train.yaml` | LLaMA-Factory SFT + LoRA training configuration |
| `03_eval_one_gec_model.py` | Single-model inference and automatic evaluation script |

---

### 2. Requirements

```bash
pip install llamafactory transformers torch sacrebleu pandas tqdm
```

---

### 3. Data Format

The dataset is stored in JSONL format. Each line is one training instance using a `messages` field in chat format, including system, user, and assistant messages.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert in the Urdu language."
    },
    {
      "role": "user",
      "content": "Please correct all the spelling and grammatical errors in the following text:\nErroneous Urdu sentence"
    },
    {
      "role": "assistant",
      "content": "Corrected Urdu sentence"
    }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `system` | Defines the model role as an Urdu language expert |
| `user` | Provides the erroneous Urdu sentence and requests correction |
| `assistant` | The gold corrected sentence |

Example:

```json
{"messages": [{"role": "system", "content": "You are an expert in the Urdu language."}, {"role": "user", "content": "Please correct all the spelling and grammatical errors in the following text:\nکہ اسکول کے زمانے مےں پڑھائی میں کمزور تھا"}, {"role": "assistant", "content": "کہ اسکول کے زمانے میں پڑھائی میں کمزور تھا"}]}
```

---

### 4. Dataset Splitting

Run dataset splitting with:

```bash
python 01_split.py
```

The script reads `UrGECtr_train.jsonl`, shuffles all samples, and splits them into train, development, and test sets.

| Parameter | Value | Description |
| --- | ---: | --- |
| `input_file` | `UrGECtr_train.jsonl` | Original complete JSONL dataset |
| `train_file` | `urgec_train.jsonl` | Output training set |
| `dev_file` | `urgec_dev.jsonl` | Output development set |
| `test_file` | `urgec_test.jsonl` | Output test set |
| `dev_size` | `5000` | Number of development samples |
| `test_size` | `5000` | Number of test samples |
| `used_train_size` | `100000` | Number of training samples used for fine-tuning |
| `seed` | `42` | Random seed for reproducibility |

| Data file | Samples | Size |
| --- | ---: | ---: |
| `data/urgec_train.jsonl` | 100,000 | Training set |
| `data/urgec_dev.jsonl` | 5,000 | ~2.28 MB |
| `data/urgec_test.jsonl` | 5,000 | ~2.27 MB |

---

### 5. Dataset Registration in LLaMA-Factory

Register the datasets in LLaMA-Factory's `data/dataset_info.json`:

```json
{
  "urgec_train": {
    "file_name": "urgec_train.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    }
  },
  "urgec_dev": {
    "file_name": "urgec_dev.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    }
  }
}
```

Place the following files in your LLaMA-Factory `data/` directory:

```
/path/to/LlamaFactory/data/urgec_train.jsonl
/path/to/LlamaFactory/data/urgec_dev.jsonl
```

Update `dataset_dir` in `02_urgec_train.yaml` to point to your local LLaMA-Factory data directory.

---

### 6. LoRA Fine-Tuning Configuration

| Parameter | Value | Description |
| --- | ---: | --- |
| `stage` | `sft` | Supervised fine-tuning stage |
| `finetuning_type` | `lora` | LoRA fine-tuning |
| `lora_target` | `all` | Apply LoRA to all linear layers |
| `lora_rank` | `16` | Rank of LoRA matrices |
| `lora_alpha` | `32` | LoRA scaling factor (scaling = 32/16 = 2) |
| `lora_dropout` | `0.05` | LoRA dropout |

---

### 7. Training Hyperparameters

| Parameter | Value | Description |
| --- | ---: | --- |
| `per_device_train_batch_size` | `2` | Per-device batch size |
| `gradient_accumulation_steps` | `8` | Gradient accumulation (effective batch = 16) |
| `learning_rate` | `2.0e-5` | Learning rate |
| `num_train_epochs` | `1.0` | Number of epochs |
| `lr_scheduler_type` | `cosine` | Cosine LR scheduler |
| `warmup_ratio` | `0.05` | Warmup ratio |
| `bf16` | `true` | BF16 mixed precision |
| `flash_attn` | `fa2` | FlashAttention-2 |
| `cutoff_len` | `2048` | Maximum sequence length |
| `packing` | `true` | Sample packing for efficiency |
| `seed` | `42` | Random seed |

> If your GPU does not support BF16, replace `bf16: true` with `fp16: true`.

---

### 8. Training Command

Run the following in the LLaMA-Factory root directory:

```bash
llamafactory-cli train /path/to/02_urgec_train.yaml
```

**Important notes before running:**
1. Update `model_name_or_path` in the YAML to your local base model path.
2. Update `dataset_dir` to your LLaMA-Factory data directory.
3. Update `output_dir` to your desired output directory.
4. Ensure `urgec_train.jsonl` and `urgec_dev.jsonl` are registered in `dataset_info.json`.

---

### 9. Inference and Evaluation

```bash
# Evaluate fine-tuned model (with LoRA adapter)
python 03_eval_one_gec_model.py \
  --model_name llama3_2_3b_lora_urgec \
  --base_model_path /path/to/base_model \
  --adapter_path /path/to/lora_adapter \
  --test_path /path/to/urgec_test.jsonl \
  --out_dir /path/to/eval_results \
  --batch_size 4 \
  --max_new_tokens 128 \
  --max_input_length 1024

# Evaluate base model only (without LoRA adapter)
python 03_eval_one_gec_model.py \
  --model_name llama3_2_3b_base \
  --base_model_path /path/to/base_model \
  --test_path /path/to/urgec_test.jsonl \
  --out_dir /path/to/eval_results
```

Inference parameters:

| Parameter | Value |
| --- | ---: |
| `do_sample` | `false` |
| `max_new_tokens` | `128` |
| `max_input_length` | `1024` |
| `batch_size` | `4` |

---

### 10. Evaluation Metrics

| Metric | Description |
| --- | --- |
| `chrF++` | Character n-gram metric; higher is better |
| `TER` | Translation Edit Rate; lower is better |
| `GLEU` | GEC n-gram metric; higher is better |
| `ExactMatch` | Exact match percentage; higher is better |
| `PredRefEditDistance` | Edit distance to reference; lower is better |
| `EditDistanceReduction` | Relative error reduction; higher is better |
| `CloserRate` | Percentage closer to reference than source; higher is better |
| `UnchangedRate` | Percentage of unchanged outputs |

Output files per model:

| File | Content |
| --- | --- |
| `{model_name}_predictions.csv` | Source, reference, prediction, edit distances |
| `{model_name}_metrics.csv` | Summary metrics (CSV) |
| `{model_name}_metrics.json` | Summary metrics (JSON) |

---

