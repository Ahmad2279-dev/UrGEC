# Dataset Construction

This folder contains the pipeline for constructing the UrGEC dataset. The process involves three stages: corpus cleaning, data splitting, and synthetic error generation using three methods (rule-based, LLM-based, and machine translation-based).


## Project Structure

```
Dataset-Construction/
├── preprocess.py                    → Urdu corpus cleaning pipeline
├── split_preprocessed_sentences.py → Split clean corpus for three methods
├── rulebased.py                     → Rule-based error generation
├── llmbased.py                      → LLM-based error generation (Qwen3-8B)
├── mtbased.py                       → MT-based error generation (M2M100)
└── README.md                        → this file
```

---

## Requirements

```bash
pip install pandas scikit-learn torch transformers tqdm
```

---

## Stage 1 — Corpus Cleaning

Cleans raw Urdu news articles into short, valid sentences (5–30 tokens).

```bash
python preprocess.py
```

| Parameter | Value |
| --- | --- |
| Input | `urdu-news-dataset-1M.csv` |
| Output | `urdu_clean_corpus.csv` |
| Min tokens | 5 |
| Max tokens | 30 |
| Random seed | 42 |

Cleaning steps: boilerplate removal, Unicode normalization, word fusion fixing, noise removal, sentence splitting, deduplication, and quality filtering.

---

## Stage 2 — Data Splitting

Splits the clean corpus into three subsets for each error generation method.

```bash
python split_preprocessed_sentences.py
```

| Output File | Split | Purpose |
| --- | ---: | --- |
| `data_splits/rule_based_input.csv` | 40% | Rule-based error generation |
| `data_splits/llm_input.csv` | 40% | LLM-based error generation |
| `data_splits/mt_input.csv` | 20% | MT-based error generation |

---

## Stage 3 — Error Generation

### Method 1 — Rule-Based

Generates synthetic errors using predefined linguistic rules for Urdu.

```bash
python rulebased.py
```

| Input | `data_splits/rule_based_input.csv` |
| --- | --- |
| Output | `rule_based_syntheticdata.csv` |

Error types: spelling errors, postposition substitution/deletion, word insertion/deletion, gender agreement errors.

---

### Method 2 — LLM-Based (Qwen3-8B)

Uses Qwen3-8B to generate 6 error  sentences.

```bash
python llmbased.py \
  --input data_splits/llm_input.csv \
  --output llm_syntheticdata.csv \
  --model_path /path/to/Qwen3-8B \
  --batch_size 4 \
  --max_new_tokens 768
```

| Error Type | Description |
| --- | --- |
| `spelling_error` | Single character change |
| `postposition_substitution` | Wrong postposition |
| `postposition_deletion` | Missing postposition |
| `word_insertion` | Extra unnecessary word |
| `word_deletion` | Missing word |
| `gender_agreement` | Wrong gender form |

> Supports checkpoint resume — rerun the same command to continue from where it stopped.

---

### Method 3 — MT-Based (M2M100)

Generates errors via Urdu→English→Urdu back-translation using M2M100.

```bash
python mtbased.py
```

| Input | `data_splits/mt_input.csv` |
| --- | --- |
| Output | `output_complete.csv` |
| Model | M2M100 |

> Requires GPU for reasonable speed. Falls back to CPU automatically.

---

## Output Format

All methods produce CSV files with the following columns:

| Column | Description |
| --- | --- |
| `correct_sentence` | Original clean Urdu sentence |
| `incorrect_sentence` | Synthetically corrupted sentence |
| `error_type` | Type of introduced error |
