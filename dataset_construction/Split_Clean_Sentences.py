import pandas as pd
from sklearn.model_selection import train_test_split
import os


INPUT_CSV = "urdu_clean_corpusC.csv"

TEXT_COLUMN = "sentence"

OUTPUT_DIR = "data_splits"

RANDOM_SEED = 42


os.makedirs(OUTPUT_DIR, exist_ok=True)


print("="*60)
print("URDU SYNTHETIC CORPUS SPLITTING PIPELINE")
print("="*60)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_CSV)

print(f"Original dataset size: {len(df)}")


print("\nCleaning dataset...")

df = df.dropna(subset=[TEXT_COLUMN])

df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

df[TEXT_COLUMN] = df[TEXT_COLUMN].str.strip()

df = df[df[TEXT_COLUMN] != ""]

df = df.drop_duplicates(subset=[TEXT_COLUMN])

print(f"Dataset after cleaning: {len(df)}")



print("\nShuffling dataset...")

df = df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)



print("\nSplitting dataset...")



rule_df, remaining_df = train_test_split(
    df,
    test_size=0.60,
    random_state=RANDOM_SEED
)

llm_df, mt_df = train_test_split(
    remaining_df,
    test_size=0.3333,
    random_state=RANDOM_SEED
)

print("\nSaving split files...")

RULE_PATH = os.path.join(
    OUTPUT_DIR,
    "rule_based_input.csv"
)

LLM_PATH = os.path.join(
    OUTPUT_DIR,
    "llm_input.csv"
)

MT_PATH = os.path.join(
    OUTPUT_DIR,
    "mt_input.csv"
)

rule_df.to_csv(
    RULE_PATH,
    index=False,
    encoding='utf-8-sig'
)

llm_df.to_csv(
    LLM_PATH,
    index=False,
    encoding='utf-8-sig'
)

mt_df.to_csv(
    MT_PATH,
    index=False,
    encoding='utf-8-sig'
)


print("\n" + "="*60)
print("SPLIT COMPLETE")
print("="*60)

total = len(df)

print(f"\nTotal Sentences: {total}")

print("\nGenerated Files:")

print(f"\n1. Rule-Based")
print(f"   Sentences: {len(rule_df)}")
print(f"   Percentage: {(len(rule_df)/total)*100:.2f}%")
print(f"   File: {RULE_PATH}")

print(f"\n2. LLM-Based")
print(f"   Sentences: {len(llm_df)}")
print(f"   Percentage: {(len(llm_df)/total)*100:.2f}%")
print(f"   File: {LLM_PATH}")

print(f"\n3. Machine Translation")
print(f"   Sentences: {len(mt_df)}")
print(f"   Percentage: {(len(mt_df)/total)*100:.2f}%")
print(f"   File: {MT_PATH}")

print("\nPipeline Finished Successfully!")