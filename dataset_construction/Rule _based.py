import random
import pandas as pd
import re
from tqdm import tqdm
from collections import Counter



INPUT_CSV = "rule_based_input.csv"
OUTPUT_CSV = "rule_based_syntheticdata.csv"

NUM_VARIATIONS = 1
RANDOM_SEED = 42

random.seed(RANDOM_SEED)



CONFUSION_SET = {
    "ب": ["پ"],
    "پ": ["ب"],
    "ت": ["ٹ", "ث"],
    "ٹ": ["ت"],
    "س": ["ص", "ث"],
    "ص": ["س"],
    "ز": ["ذ", "ض", "ظ"],
    "ذ": ["ز"],
    "ض": ["ظ", "ز"],
    "ظ": ["ض", "ز"],
    "ہ": ["ح"],
    "ح": ["ہ"],
    "ی": ["ے"],
    "ے": ["ی"]
}



COMMON_WORDS = [
    "اور",
    "لیکن",
    "کہ",
    "کو",
    "سے",
    "میں",
    "پر",
    "نے",
    "تھا",
    "ہے"
]



POSTPOSITION_ERROR_MAP = {
    "نے": ["کو", "سے", "میں", ""],
    "کو": ["نے", "سے", "پر", ""],
    "سے": ["کو", "پر", "میں", ""],
    "میں": ["پر", "سے", ""],
    "پر": ["میں", "سے", ""],
    "کا": ["کی", "کے", ""],
    "کی": ["کا", "کے", ""],
    "کے": ["کا", "کی", ""]
}



GENDER_PAIRS = {
    "رہا": "رہی",
    "رہی": "رہا",
    "گیا": "گئی",
    "گئی": "گیا",
    "کھاتا": "کھاتی",
    "کھاتی": "کھاتا",
    "جاتا": "جاتی",
    "جاتی": "جاتا",
    "کرتا": "کرتی",
    "کرتی": "کرتا"
}



def is_valid_urdu(text):

    if not isinstance(text, str):
        return False

    text = text.strip()

    if len(text) < 5:
        return False

    urdu_chars = re.findall(r'[\u0600-\u06FF]', text)

    if len(urdu_chars) < len(text) * 0.5:
        return False

    return True



def spelling_error(sentence):

    words = sentence.split()

    candidates = []

    for word in words:
        for char in word:
            if char in CONFUSION_SET:
                candidates.append(word)
                break

    if not candidates:
        return None

    target = random.choice(candidates)

    chars = list(target)

    valid_positions = [
        i for i, c in enumerate(chars)
        if c in CONFUSION_SET
    ]

    if not valid_positions:
        return None

    pos = random.choice(valid_positions)

    chars[pos] = random.choice(
        CONFUSION_SET[chars[pos]]
    )

    corrupted = ''.join(chars)

    sentence = sentence.replace(
        target,
        corrupted,
        1
    )

    return sentence, "spelling_error"



def postposition_error(sentence):

    words = sentence.split()

    positions = [
        i for i, w in enumerate(words)
        if w in POSTPOSITION_ERROR_MAP
    ]

    if not positions:
        return None

    idx = random.choice(positions)

    original = words[idx]

    replacement = random.choice(
        POSTPOSITION_ERROR_MAP[original]
    )

    if replacement == "":
        del words[idx]
        label = "postposition_deletion"
    else:
        words[idx] = replacement
        label = "postposition_substitution"

    return " ".join(words), label



def gender_error(sentence):

    words = sentence.split()

    positions = [
        i for i, w in enumerate(words)
        if w in GENDER_PAIRS
    ]

    if not positions:
        return None

    idx = random.choice(positions)

    words[idx] = GENDER_PAIRS[words[idx]]

    return " ".join(words), "gender_agreement"



def word_insertion(sentence):

    words = sentence.split()

    idx = random.randint(0, len(words))

    insert_word = random.choice(COMMON_WORDS)

    words.insert(idx, insert_word)

    return " ".join(words), "word_insertion"



def word_deletion(sentence):

    words = sentence.split()

    if len(words) <= 3:
        return None

    idx = random.randint(0, len(words)-1)

    del words[idx]

    return " ".join(words), "word_deletion"



ERROR_FUNCTIONS = [
    spelling_error,
    postposition_error,
    gender_error,
    word_insertion,
    word_deletion
]

ERROR_WEIGHTS = [
    0.35,
    0.30,
    0.20,
    0.10,
    0.05
]


print("Loading dataset...")

df = pd.read_csv(INPUT_CSV)

print(f"Rows loaded: {len(df)}")


urdu_chars = set(
    'ابپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنہوےیئ'
)

TEXT_COLUMN = None

for col in df.columns:

    sample = str(df[col].iloc[0])

    if any(c in urdu_chars for c in sample):
        TEXT_COLUMN = col
        break

if TEXT_COLUMN is None:
    raise Exception("No Urdu text column found")

print(f"Using column: {TEXT_COLUMN}")



df = df.dropna(subset=[TEXT_COLUMN])

df[TEXT_COLUMN] = (
    df[TEXT_COLUMN]
    .astype(str)
    .str.strip()
)

df = df[
    df[TEXT_COLUMN] != ""
]

df = df.drop_duplicates(
    subset=[TEXT_COLUMN]
)

print(f"Clean rows: {len(df)}")



results = []

stats = Counter()

sentences = df[TEXT_COLUMN].tolist()

print("Generating synthetic data...")

for sentence in tqdm(sentences):

    if not is_valid_urdu(sentence):
        continue

    for _ in range(NUM_VARIATIONS):

        try:

            corruption_function = random.choices(
                ERROR_FUNCTIONS,
                weights=ERROR_WEIGHTS,
                k=1
            )[0]

            result = corruption_function(sentence)

            if result is None:
                continue

            corrupted, error_type = result

            if corrupted == sentence:
                continue

            if len(corrupted.split()) < 2:
                continue

            results.append({
                "incorrect_sentence": corrupted,
                "correct_sentence": sentence,
                "error_type": error_type
            })

            stats[error_type] += 1

        except:
            continue



output_df = pd.DataFrame(results)

output_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding='utf-8-sig'
)


print("\nGeneration Complete")
print("="*60)

print(f"Generated pairs: {len(output_df)}")

print("\nError Distribution:")

for k, v in stats.items():
    print(f"{k}: {v}")

print(f"\nSaved to: {OUTPUT_CSV}")

print("\nSample Output:\n")

for i in range(min(5, len(output_df))):

    row = output_df.iloc[i]

    print(f"Correct:   {row['correct_sentence']}")
    print(f"Incorrect: {row['incorrect_sentence']}")
    print(f"Type:      {row['error_type']}")
    print("-"*50)