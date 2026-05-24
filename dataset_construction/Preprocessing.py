
import pandas as pd
import re
import hashlib
import random
import io
from collections import Counter


INPUT_FILE   = "Urdu Clean Corpora\urdu-news-dataset-1M.csv"
OUTPUT_FILE  = "Urdu Clean Corpora\urdu_clean_corpus.csv"
STATS_FILE   = "Urdu Clean Corpora\urdu_corpus_stats.txt"
REVIEW_FILE  = "Urdu Clean Corpora\manual_review.txt"


TEST_MODE = False
TEST_ROWS = 1000


def load_csv(filepath):
    print("  Method 1: pandas utf-8-sig (UTF-8 with BOM)\n")
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig', low_memory=False)

    
        sample = ' '.join(df.iloc[:3].astype(str).values.flatten())
        urdu_chars = sum(1 for c in sample if '\u0600' <= c <= '\u06FF')

        if urdu_chars > 20:
            print(f"   Method 1 SUCCESS — {urdu_chars} Urdu characters confirmed")
            return df, 'utf-8-sig'
        else:
            print(f"   Method 1: only {urdu_chars} Urdu chars — trying fallbacks...")

    except Exception as e:
        print(f"   Method 1 failed: {e}")

   
    for enc in ['utf-8', 'cp1256']:
        try:
            df = pd.read_csv(filepath, encoding=enc,
                             encoding_errors='replace', low_memory=False)
            sample = ' '.join(df.iloc[:3].astype(str).values.flatten())
            urdu_chars = sum(1 for c in sample if '\u0600' <= c <= '\u06FF')
            if urdu_chars > 20:
                print(f"   Fallback SUCCESS: pandas encoding={enc}")
                return df, enc
            else:
                print(f"  {enc}: only {urdu_chars} Urdu chars")
        except Exception as e:
            print(f"   {enc} failed: {e}")

    raise ValueError(
        "All loading methods failed. Run this to detect encoding:\n"
        "  pip install chardet\n"
        f"  python -c \"import chardet; "
        f"print(chardet.detect(open(r'{filepath}', 'rb').read(100000)))\""
    )


print("=" * 60)
print("  URDU NEWS CORPUS CLEANING PIPELINE (MAX 30 TOKENS)")
print("=" * 60)
print(f"\n[1/6] Loading dataset from:\n  {INPUT_FILE}\n")

df, detected_encoding = load_csv(INPUT_FILE)

df.columns = [re.sub(r'^[\ufeff\W]+', '', c.strip()) for c in df.columns]

print(f"\n  Total rows loaded : {len(df):,}")
print(f"  Columns found     : {df.columns.tolist()}")

if TEST_MODE:
    df = df.head(TEST_ROWS)
    print(f"\n  TEST MODE — first {TEST_ROWS:,} rows only")
    print(f"    Set TEST_MODE = False for full dataset\n")



def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {col.lower().strip(): col for col in df.columns}
    for c in candidates:
        if c.lower().strip() in lower_map:
            return lower_map[c.lower().strip()]
    return None


print("\n[2/6] Detecting column names...\n")

TEXT_COL     = find_column(df, ['News Text', 'news_text', 'NewsText',
                                 'text', 'Text', 'body', 'Body', 'content'])
HEADLINE_COL = find_column(df, ['Headline', 'headline', 'title', 'Title'])
CATEGORY_COL = find_column(df, ['Category', 'category', 'label', 'Label', 'topic'])
SOURCE_COL   = find_column(df, ['Source', 'source', 'publisher', 'Publisher'])

print(f"  Text column     → '{TEXT_COL}'")
print(f"  Headline column → '{HEADLINE_COL}'")
print(f"  Category column → '{CATEGORY_COL}'")
print(f"  Source column   → '{SOURCE_COL}'")

if TEXT_COL is None:
    print("\n   ERROR: Text column not found.")
    print("    Available columns:", df.columns.tolist())
    raise SystemExit(1)

non_empty_rows = (
    df[TEXT_COL].dropna().astype(str).str.strip().str.len().gt(20).sum()
)
print(f"\n  Rows with non-empty text (>20 chars): {non_empty_rows:,} of {len(df):,}")

print("\n  --- Raw text sample (row 0) ---")
raw_sample = str(df.iloc[0][TEXT_COL])
print(f"  {raw_sample[:300]}")
urdu_in_sample  = sum(1 for c in raw_sample if '\u0600' <= c <= '\u06FF')
latin_in_sample = sum(1 for c in raw_sample if c.isascii() and c.isalpha())
print(f"  Urdu characters  : {urdu_in_sample}")
print(f"  Latin characters : {latin_in_sample}")

if urdu_in_sample < 10:
    print("\n  ✗ CRITICAL: Text is still garbled after encoding fix.")
    print("    The output would be meaningless — stopping now.")
    print("    Share the 'Columns found' line above for further diagnosis.")
    raise SystemExit(1)
elif urdu_in_sample > latin_in_sample:
    print("   Urdu dominates sample — encoding is correct\n")
else:
    print("   WARNING: Latin chars outnumber Urdu — check sample above\n")



BOILERPLATE_PHRASES = [
    'مزید پڑھیں', 'بھی پڑھیں', 'تصویر بشکریہ', 'ویب ڈیسک',
    'نمائندہ خصوصی', 'خصوصی رپورٹ', 'فوٹو فائل', 'فائل فوٹو',
    'رپورٹ:', 'ایجنسیاں', 'ذریعہ:', 'نمائندہ',
]

INVISIBLE_CHARS = [
    '\u200b', '\u200c', '\u200d', '\u200e', '\u200f',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e', '\ufeff',
]


def remove_boilerplate(text):
    for phrase in BOILERPLATE_PHRASES:
        text = text.replace(phrase, ' ')
    return text


def normalize_urdu_unicode(text):
    
    text = re.sub(r'[آأإٱ]', 'ا', text)
   
    text = text.replace('\u064a', '\u06cc')
   
    text = text.replace('\u0643', '\u06a9')
    
    text = text.replace('\u0640', '')
  
    for c in INVISIBLE_CHARS:
        text = text.replace(c, '')
    return text


def fix_word_fusions(text):
    """Insert space when Urdu verb endings are directly fused with next word."""
    verb_endings = [
        'گی', 'گا', 'گے', 'ہے', 'ہیں', 'ہوں',
        'تھا', 'تھی', 'تھے', 'کریں', 'کرے', 'کرو',
        'ہوا', 'ہوئی', 'ہوئے', 'کیا', 'کی', 'کے', 'کا',
        'آیا', 'آئی', 'آئے',
    ]
    for ending in verb_endings:
        pattern = rf'({re.escape(ending)})([\u0600-\u06FF])'
        text = re.sub(pattern, r'\1 \2', text)
    return text


def remove_noise(text):
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'[۔]{2,}', '۔', text)
    text = re.sub(r'[؟]{2,}', '؟', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[-]{3,}', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.strip()
    if not text or text.lower() == 'nan': 
        return ''
    text = remove_boilerplate(text)
    text = normalize_urdu_unicode(text)
    text = fix_word_fusions(text)
    text = remove_noise(text)
    return text




VERB_ENDING_PATTERNS = [
    'جاسکتا ہے',
    'جاسکتی ہے',
    'کی جارہی ہے',
    'کی جارہا ہے',
    'جارہا ہے',
    'جارہی ہے',
    'جاتا ہے',
    'جاتی ہے',
    'سکتا ہے',
    'سکتی ہے',
    'کیا جاتا ہے',
    'کیا جائے',
    'کیا گیا',
    'کی گئی',
    'کے گئے',
    'دیا گیا',
    'دی گئی',
    'ہو گیا',
    'ہو گئی',
    'ہو گئے',
    'گئی تھی',
    'گئے تھے',
    'گیا تھا',
    'گیا ہے',
    'گئی ہے',
    'گئے ہیں',
    'ہوئی ہے',
    'ہوئے ہیں',
    'ہوا ہے',
    'ہیں',
    'ہوئی',
    'ہوئے',
    'ہوا',
    'گئی',
    'گئے',
    'گیا',
    'تھی',
    'تھے',
    'تھا',
    'گی',
    'گا',
    'گے',
    'ہے',
]


def split_on_verb_endings(text):
    """Split an unpunctuated Urdu blob on sentence-final verb patterns."""
    MARKER = '\x00'
    
    
    if len(text.split()) > 40:
        marked = text
        for ending in VERB_ENDING_PATTERNS:
            pattern = rf'({re.escape(ending)})(?=\s+[\u0600-\u06FF])'
            marked = re.sub(pattern, r'\1' + MARKER, marked)
        
        parts = marked.split(MARKER)
        result = [p.strip() for p in parts if p.strip()]
        if len(result) > 1:
            final_result = []
            for part in result:
                if len(part.split()) <= 30:
                    final_result.append(part)
                else:
                    
                    sub_parts = split_on_verb_endings(part)
                    final_result.extend(sub_parts)
            return final_result if final_result else [text]
    
    return [text]


def split_into_sentences(text):
    if not text:
        return []

    
    parts = re.split(r'[۔؟!।]+', text)
    result = [p.strip() for p in parts if p.strip()]

    
    if len(result) <= 1 and result:
        alt = re.split(r'\n+|  +', result[0])
        alt = [s.strip() for s in alt if s.strip()]
        if len(alt) > 1:
            return alt

    if len(result) == 1 and len(result[0].split()) > 20:
        verb_split = split_on_verb_endings(result[0])
        if len(verb_split) > 1:
            return verb_split

    return result


def break_long_sentence(sentence, max_tokens=30):
    """Break a sentence longer than max_tokens into smaller chunks"""
    words = sentence.split()
    if len(words) <= max_tokens:
        return [sentence]
    
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_tokens:
         
            chunk_text = ' '.join(current_chunk)
            for punct in ['۔', '؟', '!', '،']:
                if punct in chunk_text:
                    parts = chunk_text.split(punct)
                    for i, part in enumerate(parts[:-1]):
                        if part.strip():
                            chunks.append(part.strip() + punct)
                    current_chunk = parts[-1].strip().split()
                    if not current_chunk:
                        current_chunk = []
                    break
            else:
                chunks.append(' '.join(current_chunk[:max_tokens]))
                current_chunk = current_chunk[max_tokens:]
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return [chunk for chunk in chunks if len(chunk.split()) >= 5]



reject_counts = {
    'empty_text':     0,
    'too_short':      0,
    'too_long':       0,
    'latin_heavy':    0,
    'no_urdu_chars':  0,
    'mojibake':       0,
    'mostly_symbols': 0,
    'repetitive':     0,
    'duplicate':      0,
}


def is_valid_sentence(sent):
    words      = sent.split()
    word_count = len(words)
  
    if word_count < 5:
        reject_counts['too_short'] += 1
        return False
    if word_count > 30: 
        reject_counts['too_long'] += 1
        return False

   
    latin_chars = sum(1 for c in sent if c.isascii() and c.isalpha())
    if len(sent) > 0 and latin_chars / len(sent) > 0.20:
        reject_counts['latin_heavy'] += 1
        return False

    urdu_chars = sum(1 for c in sent if '\u0600' <= c <= '\u06FF')
    if urdu_chars < 10:
        reject_counts['no_urdu_chars'] += 1
        return False

   
    if re.search(r'[طظغ]{3,}', sent):
        reject_counts['mojibake'] += 1
        return False

   
    alpha_chars = sum(1 for c in sent if c.isalpha())
    if len(sent) > 0 and alpha_chars / len(sent) < 0.50:
        reject_counts['mostly_symbols'] += 1
        return False

    freq = Counter(words)
    if freq.most_common(1)[0][1] / word_count > 0.35:
        reject_counts['repetitive'] += 1
        return False

    return True



print("[3/6] Cleaning and segmenting sentences...\n")

all_sentences = []
seen_hashes   = set()

for idx, row in df.iterrows():

    raw_text = str(row[TEXT_COL]) if TEXT_COL else ''
    headline = str(row[HEADLINE_COL]) if HEADLINE_COL else ''
    category = str(row[CATEGORY_COL]).strip() if CATEGORY_COL else 'Unknown'
    source   = str(row[SOURCE_COL]).strip() if SOURCE_COL else ''

    if not raw_text.strip() or raw_text.lower() == 'nan':
        reject_counts['empty_text'] += 1
        continue

    cleaned   = clean_text(raw_text)
    sentences = split_into_sentences(cleaned)

    for sent in sentences:
        if not is_valid_sentence(sent):
            continue
        
       
        broken_sentences = break_long_sentence(sent, max_tokens=30)
        
        for broken_sent in broken_sentences:
            if not is_valid_sentence(broken_sent):
                continue

            h = hashlib.sha256(broken_sent.encode('utf-8')).hexdigest()
            if h in seen_hashes:
                reject_counts['duplicate'] += 1
                continue
            seen_hashes.add(h)

            all_sentences.append({
                'sentence':   broken_sent,
                'category':   category,
                'source':     source,
                'headline':   clean_text(headline),
                'word_count': len(broken_sent.split()),
            })

    if (idx + 1) % 50000 == 0:
        print(f"  Processed {idx + 1:,} articles — "
              f"{len(all_sentences):,} sentences so far...")

print(f"\n   Finished processing {len(df):,} articles")



print("\n[4/6] Rejection breakdown:\n")

total_rejected  = sum(reject_counts.values())
total_attempted = len(all_sentences) + total_rejected

print(f"  {'Reason':<25} {'Count':>10}  {'%':>8}")
print(f"  {'-'*48}")
for reason, count in reject_counts.items():
    pct   = (count / total_attempted * 100) if total_attempted > 0 else 0
    flag  = '   HIGH' if reason in ('mojibake', 'no_urdu_chars') and count > 5000 else ''
    print(f"  {reason:<25} {count:>10,}  {pct:>7.1f}%{flag}")
print(f"  {'-'*48}")
accepted_pct = (len(all_sentences) / total_attempted * 100) if total_attempted > 0 else 0
print(f"  {'ACCEPTED':<25} {len(all_sentences):>10,}  {accepted_pct:>7.1f}%")

# Sanity checks
print()
wc_list = [r['word_count'] for r in all_sentences]
if wc_list:
    mean_wc   = sum(wc_list) / len(wc_list)
    median_wc = sorted(wc_list)[len(wc_list) // 2]
    
    if mean_wc > 28:
        print(f"   SANITY WARNING: Mean word count = {mean_wc:.1f} — near max limit (30)")
    else:
        print(f"   SANITY PASS: Mean word count = {mean_wc:.1f} (max=30)")

    if max(wc_list) > 30:
        print(f"   SANITY FAIL: Found sentences with {max(wc_list)} tokens (max should be 30)")
    else:
        print(f"   SANITY PASS: Max token count = {max(wc_list)} (≤30)")

    if reject_counts['too_short'] == 0 and reject_counts['too_long'] == 0:
        print("   SANITY FAIL: Zero length rejections — splitting not working")
    else:
        print(f"   SANITY PASS: Length filters active "
              f"(short={reject_counts['too_short']:,} long={reject_counts['too_long']:,})")

if len(all_sentences) == 0:
    print("\n   CRITICAL: Zero sentences accepted — cannot continue.")
    raise SystemExit(1)



print(f"\n[5/6] Saving outputs...\n")

result_df = pd.DataFrame(all_sentences)

try:
    result_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"   Clean corpus saved  → {OUTPUT_FILE}")
except PermissionError:
    alt = OUTPUT_FILE.replace('.csv', '_new.csv')
    result_df.to_csv(alt, index=False, encoding='utf-8-sig')
    print(f"   Original file locked — close it in Excel first")
    print(f"   Saved to            → {alt}")




print("\n[6/6] Post-save quality verification...\n")

try:
    saved_df = pd.read_csv(OUTPUT_FILE, encoding='utf-8-sig')
    print(f"  Reloaded {len(saved_df):,} sentences from saved file\n")

    results = []

    def check(label, condition, detail=''):
        results.append(condition)
        icon = '✓' if condition else '✗'
        print(f"  {icon} {label}" + (f" — {detail}" if detail else ''))
        return condition

    check("File not empty",
          len(saved_df) > 0,
          f"{len(saved_df):,} sentences")

    null_count = saved_df['sentence'].isna().sum()
    check("No null sentences", null_count == 0,
          f"{null_count} nulls")

    def count_urdu(t):
        return sum(1 for c in str(t) if '\u0600' <= c <= '\u06FF')
    saved_df['urdu_count'] = saved_df['sentence'].apply(count_urdu)
    bad_urdu = (saved_df['urdu_count'] < 5).sum()
    check("Urdu chars in every sentence", bad_urdu == 0,
          f"{bad_urdu} sentences have <5 Urdu chars")

    dup_count = saved_df['sentence'].duplicated().sum()
    check("No duplicates", dup_count == 0,
          f"{dup_count} duplicates")

    wc = saved_df['sentence'].str.split().str.len()
    realistic = (wc.mean() > 4) and (wc.mean() < 31) and (wc.max() <= 30)
    check("Realistic word lengths (max 30)", realistic,
          f"min={wc.min()} max={wc.max()} mean={wc.mean():.1f} median={wc.median():.0f}")

    mojibake_pat  = re.compile(r'[طظغ]{3,}')
    garbled_count = saved_df['sentence'].str.contains(mojibake_pat, na=False).sum()
    check("No mojibake/garbled text", garbled_count == 0,
          f"{garbled_count} garbled sentences")

    cat_count = saved_df['category'].nunique()
    check("Multiple categories", cat_count > 1,
          f"{cat_count} categories")

    passed       = sum(results)
    total_checks = len(results)
    print(f"\n  Result: {passed}/{total_checks} checks passed")
    if passed == total_checks:
        print("   All checks passed — corpus is ready to use")
    else:
        print("   Some checks failed — review ✗ items above")

except Exception as e:
    print(f"   Verification error: {e}")



total         = len(result_df)
cat_counts    = Counter(result_df['category'])
source_counts = Counter(result_df['source'])
lengths       = result_df['word_count']

lines = []
lines.append("=" * 60)
lines.append("  URDU CLEAN CORPUS — STATISTICS REPORT (MAX 30 TOKENS)")
lines.append("=" * 60)
lines.append(f"\n  Input file        : {INPUT_FILE}")
lines.append(f"  Output file       : {OUTPUT_FILE}")
lines.append(f"  Encoding detected : {detected_encoding}")
lines.append(f"  Test mode         : "
             f"{'YES (first ' + str(TEST_ROWS) + ' rows)' if TEST_MODE else 'NO (full dataset)'}")
lines.append(f"\n  Total articles processed : {len(df):,}")
lines.append(f"  Total clean sentences    : {total:,}")
lines.append(f"  Unique categories        : {result_df['category'].nunique()}")
lines.append(f"  Unique sources           : {result_df['source'].nunique()}")

lines.append("\n── Rejection Summary ────────────────────────────────")
for reason, count in reject_counts.items():
    pct = (count / total_attempted * 100) if total_attempted > 0 else 0
    lines.append(f"  {reason:<25} {count:>10,}   ({pct:.1f}%)")

lines.append("\n── Category Distribution ────────────────────────────")
lines.append(f"  {'Category':<30} {'Count':>8}  {'%':>7}")
lines.append(f"  {'-'*50}")
for cat, count in cat_counts.most_common():
    pct = count / total * 100
    bar = '█' * int(pct / 2)
    lines.append(f"  {cat:<30} {count:>8,}  {pct:>6.1f}%  {bar}")

lines.append("\n── Source Distribution ──────────────────────────────")
for src, count in source_counts.most_common(20):
    pct = count / total * 100
    lines.append(f"  {src:<30} {count:>8,}  ({pct:.1f}%)")

lines.append("\n── Sentence Length Distribution (MAX 30) ────────────")
lines.append(f"  Min words    : {lengths.min()}")
lines.append(f"  Max words    : {lengths.max()}")
lines.append(f"  Mean words   : {lengths.mean():.1f}")
lines.append(f"  Median words : {lengths.median():.1f}")

lines.append("\n── Random Sample (3 sentences per category) ─────────")
for cat in list(cat_counts.keys())[:10]:
    cat_sents = result_df[result_df['category'] == cat]['sentence'].tolist()
    sample = random.sample(cat_sents, min(3, len(cat_sents)))
    lines.append(f"\n  [{cat}]")
    for s in sample:
        lines.append(f"    • {s}")

lines.append("\n" + "=" * 60)

report = "\n".join(lines)
print("\n" + report)

try:
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n  ✓ Stats report saved  → {STATS_FILE}")
except PermissionError:
    alt_stats = STATS_FILE.replace('.txt', '_new.txt')
    with open(alt_stats, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  ⚠ Stats saved to      → {alt_stats}")




review_lines = []
review_lines.append("URDU CORPUS — MANUAL REVIEW SAMPLE (MAX 30 TOKENS)")
review_lines.append("=" * 60)
review_lines.append("HOW TO USE THIS FILE:")
review_lines.append("  1. Open in Notepad++")
review_lines.append("  2. Encoding menu → should show UTF-8-BOM")
review_lines.append("  3. Read through sentences and check they look correct")
review_lines.append("")
review_lines.append("WHAT GOOD LOOKS LIKE:")
review_lines.append("   اسلام آباد میں آج موسم خوشگوار رہا اور شہریوں نے")
review_lines.append("  پاکستان کرکٹ ٹیم نے میچ جیت لیا")
review_lines.append("")
review_lines.append("WHAT BAD LOOKS LIKE:")
review_lines.append("   ط§ط³ظ„ط§ظ… ط¨ط§ط¯  ← mojibake (encoding still broken)")
review_lines.append("  ہے ہے ہے ہے      ← repetitive spam")
review_lines.append("   کیا ہے            ← too short (should be filtered)")
review_lines.append("")
review_lines.append("MAXIMUM SENTENCE LENGTH: 30 TOKENS")
review_lines.append("=" * 60)

for cat in result_df['category'].unique():
    if str(cat).lower() == 'nan':
        continue
    cat_df = result_df[result_df['category'] == cat]
    sample = cat_df.sample(min(20, len(cat_df)), random_state=42)
    review_lines.append(f"\n{'='*50}")
    review_lines.append(f"CATEGORY: {cat}  (total: {len(cat_df):,} sentences)")
    review_lines.append('=' * 50)
    for _, row in sample.iterrows():
        review_lines.append(f"  [{row['word_count']}w]  {row['sentence']}")

try:
    with open(REVIEW_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(review_lines))
    print(f"   Manual review file  → {REVIEW_FILE}")
    print(f"    Open in Notepad++ and verify sentences are readable Urdu")
    print(f"    MAX TOKEN LENGTH: 30")
except Exception as e:
    print(f"   Could not save review file: {e}")

print("\n PIPELINE COMPLETE — All sentences have 5-30 tokens.\n")