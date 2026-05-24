
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import pandas as pd
from tqdm import tqdm
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

path = "mt_input.csv"
tokenizer = M2M100Tokenizer.from_pretrained(path, local_files_only=True)
model = M2M100ForConditionalGeneration.from_pretrained(path, local_files_only=True).to(device)

def process_batch(sentences, batch_size=32):
    results = []
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i+batch_size]
        
        tokenizer.src_lang = "ur"
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("en"), num_beams=1)
        english = tokenizer.batch_decode(gen, skip_special_tokens=True)
        
        tokenizer.src_lang = "en"
        encoded = tokenizer(english, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("ur"), num_beams=1)
        corrupted = tokenizer.batch_decode(gen, skip_special_tokens=True)
        
        for j in range(len(batch)):
            results.append([batch[j], english[j], corrupted[j]])
    return results

df = pd.read_csv("mt_input.csv")
sentences = df['sentence'].astype(str).tolist()

print(f"\n Processing {len(sentences)} sentences...")
print(f" Estimated time: ~{len(sentences)/3.8/3600:.1f} hours")

start_time = time.time()
results = process_batch(sentences, batch_size=32)
total_time = time.time() - start_time

out = pd.DataFrame(results, columns=["original", "english", "corrupted"])
out.to_csv("output_complete.csv", index=False, encoding="utf-8-sig")

print(f"\n Completed in {total_time/60:.1f} minutes")
print(f" Speed: {len(sentences)/total_time:.1f} sentences/sec")
print(f" Saved to: output_complete.csv")
