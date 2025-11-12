from datasets import load_dataset, Dataset, DatasetDict
import re
from tqdm import tqdm
from pprint import pprint
import html

from collections import Counter

def normalize_text(text):
    """Lowercase and remove punctuation for ASR-style input."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

MAX_WORDS = 40 # keep low to limit chunk size for memory when we run in the browser
#MAX_WORDS = 128
def chunk_text(text, max_words=MAX_WORDS):
    """Split long text into chunks of roughly max_words."""
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# 1. Load dataset (subset if needed)
print("Loading dataset...")
#dataset = load_dataset("PleIAs/YouTube-Commons", split="train")
dataset = load_dataset(
    "parquet",
    data_files=[
        "https://huggingface.co/datasets/PleIAs/YouTube-Commons/resolve/main/cctube_0.parquet",
        "https://huggingface.co/datasets/PleIAs/YouTube-Commons/resolve/main/cctube_1.parquet",
        "https://huggingface.co/datasets/PleIAs/YouTube-Commons/resolve/main/cctube_2.parquet",
        "https://huggingface.co/datasets/PleIAs/YouTube-Commons/resolve/main/cctube_3.parquet",
    ],
    split="train"
)

# # 1️⃣ Count occurrences of each video_id
# id_counts = Counter(dataset["video_id"])

# # 2️⃣ Collect IDs that appear more than once
# dupe_ids = {vid for vid, c in id_counts.items() if c > 1}
# print(f"Found {len(dupe_ids)} duplicated video_id values")

# # 3️⃣ Filter dataset to only those duplicates
# dupes_dataset = dataset.filter(lambda x: x["video_id"] in dupe_ids)

# # 4️⃣ Convert to pandas for convenient display
# df = dupes_dataset.to_pandas()

# # 5️⃣ Select relevant columns
# df_subset = df[["video_id", "original_language", "transcription_language"]]

# # 6️⃣ Group by video_id and show
# for vid, group in df_subset.groupby("video_id"):
#     print(f"\n▶ video_id: {vid}")
#     print(group[["original_language", "transcription_language"]].to_string(index=False))

print('origina',dataset)

dataset = dataset.filter(
    lambda x: x.get("original_language") == "en" and x.get("transcription_language") == "en"
    #lambda x: x.get("original_language") == "en"
    #lambda x: x.get("transcription_language") == "en"
)

print('en only',dataset)

def uppercase_ratio(text):
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(ch.isupper() for ch in letters) / len(letters)

def non_uppercase(example):
    ratio = uppercase_ratio(example["text"])
    return ratio < 0.2  # keep only if less than 99% uppercase

def clean_html_entities(example):
    text = html.unescape(example["text"])       # Decode HTML entities
    text = text.replace("\xa0", " ")            # Replace non-breaking spaces
    example["text"] = text
    return example

def has_punctuations(example):
    punctuations = ",.;:?!"
    text = example.get("text", "") or ""
    if not text:
        return False

    # Count punctuation characters
    n = sum(1 for ch in text if ch in punctuations)
    percent = (n * 100) / len(text)
    
    # Keep texts where at least 1% of chars are punctuation
    return percent >= 1.0

# 2️⃣ Apply the filter

dataset_punctuated = (
    dataset
    .map(clean_html_entities, num_proc=4)
    .filter(non_uppercase, num_proc=4)
    .filter(has_punctuations, num_proc=4)
)

print('non uppercase with punctuations',dataset)
#dataset_punctuated = dataset.filter(has_punctuations, num_proc=4)

pprint(dataset_punctuated)
pprint(dataset_punctuated[0])

# 2. Build new samples for ASR correction
inputs, outputs = [], []

for item in tqdm(dataset_punctuated, desc="Processing samples"):
    # use available transcript text
    raw_text = item.get("text") or item.get("transcript") or ""
    if not raw_text.strip():
        continue
    
    chunks = chunk_text(raw_text)
    for chunk in chunks:
        cleaned_text = html.unescape(chunk).replace('\xa0', ' ').strip()
        #cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        noisy_input = normalize_text(cleaned_text)
        inputs.append(noisy_input)
        outputs.append(cleaned_text)

# 3. Create new dataset
new_dataset = Dataset.from_dict({
    "input_text": inputs,
    "target_text": outputs
})

pprint(new_dataset[0])
pprint(new_dataset[1])
pprint(new_dataset[2])
# 4. Optional: split into train/validation sets
dataset_dict = new_dataset.train_test_split(test_size=0.1, seed=42)
dataset_dict = DatasetDict({
    "train": dataset_dict["train"],
    "validation": dataset_dict["test"]
})

# 5. Save locally
dataset_dict.save_to_disk("t5_asr_correction_dataset")

# (Optional) Push to HF Hub:
# dataset_dict.push_to_hub("your-username/t5-asr-correction")

print("✅ New dataset created and saved to 't5_asr_correction_dataset'")
print(dataset_dict)


