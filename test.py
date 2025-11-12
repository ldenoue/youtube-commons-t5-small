
import pandas as pd
import html
import textwrap

# read parquet file into a DataFrame
df = pd.read_parquet("cctube_0.parquet")

print(df.head())        # show first rows
print(df.columns)       # see column names


from datasets import Dataset, load_dataset

pd.set_option('display.max_colwidth', None)

#import pandas as pd
#df = pd.read_parquet("/content/drive/MyDrive/cctube_0.parquet")
df_en = df[(df["original_language"] == "en") & (df["transcription_language"] == "en")].copy()

#df_en["text"] = df_en["text"].apply(lambda x: html.unescape(x))
df_en["text"] = df_en["text"].apply(
    lambda x: " ".join(html.unescape(x).split())
)
# Define punctuation regex (anything that's not word or whitespace)
punctuations = ",.;:?!"

def has_punctuations(text):
    if not text:
        return False
    n = sum(1 for c in text if c in punctuations)
    percent = n * 100 / len(text)
    return percent >= 1.0

def chunk_text(text, chunk_size=512):
    words = text.split()  # splits by any whitespace, keeps punctuation attached
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])


# Select rows where text has punctuation
#df_en = df_en[df_en["text"].str.contains(punct_pattern, regex=True, na=False)]
df_en_with_punct = df_en[df_en["text"].apply(has_punctuations)]
df_en_with_punct = df_en_with_punct.copy()
df_en_with_punct["text_clean"] = (
    df_en_with_punct["text"]
    #.str.replace(r"[^\w\s]", "", regex=True)  # remove all punctuation
    .str.translate(str.maketrans('', '', punctuations))
    .str.lower()                             # convert to lowercase
)
#print(df_en[["title", "text"]].head(10))
print(len(df_en), len(df_en_with_punct))  # number of rows with original_language = "en"
#print(df_en_with_punct.head(1))

df_subset = df_en_with_punct[["text", "text_clean"]]

#for text in df_subset["text"]:
#    print(text)
for idx, row in df_subset.iterrows():
    print("--- Original:", idx, textwrap.fill(row["text"],width=160))
    chunks = list(chunk_text(row["text"], 512))
    clean_chunks = list(chunk_text(row["text_clean"], 512))
    print("--- First sentence:", idx, textwrap.fill(chunks[0],width=160))
    print("--- First clean   :", idx, textwrap.fill(clean_chunks[0],width=160))
    print("--- Last sentence:", idx, textwrap.fill(chunks[-1],width=160))
    print("--- Last clean.   :", idx, textwrap.fill(clean_chunks[-1],width=160))
    #print("Cleaned :", idx, textwrap.fill(row["text_clean"],width=80))
    print("---")
# Convert to Hugging Face dataset
#dataset = Dataset.from_pandas(df_subset)

exit(0)
import nltk
import re
from datasets import load_dataset

# Download sentence tokenizer
nltk.download("punkt")

# Load datasets
wiki2 = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
wiki103 = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

def degrade(text):
    """Simulate ASR output: lowercase + remove punctuation"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def avg_sentence_length(dataset, num_samples=5000):
    sentences = []
    for i, row in enumerate(dataset):
        if i >= num_samples:
            break
        text = row["text"].strip()
        if text:
            sents = nltk.sent_tokenize(text)
            sentences.extend(sents)
    if not sentences:
        return 0, 0
    avg_words = sum(len(re.findall(r"\w+", s)) for s in sentences) / len(sentences)
    return len(sentences), avg_words

# Compute stats
# wiki2_stats = avg_sentence_length(wiki2)
# wiki103_stats = avg_sentence_length(wiki103)

# print("📊 WikiText-2:", wiki2_stats[0], "sentences, avg words per sentence:", wiki2_stats[1])
# print("📊 WikiText-103:", wiki103_stats[0], "sentences, avg words per sentence:", wiki103_stats[1])

# Show some degraded vs original examples
print("\n🔎 Example pairs (WikiText-2):\n")
count = 0
print(len(wiki103))
for i in range(50,100):
    print(wiki103[i], flush=True)
# for row in wiki2:
#     text = row["text"].strip()
#     if text:
#         print(text)
#         for sent in nltk.sent_tokenize(text):
#             degraded = degrade(sent)
#             print("Input (degraded):", degraded)
#             print("Target (original):", sent)
#             print("---")
#             count += 1
#             if count >= 5:
#                 break
#     if count >= 5:
#         break

