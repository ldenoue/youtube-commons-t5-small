#!/usr/bin/env python3
"""
Fine-tune T5-small to restore punctuation, capitalization, and spelling
from ASR-style transcripts built from the YouTube-Commons dataset.
"""

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate


# 1️⃣  Load processed dataset ---------------------------------------------------
print("📂 Loading dataset...")
dataset = load_from_disk("t5_asr_correction_dataset")
train_ds = dataset["train"]
val_ds = dataset["validation"]

# 2️⃣  Model / tokenizer setup -------------------------------------------------
model_name = "t5-small"
print(f"🧠 Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Auto-select best device (CUDA → MPS → CPU)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")

if torch.cuda.is_available():
    print("✅ Using CUDA GPU:", torch.cuda.get_device_name(0))
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    print("✅ Using Apple Metal (MPS)")
else:
    print("⚙️ Using CPU")

# 3️⃣  Tokenization ------------------------------------------------------------
max_input_length = 256
max_target_length = 256

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_input_length,
        truncation=True,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target_text"],
            max_length=max_target_length,
            truncation=True,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("🔤 Tokenizing...")
tokenized_train = train_ds.map(preprocess_function, batched=True)
tokenized_val = val_ds.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# 4️⃣  Metrics -----------------------------------------------------------------
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    # Filter out empty strings
    paired = [(p, l) for p, l in zip(decoded_preds, decoded_labels) if len(p) > 0 and len(l) > 0]
    if not paired:
        return {"rouge1": 0.0, "rougeL": 0.0}
    decoded_preds, decoded_labels = zip(*paired)

    result = rouge.compute(predictions=list(decoded_preds), references=list(decoded_labels))
    # Only keep a few key scores
    result = {k: round(v * 100, 2) for k, v in result.items() if k in ["rouge1", "rougeL"]}
    return result

# 5️⃣  Training arguments ------------------------------------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5_asr_correction_model",
    eval_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),  # only on CUDA
    logging_dir="./logs",
)

# 6️⃣  Trainer -----------------------------------------------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,  # replaces tokenizer=...
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 7️⃣  Evaluate base model -----------------------------------------------------
print("🔍 Evaluating base model before fine-tuning...")
pre_eval = trainer.evaluate()
print(f"Base model ROUGE-1: {pre_eval.get('eval_rouge1', 0):.2f} | ROUGE-L: {pre_eval.get('eval_rougeL', 0):.2f}")

# 8️⃣  Train -------------------------------------------------------------------
print("🚀 Starting fine-tuning...")
trainer.train()

# 9️⃣  Evaluate fine-tuned model ----------------------------------------------
print("🔍 Evaluating fine-tuned model...")
post_eval = trainer.evaluate()
print(f"Fine-tuned model ROUGE-1: {post_eval.get('eval_rouge1', 0):.2f} | ROUGE-L: {post_eval.get('eval_rougeL', 0):.2f}")

# 🔟  Save --------------------------------------------------------------------
trainer.save_model("./t5_asr_correction_model_final")
tokenizer.save_pretrained("./t5_asr_correction_model_final")

print("✅ Training complete — model saved to ./t5_asr_correction_model_final")
