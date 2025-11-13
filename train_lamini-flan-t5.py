import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from transformers import EarlyStoppingCallback
from datasets import Dataset, load_from_disk
from evaluate import load
import numpy as np

# ----------------------------
# 1️⃣ Load model & tokenizer
# ----------------------------
model_name = "MBZUAI/LaMini-Flan-T5-77M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")

if torch.cuda.is_available():
    print("✅ Using CUDA GPU:", torch.cuda.get_device_name(0))
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    print("✅ Using Apple Metal (MPS)")
else:
    print("⚙️ Using CPU")

# ----------------------------
# 2️⃣ Wrap model with LoRA
# ----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],  # attention query/value
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, lora_config)

# ----------------------------
# 3️⃣ Load / create dataset
# ----------------------------
dataset = load_from_disk("t5_asr_correction_dataset")
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

prefix = "Correct this ASR transcript: "

def preprocess(batch):
    # Combine prefix with each example
    inputs = [prefix + x for x in batch["input_text"]]
    targets = batch["target_text"]

    # Tokenize both sides
    model_inputs = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    labels = tokenizer(
        targets,
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in model_inputs["labels"]
    ]
    return model_inputs

#tokenized_datasets = dataset.map(preprocess, batched=True)
tokenized_datasets = dataset.map(
    preprocess,
    batched=True,
    remove_columns=["input_text", "target_text"],
)
print(dataset.column_names)
print(tokenized_datasets["train"].column_names)
# ----------------------------
# 4️⃣ Metrics: WER + CER
# ----------------------------
wer_metric = load("wer")
cer_metric = load("cer")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)  # ✅ fix here

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"wer": wer, "cer": cer}

# ----------------------------
# 5️⃣ Training arguments
# ----------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="./asr_lora",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=2,
    fp16=False,
    bf16=False,
    optim="adamw_torch",               # ✅ ensure no bitsandbytes
    report_to="none",
)

# ----------------------------
# 6️⃣ Data collator
# ----------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ----------------------------
# 7️⃣ Trainer with early stopping
# ----------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

# ----------------------------
# 8️⃣ Train LoRA adapters
# ----------------------------
trainer.train()

# ----------------------------
# 9️⃣ Merge LoRA adapters into base model for inference
# ----------------------------
model = model.merge_and_unload()
model.save_pretrained("./asr_lora_merged")
tokenizer.save_pretrained("./asr_lora_merged")

# Now you can quantize this model and run it in the browser
