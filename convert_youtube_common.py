#!/usr/bin/env python3
"""
Export a fine-tuned T5-small model to ONNX for use with transformers.js
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager
from pathlib import Path
import onnxruntime as ort

# 1️⃣  Paths -------------------------------------------------------------------
model_dir = Path("./t5_asr_correction_model_final")
onnx_dir = Path("./onnx_export")
onnx_dir.mkdir(parents=True, exist_ok=True)

# 2️⃣  Load model and tokenizer ------------------------------------------------
model_name = str(model_dir)
print(f"📦 Loading fine-tuned model from {model_name}")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3️⃣  Determine ONNX configuration -------------------------------------------
feature = "seq2seq-lm"
model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
    model, feature=feature
)
onnx_config = model_onnx_config(model.config)

# 4️⃣  Export ------------------------------------------------------------------
onnx_path = onnx_dir / "model.onnx"
print(f"🧠 Exporting to {onnx_path}")

# Dummy input
dummy_input = tokenizer("this is a test", return_tensors="pt")

export(
    preprocessor=tokenizer,  # ✅ this handles tokenization
    model=model,
    config=onnx_config,
    opset=17,
    output=onnx_path,
    device="cpu",
)

print("✅ Model exported successfully!")

# 5️⃣  Quick sanity check ------------------------------------------------------
print("🔍 Validating ONNX model...")

# Prepare dummy encoder + decoder inputs
dummy_inputs = tokenizer("this is a test", return_tensors="pt")
decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]])  # simple start
decoder_attention_mask = torch.ones_like(decoder_input_ids)

inputs = {
    "input_ids": dummy_inputs["input_ids"].cpu().numpy(),
    "attention_mask": dummy_inputs["attention_mask"].cpu().numpy(),
    "decoder_input_ids": decoder_input_ids.cpu().numpy(),
    "decoder_attention_mask": decoder_attention_mask.cpu().numpy(),
}

ort_session = ort.InferenceSession(str(onnx_path))
ort_outs = ort_session.run(None, inputs)
print(f"✅ ONNX model ran successfully. Output tensors: {len(ort_outs)}")
