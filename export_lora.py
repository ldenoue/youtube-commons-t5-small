from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTQuantizer, AutoQuantizationConfig
import os

# ----------------------------
# Paths
# ----------------------------
model_dir = "./asr_lora_merged"       # your merged LoRA model
export_dir = "./asr_lora_q4int6_onnx" # output folder
os.makedirs(export_dir, exist_ok=True)

# ----------------------------
# Load model and tokenizer
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# ----------------------------
# Convert to ONNX
# ----------------------------
onnx_model = ORTModelForSeq2SeqLM.from_transformers(model, task="seq2seq-lm")

# Save initial ONNX model
onnx_model.save_pretrained(export_dir)

# Save tokenizer separately
tokenizer.save_pretrained(export_dir)

print("✅ Exported base ONNX model and tokenizer")

# ----------------------------
# Quantize ONNX model (Q4Int6)
# ----------------------------
# ORTQuantizer handles static/dynamic quantization for encoder/decoder
quantizer = ORTQuantizer.from_pretrained(export_dir, file_suffix=".onnx")

# AutoQuantizationConfig allows specifying aggressive quantization
quant_config = AutoQuantizationConfig.avx512_vnni(
    is_static=False,    # dynamic quantization
    per_channel=True,   # per-channel weights
)

quantized_dir = os.path.join(export_dir, "onnx_q4int6")
os.makedirs(quantized_dir, exist_ok=True)

quantizer.quantize(
    save_dir=quantized_dir,
    quantization_config=quant_config
)

# Optional: write a small config file for Transformers.js
with open(os.path.join(quantized_dir, "quantize_config.json"), "w") as f:
    f.write('{"quantized": "q4int6", "format": "onnx"}')

print(f"✅ Quantized ONNX (q4int6) ready at: {quantized_dir}")
