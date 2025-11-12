from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")

# Base model (pretrained)
base_model_name = "t5-small"
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to(device)
base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Fine-tuned model
finetuned_path = "./t5_asr_correction_model_final"
finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(finetuned_path).to(device)
finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_path)

print("✅ Models loaded")

def correct_text(model, tokenizer, text, max_new_tokens=128):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

samples = [
    "i went to new york last week and met john smith",
    "the weather today is amazing i think we should go outside",
    "welcome to toyota research institute this is our latest robot",
    "uh so yeah i guess it was kind of fun but a little bit scary"
]

for s in samples:
    print("🟡 Input:", s)
    base_out = correct_text(base_model, base_tokenizer, s)
    tuned_out = correct_text(finetuned_model, finetuned_tokenizer, s)
    print("🔵 Base output:     ", base_out)
    print("🟢 Fine-tuned output:", tuned_out)
    print("-" * 80)

