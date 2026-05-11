"""Quick diagnostic: verify that preprocess_for_sft produces correct label masking."""
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)


def apply_template(tok, messages, **kw):
    try:
        return tok.apply_chat_template(messages, enable_thinking=False, **kw)
    except TypeError:
        return tok.apply_chat_template(messages, **kw)


with open("/scratch/safety_data/pkusafe_sft_train.jsonl") as f:
    ex = json.loads(f.readline())

messages = ex["messages"]
full_text   = apply_template(tok, messages,      tokenize=False, add_generation_prompt=False)
prompt_text = apply_template(tok, messages[:-1], tokenize=False, add_generation_prompt=True)

full_ids   = tok(full_text,   add_special_tokens=False)["input_ids"]
prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
n_prompt   = len(prompt_ids)
labels     = [-100] * n_prompt + full_ids[n_prompt:]

print("Full tokens            :", len(full_ids))
print("Prompt tokens (masked) :", n_prompt)
print("Response tokens (loss) :", sum(1 for l in labels if l != -100))
print("Response text          :", repr(tok.decode(full_ids[n_prompt:])))
print()
if n_prompt >= len(full_ids):
    print("ERROR: prompt length >= full length — all tokens masked, loss will be NaN")
elif sum(1 for l in labels if l != -100) == 0:
    print("ERROR: zero non-masked tokens — label masking is broken")
else:
    print("OK: label masking looks correct")

# ── Direct model loss test ────────────────────────────────────────────────────
print("\n--- Direct model loss test ---")
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B", trust_remote_code=True,
    torch_dtype=torch.bfloat16, attn_implementation="eager"
).cuda()
print("Model weight dtype:", next(model.parameters()).dtype)
print("Model config vocab_size:", model.config.vocab_size)
print("Max label token ID:", max(l for l in labels if l != -100))

input_ids_t = torch.tensor([full_ids]).cuda()
labels_t    = torch.tensor([labels]).cuda()

with torch.no_grad():
    out = model(input_ids=input_ids_t, labels=labels_t)

print("Model loss :", out.loss.item())
print("Is NaN     :", torch.isnan(out.loss).item())
print("Is Inf     :", torch.isinf(out.loss).item())

# Check logit range — overflow here explains impossible loss values
logits = out.logits  # [1, seq_len, vocab_size]
print(f"Logits dtype : {logits.dtype}")
print(f"Logits max   : {logits.max().item():.2f}")
print(f"Logits min   : {logits.min().item():.2f}")
print(f"Logits NaN   : {torch.isnan(logits).any().item()}")
print(f"Logits Inf   : {torch.isinf(logits).any().item()}")

# Manual per-token cross-entropy to see exact values
import torch.nn.functional as F
logits_2d  = logits[0].float()           # [seq_len, vocab] in fp32
labels_1d  = labels_t[0]                 # [seq_len]
per_tok    = F.cross_entropy(logits_2d, labels_1d, ignore_index=-100, reduction="none")
valid      = labels_1d != -100
print(f"\nPer-token losses on response tokens: {per_tok[valid].tolist()}")
print(f"Mean manual CE loss: {per_tok[valid].mean().item():.4f}")

# ── FP32 comparison ───────────────────────────────────────────────────────────
print("\n--- FP32 model comparison ---")
model_fp32 = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B", trust_remote_code=True,
    attn_implementation="eager"
).cuda()
with torch.no_grad():
    out32 = model_fp32(input_ids=input_ids_t, labels=labels_t)
print(f"FP32 loss        : {out32.loss.item():.4f}")
print(f"FP32 logits max  : {out32.logits.max().item():.2f}")
per_tok32 = F.cross_entropy(out32.logits[0], labels_t[0], ignore_index=-100, reduction="none")
print(f"FP32 per-token   : {per_tok32[valid].tolist()}")
