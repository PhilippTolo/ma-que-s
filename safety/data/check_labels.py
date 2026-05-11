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
