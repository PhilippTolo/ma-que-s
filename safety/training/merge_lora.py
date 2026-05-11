"""
Merge a LoRA adapter into the base model weights.

vLLM is most reliable with a fully merged checkpoint (no adapter separation).
Run this after sft_train.py (or grpo_train.py) to produce the final model.

Also writes the required generation_config.json so the checkpoint is
ready for push_to_hub.py.

Usage:
    python merge_lora.py \\
        --adapter ./checkpoints/sft_v1/lora_adapter \\
        --base    Qwen/Qwen3-1.7B \\
        --output  ./checkpoints/sft_v1/merged \\
        --temperature 0.6 --top-k 20 --top-p 0.9

    # Then push:
    python ../../shared/push_to_hub.py \\
        --checkpoint ./checkpoints/sft_v1/merged \\
        --repo-id <your-hf-user>/cs552-safety-individual
"""

import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from safety.constants import REQUIRED_GEN_CONFIG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help="Path to saved LoRA adapter")
    p.add_argument("--base",    required=True, help="Base model HF ID or path")
    p.add_argument("--output",  required=True, help="Directory for merged model")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-k",       type=int,   default=20)
    p.add_argument("--top-p",       type=float, default=0.9)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Merging LoRA adapter into base model")
    print(f"  Base    : {args.base}")
    print(f"  Adapter : {args.adapter}")
    print(f"  Output  : {args.output}")
    print(f"{'='*60}\n")

    # ── Load base + adapter ───────────────────────────────────────────────────
    print("[1/4] Loading base model...")
    # Use GPU when available (A100 40 GB handles 1.7B trivially); fall back to
    # CPU only if no CUDA device is found (e.g. local laptop runs).
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device_map,
    )
    print(f"  device_map={device_map!r}")

    print("[2/4] Loading LoRA adapter and merging...")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter)
    merged = peft_model.merge_and_unload()
    print(f"  Merged parameters: {sum(p.numel() for p in merged.parameters()) / 1e6:.1f}M")

    # ── Save merged model ─────────────────────────────────────────────────────
    print("[3/4] Saving merged model...")
    merged.save_pretrained(args.output, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)

    # ── Patch chat_template to force non-thinking mode ────────────────────────
    # The CI never passes enable_thinking as a kwarg, so the template must
    # bake in the default.  Prepending this line overrides whatever the base
    # Qwen3 template sets (which defaults to enable_thinking=True).
    print("[3b/4] Patching chat_template to force enable_thinking=false...")
    _THINKING_OVERRIDE = "{%- set enable_thinking = false %}"

    # Patch tokenizer_config.json (embedded template)
    tok_cfg_path = os.path.join(args.output, "tokenizer_config.json")
    if os.path.exists(tok_cfg_path):
        with open(tok_cfg_path, encoding="utf-8") as _f:
            tok_cfg = json.load(_f)
        if "chat_template" in tok_cfg and _THINKING_OVERRIDE not in tok_cfg["chat_template"]:
            tok_cfg["chat_template"] = _THINKING_OVERRIDE + "\n" + tok_cfg["chat_template"]
            with open(tok_cfg_path, "w", encoding="utf-8") as _f:
                json.dump(tok_cfg, _f, ensure_ascii=False, indent=2)
            print("  Patched tokenizer_config.json chat_template.")
        else:
            print("  tokenizer_config.json: patch already present or no chat_template field.")

    # Also patch a standalone chat_template.jinja if it exists
    jinja_path = os.path.join(args.output, "chat_template.jinja")
    if os.path.exists(jinja_path):
        with open(jinja_path, encoding="utf-8") as _f:
            jinja_src = _f.read()
        if _THINKING_OVERRIDE not in jinja_src:
            with open(jinja_path, "w", encoding="utf-8") as _f:
                _f.write(_THINKING_OVERRIDE + "\n" + jinja_src)
            print("  Patched chat_template.jinja.")

    # ── Write generation_config.json ──────────────────────────────────────────
    print("[4/4] Writing generation_config.json...")
    gen_cfg = GenerationConfig(
        **REQUIRED_GEN_CONFIG,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    gen_cfg.save_pretrained(args.output)

    # Verify it's correct
    cfg_path = os.path.join(args.output, "generation_config.json")
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    print("\n  generation_config.json:")
    print(json.dumps(cfg_dict, indent=4))

    # Sanity check required fields
    missing = [k for k in REQUIRED_GEN_CONFIG if k not in cfg_dict]
    if missing:
        print(f"\n  WARNING: missing required fields: {missing}")
    else:
        print("\n  All required fields present.")

    print(f"\nDone. Merged model at: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Test  : python ../../shared/test_vllm_base.py --model {args.output}")
    print(f"  2. Push  : python ../../shared/push_to_hub.py \\")
    print(f"                 --checkpoint {args.output} \\")
    print(f"                 --repo-id <your-hf-user>/cs552-safety-individual")


if __name__ == "__main__":
    main()
