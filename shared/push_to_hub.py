"""
Push a trained checkpoint to HuggingFace Hub with the required generation_config.

Usage:
    python push_to_hub.py \
        --checkpoint ./safety_sft_checkpoint \
        --repo-id <your-hf-username>/cs552-safety-individual \
        --temperature 0.6 --top-k 20 --top-p 0.9

The script:
  1. Loads the model and tokenizer from --checkpoint
  2. Writes / overwrites generation_config.json with the required fields
  3. Pushes everything to HuggingFace Hub as a public repo
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from safety.constants import REQUIRED_GEN_CONFIG as REQUIRED_FIELDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Local path to checkpoint")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo, e.g. user/model-name")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--private", action="store_true", help="Make repo private (default: public)")
    args = parser.parse_args()

    print(f"Loading checkpoint from: {args.checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)

    # Build generation config with all required fields
    gen_config = GenerationConfig(
        **REQUIRED_FIELDS,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    model.generation_config = gen_config

    # Verify the config matches spec
    cfg_path = os.path.join(args.checkpoint, "generation_config.json")
    with open(cfg_path, "w") as f:
        json.dump(gen_config.to_dict(), f, indent=4)
    print(f"generation_config.json written to {cfg_path}")
    print(json.dumps(gen_config.to_dict(), indent=2))

    # Push
    print(f"\nPushing to: {args.repo_id}  (private={args.private})")
    model.push_to_hub(args.repo_id, private=args.private)
    tokenizer.push_to_hub(args.repo_id, private=args.private)
    gen_config.push_to_hub(args.repo_id, private=args.private)

    print(f"\nDone. Model available at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
