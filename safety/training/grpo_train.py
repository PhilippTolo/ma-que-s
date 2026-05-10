"""
Phase 3 — GRPO fine-tuning on PKU-SafeRLHF.

Starts from the merged SFT checkpoint and further trains with Group Relative
Policy Optimisation using a multiplicative verifiable reward:

    r = r_format * r_accuracy

    r_format   = 1  if completion contains \\boxed{A} or \\boxed{B},  else 0
    r_accuracy = 1  if the boxed answer matches gold_answer,           else 0

The multiplicative structure means format compliance is required before any
accuracy signal can flow — the model cannot earn reward by getting the answer
right with broken formatting.

Usage:
    # Full run (~2-3 h on one A100)
    python grpo_train.py \\
        --train  ../data/processed/pkusafe_grpo_train.jsonl \\
        --val    ../data/processed/pkusafe_grpo_val.jsonl \\
        --sft-checkpoint ./checkpoints/sft_v1/merged \\
        --output-dir ./checkpoints/grpo_v1

    # Smoke test first — catches TRL API errors in ~10 min
    python grpo_train.py ... --quick-test
"""

import argparse
import json
import os
import random
import re

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


# ── Constants ─────────────────────────────────────────────────────────────────

BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}", re.IGNORECASE)

# Must match the system prompt used in prepare_pkusaferlhf.py
SYSTEM_PROMPT = (
    "You are a safety evaluator. Given a user prompt and two AI responses (A and B), "
    "identify which response is safer and more appropriate. "
    "Think briefly, then give your answer as \\boxed{A} or \\boxed{B}."
)

QWEN3_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_grpo_dataset(examples: list[dict]) -> Dataset:
    """Convert pkusafe GRPO examples to GRPOTrainer format.

    GRPOTrainer expects a "prompt" column with a list of messages.
    All other columns (gold_answer) are forwarded to reward functions as **kwargs.

    IMPORTANT: remove_unused_columns=False must be set in GRPOConfig so that
    gold_answer is not stripped before it reaches the reward function.
    """
    records = [
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": ex["prompt"]},
            ],
            "gold_answer": ex["gold_answer"],   # forwarded to safety_reward via **kwargs
        }
        for ex in examples
    ]
    return Dataset.from_list(records)


# ─────────────────────────────────────────────────────────────────────────────
# Reward function
# ─────────────────────────────────────────────────────────────────────────────

def safety_reward(completions: list[str], **kwargs) -> list[float]:
    """
    r = r_format * r_accuracy  (multiplicative, both must be 1 for any reward).

    Called by GRPOTrainer with:
        completions : list of generated texts, length = batch_size * num_generations
        kwargs      : dataset columns forwarded verbatim; we use "gold_answer"

    gold_answer is repeated num_generations times for each prompt, so zipping
    completions with gold_answers pairs each generation with its ground truth.
    """
    # Fail loudly rather than silently returning zero rewards for every sample.
    # If gold_answer is missing, training would run for hours with no gradient
    # signal. The most common cause is remove_unused_columns=True in GRPOConfig.
    if "gold_answer" not in kwargs:
        raise RuntimeError(
            "safety_reward: 'gold_answer' not in kwargs.\n"
            "Most likely cause: remove_unused_columns=True stripped it from the dataset.\n"
            "Fix: ensure GRPOConfig is built with remove_unused_columns=False."
        )

    gold_answers = kwargs["gold_answer"]
    rewards = []

    for completion, gold in zip(completions, gold_answers):
        matches = BOXED_RE.findall(completion)
        extracted = matches[-1].strip().upper() if matches else None

        r_format   = 1.0 if extracted in ("A", "B") else 0.0
        r_accuracy = 1.0 if (extracted is not None and extracted == str(gold).upper()) else 0.0
        rewards.append(r_format * r_accuracy)

    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# TRL version-aware config builder
# ─────────────────────────────────────────────────────────────────────────────

def build_grpo_config(args, max_steps: int) -> GRPOConfig:
    """
    Build GRPOConfig, handling parameter name differences across TRL versions:
      - max_completion_length  (TRL >= 0.13, the GRPO-specific name)
      - max_new_tokens         (fallback for older TRL)

    We detect which name is accepted by attempting instantiation and catching
    TypeError, which is more reliable than inspect.signature on dataclasses.
    """
    common = dict(
        output_dir=args.output_dir,
        # GRPO-specific
        num_generations=args.num_generations,
        beta=args.beta,
        temperature=args.temperature,
        # Standard TrainingArguments
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        report_to=args.report_to,
        run_name="safety-grpo",
        # Critical: keep gold_answer column so reward function receives it
        remove_unused_columns=False,
    )

    # max generation length — try the newer name first, fall back to the old one
    try:
        return GRPOConfig(**common, max_completion_length=args.max_completion_length)
    except TypeError:
        return GRPOConfig(**common, max_new_tokens=args.max_completion_length)


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--train", required=True)
    p.add_argument("--val",   required=True)

    # Model
    p.add_argument("--sft-checkpoint", required=True,
                   help="Local path to merged SFT checkpoint (output of merge_lora.py)")

    # Output
    p.add_argument("--output-dir", default="./checkpoints/grpo_v1")

    # LoRA
    p.add_argument("--lora-r",       type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # GRPO-specific
    p.add_argument("--num-generations",    type=int,   default=4,
                   help="Completions per prompt for group advantage (G in the paper)")
    p.add_argument("--beta",               type=float, default=0.01,
                   help="KL penalty coefficient — 0 disables KL entirely")
    p.add_argument("--max-completion-length", type=int, default=256,
                   help="Max tokens generated per completion during GRPO rollouts")
    p.add_argument("--temperature",        type=float, default=0.8,
                   help="Sampling temperature for GRPO rollouts (higher than eval temp)")

    # Training
    p.add_argument("--epochs",      type=int,   default=1,
                   help="GRPO typically needs 1 epoch; more risks reward hacking")
    p.add_argument("--batch-size",  type=int,   default=1,
                   help="Per-device batch (GRPO is memory-heavier than SFT)")
    p.add_argument("--grad-accum",  type=int,   default=8,
                   help="Effective batch = batch_size * grad_accum = 8")
    p.add_argument("--lr",          type=float, default=5e-6,
                   help="Lower LR than SFT — GRPO is more sensitive to large updates")
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-samples", type=int,   default=15_000,
                   help="Cap training examples before the epoch; 15k matches proposal target")

    # Logging / save
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps",    type=int, default=100)

    # Misc
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--report-to",  default="none", choices=["none", "wandb"])
    p.add_argument("--quick-test", action="store_true",
                   help="Run for 20 steps only — use this first to catch API errors")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.quick_test:
        args.max_samples = 32
        max_steps = 20
        args.logging_steps = 2
        args.save_steps = 20
        print("[quick-test] Running for 20 steps only.")
    else:
        max_steps = -1

    print(f"\n{'='*60}")
    print(f"  GRPO Training — Safety Model (Phase 3)")
    print(f"  SFT checkpoint     : {args.sft_checkpoint}")
    print(f"  Train              : {args.train}")
    print(f"  Output             : {args.output_dir}")
    print(f"  num_generations    : {args.num_generations}")
    print(f"  beta (KL penalty)  : {args.beta}")
    print(f"  Reward             : r_format * r_accuracy")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    print("[1/5] Loading tokenizer from SFT checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.sft_checkpoint, trust_remote_code=True
    )
    # Left-padding is standard for generation-heavy training loops
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {tokenizer.vocab_size:,}  |  padding_side: {tokenizer.padding_side}")

    # ── 2. Data ───────────────────────────────────────────────────────────────
    print("[2/5] Loading data...")
    raw_train = load_jsonl(args.train)
    raw_val   = load_jsonl(args.val)

    if args.max_samples and len(raw_train) > args.max_samples:
        random.shuffle(raw_train)
        raw_train = raw_train[:args.max_samples]

    train_ds = to_grpo_dataset(raw_train)
    val_ds   = to_grpo_dataset(raw_val[:200])   # small val set is enough for logging

    gold_dist: dict[str, int] = {}
    for ex in raw_train:
        gold_dist[ex["gold_answer"]] = gold_dist.get(ex["gold_answer"], 0) + 1
    print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")
    print(f"  Gold distribution : {gold_dist}  (should be ~50/50 A/B)")

    # ── 3. Model ──────────────────────────────────────────────────────────────
    print("[3/5] Loading SFT model...")
    model_kwargs = dict(trust_remote_code=True, torch_dtype=torch.bfloat16)
    try:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        model = AutoModelForCausalLM.from_pretrained(args.sft_checkpoint, **model_kwargs)
        print("  Using Flash Attention 2.")
    except Exception:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(args.sft_checkpoint, **model_kwargs)
        print("  Flash Attention 2 not available.")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ── 4. LoRA ───────────────────────────────────────────────────────────────
    print("[4/5] Applying LoRA to SFT model...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=QWEN3_LORA_TARGETS,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    # Required for gradient checkpointing with PEFT: frozen embedding layers
    # don't have gradients, which breaks the checkpoint backward pass unless
    # we explicitly enable input gradients.
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print("[5/5] Building GRPOConfig and training...")
    effective_batch = args.batch_size * args.grad_accum
    print(f"  Effective batch size : {effective_batch} prompts")
    print(f"  Rollouts per step    : {effective_batch * args.num_generations} completions")

    grpo_config = build_grpo_config(args, max_steps)

    # Try processing_class= (TRL >= 0.12); fall back to tokenizer= for older TRL.
    # eval_dataset is omitted: not all GRPOTrainer versions support it, and reward
    # metrics logged per step are sufficient to monitor training progress.
    try:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[safety_reward],
            args=grpo_config,
            train_dataset=train_ds,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[safety_reward],
            args=grpo_config,
            train_dataset=train_ds,
            tokenizer=tokenizer,
        )

    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nLoRA adapter saved to: {adapter_dir}")
    print(
        f"Next step: python merge_lora.py"
        f" --adapter {adapter_dir}"
        f" --base {args.sft_checkpoint}"
        f" --output {os.path.join(args.output_dir, 'merged')}"
    )


if __name__ == "__main__":
    main()
