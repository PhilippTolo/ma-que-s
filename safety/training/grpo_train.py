"""
Phase 3 — GRPO fine-tuning on PKU-SafeRLHF.

Starts from the merged SFT checkpoint and further trains with Group Relative
Policy Optimisation using an additive verifiable reward:

    r = 0.5 * r_format + 0.5 * r_accuracy

    r_format   = 1  if completion contains \\boxed{A} or \\boxed{B},  else 0
    r_accuracy = 1  if the boxed answer matches gold_answer,           else 0

Additive (not multiplicative) so format signal flows even when the model is
not yet accurate — fully multiplicative rewards collapse to all-zero batches
early in training (std=0 → NaN GRPO advantage).

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
import sys

import torch
from datasets import Dataset

# bitsandbytes is not compiled for CUDA 12.8 and raises RuntimeError on import.
# PEFT + TRL import it unconditionally, but standard LoRA never calls any bnb
# functions — so mocking it is safe for our use case.
# IMPORTANT: must use types.ModuleType with a real ModuleSpec, NOT MagicMock.
# MagicMock().__spec__ raises AttributeError; importlib.util.find_spec (called
# by TRL's lazy importer) converts that to ValueError("__spec__ is not set").
if "bitsandbytes" not in sys.modules:
    try:
        import bitsandbytes  # noqa: F401
    except (RuntimeError, ImportError):
        import types as _types
        import importlib.machinery as _im
        from unittest.mock import MagicMock as _M

        def _bnb_mod(name: str):
            m = _types.ModuleType(name)
            m.__spec__ = _im.ModuleSpec(name, loader=None)
            m.__version__ = "0.41.0"
            m.__getattr__ = lambda attr: _M()
            return m

        for _mod_name in (
            "bitsandbytes", "bitsandbytes.nn", "bitsandbytes.optim",
            "bitsandbytes.functional", "bitsandbytes.cuda_setup",
            "bitsandbytes.autograd", "bitsandbytes.nn.modules",
            "bitsandbytes.utils",
        ):
            sys.modules[_mod_name] = _bnb_mod(_mod_name)
        print("[bnb-mock] bitsandbytes unavailable on CUDA 12.8; mocked for PEFT LoRA.")

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from safety.utils import extract_boxed

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
    r = 0.5 * r_format + 0.5 * r_accuracy  (additive).

    Additive rather than multiplicative so format signal flows even when the
    model is not yet accurate — multiplicative rewards collapse to all-zero
    batches at training start (std=0 → NaN GRPO advantage).

    r_format   = 1.0 if \\boxed{A} or \\boxed{B} present, else 0.0
    r_accuracy = 1.0 if boxed answer matches gold,         else 0.0

    Called by GRPOTrainer with:
        completions : list of generated texts, length = batch_size * num_generations
        kwargs      : dataset columns forwarded verbatim; we use "gold_answer"
    """
    if "gold_answer" not in kwargs:
        raise RuntimeError(
            "safety_reward: 'gold_answer' not in kwargs.\n"
            "Most likely cause: remove_unused_columns=True stripped it from the dataset.\n"
            "Fix: ensure GRPOConfig is built with remove_unused_columns=False."
        )

    gold_answers = kwargs["gold_answer"]
    rewards = []

    for completion, gold in zip(completions, gold_answers):
        # TRL >= 1.0 passes completions as lists of message dicts;
        # older TRL passes plain strings. Handle both.
        if isinstance(completion, list):
            text = "".join(
                msg.get("content", "") if isinstance(msg, dict) else str(msg)
                for msg in completion
            )
        else:
            text = completion

        extracted = extract_boxed(text)

        r_format   = 1.0 if extracted in ("A", "B") else 0.0
        r_accuracy = 1.0 if (extracted is not None and extracted == str(gold).upper()) else 0.0
        rewards.append(0.5 * r_format + 0.5 * r_accuracy)

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
        max_grad_norm=0.3,
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
    p.add_argument("--no-lora", action="store_true",
                   help="Skip LoRA and do full fine-tuning (required on CUDA 12.8 — bitsandbytes incompatible)")

    # GRPO-specific
    p.add_argument("--num-generations",    type=int,   default=4,
                   help="Completions per prompt for group advantage (G in the paper)")
    p.add_argument("--beta",               type=float, default=0.0,
                   help="KL penalty coefficient — 0 disables KL entirely (default); "
                        "if >0 an explicit frozen ref_model is loaded from --sft-checkpoint")
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
    p.add_argument("--lr",          type=float, default=2e-4,
                   help="LoRA GRPO: ~2e-4; full-FT GRPO: ~5e-6 (pass --no-lora to use full FT)")
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
    print(f"  Reward             : 0.5*r_format + 0.5*r_accuracy")
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

    # Patch chat_template to force enable_thinking=False.
    # Without this, Qwen3's template injects <think>\n before the assistant turn
    # during GRPOTrainer's internal apply_chat_template calls, causing the model to
    # generate a full 256-token thinking chain instead of \boxed{A}/\boxed{B} →
    # reward=0 for every completion → reward_std=0 → GRPO advantage=0 → no learning.
    _THINKING_OVERRIDE = "{%- set enable_thinking = false %}"
    if tokenizer.chat_template and _THINKING_OVERRIDE not in tokenizer.chat_template:
        tokenizer.chat_template = _THINKING_OVERRIDE + "\n" + tokenizer.chat_template
        print("  Patched chat_template: enable_thinking=false")

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
    # Force eager attention — FA2 produces NaN logits with Qwen3-1.7B in BF16,
    # causing the generation probability tensor to contain inf/nan (same issue as SFT).
    model_kwargs = dict(trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="eager")
    model = AutoModelForCausalLM.from_pretrained(args.sft_checkpoint, **model_kwargs)
    print("  Using BF16 + eager attention (NaN-safe).")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Guard every transformer layer against NaN cascade.
    # Key insight: torch.clamp(NaN) == NaN — clamping does NOT replace NaN.
    # NaN can originate in any attention/MLP layer and propagate forward through
    # the residual stream into lm_head, making logit-only clamping useless.
    # Registering nan_to_num+clamp on every decoder layer output stops the cascade
    # at its source.  Value bounds (-3e4, 3e4) are well within BF16 range but far
    # above typical healthy activations (<100), so normal training is unaffected.
    def _clamp_hidden(module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            if isinstance(h, torch.Tensor):
                h = h.nan_to_num(0.0).clamp(-3e4, 3e4)
            return (h,) + out[1:]
        if isinstance(out, torch.Tensor):
            return out.nan_to_num(0.0).clamp(-3e4, 3e4)
        return out

    n_hooks = 0
    for layer in model.model.layers:
        layer.register_forward_hook(_clamp_hidden)
        n_hooks += 1
    print(f"  Registered NaN-guard on {n_hooks} transformer layers.")

    # Also guard lm_head: nan_to_num first (clamp alone passes NaN through),
    # then clamp logits to ±100 so softmax sampling always gets finite input.
    def _clamp_logits(module, inp, out):
        return out.nan_to_num(0.0).clamp(-100.0, 100.0)
    model.lm_head.register_forward_hook(_clamp_logits)
    print("  Registered NaN-guard + logit-clamp on lm_head.")

    # ── 4. LoRA ───────────────────────────────────────────────────────────────
    if args.no_lora:
        print("[4/5] Skipping LoRA (full fine-tuning — CUDA 12.8 bitsandbytes workaround).")
        model.enable_input_require_grads()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  trainable params: {trainable:,} || all params: {total:,} || trainable%: {100*trainable/total:.4f}")
    else:
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

    # ── 4b. Reference model ───────────────────────────────────────────────────
    # With beta=0 (default) KL is disabled — no ref_model needed and none loaded.
    # With beta>0 we load an explicit frozen copy from the SFT checkpoint rather
    # than relying on TRL's PeftModel auto-detection (which varies across TRL
    # versions and could silently compute reference log-probs from the wrong
    # distribution).
    ref_model = None
    if args.beta > 0:
        print(f"[4b/5] Loading frozen reference model (beta={args.beta})...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.sft_checkpoint,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        print("  Reference model loaded and frozen.")
    else:
        print("[4b/5] beta=0 — KL penalty disabled, no reference model loaded.")

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print("[5/5] Building GRPOConfig and training...")
    effective_batch = args.batch_size * args.grad_accum
    print(f"  Effective batch size : {effective_batch} prompts")
    print(f"  Rollouts per step    : {effective_batch * args.num_generations} completions")

    grpo_config = build_grpo_config(args, max_steps)

    # Try processing_class= (TRL >= 0.12); fall back to tokenizer= for older TRL.
    # ref_model is only passed when explicitly loaded (beta>0); TRL 1.3.0 removed
    # the ref_model parameter from GRPOTrainer — passing None causes TypeError.
    # eval_dataset is passed where supported; older GRPOTrainer versions ignore it.
    def _make_trainer(processing_class_kwarg, eval_kwarg, ref_kwarg):
        return GRPOTrainer(
            model=model,
            reward_funcs=[safety_reward],
            args=grpo_config,
            train_dataset=train_ds,
            **ref_kwarg,
            **processing_class_kwarg,
            **eval_kwarg,
        )

    ref_kwargs_options = (
        ({"ref_model": ref_model},) if ref_model is not None else ({},)
    )
    trainer = None
    for ref_kw in ref_kwargs_options:
        for pc_kwargs in ({"processing_class": tokenizer}, {"tokenizer": tokenizer}):
            for ev_kwargs in ({"eval_dataset": val_ds}, {}):
                try:
                    trainer = _make_trainer(pc_kwargs, ev_kwargs, ref_kw)
                    break
                except TypeError:
                    continue
            else:
                continue
            break
        if trainer is not None:
            break

    if trainer is None:
        raise RuntimeError(
            "GRPOTrainer construction failed for all argument combinations. "
            "Check TRL version compatibility."
        )

    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    save_dir = os.path.join(args.output_dir, "lora_adapter")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    if args.no_lora:
        print(f"\nFull model saved to: {save_dir}")
        print(
            f"Next step (no merge needed — full model):\n"
            f"  python ../../shared/push_to_hub.py \\\n"
            f"      --checkpoint {save_dir} \\\n"
            f"      --repo-id <your-hf-user>/cs552-safety-individual"
        )
    else:
        print(f"\nLoRA adapter saved to: {save_dir}")
        print(
            f"Next step: python merge_lora.py"
            f" --adapter {save_dir}"
            f" --base {args.sft_checkpoint}"
            f" --output {os.path.join(args.output_dir, 'merged')}"
        )


if __name__ == "__main__":
    main()
