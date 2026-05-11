"""
Phase 2 — SFT fine-tuning of Qwen3-1.7B on SafetyBench.

Uses TRL SFTTrainer + PEFT LoRA.  Only the assistant turns (rationale +
\\boxed{X}) contribute to the loss; system and user turns are masked.

Outputs a LoRA adapter in --output-dir/lora_adapter/.
Run merge_lora.py afterward to produce a vLLM-compatible merged checkpoint.

Usage:
    # Full run
    python sft_train.py \\
        --train  ../data/processed/safetybench_train.jsonl \\
        --val    ../data/processed/safetybench_val.jsonl \\
        --output-dir ./checkpoints/sft_v1

    # Smoke test (trains for --max-steps steps only)
    python sft_train.py --train ... --val ... --output-dir ./smoke --quick-test
"""

import argparse
import json
import os
import re

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from trl import SFTConfig, SFTTrainer

# ── Regex for \\boxed{} extraction ───────────────────────────────────────────
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}", re.IGNORECASE)

# ── Default LoRA targets for Qwen3 ────────────────────────────────────────────
QWEN3_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ─────────────────────────────────────────────────────────────────────────────
# Chat template helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_template(tokenizer, messages, **kwargs) -> str:
    """
    Call apply_chat_template with enable_thinking=False when supported.

    Qwen3 is a thinking model: with add_generation_prompt=True its template
    inserts '<think>\\n' tokens that are NOT present when the assistant turn is
    pre-filled (add_generation_prompt=False + full messages).  That extra token
    inflates the measured prompt length and causes TRL's auto-masking to mask
    assistant tokens as -100, giving loss = 0/0 = NaN.  Disabling thinking
    mode makes both tokenisations consistent.
    """
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def filter_by_length(
    examples: list[dict],
    tokenizer,
    max_seq_length: int,
) -> list[dict]:
    """Drop examples whose tokenized length exceeds max_seq_length."""
    kept, dropped = [], 0
    for ex in examples:
        text = apply_template(
            tokenizer, ex["messages"],
            tokenize=False, add_generation_prompt=False
        )
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if n_tokens <= max_seq_length:
            kept.append(ex)
        else:
            dropped += 1
    if dropped:
        print(f"  [filter] Dropped {dropped} examples exceeding {max_seq_length} tokens.")
    return kept


def preprocess_for_sft(
    examples: list[dict],
    tokenizer,
    max_seq_length: int,
) -> Dataset:
    """
    Tokenize and create labels with explicit masking.

    System + user tokens → label = -100  (masked, not included in loss)
    Assistant tokens     → label = token_id  (loss computed here)

    We do NOT rely on TRL's auto-masking because it derives prompt length from
    the prompt-only tokenization (add_generation_prompt=True).  For Qwen3 in
    thinking mode that injects '<think>\\n' tokens which are absent in the full
    conversation, causing the split point to be off and assistant tokens to be
    erroneously masked to -100 → loss = 0/0 = NaN → NaN gradients → corrupted
    weights.
    """
    result: dict[str, list] = {"input_ids": [], "attention_mask": [], "labels": []}
    skipped = 0

    for ex in examples:
        messages = ex["messages"]

        full_text   = apply_template(tokenizer, messages,
                                     tokenize=False, add_generation_prompt=False)
        prompt_text = apply_template(tokenizer, messages[:-1],
                                     tokenize=False, add_generation_prompt=True)

        full_ids   = tokenizer(full_text,   add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        n_prompt = len(prompt_ids)

        if n_prompt >= len(full_ids):
            # No response tokens after the prompt — skip degenerate example.
            skipped += 1
            continue

        labels = [-100] * n_prompt + full_ids[n_prompt:]

        # Truncate to max sequence length
        full_ids = full_ids[:max_seq_length]
        labels   = labels[:max_seq_length]

        # Guard: after truncation there must still be at least one valid label.
        if all(l == -100 for l in labels):
            skipped += 1
            continue

        result["input_ids"].append(full_ids)
        result["attention_mask"].append([1] * len(full_ids))
        result["labels"].append(labels)

    if skipped:
        print(f"  [preprocess] Skipped {skipped} examples (no response tokens or all masked).")

    print(f"  [preprocess] {len(result['input_ids'])} examples ready for training.")
    return Dataset.from_dict(result)


# ─────────────────────────────────────────────────────────────────────────────
# Format compliance callback
# ─────────────────────────────────────────────────────────────────────────────

class FormatComplianceCallback(TrainerCallback):
    """
    Every `eval_every` steps: sample `n_eval` val examples, run greedy
    inference, and log format_rate and accuracy.

    Runs only on the main process and only if not using DDP
    (to keep it simple and non-blocking).
    """

    def __init__(
        self,
        val_examples: list[dict],
        tokenizer,
        eval_every: int = 200,
        n_eval: int = 30,
    ):
        self.val_examples = val_examples[:n_eval]
        self.tokenizer = tokenizer
        self.eval_every = eval_every

    def on_step_end(
        self,
        args: SFTConfig,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return control
        if state.global_step == 0 or state.global_step % self.eval_every != 0:
            return control

        # Unwrap DDP/FSDP so .generate() and .device work on the raw model.
        raw_model = getattr(model, "module", model)
        device = next(raw_model.parameters()).device

        raw_model.eval()
        n_boxed, n_correct = 0, 0

        with torch.no_grad():
            for ex in self.val_examples:
                # Build prompt (system + user only, no assistant turn)
                chat = ex["messages"][:-1]
                prompt = apply_template(
                    self.tokenizer, chat,
                    tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=512
                ).to(device)

                out_ids = raw_model.generate(
                    **inputs,
                    max_new_tokens=512,   # enough for thinking tokens + boxed answer
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
                generated = self.tokenizer.decode(new_ids, skip_special_tokens=True)

                matches = BOXED_RE.findall(generated)
                if matches:
                    n_boxed += 1
                    if matches[-1].strip().upper() == ex.get("answer", "").upper():
                        n_correct += 1

        n = len(self.val_examples)
        fmt_rate = n_boxed / n
        acc = n_correct / n
        msg = (
            f"\n  [step {state.global_step:>6}]  "
            f"format_rate={fmt_rate:.1%}  accuracy={acc:.1%}  "
            f"(n={n})\n"
        )
        print(msg)

        # Log to WandB if active
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(
                    {"eval/format_rate": fmt_rate, "eval/accuracy": acc},
                    step=state.global_step,
                )
        except ImportError:
            pass

        raw_model.train()
        return control


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--train", required=True, help="Path to safetybench_train.jsonl")
    p.add_argument("--val",   required=True, help="Path to safetybench_val.jsonl")

    # Model
    p.add_argument("--model", default="Qwen/Qwen3-1.7B",
                   help="Base model HF ID or local path")

    # Output
    p.add_argument("--output-dir", default="./checkpoints/sft_v1")

    # LoRA
    p.add_argument("--lora-r",     type=int,   default=16)
    p.add_argument("--lora-alpha", type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--no-lora",    action="store_true",
                   help="Full fine-tuning (not recommended unless you have >40 GB VRAM)")

    # Training hyperparams
    p.add_argument("--epochs",         type=int,   default=3)
    p.add_argument("--batch-size",     type=int,   default=8,
                   help="Per-device train batch size")
    p.add_argument("--grad-accum",     type=int,   default=4)
    p.add_argument("--lr",             type=float, default=1e-4,
                   help="LoRA LR — ~10x full fine-tuning LR per 'LoRA Without Regret'")
    p.add_argument("--warmup-ratio",   type=float, default=0.05)
    p.add_argument("--max-seq-length", type=int,   default=1024)
    p.add_argument("--weight-decay",   type=float, default=0.01)

    # Logging / eval / save
    p.add_argument("--logging-steps",   type=int, default=10)
    p.add_argument("--eval-steps",      type=int, default=200)
    p.add_argument("--save-steps",      type=int, default=400)
    p.add_argument("--format-eval-n",   type=int, default=30,
                   help="Number of val examples for format compliance check")

    # Misc
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--report-to",    default="none",
                   choices=["none", "wandb", "tensorboard"])
    p.add_argument("--quick-test",   action="store_true",
                   help="Train for 20 steps only (smoke test)")
    p.add_argument("--max-steps",    type=int, default=-1,
                   help="Override max training steps (ignored unless quick-test or set explicitly)")

    return p.parse_args()


def main():
    args = parse_args()

    if args.quick_test:
        args.max_steps = 20
        args.eval_steps = 10
        args.save_steps = 20
        args.logging_steps = 5
        print("[quick-test] Running for 20 steps only.")

    # Full fine-tuning needs a much lower LR than LoRA
    if args.no_lora and args.lr == 1e-4:
        args.lr = 2e-5
        print(f"[no-lora] LR auto-adjusted to {args.lr} (LoRA default 1e-4 is too high for full FT)")

    print(f"\n{'='*60}")
    print(f"  SFT Training — Safety Model")
    print(f"  Base model : {args.model}")
    print(f"  Train data : {args.train}")
    print(f"  Val data   : {args.val}")
    print(f"  Output     : {args.output_dir}")
    print(f"  LoRA       : {'disabled' if args.no_lora else f'r={args.lora_r}, alpha={args.lora_alpha}'}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {tokenizer.vocab_size:,}  |  pad_token: {tokenizer.pad_token!r}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("[2/6] Loading and filtering data...")
    raw_train = load_jsonl(args.train)
    raw_val   = load_jsonl(args.val)

    # For quick test, shrink data
    if args.quick_test:
        raw_train = raw_train[:200]
        raw_val   = raw_val[:50]

    train_examples = filter_by_length(raw_train, tokenizer, args.max_seq_length)
    val_examples   = filter_by_length(raw_val,   tokenizer, args.max_seq_length)
    print(f"  Train: {len(train_examples):,}  |  Val: {len(val_examples):,}")

    # Pre-tokenize with explicit label masking (system+user → -100, assistant → loss).
    # This replaces TRL's auto-masking, which breaks for Qwen3: the thinking-mode
    # template injects <think>\n in the prompt-only tokenization but NOT in the
    # full-conversation tokenization, causing the split point to be off.
    print("  Pre-tokenising with explicit label masking...")
    train_ds = preprocess_for_sft(train_examples, tokenizer, args.max_seq_length)
    val_ds   = preprocess_for_sft(val_examples,   tokenizer, args.max_seq_length)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("[3/6] Loading model...")
    model_kwargs = dict(
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    # Use Flash Attention 2 if available (A100+)
    try:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        print("  Using Flash Attention 2.")
    except Exception:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        print("  Flash Attention 2 not available, using default attention.")

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ── LoRA ──────────────────────────────────────────────────────────────────
    peft_config = None
    if not args.no_lora:
        print("[4/6] Configuring LoRA...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=QWEN3_LORA_TARGETS,
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}, "
              f"targets={QWEN3_LORA_TARGETS}")
    else:
        print("[4/6] Skipping LoRA (full fine-tuning).")

    # ── SFT config ────────────────────────────────────────────────────────────
    print("[5/6] Building SFTConfig...")
    effective_batch = (
        args.batch_size * args.grad_accum
        * max(1, torch.cuda.device_count())
    )
    print(f"  Effective batch size: {effective_batch}")

    max_steps = args.max_steps if (args.quick_test or args.max_steps > 0) else -1

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        # Epochs / steps
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        # Batch / accumulation
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        # Optimizer
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        # Precision
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        # Memory
        gradient_checkpointing=True,
        # Logging / eval / save
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Dataset — data is already tokenized; tell TRL to skip its prepare step
        remove_unused_columns=True,
        dataset_kwargs={"skip_prepare_dataset": True},
        # Misc
        seed=args.seed,
        report_to=args.report_to,
        run_name="safety-sft",
    )

    # ── Collator ──────────────────────────────────────────────────────────────
    # DataCollatorForSeq2Seq pads input_ids (pad_token_id), attention_mask (0),
    # and labels (-100) to the longest sequence in each batch.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    print("[6/6] Initializing SFTTrainer and training...")

    format_callback = FormatComplianceCallback(
        val_examples=val_examples,
        tokenizer=tokenizer,
        eval_every=args.eval_steps,
        n_eval=args.format_eval_n,
    )

    try:
        # TRL >= 0.12 uses processing_class
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
            data_collator=data_collator,
            callbacks=[format_callback],
        )
    except TypeError:
        # Fallback for TRL 0.9.x — uses tokenizer=
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            peft_config=peft_config,
            data_collator=data_collator,
            callbacks=[format_callback],
        )

    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nLoRA adapter saved to: {adapter_dir}")
    print(f"Next step: python merge_lora.py --adapter {adapter_dir} --base {args.model}")


if __name__ == "__main__":
    main()
