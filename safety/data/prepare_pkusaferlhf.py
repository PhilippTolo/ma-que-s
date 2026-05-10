"""
Phase 3 — Prepare PKU-SafeRLHF for GRPO training.

PKU-SafeRLHF (Dai et al., 2023): preference pairs with safety labels.
HuggingFace: PKU-Alignment/PKU-SafeRLHF

Strategy from proposal: reformat preference pairs as binary safe/unsafe MCQs.
For each pair we know which response is safer (from is_response_X_safe labels).
The GRPO reward function then checks if the model's \\boxed{} answer matches.

Output schema (one JSON object per line):
    {
        "prompt": "<chat-formatted prompt asking 'which is safer?'>",
        "gold_answer": "A",     # letter of the safer response
        "response_a": "...",    # truncated for token budget
        "response_b": "...",
        "original_prompt": "..."
    }

The GRPO trainer calls a reward_fn(completions, gold_answers) that returns
per-sample scalar rewards — see grpo_train.py.

Usage:
    python prepare_pkusaferlhf.py --output-dir ./processed
    python prepare_pkusaferlhf.py --output-dir ./processed --max-resp-tokens 200
"""

import argparse
import json
import os
import random
import textwrap
from collections import Counter

from datasets import load_dataset

DATASET_NAME = "PKU-Alignment/PKU-SafeRLHF"

SYSTEM_PROMPT = (
    "You are a safety evaluator. Given a user prompt and two AI responses (A and B), "
    "identify which response is safer and more appropriate. "
    "Think briefly, then give your answer as \\boxed{A} or \\boxed{B}."
)

# Responses in SafeRLHF can be very long — truncate to keep token budget manageable.
MAX_RESP_CHARS = 800


def truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def build_mcq_prompt(original_prompt: str, resp_a: str, resp_b: str) -> str:
    user_msg = textwrap.dedent(f"""\
        User prompt: {original_prompt.strip()}

        Response A:
        {resp_a}

        Response B:
        {resp_b}

        Which response (A or B) is safer and more appropriate? \
Enclose your answer in \\boxed{{}}. Example: \\boxed{{A}}""")
    return user_msg


def process_dataset(rows: list[dict], max_resp_chars: int) -> list[dict]:
    """
    Filter for pairs where safety is unambiguous (one safe, one unsafe),
    then build MCQ prompts.
    """
    examples = []
    skipped_ambiguous = 0
    skipped_missing = 0

    for row in rows:
        # ── Extract fields ────────────────────────────────────────────────────
        prompt_text = row.get("prompt") or row.get("instruction") or ""
        r0 = row.get("response_0") or row.get("chosen") or ""
        r1 = row.get("response_1") or row.get("rejected") or ""

        if not (prompt_text and r0 and r1):
            skipped_missing += 1
            continue

        # ── Determine which response is safer ─────────────────────────────────
        # PKU-SafeRLHF labels: is_response_0_safe (bool), is_response_1_safe (bool)
        safe0 = row.get("is_response_0_safe")
        safe1 = row.get("is_response_1_safe")

        if safe0 is None or safe1 is None:
            # Fallback: use better_response_id if available
            better = row.get("better_response_id")
            if better is not None:
                # 'better' means preferred, not necessarily 'safe', but use it
                safe0 = (better == 0)
                safe1 = (better == 1)
            else:
                skipped_ambiguous += 1
                continue

        # Only use pairs where exactly one response is safe (clear signal)
        if bool(safe0) == bool(safe1):
            skipped_ambiguous += 1
            continue

        # Randomly assign safe/unsafe to A/B to avoid positional bias
        if random.random() < 0.5:
            resp_a = truncate(str(r0), max_resp_chars)
            resp_b = truncate(str(r1), max_resp_chars)
            gold = "A" if safe0 else "B"
        else:
            resp_a = truncate(str(r1), max_resp_chars)
            resp_b = truncate(str(r0), max_resp_chars)
            gold = "A" if safe1 else "B"

        user_msg = build_mcq_prompt(str(prompt_text), resp_a, resp_b)

        examples.append(
            {
                "prompt": user_msg,
                "gold_answer": gold,
                "response_a": resp_a,
                "response_b": resp_b,
                "original_prompt": str(prompt_text),
            }
        )

    print(f"  Skipped — ambiguous safety labels : {skipped_ambiguous:,}")
    print(f"  Skipped — missing required fields : {skipped_missing:,}")
    return examples


def save_jsonl(data: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data):,} examples → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--output-dir", default="./processed")
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-resp-chars", type=int, default=MAX_RESP_CHARS,
                        help="Truncate response text to this many characters")
    parser.add_argument("--max-samples", type=int, default=20_000,
                        help="Cap total pairs (PKU-SafeRLHF is large; 20k is plenty)")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  PKU-SafeRLHF data preparation (GRPO)")
    print(f"  Dataset     : {args.dataset_name}")
    print(f"  Max samples : {args.max_samples:,}")
    print(f"  Max resp chars: {args.max_resp_chars}")
    print(f"{'='*60}\n")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("[1/4] Downloading dataset (this may take a while)...")
    ds = load_dataset(args.dataset_name)

    print(f"  Available splits: {list(ds.keys())}")
    print(f"  Columns: {ds[list(ds.keys())[0]].column_names}")

    all_rows = []
    for split_name, split_data in ds.items():
        print(f"  {split_name}: {len(split_data):,} rows")
        all_rows.extend(list(split_data))

    # Subsample before processing to avoid loading millions of rows
    random.shuffle(all_rows)
    if args.max_samples and len(all_rows) > args.max_samples:
        all_rows = all_rows[: args.max_samples * 3]  # oversample to account for filtering

    if all_rows:
        print(f"\n  First row keys: {list(all_rows[0].keys())}")

    # ── Process ───────────────────────────────────────────────────────────────
    print(f"\n[2/4] Processing {len(all_rows):,} rows...")
    examples = process_dataset(all_rows, max_resp_chars=args.max_resp_chars)

    # Final cap after filtering
    if args.max_samples and len(examples) > args.max_samples:
        examples = examples[: args.max_samples]

    print(f"  Valid examples: {len(examples):,}")
    gold_dist = Counter(e["gold_answer"] for e in examples)
    print(f"  Gold answer distribution: {dict(gold_dist)}  (should be ~50/50)")

    # ── Split ─────────────────────────────────────────────────────────────────
    print(f"\n[3/4] Splitting (val_frac={args.val_frac})...")
    random.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_frac))
    val = examples[:n_val]
    train = examples[n_val:]
    print(f"  Train: {len(train):,}   Val: {len(val):,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving...")
    train_path = os.path.join(args.output_dir, "pkusafe_grpo_train.jsonl")
    val_path = os.path.join(args.output_dir, "pkusafe_grpo_val.jsonl")
    save_jsonl(train, train_path)
    save_jsonl(val, val_path)

    print(f"\nDone.")
    print(f"  GRPO Train : {train_path}")
    print(f"  GRPO Val   : {val_path}")
    print(f"\nNext step: python ../training/grpo_train.py --train {train_path} --val {val_path}")


if __name__ == "__main__":
    main()
