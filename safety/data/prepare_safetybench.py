"""
Phase 1 — Prepare SafetyBench for SFT training.

Downloads thu-coai/SafetyBench from HuggingFace, converts to Qwen3 chat
message format with \\boxed{X} answers, and saves train / val JSONL splits.

Output schema (one JSON object per line):
    {
        "messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "<question+choices>"},
            {"role": "assistant", "content": "<rationale>\\n\\boxed{X}"}
        ],
        "category": "...",
        "answer": "A"
    }

Usage:
    python prepare_safetybench.py --output-dir ./processed
    python prepare_safetybench.py --output-dir ./processed --val-frac 0.15 --seed 42
    python prepare_safetybench.py --output-dir ./processed --no-rationale   # bare \\boxed{X} only
"""

import argparse
import json
import os
import random
import textwrap
from collections import Counter
from typing import Optional

from datasets import load_dataset

# ── Constants ──────────────────────────────────────────────────────────────────

DATASET_NAME = "thu-coai/SafetyBench"

# Qwen3 non-thinking system prompt
SYSTEM_PROMPT = (
    "You are a safety-aware AI assistant. For each multiple-choice question, "
    "briefly reason through the options, then state your final answer enclosed "
    "in \\boxed{} — for example: \\boxed{A}. "
    "Do not use a thinking block."
)

CHOICES = [chr(ord("A") + i) for i in range(20)]  # A–T; CI has 2-20 options

# Per-category one-sentence justification to include in the rationale.
# Keys are lowercase, partial matches are used if exact key not found.
CATEGORY_RATIONALE = {
    "offensiveness": (
        "Offensive, insulting, or degrading language causes real harm "
        "and has no place in responsible AI output."
    ),
    "unfairness": (
        "Discriminatory content that targets people based on protected "
        "characteristics is both harmful and unjust."
    ),
    "discrimination": (
        "Discriminatory content that targets people based on protected "
        "characteristics is both harmful and unjust."
    ),
    "physical": (
        "Content that could cause or encourage physical harm must always be refused."
    ),
    "mental": (
        "Protecting mental wellbeing means avoiding content that could cause "
        "psychological harm or distress."
    ),
    "illegal": (
        "Assisting with illegal activities endangers individuals and society "
        "and must be refused."
    ),
    "crimes": (
        "Assisting with crimes or illegal activities is fundamentally unsafe."
    ),
    "privacy": (
        "Protecting personal privacy and refusing to enable unauthorized data "
        "access is a core safety responsibility."
    ),
    "ethics": (
        "Ethical AI behavior requires refusing requests that violate moral "
        "principles or could cause moral harm."
    ),
    "default": (
        "Responsible AI behavior requires avoiding harmful, dangerous, "
        "or inappropriate content."
    ),
}


def get_category_rationale(category: Optional[str]) -> str:
    if not category:
        return CATEGORY_RATIONALE["default"]
    cat_lower = str(category).lower()
    for key, rationale in CATEGORY_RATIONALE.items():
        if key in cat_lower:
            return rationale
    return CATEGORY_RATIONALE["default"]


def build_user_message(row: dict) -> Optional[str]:
    """Extract question and choices from a row; return formatted user message or None."""

    # ── Detect question field ─────────────────────────────────────────────────
    question = None
    for field in ("prompt", "question", "input"):
        if field in row and row[field]:
            question = str(row[field]).strip()
            break
    if not question:
        return None

    # ── Detect choice fields ──────────────────────────────────────────────────
    # Case 1: separate A/B/C/D columns
    choices_dict = {}
    for letter in CHOICES:
        if letter in row and row[letter]:
            choices_dict[letter] = str(row[letter]).strip()
        elif letter.lower() in row and row[letter.lower()]:
            choices_dict[letter] = str(row[letter.lower()]).strip()

    # Case 2: choices in a nested dict/list
    if not choices_dict:
        for field in ("choices", "options"):
            val = row.get(field)
            if isinstance(val, dict):
                choices_dict = {k.upper(): str(v) for k, v in val.items()}
                break
            if isinstance(val, list) and len(val) >= 2:
                choices_dict = {
                    CHOICES[i]: str(v) for i, v in enumerate(val) if i < len(CHOICES)
                }
                break

    if len(choices_dict) < 2:
        return None

    choices_text = "\n".join(f"{letter}. {text}" for letter, text in sorted(choices_dict.items()))
    return f"{question}\n\n{choices_text}"


def detect_answer(row: dict) -> Optional[str]:
    """Return the correct answer letter (A/B/C/D) or None."""
    for field in ("answer", "label", "correct_answer", "gold"):
        val = row.get(field)
        if val is not None:
            val_str = str(val).strip().upper()
            # Could be "A", "B", "C", "D" or "0","1","2","3" (index)
            if val_str in CHOICES:
                return val_str
            if val_str.isdigit():
                idx = int(val_str)
                if 0 <= idx < len(CHOICES):
                    return CHOICES[idx]
    return None


_RATIONALE_TEMPLATES = [
    # Template 0 — "Analyzing the options"
    lambda answer, cat: textwrap.dedent(f"""\
        Analyzing the options:

        Option {answer} is the appropriate response. {cat} \
The other options involve potentially harmful, illegal, or inappropriate behavior \
that a responsible AI assistant must avoid.

        \\boxed{{{answer}}}"""),

    # Template 1 — direct reasoning lead
    lambda answer, cat: textwrap.dedent(f"""\
        Let me reason through each choice:

        The correct answer is {answer}. {cat} \
Selecting any other option would risk causing harm or violating ethical guidelines.

        \\boxed{{{answer}}}"""),

    # Template 2 — elimination framing
    lambda answer, cat: textwrap.dedent(f"""\
        Evaluating the choices:

        Options other than {answer} are unsafe or inappropriate. {cat} \
Therefore the only responsible choice is {answer}.

        \\boxed{{{answer}}}"""),

    # Template 3 — principle-first framing
    lambda answer, cat: textwrap.dedent(f"""\
        Applying safety principles:

        {cat} Based on this, option {answer} is the correct and responsible answer.

        \\boxed{{{answer}}}"""),
]


def build_rationale(answer: str, category: Optional[str], add_rationale: bool) -> str:
    """Build the assistant response ending with \\boxed{answer}."""
    if not add_rationale:
        return f"\\boxed{{{answer}}}"

    cat_sentence = get_category_rationale(category)
    template = random.choice(_RATIONALE_TEMPLATES)
    return template(answer, cat_sentence)


def process_dataset(dataset, add_rationale: bool = True) -> list[dict]:
    """Convert a HuggingFace dataset split into a list of training examples."""
    examples = []
    skipped = 0

    for row in dataset:
        user_msg = build_user_message(row)
        if user_msg is None:
            skipped += 1
            continue

        answer = detect_answer(row)
        if answer is None:
            skipped += 1
            continue

        category = row.get("category") or row.get("type") or row.get("task")
        assistant_msg = build_rationale(answer, category, add_rationale)

        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ],
                "category": str(category) if category else "unknown",
                "answer": answer,
            }
        )

    if skipped:
        print(f"  [warn] Skipped {skipped} rows (missing question or answer field).")
    return examples


def save_jsonl(data: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data):,} examples → {path}")


def print_stats(examples: list[dict], label: str) -> None:
    cats = Counter(e["category"] for e in examples)
    answers = Counter(e["answer"] for e in examples)
    print(f"\n  {label} statistics ({len(examples):,} examples)")
    print(f"  Answer distribution: {dict(sorted(answers.items()))}")
    print(f"  Category distribution (top 8):")
    for cat, count in cats.most_common(8):
        print(f"    {cat:<40} {count:>5}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--lang", default="en", choices=["en", "zh"],
                        help="Language version of SafetyBench")
    parser.add_argument("--output-dir", default="./processed")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Fraction of data reserved for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-rationale", action="store_true",
                        help="Skip rationale, just output \\boxed{X}")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap total samples (useful for quick tests)")
    args = parser.parse_args()

    add_rationale = not args.no_rationale
    random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  SafetyBench data preparation")
    print(f"  Dataset : {args.dataset_name}")
    print(f"  Language: {args.lang}")
    print(f"  Rationale: {add_rationale}")
    print(f"{'='*60}\n")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"[1/4] Downloading {args.dataset_name} ...")
    try:
        ds = load_dataset(args.dataset_name, args.lang)
    except Exception:
        # Fallback: try without language specifier
        print(f"  Retry without language split...")
        ds = load_dataset(args.dataset_name)

    print(f"  Available splits: {list(ds.keys())}")
    print(f"  Columns: {ds[list(ds.keys())[0]].column_names}")

    # Merge all splits into one pool, then re-split (SafetyBench may only
    # have a test set, which we treat as our training pool).
    all_rows = []
    for split_name, split_data in ds.items():
        print(f"  {split_name}: {len(split_data):,} rows")
        all_rows.extend(list(split_data))

    if args.max_samples:
        random.shuffle(all_rows)
        all_rows = all_rows[: args.max_samples]

    # ── Process ───────────────────────────────────────────────────────────────
    print(f"\n[2/4] Processing {len(all_rows):,} rows...")

    # Peek at the first row to diagnose structure
    if all_rows:
        print(f"  First row keys: {list(all_rows[0].keys())}")
        print(f"  First row sample: {json.dumps({k: str(v)[:60] for k, v in list(all_rows[0].items())[:6]}, ensure_ascii=False)}")

    examples = process_dataset(all_rows, add_rationale=add_rationale)
    print(f"  Processed: {len(examples):,} valid examples")

    # ── Train / val split ─────────────────────────────────────────────────────
    print(f"\n[3/4] Splitting (val_frac={args.val_frac}, seed={args.seed})...")
    random.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_frac))
    val = examples[:n_val]
    train = examples[n_val:]

    print_stats(train, "Train")
    print_stats(val, "Val")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving...")
    train_path = os.path.join(args.output_dir, "safetybench_train.jsonl")
    val_path = os.path.join(args.output_dir, "safetybench_val.jsonl")
    save_jsonl(train, train_path)
    save_jsonl(val, val_path)

    print(f"\nDone.")
    print(f"  Train : {train_path}  ({len(train):,} examples)")
    print(f"  Val   : {val_path}  ({len(val):,} examples)")
    print(f"\nNext step: python ../training/sft_train.py --train {train_path} --val {val_path}")


if __name__ == "__main__":
    main()
