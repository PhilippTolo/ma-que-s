"""
Prepare MMLU safety-relevant subsets for SFT training.

Downloads 6 safety-adjacent MMLU subsets from cais/mmlu, converts to
Qwen3 chat-message format with \\boxed{X} answers.

Why MMLU instead of SafetyBench:
  - SafetyBench (thu-coai/SafetyBench) HF release withholds ground-truth labels
    → zero usable training examples despite 24k questions.
  - MMLU safety subsets are fully labeled, 4-option (A-D), cover ethics/harm
    categories that improve generalisation beyond PKU-SafeRLHF binary A/B format.

Subsets used:
  moral_scenarios, moral_disputes, medical_ethics,
  business_ethics, jurisprudence, philosophy

Output schema (same as prepare_safetybench.py / pkusafe_sft):
  {"messages": [...], "category": "...", "answer": "A"}

Usage:
    python prepare_mmlu_safety.py --output-dir /scratch/safety/data
"""

import argparse
import ast
import json
import os
import random

from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a safety-aware AI assistant. For each multiple-choice question, "
    "briefly reason through the options, then state your final answer enclosed "
    "in \\boxed{} — for example: \\boxed{A}. "
    "Do not use a thinking block."
)

SAFETY_SUBSETS = [
    "moral_scenarios",
    "moral_disputes",
    "medical_ethics",
    "business_ethics",
    "jurisprudence",
    "philosophy",
]

CHOICES = ["A", "B", "C", "D"]

RATIONALE_TEMPLATES = [
    lambda answer, cat: (
        f"Analyzing the options:\n\n"
        f"Option {answer} is correct. This question involves {cat} reasoning. "
        f"The other options are either factually wrong, ethically problematic, "
        f"or less appropriate given the context.\n\n\\boxed{{{answer}}}"
    ),
    lambda answer, cat: (
        f"Let me reason through each choice:\n\n"
        f"The correct answer is {answer}. In the context of {cat}, "
        f"this option best reflects sound ethical and factual reasoning.\n\n"
        f"\\boxed{{{answer}}}"
    ),
    lambda answer, cat: (
        f"Evaluating the choices:\n\n"
        f"Option {answer} is the most appropriate response to this {cat} question. "
        f"The remaining options contain inaccuracies or ethically questionable content.\n\n"
        f"\\boxed{{{answer}}}"
    ),
]


def process_subset(subset_name: str, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    examples = []
    skipped = 0

    try:
        ds = load_dataset("cais/mmlu", subset_name, trust_remote_code=True)
    except Exception as e:
        print(f"  Warning: could not load subset '{subset_name}': {e}")
        return []

    for split_name, split in ds.items():
        for row in split:
            question = row.get("question", "").strip()
            choices_raw = row.get("choices", [])

            # choices can be a list or a string repr of a list
            if isinstance(choices_raw, str):
                try:
                    choices_raw = ast.literal_eval(choices_raw)
                except Exception:
                    skipped += 1
                    continue

            if not isinstance(choices_raw, list) or len(choices_raw) < 2:
                skipped += 1
                continue

            answer_idx = row.get("answer")
            if answer_idx is None:
                skipped += 1
                continue
            try:
                answer_idx = int(answer_idx)
            except (ValueError, TypeError):
                skipped += 1
                continue
            if not (0 <= answer_idx < len(choices_raw)):
                skipped += 1
                continue

            answer_letter = CHOICES[answer_idx]
            choices_text = "\n".join(
                f"{CHOICES[i]}. {str(c).strip()}"
                for i, c in enumerate(choices_raw)
                if i < len(CHOICES)
            )
            user_msg = f"{question}\n\n{choices_text}"

            template = rng.choice(RATIONALE_TEMPLATES)
            assistant_msg = template(answer_letter, subset_name.replace("_", " "))

            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ],
                "category": subset_name,
                "answer": answer_letter,
            })

    if skipped:
        print(f"  [{subset_name}] Skipped {skipped} rows.")
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./processed")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MMLU Safety Subsets — data preparation")
    print(f"  Subsets: {SAFETY_SUBSETS}")
    print(f"{'='*60}\n")

    all_examples = []
    for subset in SAFETY_SUBSETS:
        print(f"  Loading {subset}...")
        exs = process_subset(subset, seed=args.seed)
        print(f"  → {len(exs)} examples")
        all_examples.extend(exs)

    print(f"\n  Total: {len(all_examples)} examples")

    random.shuffle(all_examples)
    n_val = max(1, int(len(all_examples) * args.val_frac))
    val = all_examples[:n_val]
    train = all_examples[n_val:]

    from collections import Counter
    answers = Counter(e["answer"] for e in train)
    cats = Counter(e["category"] for e in train)
    print(f"\n  Train answer distribution: {dict(sorted(answers.items()))}")
    print(f"  Train category distribution:")
    for cat, count in cats.most_common():
        print(f"    {cat:<30} {count:>5}")

    train_path = os.path.join(args.output_dir, "mmlu_safety_train.jsonl")
    val_path = os.path.join(args.output_dir, "mmlu_safety_val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n  Saved {len(train)} train → {train_path}")
    print(f"  Saved {len(val)} val   → {val_path}")
    print(f"\nNext step:")
    print(f"  python sft_train.py \\")
    print(f"    --train /scratch/safety/data/pkusafe_sft_train.jsonl {train_path} \\")
    print(f"    --val   {val_path} \\")
    print(f"    --output-dir /scratch/safety/sft_v3 --epochs 2 --lr 1e-5")


if __name__ == "__main__":
    main()
