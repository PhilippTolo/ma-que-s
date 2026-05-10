"""
Quick data inspection utility.

Usage:
    python inspect_data.py processed/safetybench_train.jsonl
    python inspect_data.py processed/pkusafe_grpo_train.jsonl --n 3
    python inspect_data.py processed/safetybench_val.jsonl --check-boxed
"""

import argparse
import json
import re
from collections import Counter

BOXED_PATTERN = re.compile(r"\\boxed\{([^}]+)\}", re.IGNORECASE)


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_boxed(examples: list[dict]) -> None:
    """Report how many examples have \\boxed{} in the assistant turn."""
    found = 0
    for ex in examples:
        # SFT format: messages[-1] is assistant
        if "messages" in ex:
            text = ex["messages"][-1].get("content", "")
        elif "prompt" in ex:
            # GRPO format — check gold_answer field instead
            found += 1 if ex.get("gold_answer") else 0
            continue
        else:
            text = ""
        if BOXED_PATTERN.search(text):
            found += 1
    total = len(examples)
    pct = 100.0 * found / total if total else 0
    status = "OK" if found == total else "WARN"
    print(f"  [\\boxed{{}} check] {found}/{total} ({pct:.1f}%) — {status}")


def show_example(ex: dict, idx: int) -> None:
    print(f"\n  {'─'*55}")
    print(f"  Example #{idx}")
    if "messages" in ex:
        for msg in ex["messages"]:
            role = msg["role"].upper()
            content = msg["content"]
            # Truncate long content
            if len(content) > 300:
                content = content[:300] + "…"
            print(f"  [{role}]\n  {content}\n")
        meta_keys = [k for k in ex if k != "messages"]
        if meta_keys:
            print(f"  Meta: { {k: ex[k] for k in meta_keys} }")
    elif "prompt" in ex:
        prompt = ex["prompt"]
        if len(prompt) > 400:
            prompt = prompt[:400] + "…"
        print(f"  [PROMPT]\n  {prompt}")
        print(f"  [GOLD ANSWER] {ex.get('gold_answer', '?')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to JSONL file")
    parser.add_argument("--n", type=int, default=2, help="Number of examples to print")
    parser.add_argument("--check-boxed", action="store_true",
                        help="Verify \\boxed{} present in all assistant outputs")
    args = parser.parse_args()

    print(f"\nLoading: {args.path}")
    examples = load_jsonl(args.path)
    print(f"Total examples: {len(examples):,}")

    # Category distribution (SFT data)
    if examples and "category" in examples[0]:
        cats = Counter(e.get("category", "unknown") for e in examples)
        print(f"\n  Category distribution (top 10):")
        for cat, count in cats.most_common(10):
            print(f"    {cat:<40} {count:>5}")

    # Answer distribution
    answer_field = "answer" if "answer" in (examples[0] if examples else {}) else "gold_answer"
    if examples and answer_field in examples[0]:
        answers = Counter(e.get(answer_field) for e in examples)
        print(f"\n  Answer distribution: {dict(sorted(answers.items()))}")

    # Boxed check
    if args.check_boxed or True:
        check_boxed(examples)

    # Print examples
    print(f"\n  Showing {min(args.n, len(examples))} example(s):")
    for i in range(min(args.n, len(examples))):
        show_example(examples[i], i + 1)

    print()


if __name__ == "__main__":
    main()
