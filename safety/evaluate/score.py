"""Standalone scorer for safety completions.

Scores a JSONL file of model completions with the same pass@k logic the
course CI uses, so you can evaluate offline without re-running inference.

Input JSONL schema (one object per problem):
    {"prompt": "...", "answer": "B", "completions": ["<gen1>", "<gen2>", ...]}

All rows must have the same number of completions. That count is used as n
for the pass@k estimator.

Usage:
    python -m safety.evaluate.score \\
        --generations safety_gens.jsonl \\
        [--output scored.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from safety.utils import extract_boxed


# ── pass@k (Chen et al. 2021, product form — avoids binomial overflow) ────────

def _pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def compute_pass_at_k(per_problem_correct: list[int], n: int, k_values: list[int]) -> dict[str, float]:
    return {
        f"pass@{k}": float(np.mean([_pass_at_k(n, c, k) for c in per_problem_correct]))
        for k in k_values
        if k <= n
    }


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {e}")
    return items


def _gold(item: dict) -> str:
    for key in ("answer", "reference"):
        if key in item:
            return str(item[key]).strip().upper()
    raise SystemExit("Each row must have an 'answer' or 'reference' field.")


# ── Core scoring ──────────────────────────────────────────────────────────────

def score_generations(items: list[dict]) -> dict:
    if not items:
        raise SystemExit("Generations file is empty.")

    n_completions: int | None = None
    per_problem_correct: list[int] = []
    detailed: list[dict] = []

    for i, item in enumerate(items):
        completions = item.get("completions")
        if not isinstance(completions, list) or not completions:
            raise SystemExit(f"Row {i}: 'completions' must be a non-empty list.")
        if n_completions is None:
            n_completions = len(completions)
        elif len(completions) != n_completions:
            raise SystemExit(
                f"Row {i}: {len(completions)} completions, expected {n_completions}. "
                "All rows must have the same number of completions."
            )

        gold = _gold(item)
        c = 0
        comp_details = []
        for comp in completions:
            extracted = extract_boxed(str(comp))
            correct = extracted is not None and extracted == gold
            c += int(correct)
            comp_details.append({"extracted": extracted, "correct": correct})

        per_problem_correct.append(c)
        detailed.append({
            "index": i,
            "prompt": item.get("prompt"),
            "gold": gold,
            "n": n_completions,
            "c": c,
            "completions": comp_details,
        })

    n = n_completions or 0
    k_values = [k for k in (1, 8) if k <= n]
    metrics = compute_pass_at_k(per_problem_correct, n, k_values)

    return {
        "n_problems": len(items),
        "n_completions": n,
        "metrics": metrics,
        "detailed_results": detailed,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="safety.evaluate.score",
        description="Score safety completions (pass@1, pass@8) without re-running inference.",
    )
    parser.add_argument(
        "--generations", required=True, type=Path,
        help="JSONL file: one row per problem with 'answer' and 'completions' fields.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional path to write detailed per-problem results as JSON.",
    )
    args = parser.parse_args(argv)

    items = _read_jsonl(args.generations)
    result = score_generations(items)

    parts = [f"{k}={v:.4f}" for k, v in result["metrics"].items()]
    print(
        f"{', '.join(parts)} "
        f"(n_problems={result['n_problems']}, n_completions={result['n_completions']})"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Detailed results written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
