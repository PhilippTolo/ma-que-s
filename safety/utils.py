"""Shared utilities for the safety training pipeline."""

import re

BOXED_RE = re.compile(r"\\boxed\{+([^}]+?)\}+", re.IGNORECASE)


def extract_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{} in text, uppercased, or None."""
    matches = BOXED_RE.findall(text)
    return matches[-1].strip().upper() if matches else None
