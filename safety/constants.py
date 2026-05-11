"""Shared constants for Qwen3 generation config.

Token IDs are Qwen3-specific. Call verify() at runtime to confirm they
match the loaded tokenizer before pushing to hub.
"""

BOS_TOKEN_ID = 151643
EOS_TOKEN_IDS = [151645, 151643]
PAD_TOKEN_ID = 151643

REQUIRED_GEN_CONFIG = {
    "bos_token_id": BOS_TOKEN_ID,
    "eos_token_id": EOS_TOKEN_IDS,
    "pad_token_id": PAD_TOKEN_ID,
    "do_sample": True,
    "transformers_version": "4.51.0",
}


def verify(tokenizer) -> None:
    """Assert that the loaded tokenizer's special token IDs match our constants."""
    assert tokenizer.bos_token_id == BOS_TOKEN_ID, (
        f"bos_token_id mismatch: tokenizer={tokenizer.bos_token_id}, "
        f"constants={BOS_TOKEN_ID}"
    )
    eos = tokenizer.eos_token_id
    eos_set = {eos} if isinstance(eos, int) else set(eos)
    assert eos_set & set(EOS_TOKEN_IDS), (
        f"eos_token_id mismatch: tokenizer={eos}, constants={EOS_TOKEN_IDS}"
    )
    print("  [constants] Token ID check passed.")
