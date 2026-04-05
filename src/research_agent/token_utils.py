from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=16)
def _encoding_for_model(model: str):
    try:
        import tiktoken  # type: ignore

        return tiktoken.encoding_for_model(model)
    except Exception:
        try:
            import tiktoken  # type: ignore

            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def estimate_tokens(text: str, model: str = "gpt-4.1") -> int:
    if not text:
        return 0
    enc = _encoding_for_model(model)
    if enc is None:
        # Conservative rough approximation.
        return max(1, len(text) // 4)
    try:
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)
