import numpy as np


def token_length_scores(prompts, encode) -> np.ndarray:
    """Token count per prompt, row-aligned to `prompts`.

    `encode` is any callable that turns one string into a sequence of token ids, so
    the caller controls the tokenizer.
    """
    if len(prompts) == 0:
        raise ValueError("token_length_scores got no prompts")
    scores = np.asarray([len(encode(str(p))) for p in prompts], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("token_length_scores produced non-finite output")
    if scores.shape != (len(prompts),):
        raise ValueError(f"shape {scores.shape} does not match {len(prompts)} prompts")
    return scores
