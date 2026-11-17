"""The reference poisoning budget and deterministic row selection, shared by both attacks."""

import math
import random


def select_poison_indices(size, *, fraction=None, count=None, seed=42):
    """Count takes precedence; fractions round down without changing the global RNG."""
    if count is None:
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('Supply a count or a finite fraction between zero and one.')
        count = int(fraction * size)
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= size:
        raise ValueError('Poison count must be an integer between zero and the training size.')
    # A local RNG preserves the reference selection without resetting other randomness.
    return random.Random(seed).sample(range(size), count)
