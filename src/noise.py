#!/usr/bin/env python3
"""
CENG467 UNMT — Noise Functions for Denoising Autoencoder
=========================================================
Implements the three noise operations from Lample et al. (2018)
that corrupt input sequences for denoising autoencoder training:

  1. **Word Shuffle**  — permute tokens within a local window
  2. **Word Dropout**   — randomly delete tokens
  3. **Word Masking**   — replace tokens with a <MASK> symbol

These functions operate on Python lists of token IDs (before
batching / padding) so they can be applied per-sentence with
varying lengths.
"""

import random
from typing import List, Optional, Set


# ════════════════════════════════════════════════════════════
#  INDIVIDUAL NOISE OPERATIONS
# ════════════════════════════════════════════════════════════

def word_shuffle(token_ids: List[int], k: int = 3) -> List[int]:
    """Locally shuffle tokens within a window of *k* positions.

    For each token at position *i*, a random value is sampled from
    ``Uniform(i, i + k + 1)``.  Tokens are then sorted by these values,
    producing a permutation that roughly preserves word order.

    Args:
        token_ids: Original token ID sequence.
        k: Maximum shuffle distance.  ``k=0`` means no shuffle.

    Returns:
        Shuffled copy of *token_ids*.
    """
    n = len(token_ids)
    if n <= 1 or k == 0:
        return list(token_ids)

    # Assign noisy positions and sort by them
    noise = [i + random.uniform(0, k + 1) for i in range(n)]
    permutation = sorted(range(n), key=lambda i: noise[i])
    return [token_ids[i] for i in permutation]


def word_dropout(
    token_ids: List[int],
    p: float = 0.1,
    keep_ids: Optional[Set[int]] = None,
) -> List[int]:
    """Randomly remove tokens with probability *p*.

    Tokens whose ID is in *keep_ids* (e.g. BOS, EOS, language tokens)
    are **never** dropped.  If all content tokens are dropped, at
    least one is retained to avoid empty sequences.

    Args:
        token_ids: Original token ID sequence.
        p: Dropout probability per token.
        keep_ids: Set of IDs that must be preserved.

    Returns:
        Filtered copy of *token_ids*.
    """
    if p <= 0.0 or not token_ids:
        return list(token_ids)

    keep = keep_ids or set()
    result = [t for t in token_ids if t in keep or random.random() >= p]

    # Safety: never return an empty sequence
    if not result:
        result = [random.choice(token_ids)]

    return result


def word_mask(
    token_ids: List[int],
    mask_id: int,
    p: float = 0.1,
    keep_ids: Optional[Set[int]] = None,
) -> List[int]:
    """Replace tokens with *mask_id* with probability *p*.

    Tokens in *keep_ids* are never masked.

    Args:
        token_ids: Original token ID sequence.
        mask_id: ID of the ``<MASK>`` token in the vocabulary.
        p: Masking probability per token.
        keep_ids: Set of IDs that must be preserved.

    Returns:
        Masked copy of *token_ids*.
    """
    if p <= 0.0:
        return list(token_ids)

    keep = keep_ids or set()
    return [
        (mask_id if t not in keep and random.random() < p else t)
        for t in token_ids
    ]


# ════════════════════════════════════════════════════════════
#  COMPOSITE NOISE FUNCTION
# ════════════════════════════════════════════════════════════

def apply_noise(
    token_ids: List[int],
    shuffle_k: int = 3,
    dropout_p: float = 0.1,
    mask_p: float = 0.1,
    mask_id: int = 6,
    special_ids: Optional[Set[int]] = None,
) -> List[int]:
    """Apply all three noise operations sequentially.

    Order (following Lample et al.):
      1. Shuffle  — reorder tokens locally
      2. Dropout  — remove some tokens
      3. Masking  — replace some tokens with <MASK>

    Args:
        token_ids: Clean token ID sequence.
        shuffle_k: Shuffle window size.
        dropout_p: Token dropout probability.
        mask_p: Token masking probability.
        mask_id: ID of the ``<MASK>`` token.
        special_ids: Token IDs to protect from dropout / masking
                     (e.g. ``{pad_id, bos_id, eos_id, lang_ids}``).

    Returns:
        Noised copy of *token_ids*.

    Example::

        >>> apply_noise([10, 20, 30, 40, 50], shuffle_k=2,
        ...             dropout_p=0.1, mask_p=0.1, mask_id=6)
        [10, 30, 6, 50]   # shuffled, one dropped, one masked
    """
    keep = special_ids or set()

    result = word_shuffle(token_ids, k=shuffle_k)
    result = word_dropout(result, p=dropout_p, keep_ids=keep)
    result = word_mask(result, mask_id=mask_id, p=mask_p, keep_ids=keep)

    return result


def make_noise_fn(cfg: dict, special_ids: Set[int], mask_id: int):
    """Return a pre-configured noise function from a config dict.

    The returned callable has signature ``fn(token_ids) → noised_ids``,
    suitable for passing directly to the training loop.

    Args:
        cfg: The ``noise`` section of config.yaml.
        special_ids: Token IDs to protect.
        mask_id: <MASK> token ID.

    Returns:
        A callable ``(List[int]) → List[int]``.
    """
    shuffle_k = cfg.get("word_shuffle_k", 3)
    dropout_p = cfg.get("word_dropout", 0.1)
    mask_p = cfg.get("word_mask", 0.1)

    def noise_fn(token_ids: List[int]) -> List[int]:
        return apply_noise(
            token_ids,
            shuffle_k=shuffle_k,
            dropout_p=dropout_p,
            mask_p=mask_p,
            mask_id=mask_id,
            special_ids=special_ids,
        )

    return noise_fn
