#!/usr/bin/env python3
"""
CENG467 UNMT — Monolingual Dataset & Data Utilities
=====================================================
Provides :class:`MonolingualDataset` for loading cleaned monolingual
text, encoding it with SentencePiece, and caching the encoded IDs
as compact NumPy arrays for fast reloading.

Also includes batching helpers used by the training loop:

  • :func:`pad_sequences` — variable-length lists → padded tensor
  • :func:`prepare_dae_batch` — build encoder/decoder I/O for DAE
  • :func:`alternating_loader` — interleave TR / EN batches
"""

import os
import sys
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import ensure_dir, setup_logging

log = setup_logging("dataset")


# ════════════════════════════════════════════════════════════
#  MONOLINGUAL DATASET
# ════════════════════════════════════════════════════════════

class MonolingualDataset(Dataset):
    """Memory-efficient dataset for monolingual sentences.

    On first load the cleaned text file is encoded with SentencePiece
    and saved as a compact ``.npz`` cache.  Subsequent loads read the
    cache directly, skipping the encoding step entirely.

    Storage layout (in memory & on disk):
      • ``ids``     — flat ``int32`` array of all token IDs
      • ``offsets`` — ``int64`` array of sentence boundaries

    Memory usage for 1 M sentences × ~50 tokens ≈ 200 MB (int32).

    Args:
        data_path: Path to ``clean.{lang}.txt``.
        sp_model_path: Path to ``spm.model``.
        max_len: Maximum tokens per sentence (longer are dropped).
        min_len: Minimum tokens per sentence (shorter are dropped).
        cache_dir: Where to store the ``.npz`` cache (default: same as data).
    """

    def __init__(
        self,
        data_path: str,
        sp_model_path: str,
        max_len: int = 126,
        min_len: int = 3,
        cache_dir: Optional[str] = None,
    ):
        self.max_len = max_len
        self.min_len = min_len

        cache_dir = cache_dir or os.path.dirname(data_path)
        basename = os.path.splitext(os.path.basename(data_path))[0]
        cache_path = os.path.join(cache_dir, f"{basename}.ids.npz")

        if os.path.exists(cache_path):
            self._load_cache(cache_path)
        else:
            self._encode_and_cache(data_path, sp_model_path, cache_path)

    # ── Cache I/O ───────────────────────────────────────────

    def _load_cache(self, path: str) -> None:
        log.info(f"Loading cached encodings ← {path}")
        data = np.load(path)
        self.ids = data["ids"]
        self.offsets = data["offsets"]
        log.info(f"  {len(self):,} sentences, {len(self.ids):,} tokens")

    def _encode_and_cache(
        self, data_path: str, sp_model_path: str, cache_path: str
    ) -> None:
        log.info(f"Encoding with SentencePiece: {data_path}")
        sp = spm.SentencePieceProcessor(model_file=sp_model_path)

        all_ids: List[int] = []
        offsets: List[int] = [0]
        kept, skipped = 0, 0

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids = sp.encode(line, out_type=int)
                if self.min_len <= len(ids) <= self.max_len:
                    all_ids.extend(ids)
                    offsets.append(len(all_ids))
                    kept += 1
                else:
                    skipped += 1

                if kept % 200_000 == 0 and kept > 0:
                    log.info(f"  encoded {kept:,} sentences …")

        self.ids = np.array(all_ids, dtype=np.int32)
        self.offsets = np.array(offsets, dtype=np.int64)

        ensure_dir(os.path.dirname(cache_path))
        np.savez(cache_path, ids=self.ids, offsets=self.offsets)
        log.info(
            f"  ✓ {kept:,} sentences cached (skipped {skipped:,}) → {cache_path}"
        )

    # ── Dataset interface ───────────────────────────────────

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, idx: int) -> List[int]:
        """Return token IDs for sentence *idx* as a Python list."""
        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1])
        return self.ids[start:end].tolist()


# ════════════════════════════════════════════════════════════
#  BATCHING UTILITIES
# ════════════════════════════════════════════════════════════

def _identity_collate(batch: List[List[int]]) -> List[List[int]]:
    """Pass-through collate — keeps each sample as a list of ints.

    Padding is deferred to :func:`prepare_dae_batch` where we know
    the language token and noise function.
    """
    return batch


def build_dataloader(
    dataset: MonolingualDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader that yields ``List[List[int]]`` batches."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_identity_collate,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )


def pad_sequences(
    seqs: List[List[int]], pad_id: int = 0
) -> torch.Tensor:
    """Pad variable-length int lists into a ``(B, T_max)`` LongTensor."""
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def prepare_dae_batch(
    clean_ids_batch: List[List[int]],
    lang_token_id: int,
    noise_fn,
    pad_id: int = 0,
    bos_id: int = 2,
    eos_id: int = 3,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """Prepare encoder / decoder tensors for one DAE training step.

    Builds:
      • **Encoder input**  ``[LANG] + noise(clean)``
      • **Decoder input**  ``[BOS] + clean``
      • **Decoder target** ``clean + [EOS]``

    All three are padded to the maximum length in the batch and
    moved to *device*.

    Returns:
        ``(enc_input, dec_input, dec_target,
          enc_padding_mask, dec_padding_mask)``
    """
    # Apply noise to each sentence
    noised_batch = [noise_fn(ids) for ids in clean_ids_batch]

    # Encoder: [LANG] + noised tokens
    enc_seqs = [[lang_token_id] + n for n in noised_batch]

    # Decoder input: [BOS] + clean
    dec_in_seqs = [[bos_id] + c for c in clean_ids_batch]

    # Decoder target: clean + [EOS]
    dec_tgt_seqs = [c + [eos_id] for c in clean_ids_batch]

    enc_input = pad_sequences(enc_seqs, pad_id).to(device)
    dec_input = pad_sequences(dec_in_seqs, pad_id).to(device)
    dec_target = pad_sequences(dec_tgt_seqs, pad_id).to(device)

    enc_padding_mask = (enc_input == pad_id)
    dec_padding_mask = (dec_input == pad_id)

    return enc_input, dec_input, dec_target, enc_padding_mask, dec_padding_mask


# ════════════════════════════════════════════════════════════
#  ALTERNATING (TR / EN) BATCH ITERATOR
# ════════════════════════════════════════════════════════════

def alternating_loader(
    loader_a: DataLoader,
    loader_b: DataLoader,
    lang_id_a: int,
    lang_id_b: int,
) -> Iterator[Tuple[List[List[int]], int]]:
    """Yield batches alternating between two languages.

    Stops when the **shorter** loader is exhausted, ensuring
    balanced training per epoch.

    Yields:
        ``(batch, lang_token_id)``
    """
    iter_a = iter(loader_a)
    iter_b = iter(loader_b)

    while True:
        try:
            batch_a = next(iter_a)
            yield batch_a, lang_id_a
        except StopIteration:
            return

        try:
            batch_b = next(iter_b)
            yield batch_b, lang_id_b
        except StopIteration:
            return
