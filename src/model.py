#!/usr/bin/env python3
"""
CENG467 UNMT — Shared Transformer Encoder–Decoder
===================================================
Lightweight Transformer designed for Colab T4 (16 GB VRAM):

  ┌───────────────────────────────────────────────────┐
  │  4 encoder layers  ·  4 decoder layers            │
  │  d_model = 512     ·  n_heads = 8                 │
  │  d_ff    = 2048    ·  dropout = 0.1               │
  │  Pre-LN (norm_first)  ·  Weight tying             │
  │  ~46 M parameters                                 │
  └───────────────────────────────────────────────────┘

Both languages share every parameter.  The target language is
indicated by a prefix token (``<TR>`` or ``<EN>``) prepended to
the encoder input.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


# ════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to *x* of shape ``(B, T, D)``."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ════════════════════════════════════════════════════════════
#  UNMT TRANSFORMER
# ════════════════════════════════════════════════════════════

class UNMTTransformer(nn.Module):
    """Shared encoder–decoder Transformer for unsupervised NMT.

    Both Turkish and English share:
      • Embedding layer (input + output weight tying)
      • Encoder stack
      • Decoder stack

    The target language is communicated by prepending a language
    token (``<TR>`` / ``<EN>``) to the encoder input.

    Args:
        vocab_size: SentencePiece vocabulary size.
        d_model:    Embedding / hidden dimension.
        n_heads:    Number of attention heads.
        num_layers: Number of encoder AND decoder layers.
        d_ff:       Feed-forward inner dimension.
        dropout:    Dropout probability.
        max_seq_len: Maximum sequence length.
        pad_id:     Padding token ID.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.vocab_size = vocab_size

        # ── Shared embedding ────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len + 64, dropout)
        self.embed_scale = math.sqrt(d_model)

        # ── Encoder (Pre-LN) ────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ── Decoder (Pre-LN) ────────────────────────────────
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ── Output projection (weight-tied to embedding) ────
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.embedding.weight

        # ── Initialise weights ───────────────────────────────
        self._init_weights()

    # ────────────────────────────────────────────────────────
    #  Initialisation
    # ────────────────────────────────────────────────────────

    def _init_weights(self):
        """Stable Transformer initialisation.

        Important:
        The embedding matrix is shared with the output projection.
        If the embedding keeps PyTorch's default N(0, 1) init while
        inputs are scaled by sqrt(d_model), logits become extremely
        large at step 0. This causes CE loss in the hundreds and makes
        greedy decoding collapse into repetition loops.

        We initialise embeddings with std = d_model^-0.5, so after
        sqrt(d_model) scaling the embedded activations have roughly
        unit scale, while the tied output projection remains stable.
        """
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.d_model ** -0.5)

        if self.pad_id is not None:
            with torch.no_grad():
                self.embedding.weight[self.pad_id].fill_(0)

        for name, p in self.named_parameters():
            if name == "embedding.weight":
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ────────────────────────────────────────────────────────
    #  Helpers
    # ────────────────────────────────────────────────────────

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed tokens and add positional encoding."""
        return self.pos_encoding(self.embedding(token_ids) * self.embed_scale)

    @staticmethod
    def _causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask (``-inf`` above diagonal)."""
        mask = torch.triu(
            torch.ones(sz, sz, device=device), diagonal=1
        )
        return mask.masked_fill(mask == 1, float("-inf"))

    # ────────────────────────────────────────────────────────
    #  Encode / Decode / Forward
    # ────────────────────────────────────────────────────────

    def encode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode source tokens → memory.

        Args:
            src: ``(B, S)`` source token IDs.
            src_key_padding_mask: ``(B, S)`` bool, True where padded.

        Returns:
            ``(B, S, D)`` encoder memory.
        """
        return self.encoder(
            self._embed(src),
            src_key_padding_mask=src_key_padding_mask,
        )

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode target tokens with cross-attention to encoder memory.

        A causal mask is applied automatically so that each position
        can only attend to itself and earlier positions.

        Args:
            tgt: ``(B, T)`` target token IDs.
            memory: ``(B, S, D)`` encoder output.
            tgt_key_padding_mask: ``(B, T)`` bool.
            memory_key_padding_mask: ``(B, S)`` bool.

        Returns:
            ``(B, T, D)`` decoded representations.
        """
        tgt_mask = self._causal_mask(tgt.size(1), tgt.device)

        return self.decoder(
            self._embed(tgt),
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full forward pass: encode → decode → project to logits.

        Args:
            src: ``(B, S)`` encoder input (language token + noised tokens).
            tgt: ``(B, T)`` decoder input (BOS + clean tokens).
            src_key_padding_mask: ``(B, S)`` bool.
            tgt_key_padding_mask: ``(B, T)`` bool.

        Returns:
            ``(B, T, V)`` logits over vocabulary.
        """
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)
        decoded = self.decode(
            tgt,
            memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output_proj(decoded)

    # ────────────────────────────────────────────────────────
    #  Greedy decoding (for quick inference / back-translation)
    # ────────────────────────────────────────────────────────

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        max_len: int = 128,
        bos_id: int = 2,
        eos_id: int = 3,
    ) -> torch.Tensor:
        """Autoregressively generate with greedy (argmax) decoding.

        Args:
            src: ``(B, S)`` encoder input.
            src_key_padding_mask: ``(B, S)`` bool.
            max_len: Maximum output length (including BOS).
            bos_id: Beginning-of-sentence token ID.
            eos_id: End-of-sentence token ID.

        Returns:
            ``(B, T)`` generated token IDs (including BOS, up to EOS / max_len).
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device

        # Encode
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)

        # Start with BOS
        ys = torch.full(
            (batch_size, 1), bos_id, dtype=torch.long, device=device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt_mask = self._causal_mask(ys.size(1), device)
            tgt_pad_mask = (ys == self.pad_id)

            decoded = self.decoder(
                self._embed(ys),
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            logits = self.output_proj(decoded[:, -1, :])   # last step
            next_tokens = logits.argmax(dim=-1)             # (B,)

            # Finished sequences emit PAD
            next_tokens = next_tokens.masked_fill(finished, self.pad_id)
            ys = torch.cat([ys, next_tokens.unsqueeze(1)], dim=1)

            finished = finished | (next_tokens == eos_id)
            if finished.all():
                break

        return ys


# ════════════════════════════════════════════════════════════
#  FACTORY
# ════════════════════════════════════════════════════════════

def build_model(cfg: dict, vocab_size: int, pad_id: int = 0) -> UNMTTransformer:
    """Construct a :class:`UNMTTransformer` from config.yaml's ``model`` section.

    Args:
        cfg: The full config dict (``model`` key is read).
        vocab_size: SentencePiece vocabulary size.
        pad_id: Padding token ID.

    Returns:
        An uninitialised (randomly-weighted) :class:`UNMTTransformer`.
    """
    m = cfg["model"]
    return UNMTTransformer(
        vocab_size=vocab_size,
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        num_layers=m["num_layers"],
        d_ff=m["d_ff"],
        dropout=m["dropout"],
        max_seq_len=m["max_seq_len"],
        pad_id=pad_id,
    )
