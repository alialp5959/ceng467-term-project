#!/usr/bin/env python3
"""
CENG467 UNMT — Generation & Decoding Utilities
==============================================
Provides inference utilities for the trained UNMT Transformer:
  • Greedy decoding (fast, used during back-translation)
  • Beam search decoding (better quality, used for final evaluation)

Usage:
  # Translate a file using greedy search
  python src/generate.py --checkpoint checkpoints/checkpoint_ibt_latest.pt \
                         --source-file data/raw/test.tr \
                         --output-file results/test.en.out \
                         --direction tr-en --strategy greedy

  # Translate using beam search (beam size 4)
  python src/generate.py --checkpoint checkpoints/checkpoint_ibt_latest.pt \
                         --source-file data/raw/test.tr \
                         --output-file results/test.en.out \
                         --direction tr-en --strategy beam --beam-size 4
"""

import os
import sys
import argparse
from typing import List

import torch
import sentencepiece as spm
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import load_config, get_device, setup_logging, add_base_args
from src.model import build_model

log = setup_logging("generate")


# ════════════════════════════════════════════════════════════
#  BEAM SEARCH DECODING
# ════════════════════════════════════════════════════════════

@torch.no_grad()
def beam_search_decode(
    model: torch.nn.Module,
    src: torch.Tensor,
    src_key_padding_mask: torch.Tensor,
    beam_size: int = 4,
    max_len: int = 128,
    bos_id: int = 2,
    eos_id: int = 3,
    pad_id: int = 0,
    length_penalty: float = 0.6,
) -> torch.Tensor:
    """Batched beam search with EOS handling and length normalization."""
    if beam_size == 1:
        return model.greedy_decode(src, src_key_padding_mask, max_len, bos_id, eos_id)

    model.eval()
    batch_size = src.size(0)
    device = src.device

    memory = model.encode(src, src_key_padding_mask=src_key_padding_mask)
    alive_seqs = torch.full((batch_size, beam_size, 1), bos_id, dtype=torch.long, device=device)
    alive_log_probs = torch.zeros(batch_size, beam_size, device=device)
    alive_log_probs[:, 1:] = float('-inf')

    memory_beam = memory.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(batch_size * beam_size, memory.size(1), -1)
    if src_key_padding_mask is not None:
        src_mask_beam = src_key_padding_mask.unsqueeze(1).expand(-1, beam_size, -1).reshape(batch_size * beam_size, -1)
    else:
        src_mask_beam = None

    finished_seqs = torch.full(
        (batch_size, beam_size, max_len),
        pad_id,
        dtype=torch.long,
        device=device,
    )
    finished_scores = torch.full((batch_size, beam_size), float('-inf'), device=device)

    for _ in range(1, max_len):
        flat_seqs = alive_seqs.reshape(batch_size * beam_size, -1)
        tgt_mask = model._causal_mask(flat_seqs.size(1), device)
        tgt_pad_mask = (flat_seqs == pad_id)

        decoded = model.decoder(
            model._embed(flat_seqs),
            memory_beam,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask_beam,
        )
        logits = model.output_proj(decoded[:, -1, :])
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs = log_probs.reshape(batch_size, beam_size, -1)

        next_scores = alive_log_probs.unsqueeze(-1) + log_probs
        next_scores_flat = next_scores.reshape(batch_size, -1)
        candidate_count = min(beam_size * 2, next_scores_flat.size(1))
        topk_scores, topk_indices = torch.topk(
            next_scores_flat, candidate_count, dim=-1
        )

        vocab_size = log_probs.size(-1)
        prev_beam_indices = topk_indices // vocab_size
        next_tokens = topk_indices % vocab_size

        prev_seqs = torch.gather(
            alive_seqs,
            1,
            prev_beam_indices.unsqueeze(-1).expand(-1, -1, alive_seqs.size(-1))
        )
        candidate_seqs = torch.cat(
            [prev_seqs, next_tokens.unsqueeze(-1)], dim=-1
        )
        is_eos = next_tokens == eos_id

        normalized_scores = topk_scores / (
            ((5.0 + candidate_seqs.size(-1)) / 6.0) ** length_penalty
        )
        new_finished_scores = normalized_scores.masked_fill(
            ~is_eos, float("-inf")
        )
        new_finished_seqs = torch.full(
            (batch_size, candidate_count, max_len),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        new_finished_seqs[:, :, :candidate_seqs.size(-1)] = candidate_seqs

        combined_finished_scores = torch.cat(
            [finished_scores, new_finished_scores], dim=1
        )
        combined_finished_seqs = torch.cat(
            [finished_seqs, new_finished_seqs], dim=1
        )
        finished_scores, finished_indices = torch.topk(
            combined_finished_scores, beam_size, dim=1
        )
        finished_seqs = torch.gather(
            combined_finished_seqs,
            1,
            finished_indices.unsqueeze(-1).expand(-1, -1, max_len),
        )

        alive_candidate_scores = topk_scores.masked_fill(
            is_eos, float("-inf")
        )
        alive_log_probs, alive_indices = torch.topk(
            alive_candidate_scores, beam_size, dim=1
        )
        alive_seqs = torch.gather(
            candidate_seqs,
            1,
            alive_indices.unsqueeze(-1).expand(
                -1, -1, candidate_seqs.size(-1)
            ),
        )

    best_finished_idx = finished_scores.argmax(dim=-1)
    best_finished = torch.gather(
        finished_seqs,
        1,
        best_finished_idx.view(-1, 1, 1).expand(-1, 1, max_len),
    ).squeeze(1)

    alive_scores = alive_log_probs / (
        ((5.0 + alive_seqs.size(-1)) / 6.0) ** length_penalty
    )
    best_alive_idx = alive_scores.argmax(dim=-1)
    best_alive = torch.gather(
        alive_seqs,
        1,
        best_alive_idx.view(-1, 1, 1).expand(-1, 1, alive_seqs.size(-1)),
    ).squeeze(1)
    best_alive_padded = torch.full(
        (batch_size, max_len), pad_id, dtype=torch.long, device=device
    )
    best_alive_padded[:, :best_alive.size(1)] = best_alive

    has_finished = torch.isfinite(finished_scores[:, 0])
    return torch.where(
        has_finished.unsqueeze(1), best_finished, best_alive_padded
    )


# ════════════════════════════════════════════════════════════
#  TRANSLATION INTERFACE
# ════════════════════════════════════════════════════════════

def translate_lines(
    lines: List[str],
    model: torch.nn.Module,
    sp: spm.SentencePieceProcessor,
    src_lang: str,
    device: torch.device,
    strategy: str = "greedy",
    beam_size: int = 4,
    batch_size: int = 32,
    max_len: int = 128,
    length_penalty: float = 0.6,
) -> List[str]:
    """Translate text while conditioning on the requested target language."""
    model.eval()

    normalized_src_lang = src_lang.lower()
    target_by_source = {"tr": "en", "en": "tr"}
    if normalized_src_lang not in target_by_source:
        raise ValueError(f"Unsupported source language: {src_lang}")
    target_lang = target_by_source[normalized_src_lang]
    target_lang_id = sp.piece_to_id(f"<{target_lang.upper()}>")
    if target_lang_id == sp.unk_id():
        raise ValueError(
            f"Target language token <{target_lang.upper()}> is missing"
        )

    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    pad_id = sp.pad_id()

    outputs = []
    
    for i in tqdm(range(0, len(lines), batch_size), desc="Translating"):
        batch_lines = lines[i:i + batch_size]
        
        # Encode with SP
        batch_ids = [sp.encode(line, out_type=int) for line in batch_lines]
        
        # The prefix tells the shared model which language to generate.
        batch_ids = [[target_lang_id] + ids for ids in batch_ids]
        
        # Pad sequences
        max_batch_len = min(max_len, max(len(ids) for ids in batch_ids))
        padded = []
        for ids in batch_ids:
            if len(ids) > max_batch_len:
                padded.append(ids[:max_batch_len])
            else:
                padded.append(ids + [pad_id] * (max_batch_len - len(ids)))
                
        src_tensor = torch.tensor(padded, dtype=torch.long, device=device)
        src_mask = (src_tensor == pad_id)
        
        with torch.no_grad():
            if strategy == "greedy":
                pred_ids = model.greedy_decode(
                    src_tensor, src_mask, max_len=max_len, bos_id=bos_id, eos_id=eos_id
                )
            else:
                pred_ids = beam_search_decode(
                    model, src_tensor, src_mask, beam_size=beam_size, 
                    max_len=max_len, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
                    length_penalty=length_penalty,
                )
                
        # Decode SP IDs back to text
        pred_ids_list = pred_ids.cpu().tolist()
        for seq in pred_ids_list:
            # truncate at EOS
            if eos_id in seq:
                seq = seq[:seq.index(eos_id)]
            # remove BOS if present at start
            if seq and seq[0] == bos_id:
                seq = seq[1:]
            seq = [token_id for token_id in seq if token_id != pad_id]
            text = sp.decode(seq)
            outputs.append(text)
            
    return outputs


def main():
    parser = argparse.ArgumentParser(description="UNMT Generation Script")
    add_base_args(parser)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--source-file", type=str, required=True, help="Input text file")
    parser.add_argument("--output-file", type=str, required=True, help="Output translations file")
    parser.add_argument("--direction", type=str, required=True, choices=["tr-en", "en-tr"], help="Translation direction")
    parser.add_argument("--strategy", type=str, default="greedy", choices=["greedy", "beam"], help="Decoding strategy")
    parser.add_argument("--beam-size", type=int, default=4, help="Beam size (if strategy is beam)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--length-penalty", type=float, default=0.6, help="Beam-search length penalty")
    args = parser.parse_args()

    cfg = load_config(args.config, args.base_dir)
    device = get_device()
    
    log.info(f"Loading SentencePiece model...")
    proc_dir = os.path.join(cfg["paths"]["base_dir"], cfg["data"]["processed_subdir"])
    sp_path = os.path.join(proc_dir, f"{cfg['vocab']['model_prefix']}.model")
    sp = spm.SentencePieceProcessor(model_file=sp_path)
    
    log.info(f"Loading checkpoint {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = build_model(cfg, vocab_size=sp.get_piece_size(), pad_id=sp.pad_id())
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    src_lang = args.direction.split("-")[0]
    
    log.info(f"Reading {args.source_file}...")
    with open(args.source_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    log.info(f"Translating {len(lines)} lines ({args.direction}) using {args.strategy} search...")
    translations = translate_lines(
        lines, model, sp, src_lang, device, 
        strategy=args.strategy,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
        max_len=cfg["model"]["max_seq_len"],
        length_penalty=args.length_penalty,
    )
    
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for t in translations:
            f.write(t + "\n")
            
    log.info(f"Done! Translations saved to {args.output_file}")


if __name__ == "__main__":
    main()
