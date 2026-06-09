#!/usr/bin/env python3
"""
CENG467 UNMT — Evaluation Script
================================
Evaluates the trained UNMT models on the FLORES-200 devtest set.

Computes:
  • BLEU score
  • chrF score
for both TR->EN and EN->TR directions.

Usage:
  python src/evaluate.py --checkpoint checkpoints/checkpoint_ibt_latest.pt --strategy beam
"""

import os
import sys
import argparse
import pandas as pd
from datasets import load_dataset
import sentencepiece as spm
import sacrebleu
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import load_config, get_device, setup_logging, add_base_args, ensure_dir, resolve_path
from src.model import build_model
from src.generate import translate_lines

log = setup_logging("evaluate")


def get_flores_data(cfg):
    """Load FLORES-200 devtest for TR-EN."""
    from src.utils import download_flores_if_needed
    log.info("Loading FLORES-200 devtest dataset...")
    
    tr_path, en_path = download_flores_if_needed(cfg)
    
    with open(tr_path, "r", encoding="utf-8") as f:
        tr_sentences = [line.strip() for line in f]
    with open(en_path, "r", encoding="utf-8") as f:
        en_sentences = [line.strip() for line in f]
    
    assert len(tr_sentences) == len(en_sentences), "Mismatched FLORES-200 lengths!"
    return tr_sentences, en_sentences


def evaluate_direction(src_lines, ref_lines, model, sp, src_lang, device, args):
    """Generate translations and compute BLEU / chrF."""
    log.info(f"Translating {len(src_lines)} sentences ({src_lang.upper()} -> Target)...")
    
    hypotheses = translate_lines(
        src_lines, model, sp, src_lang, device, 
        strategy=args.strategy,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
        max_len=args.max_len,
        length_penalty=args.length_penalty,
    )
    
    log.info("Computing metrics...")
    # SacreBLEU expects a list of hypotheses and a list of lists of references
    refs = [ref_lines]
    
    bleu = sacrebleu.corpus_bleu(hypotheses, refs)
    chrf = sacrebleu.corpus_chrf(hypotheses, refs)
    
    return hypotheses, bleu, chrf


def main():
    parser = argparse.ArgumentParser(description="UNMT Evaluation")
    add_base_args(parser)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--strategy", type=str, default="beam", choices=["greedy", "beam"], help="Decoding strategy")
    parser.add_argument("--beam-size", type=int, default=None, help="Beam size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for generation")
    parser.add_argument("--max-len", type=int, default=None, help="Maximum output tokens")
    parser.add_argument("--length-penalty", type=float, default=None, help="Beam-search length penalty")
    parser.add_argument("--output-csv", type=str, default="results/evaluation.csv", help="Where to save results")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N FLORES examples for quick sanity checks")
    args = parser.parse_args()

    cfg = load_config(args.config, args.base_dir)
    device = get_device()
    args.beam_size = args.beam_size or cfg["evaluation"].get("beam_size", 4)
    args.max_len = args.max_len or cfg["model"]["max_seq_len"]
    if args.length_penalty is None:
        args.length_penalty = cfg["evaluation"].get("length_penalty", 0.6)
    
    # 1. Load Data
    tr_refs, en_refs = get_flores_data(cfg)
    if args.limit is not None:
        log.info(f"Using only first {args.limit} examples for evaluation.")
        tr_refs = tr_refs[:args.limit]
        en_refs = en_refs[:args.limit]
    
    # 2. Load SP
    proc_dir = resolve_path(cfg, "data", "processed_subdir")
    sp_path = os.path.join(proc_dir, f"{cfg['vocab']['model_prefix']}.model")
    sp = spm.SentencePieceProcessor(model_file=sp_path)
    
    # 3. Load Model
    log.info(f"Loading checkpoint {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "iteration" in checkpoint:
        log.info(f"Evaluating IBT iteration {checkpoint['iteration']}")
    else:
        log.warning("Checkpoint has no IBT iteration marker; this may be a DAE-only model.")
    model = build_model(cfg, vocab_size=sp.get_piece_size(), pad_id=sp.pad_id())
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # 4. Evaluate TR -> EN
    en_hyps, tr2en_bleu, tr2en_chrf = evaluate_direction(
        tr_refs, en_refs, model, sp, "tr", device, args
    )
    log.info(f"TR->EN | BLEU: {tr2en_bleu.score:.2f} | chrF: {tr2en_chrf.score:.2f}")
    
    # 5. Evaluate EN -> TR
    tr_hyps, en2tr_bleu, en2tr_chrf = evaluate_direction(
        en_refs, tr_refs, model, sp, "en", device, args
    )
    log.info(f"EN->TR | BLEU: {en2tr_bleu.score:.2f} | chrF: {en2tr_chrf.score:.2f}")
    
    # 6. Save results to CSV for analysis
    out_path = cfg["paths"]["base_dir"]
    out_file = os.path.join(out_path, args.output_csv)
    ensure_dir(os.path.dirname(out_file))
    
    df = pd.DataFrame({
        "TR_Source": tr_refs,
        "EN_Reference": en_refs,
        "EN_Hypothesis": en_hyps,
        "EN_Source": en_refs,
        "TR_Reference": tr_refs,
        "TR_Hypothesis": tr_hyps,
    })
    df.to_csv(out_file, index=False)
    log.info(f"Translations saved to {out_file}")
    
    # 7. Print Final Table
    print("\n" + "="*40)
    print("FINAL EVALUATION RESULTS")
    print("="*40)
    print(f"Decoding: {args.strategy.upper()} (Beam: {args.beam_size})")
    print("-" * 40)
    print(f"{'Direction':<10} | {'BLEU':<10} | {'chrF':<10}")
    print("-" * 40)
    print(f"{'TR -> EN':<10} | {tr2en_bleu.score:<10.2f} | {tr2en_chrf.score:<10.2f}")
    print(f"{'EN -> TR':<10} | {en2tr_bleu.score:<10.2f} | {en2tr_chrf.score:<10.2f}")
    print("="*40 + "\n")


if __name__ == "__main__":
    main()
