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
  python src/evaluate.py --checkpoint checkpoints/checkpoint_latest.pt --strategy beam
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


def get_flores_data():
    """Load FLORES-200 devtest for TR-EN."""
    log.info("Loading FLORES-200 devtest dataset...")
    # 'tur_Latn' = Turkish, 'eng_Latn' = English
    ds_tr = load_dataset("facebook/flores", "tur_Latn", split="devtest")
    ds_en = load_dataset("facebook/flores", "eng_Latn", split="devtest")
    
    # Sentence list
    tr_sentences = ds_tr["sentence"]
    en_sentences = ds_en["sentence"]
    
    assert len(tr_sentences) == len(en_sentences), "Mismatched FLORES-200 lengths!"
    return tr_sentences, en_sentences


def evaluate_direction(src_lines, ref_lines, model, sp, src_lang, device, args):
    """Generate translations and compute BLEU / chrF."""
    log.info(f"Translating {len(src_lines)} sentences ({src_lang.upper()} -> Target)...")
    
    hypotheses = translate_lines(
        src_lines, model, sp, src_lang, device, 
        strategy=args.strategy, beam_size=args.beam_size, batch_size=args.batch_size
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
    parser.add_argument("--beam-size", type=int, default=4, help="Beam size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for generation")
    parser.add_argument("--output-csv", type=str, default="results/evaluation.csv", help="Where to save results")
    args = parser.parse_args()

    cfg = load_config(args.config, args.base_dir)
    device = get_device()
    
    # 1. Load Data
    tr_refs, en_refs = get_flores_data()
    
    # 2. Load SP
    proc_dir = resolve_path(cfg, "data", "processed_subdir")
    sp_path = os.path.join(proc_dir, f"{cfg['vocab']['model_prefix']}.model")
    sp = spm.SentencePieceProcessor(model_file=sp_path)
    
    # 3. Load Model
    log.info(f"Loading checkpoint {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
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
    out_path = resolve_path(cfg, "paths", "base_dir")
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
