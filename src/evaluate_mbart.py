#!/usr/bin/env python3
"""
CENG467 UNMT — mBART-50 Zero-Shot Evaluation
=============================================
Evaluates a pre-trained multilingual BART model on FLORES-200.
"""

import os
import sys
import argparse
import pandas as pd
import sacrebleu
import torch
from tqdm import tqdm
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import load_config, get_device, setup_logging, add_base_args, ensure_dir
from src.evaluate import get_flores_data

log = setup_logging("evaluate_mbart")


def evaluate_mbart_direction(src_lines, ref_lines, model, tokenizer, src_lang, tgt_lang, device, batch_size=16):
    """Translate lines using mBART-50 and evaluate using Sacrebleu."""
    log.info(f"Translating {len(src_lines)} sentences ({src_lang.upper()} -> {tgt_lang.upper()}) using mBART-50...")
    
    # Map 'tr' / 'en' to mBART-50 language codes
    mbart_src = "tr_TR" if src_lang == "tr" else "en_XX"
    mbart_tgt = "en_XX" if src_lang == "tr" else "tr_TR"
    
    tokenizer.src_lang = mbart_src
    
    hypotheses = []
    
    # Process in batches
    for i in tqdm(range(0, len(src_lines), batch_size), desc=f"{src_lang.upper()}->{tgt_lang.upper()}"):
        batch_src = src_lines[i : i + batch_size]
        
        inputs = tokenizer(batch_src, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[mbart_tgt],
                num_beams=4,
                max_length=128
            )
            
        batch_hyps = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        hypotheses.extend(batch_hyps)
        
    log.info("Computing metrics...")
    refs = [ref_lines]
    bleu = sacrebleu.corpus_bleu(hypotheses, refs)
    chrf = sacrebleu.corpus_chrf(hypotheses, refs)
    
    return hypotheses, bleu, chrf


def main():
    parser = argparse.ArgumentParser(description="mBART Evaluation")
    add_base_args(parser)
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--output-csv", type=str, default="results/eval_mbart_zeroshot.csv", help="Where to save results")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N FLORES examples for quick sanity checks")
    args = parser.parse_args()

    cfg = load_config(args.config, args.base_dir)
    device = get_device()
    
    # 1. Load Data
    tr_refs, en_refs = get_flores_data(cfg)
    if args.limit is not None:
        log.info(f"Using only first {args.limit} examples for evaluation.")
        tr_refs = tr_refs[:args.limit]
        en_refs = en_refs[:args.limit]
        
    # 2. Load Model & Tokenizer
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    log.info(f"Loading pre-trained {model_name}...")
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    
    # 3. Evaluate TR -> EN
    en_hyps, tr2en_bleu, tr2en_chrf = evaluate_mbart_direction(
        tr_refs, en_refs, model, tokenizer, "tr", "en", device, batch_size=args.batch_size
    )
    log.info(f"TR->EN | BLEU: {tr2en_bleu.score:.2f} | chrF: {tr2en_chrf.score:.2f}")
    
    # 4. Evaluate EN -> TR
    tr_hyps, en2tr_bleu, en2tr_chrf = evaluate_mbart_direction(
        en_refs, tr_refs, model, tokenizer, "en", "tr", device, batch_size=args.batch_size
    )
    log.info(f"EN->TR | BLEU: {en2tr_bleu.score:.2f} | chrF: {en2tr_chrf.score:.2f}")
    
    # 5. Save results to CSV for analysis
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
    
    # 6. Print Final Table
    print("\n" + "="*50)
    print("mBART-50 ZERO-SHOT EVALUATION RESULTS")
    print("="*50)
    print(f"{'Direction':<10} | {'BLEU':<10} | {'chrF':<10}")
    print("-" * 50)
    print(f"{'TR -> EN':<10} | {tr2en_bleu.score:<10.2f} | {tr2en_chrf.score:<10.2f}")
    print(f"{'EN -> TR':<10} | {en2tr_bleu.score:<10.2f} | {en2tr_chrf.score:<10.2f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
