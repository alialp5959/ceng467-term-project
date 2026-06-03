#!/usr/bin/env python3
"""
CENG467 UNMT — Error Analysis Helper
====================================
Randomly samples sentences from the evaluation results for manual review.
Adds empty columns for rubric-compliant annotation.

Usage:
  python src/error_analysis.py --input results/evaluation.csv --sample 50
"""

import os
import sys
import argparse
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import setup_logging

log = setup_logging("error_analysis")

def main():
    parser = argparse.ArgumentParser(description="UNMT Error Analysis Sampler")
    parser.add_argument("--input", type=str, default="results/evaluation.csv", help="Input CSV from evaluate.py")
    parser.add_argument("--output", type=str, default="results/error_analysis_sample.csv", help="Output sample CSV")
    parser.add_argument("--sample", type=int, default=50, help="Number of samples to draw")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        log.error(f"Input file {args.input} not found! Run evaluate.py first.")
        sys.exit(1)
        
    log.info(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    
    log.info(f"Sampling {args.sample} sentences (Seed: {args.seed})...")
    sampled = df.sample(n=min(args.sample, len(df)), random_state=args.seed).copy()
    
    # We only need the TR->EN direction for the manual error analysis (or both, let's keep it simple)
    # Let's extract TR_Source, EN_Reference, EN_Hypothesis for analysis
    analysis_df = sampled[["TR_Source", "EN_Reference", "EN_Hypothesis"]].copy()
    
    # Add empty columns for annotation
    analysis_df["Adequacy_Score (1-5)"] = ""
    analysis_df["Fluency_Score (1-5)"] = ""
    analysis_df["Error_Category"] = ""
    analysis_df["Notes"] = ""
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    analysis_df.to_csv(args.output, index=False)
    
    log.info(f"Sampled file saved to {args.output}")
    print("\n" + "="*60)
    print("MANUAL ERROR ANALYSIS INSTRUCTIONS")
    print("="*60)
    print(f"Open '{args.output}' in Excel or Google Sheets.")
    print("For each of the translated sentences, fill in the following columns:")
    print("  1. Adequacy (1-5): How much meaning from the source is preserved?")
    print("  2. Fluency (1-5): How natural is the English translation?")
    print("  3. Error Category: Classify the main error if any:")
    print("      - OOV (Out of Vocabulary)")
    print("      - Semantik Kayma (Semantic Shift)")
    print("      - Sözdizimi (Syntax/Grammar Error)")
    print("      - Hallucination (Uydurma/Halüsinasyon)")
    print("      - İsim/Sayı (Entity/Number Error)")
    print("      - Tekrar (Repetition)")
    print("  4. Notes: Any other observation.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
