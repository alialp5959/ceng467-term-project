#!/usr/bin/env python3
"""
CENG467 UNMT — Iterative Back-Translation (IBT)
===============================================
The core algorithm of Unsupervised NMT. It alternates between generating
synthetic parallel data and training the model on it.

Pipeline (per iteration):
  1. Generate TR->EN (synthetic EN) from monolingual TR
  2. Generate EN->TR (synthetic TR) from monolingual EN
  3. Train the model using:
     a) Cross-Entropy on synthetic EN -> real TR
     b) Cross-Entropy on synthetic TR -> real EN
     c) DAE on real TR (optional, but stabilizes training)
     d) DAE on real EN (optional, but stabilizes training)

Usage:
  python src/backtranslate.py --checkpoint checkpoints/checkpoint_latest.pt --iterations 3
"""

import os
import sys
import argparse
import time

import torch
import torch.nn.functional as F
import sentencepiece as spm
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import load_config, resolve_path, ensure_dir, get_device, setup_logging, add_base_args
from src.model import build_model
from src.dataset import MonolingualDataset, pad_sequences, alternating_loader
from src.generate import translate_lines
from src.noise import make_noise_fn

log = setup_logging("backtranslate")


# ════════════════════════════════════════════════════════════
#  PARALLEL DATASET (SYNTHETIC -> REAL)
# ════════════════════════════════════════════════════════════

class SyntheticParallelDataset(Dataset):
    """Loads synthetic source and real target sentences."""
    def __init__(self, synthetic_lines: list, real_lines: list, sp: spm.SentencePieceProcessor):
        assert len(synthetic_lines) == len(real_lines)
        self.synthetic = synthetic_lines
        self.real = real_lines
        self.sp = sp

    def __len__(self):
        return len(self.synthetic)

    def __getitem__(self, idx):
        src_ids = self.sp.encode(self.synthetic[idx], out_type=int)
        tgt_ids = self.sp.encode(self.real[idx], out_type=int)
        return src_ids, tgt_ids

def collate_parallel(batch, pad_id, lang_id, bos_id, eos_id):
    """Prepares parallel batch: [LANG] + src -> encoder, [BOS] + tgt -> decoder."""
    src_list, tgt_list = [], []
    for src, tgt in batch:
        src_list.append([lang_id] + src)
        tgt_list.append([bos_id] + tgt + [eos_id])
        
    src_tensor = pad_sequences(src_list, pad_id)
    tgt_tensor = pad_sequences(tgt_list, pad_id)
    return src_tensor, tgt_tensor


# ════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ════════════════════════════════════════════════════════════

def train_iteration(
    cfg, model, sp, device, 
    tr_dataset, en_dataset, 
    synthetic_en, real_tr, 
    synthetic_tr, real_en,
    iteration
):
    """Run one training iteration of IBT."""
    m = cfg["training"]
    batch_size = m["batch_size"]
    grad_acc = m.get("gradient_accumulation_steps", 1)
    
    # Optimizers
    optimizer = torch.optim.AdamW(model.parameters(), lr=m["learning_rate"], weight_decay=m["weight_decay"])
    scaler = GradScaler(enabled=m["fp16"])
    
    # Datasets
    pad_id = sp.pad_id()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    tr_lang_id = sp.piece_to_id("<TR>")
    en_lang_id = sp.piece_to_id("<EN>")
    
    # 1. Synthetic EN -> Real TR
    ds_en_tr = SyntheticParallelDataset(synthetic_en, real_tr, sp)
    dl_en_tr = DataLoader(
        ds_en_tr, batch_size=batch_size, shuffle=True, drop_last=True,
        collate_fn=lambda b: collate_parallel(b, pad_id, en_lang_id, bos_id, eos_id)
    )
    
    # 2. Synthetic TR -> Real EN
    ds_tr_en = SyntheticParallelDataset(synthetic_tr, real_en, sp)
    dl_tr_en = DataLoader(
        ds_tr_en, batch_size=batch_size, shuffle=True, drop_last=True,
        collate_fn=lambda b: collate_parallel(b, pad_id, tr_lang_id, bos_id, eos_id)
    )
    
    model.train()
    total_loss = 0
    steps = 0
    
    # Determine shortest loader length to interleave
    min_len = min(len(dl_en_tr), len(dl_tr_en))
    
    it_en_tr = iter(dl_en_tr)
    it_tr_en = iter(dl_tr_en)
    
    log.info(f"Training IBT Iteration {iteration} for {min_len * 2} batches...")
    
    pbar = tqdm(range(min_len * 2), desc=f"IBT Iter {iteration}")
    optimizer.zero_grad()
    
    for i in pbar:
        # Alternate batches
        if i % 2 == 0:
            src, tgt = next(it_en_tr)
            src, tgt = src.to(device), tgt.to(device)
        else:
            src, tgt = next(it_tr_en)
            src, tgt = src.to(device), tgt.to(device)
            
        src_mask = (src == pad_id)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_mask = (tgt_in == pad_id)
        
        with autocast(enabled=m["fp16"]):
            logits = model(src, tgt_in, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1), ignore_index=pad_id)
            loss = loss / grad_acc
            
        scaler.scale(loss).backward()
        total_loss += loss.item() * grad_acc
        steps += 1
        
        if (i + 1) % grad_acc == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            pbar.set_postfix({"loss": total_loss / steps})
            
    # Save checkpoint
    out_dir = ensure_dir(resolve_path(cfg, cfg["training"]["checkpoint_subdir"]))
    ckpt_path = os.path.join(out_dir, f"checkpoint_ibt_iter{iteration}.pt")
    torch.save({
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }, ckpt_path)
    log.info(f"Saved IBT checkpoint to {ckpt_path}")


# ════════════════════════════════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="UNMT Iterative Back-Translation")
    add_base_args(parser)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to initial DAE checkpoint")
    parser.add_argument("--iterations", type=int, default=3, help="Number of IBT iterations")
    parser.add_argument("--sample-size", type=int, default=200000, help="Number of sentences to translate per iteration")
    args = parser.parse_args()

    cfg = load_config(args.config, args.base_dir)
    device = get_device()
    
    proc_dir = resolve_path(cfg, "data", "processed_subdir")
    synth_dir = ensure_dir(resolve_path(cfg, "backtranslation", "synthetic_subdir"))
    
    log.info("Loading SentencePiece...")
    sp = spm.SentencePieceProcessor(model_file=os.path.join(proc_dir, f"{cfg['vocab']['model_prefix']}.model"))
    
    log.info("Loading datasets...")
    tr_dataset = MonolingualDataset(os.path.join(proc_dir, "clean.tr.txt"), sp)
    en_dataset = MonolingualDataset(os.path.join(proc_dir, "clean.en.txt"), sp)
    
    model = build_model(cfg, vocab_size=sp.get_piece_size(), pad_id=sp.pad_id())
    log.info(f"Loading initial checkpoint {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    
    for it in range(1, args.iterations + 1):
        log.info(f"\n{'='*50}\n IBT ITERATION {it}\n{'='*50}")
        
        # 1. Sample lines to translate
        # Read from clean files directly (much faster than decoding SP IDs)
        log.info("Sampling sentences for back-translation...")
        with open(os.path.join(proc_dir, "clean.tr.txt"), "r", encoding="utf-8") as f:
            tr_lines = [next(f).strip() for _ in range(args.sample_size)]
        with open(os.path.join(proc_dir, "clean.en.txt"), "r", encoding="utf-8") as f:
            en_lines = [next(f).strip() for _ in range(args.sample_size)]
            
        # 2. Generate Synthetic EN from Real TR
        log.info(f"Generating Synthetic EN from TR (Iter {it})...")
        synth_en = translate_lines(tr_lines, model, sp, src_lang="tr", device=device, strategy="greedy", batch_size=cfg["training"]["batch_size"])
        with open(os.path.join(synth_dir, f"synth.en.iter{it}.txt"), "w", encoding="utf-8") as f:
            for line in synth_en: f.write(line + "\n")
            
        # 3. Generate Synthetic TR from Real EN
        log.info(f"Generating Synthetic TR from EN (Iter {it})...")
        synth_tr = translate_lines(en_lines, model, sp, src_lang="en", device=device, strategy="greedy", batch_size=cfg["training"]["batch_size"])
        with open(os.path.join(synth_dir, f"synth.tr.iter{it}.txt"), "w", encoding="utf-8") as f:
            for line in synth_tr: f.write(line + "\n")
            
        # 4. Train Model
        log.info(f"Training on synthetic data (Iter {it})...")
        train_iteration(
            cfg, model, sp, device, 
            tr_dataset, en_dataset, 
            synth_en, tr_lines, 
            synth_tr, en_lines,
            it
        )
        
    log.info("IBT Complete!")

if __name__ == "__main__":
    main()
