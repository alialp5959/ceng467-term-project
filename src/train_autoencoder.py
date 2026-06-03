#!/usr/bin/env python3
"""
CENG467 UNMT — Denoising Autoencoder Training
===============================================
Trains a shared Transformer encoder–decoder as a denoising autoencoder
on Turkish and English monolingual corpora.

Training procedure (per step):
  1. Sample a batch from one language (TR and EN alternate).
  2. Corrupt the tokens with shuffle / dropout / masking.
  3. Feed the corrupted sequence to the encoder (with language prefix).
  4. The decoder reconstructs the original clean sequence (teacher forcing).
  5. Minimise cross-entropy reconstruction loss.

Key Colab-friendly features:
  • Gradient accumulation (effective batch = 32 × 4 = 128)
  • FP16 mixed precision (halves VRAM on T4)
  • Checkpoint every N steps (model + optimizer + scheduler → Drive)
  • Full resume capability from latest checkpoint
  • Warmup + cosine LR decay schedule

Usage:
  python src/train_autoencoder.py                            # Default (Colab)
  python src/train_autoencoder.py --base-dir .               # Local dev
  python src/train_autoencoder.py --epochs 5 --batch-size 16 # Override

Google Colab:
  Drive is auto-mounted.  Checkpoints are saved to the shared
  Drive folder so the other teammate can resume training.
"""

import math
import os
import sys
import time
import argparse
from typing import Optional

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

# ── Project imports ──────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import (
    load_config,
    resolve_path,
    ensure_dir,
    set_seed,
    get_device,
    setup_logging,
    mount_drive_if_colab,
    add_base_args,
)
from src.model import UNMTTransformer, build_model
from src.noise import make_noise_fn
from src.dataset import (
    MonolingualDataset,
    build_dataloader,
    prepare_dae_batch,
    alternating_loader,
)

log = setup_logging("train_dae")


# ════════════════════════════════════════════════════════════
#  LEARNING-RATE SCHEDULE
# ════════════════════════════════════════════════════════════

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int
) -> LambdaLR:
    """Linear warmup → cosine decay to 0."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ════════════════════════════════════════════════════════════
#  CHECKPOINTING
# ════════════════════════════════════════════════════════════

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    save_dir: str,
    tag: str = "latest",
) -> str:
    """Persist full training state to Google Drive (or local disk).

    Saves both a tagged file (``checkpoint_{tag}.pt``) and a
    ``checkpoint_latest.pt`` symlink / copy for easy resume.
    """
    ensure_dir(save_dir)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }

    tag_path = os.path.join(save_dir, f"checkpoint_{tag}.pt")
    torch.save(state, tag_path)

    # Always update "latest"
    latest_path = os.path.join(save_dir, "checkpoint_latest.pt")
    torch.save(state, latest_path)

    log.info(f"  💾 Checkpoint saved → {tag_path}")
    return tag_path


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: GradScaler,
    save_dir: str,
    device: torch.device,
) -> tuple:
    """Resume from latest checkpoint if it exists.

    Returns ``(start_epoch, global_step)``.
    """
    latest = os.path.join(save_dir, "checkpoint_latest.pt")
    if not os.path.exists(latest):
        log.info("  No checkpoint found — training from scratch.")
        return 0, 0

    log.info(f"  Resuming from {latest} …")
    ckpt = torch.load(latest, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    epoch = ckpt["epoch"]
    step = ckpt["global_step"]
    log.info(f"  ✓ Resumed at epoch {epoch}, step {step:,}")
    return epoch, step


# ════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    tr_loader,
    en_loader,
    tr_lang_id: int,
    en_lang_id: int,
    noise_fn,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: GradScaler,
    cfg: dict,
    epoch: int,
    global_step: int,
    device: torch.device,
    ckpt_dir: str,
) -> tuple:
    """Train for one full epoch (alternating TR / EN DAE batches).

    Returns ``(avg_loss, global_step)``.
    """
    model.train()

    tcfg = cfg["training"]
    pad_id = 0
    bos_id = 2
    eos_id = 3
    accum_steps = tcfg["gradient_accumulation_steps"]
    log_every = tcfg["log_every_n_steps"]
    ckpt_every = tcfg["checkpoint_every_n_steps"]
    use_fp16 = tcfg["fp16"] and device.type == "cuda"
    vocab_size = model.vocab_size

    total_loss = 0.0
    n_tokens = 0
    n_batches = 0
    t0 = time.time()

    optimizer.zero_grad()

    for batch, lang_id in alternating_loader(
        tr_loader, en_loader, tr_lang_id, en_lang_id
    ):
        # ── Prepare DAE batch ────────────────────────────────
        enc_in, dec_in, dec_tgt, enc_mask, dec_mask = prepare_dae_batch(
            batch, lang_id, noise_fn,
            pad_id=pad_id, bos_id=bos_id, eos_id=eos_id, device=device,
        )

        # ── Forward + loss ───────────────────────────────────
        with autocast(enabled=use_fp16):
            logits = model(
                enc_in, dec_in,
                src_key_padding_mask=enc_mask,
                tgt_key_padding_mask=dec_mask,
            )
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                dec_tgt.reshape(-1),
                ignore_index=pad_id,
            )
            scaled_loss = loss / accum_steps

        # ── Backward ─────────────────────────────────────────
        scaler.scale(scaled_loss).backward()

        # Count non-pad tokens for accurate PPL
        n_tok = (dec_tgt != pad_id).sum().item()
        total_loss += loss.item() * n_tok
        n_tokens += n_tok
        n_batches += 1

        # ── Optimizer step (every accum_steps) ───────────────
        if n_batches % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

        # ── Logging ──────────────────────────────────────────
        if n_batches % (log_every * accum_steps) == 0 and n_tokens > 0:
            avg = total_loss / n_tokens
            ppl = math.exp(min(avg, 100))
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            tok_per_sec = n_tokens / elapsed
            log.info(
                f"  E{epoch+1} step {global_step:>7,} │ "
                f"loss {avg:.4f} │ ppl {ppl:8.2f} │ "
                f"lr {lr:.6f} │ {tok_per_sec:,.0f} tok/s"
            )

        # ── Periodic checkpoint ──────────────────────────────
        if (
            ckpt_every > 0
            and n_batches % (ckpt_every * accum_steps) == 0
            and n_batches > 0
        ):
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, global_step, ckpt_dir,
                tag=f"step_{global_step}",
            )

    # Final loss for the epoch
    avg_epoch_loss = total_loss / max(n_tokens, 1)
    return avg_epoch_loss, global_step


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CENG467 UNMT — Denoising Autoencoder Training",
    )
    add_base_args(parser)
    parser.add_argument("--epochs", type=int, default=None, help="Override max_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate")
    args = parser.parse_args()

    # ── Environment ──────────────────────────────────────────
    mounted = mount_drive_if_colab()
    cfg = load_config(config_path=args.config, base_dir_override=args.base_dir)
    set_seed(cfg["project"]["seed"])
    device = get_device()

    tcfg = cfg["training"]
    if args.epochs:
        tcfg["max_epochs"] = args.epochs
    if args.batch_size:
        tcfg["batch_size"] = args.batch_size
    if args.lr:
        tcfg["learning_rate"] = args.lr

    base_dir = cfg["paths"]["base_dir"]
    proc_dir = resolve_path(cfg, "data", "processed_subdir")
    ckpt_dir = os.path.join(base_dir, tcfg["checkpoint_subdir"])

    log.info("=" * 65)
    log.info("  CENG467 UNMT — Denoising Autoencoder Training")
    log.info(f"  Device   : {device}")
    log.info(f"  Base dir : {base_dir}")
    log.info(f"  Colab    : {'yes' if mounted else 'no'}")
    log.info(f"  Epochs   : {tcfg['max_epochs']}")
    log.info(f"  Batch    : {tcfg['batch_size']} × {tcfg['gradient_accumulation_steps']} "
             f"= {tcfg['batch_size'] * tcfg['gradient_accumulation_steps']} effective")
    log.info(f"  FP16     : {tcfg['fp16']}")
    log.info("=" * 65)

    # ── SentencePiece model ──────────────────────────────────
    sp_model_path = os.path.join(proc_dir, f"{cfg['vocab']['model_prefix']}.model")
    if not os.path.exists(sp_model_path):
        log.error(f"SentencePiece model not found: {sp_model_path}")
        log.error("Run  python src/preprocess.py --step vocab  first.")
        sys.exit(1)

    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    vocab_size = sp.get_piece_size()
    pad_id = sp.pad_id()       # 0
    bos_id = sp.bos_id()       # 2
    eos_id = sp.eos_id()       # 3
    tr_lang_id = sp.piece_to_id("<TR>")
    en_lang_id = sp.piece_to_id("<EN>")
    mask_id = sp.piece_to_id("<MASK>")

    log.info(f"  Vocab    : {vocab_size:,} pieces")
    log.info(f"  Tokens   : PAD={pad_id} BOS={bos_id} EOS={eos_id} "
             f"<TR>={tr_lang_id} <EN>={en_lang_id} <MASK>={mask_id}")

    # ── Datasets ─────────────────────────────────────────────
    max_len = cfg["model"]["max_seq_len"] - 2  # room for LANG + BOS/EOS
    tr_data = MonolingualDataset(
        os.path.join(proc_dir, "clean.tr.txt"), sp_model_path,
        max_len=max_len, cache_dir=proc_dir,
    )
    en_data = MonolingualDataset(
        os.path.join(proc_dir, "clean.en.txt"), sp_model_path,
        max_len=max_len, cache_dir=proc_dir,
    )
    log.info(f"  TR data  : {len(tr_data):,} sentences")
    log.info(f"  EN data  : {len(en_data):,} sentences")

    tr_loader = build_dataloader(tr_data, tcfg["batch_size"], shuffle=True)
    en_loader = build_dataloader(en_data, tcfg["batch_size"], shuffle=True)

    # ── Model ────────────────────────────────────────────────
    model = build_model(cfg, vocab_size, pad_id=pad_id).to(device)
    log.info(f"  Model    : {model.count_parameters():,} parameters")

    # ── Optimizer / Scheduler / Scaler ───────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    steps_per_epoch = min(len(tr_loader), len(en_loader)) * 2 // tcfg["gradient_accumulation_steps"]
    total_steps = steps_per_epoch * tcfg["max_epochs"]

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=tcfg["warmup_steps"],
        num_training_steps=total_steps,
    )

    scaler = GradScaler(enabled=tcfg["fp16"] and device.type == "cuda")

    # ── Noise function ───────────────────────────────────────
    special_ids = {pad_id, bos_id, eos_id, tr_lang_id, en_lang_id}
    noise_fn = make_noise_fn(cfg["noise"], special_ids, mask_id)

    # ── Resume from checkpoint ───────────────────────────────
    start_epoch, global_step = load_checkpoint(
        model, optimizer, scheduler, scaler, ckpt_dir, device
    )

    # ── Training ─────────────────────────────────────────────
    log.info("")
    log.info(f"  Steps/epoch    : ~{steps_per_epoch:,}")
    log.info(f"  Total steps    : ~{total_steps:,}")
    log.info(f"  Warmup steps   : {tcfg['warmup_steps']:,}")
    log.info(f"  Starting epoch : {start_epoch}")
    log.info("")

    best_loss = float("inf")

    for epoch in range(start_epoch, tcfg["max_epochs"]):
        log.info(f"{'━' * 65}")
        log.info(f"  EPOCH {epoch + 1} / {tcfg['max_epochs']}")
        log.info(f"{'━' * 65}")

        t_start = time.time()
        avg_loss, global_step = train_one_epoch(
            model=model,
            tr_loader=tr_loader,
            en_loader=en_loader,
            tr_lang_id=tr_lang_id,
            en_lang_id=en_lang_id,
            noise_fn=noise_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            cfg=cfg,
            epoch=epoch,
            global_step=global_step,
            device=device,
            ckpt_dir=ckpt_dir,
        )
        t_elapsed = time.time() - t_start

        ppl = math.exp(min(avg_loss, 100))
        log.info(
            f"  Epoch {epoch + 1} done │ "
            f"loss {avg_loss:.4f} │ ppl {ppl:.2f} │ "
            f"{t_elapsed / 60:.1f} min"
        )

        # ── End-of-epoch checkpoint ──────────────────────────
        tag = f"epoch_{epoch + 1}"
        if avg_loss < best_loss:
            best_loss = avg_loss
            tag = f"best_epoch_{epoch + 1}"
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch + 1, global_step, ckpt_dir, tag=tag,
        )

    log.info("")
    log.info("═" * 65)
    log.info(f"  ✅  DAE training complete.  Best loss: {best_loss:.4f}")
    log.info(f"  Checkpoints at: {ckpt_dir}")
    log.info("═" * 65)


if __name__ == "__main__":
    main()
