#!/usr/bin/env python3
"""
CENG467 UNMT - Iterative Back-Translation (IBT)
================================================

Each iteration:
  1. Samples fresh Turkish and English monolingual sentences.
  2. Generates synthetic translations using the target-language prefix.
  3. Trains on both synthetic parallel directions.
  4. Mixes in denoising-autoencoder loss to reduce language forgetting.
  5. Saves a resumable checkpoint.
"""

import argparse
import hashlib
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dataset import (
    MonolingualDataset,
    build_dataloader,
    pad_sequences,
    prepare_dae_batch,
)
from src.generate import translate_lines
from src.model import build_model
from src.noise import make_noise_fn
from src.utils import (
    add_base_args,
    ensure_dir,
    get_device,
    load_config,
    resolve_path,
    set_seed,
    setup_logging,
)

log = setup_logging("backtranslate")


class SyntheticParallelDataset(Dataset):
    """Synthetic source sentences paired with real target sentences."""

    def __init__(
        self,
        synthetic_lines: List[str],
        real_lines: List[str],
        sp: spm.SentencePieceProcessor,
    ):
        if len(synthetic_lines) != len(real_lines):
            raise ValueError("Synthetic and real corpora must have equal lengths")
        self.synthetic = synthetic_lines
        self.real = real_lines
        self.sp = sp

    def __len__(self):
        return len(self.synthetic)

    def __getitem__(self, idx):
        return (
            self.sp.encode(self.synthetic[idx], out_type=int),
            self.sp.encode(self.real[idx], out_type=int),
        )


def collate_parallel(
    batch,
    pad_id: int,
    target_lang_id: int,
    bos_id: int,
    eos_id: int,
    max_len: int,
):
    """Build ``[TARGET_LANG] + source`` and teacher-forced target tensors."""
    src_list = []
    dec_in_list = []
    dec_tgt_list = []
    for src_ids, tgt_ids in batch:
        src_list.append([target_lang_id] + src_ids[: max_len - 1])
        dec_in_list.append([bos_id] + tgt_ids[: max_len - 1])
        dec_tgt_list.append(tgt_ids[: max_len - 1] + [eos_id])
    return pad_sequences(src_list, pad_id), pad_sequences(dec_in_list, pad_id), pad_sequences(dec_tgt_list, pad_id)


def model_fingerprint(model: torch.nn.Module) -> str:
    """Cheap fingerprint used to reject stale synthetic-translation caches."""
    digest = hashlib.sha256()
    for index, (name, tensor) in enumerate(model.state_dict().items()):
        if index >= 12:
            break
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        sample = tensor.detach().reshape(-1)[:256].float().cpu().numpy()
        digest.update(sample.tobytes())
    return digest.hexdigest()[:16]


def index_fingerprint(indices: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(indices, dtype=np.int64).tobytes()
    ).hexdigest()[:16]


def sample_lines(
    dataset: MonolingualDataset,
    sp: spm.SentencePieceProcessor,
    indices: np.ndarray,
) -> List[str]:
    return [sp.decode(dataset[int(index)]) for index in indices]


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def write_lines(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


def cache_paths(
    synth_dir: str, cache_version: str, iteration: int
) -> Tuple[str, str, str]:
    prefix = "ibt_{}_iter{}".format(cache_version, iteration)
    return (
        os.path.join(synth_dir, prefix + ".en.txt"),
        os.path.join(synth_dir, prefix + ".tr.txt"),
        os.path.join(synth_dir, prefix + ".json"),
    )


def load_synthetic_cache(
    paths: Tuple[str, str, str],
    expected_metadata: dict,
) -> Tuple[List[str], List[str]]:
    synth_en_path, synth_tr_path, metadata_path = paths
    if not all(os.path.exists(path) for path in paths):
        return None, None

    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata != expected_metadata:
            log.info("Synthetic cache metadata changed; regenerating.")
            return None, None

        synth_en = read_lines(synth_en_path)
        synth_tr = read_lines(synth_tr_path)
        expected_size = expected_metadata["sample_size"]
        if len(synth_en) != expected_size or len(synth_tr) != expected_size:
            log.warning("Synthetic cache is incomplete; regenerating.")
            return None, None
        return synth_en, synth_tr
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        log.warning("Could not load synthetic cache: %s", exc)
        return None, None


def save_synthetic_cache(
    paths: Tuple[str, str, str],
    metadata: dict,
    synth_en: List[str],
    synth_tr: List[str],
) -> None:
    synth_en_path, synth_tr_path, metadata_path = paths
    write_lines(synth_en_path, synth_en)
    write_lines(synth_tr_path, synth_tr)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def translation_loss(model, src, dec_in, dec_tgt, pad_id):
    src_mask = src == pad_id
    tgt_mask = dec_in == pad_id
    logits = model(
        src,
        dec_in,
        src_key_padding_mask=src_mask,
        tgt_key_padding_mask=tgt_mask,
    )
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        dec_tgt.reshape(-1),
        ignore_index=pad_id,
    )


def train_iteration(
    cfg,
    model,
    sp,
    device,
    tr_dataset,
    en_dataset,
    synthetic_en,
    real_tr,
    synthetic_tr,
    real_en,
    noise_fn,
    optimizer,
    scaler,
    iteration,
):
    """Run one IBT iteration with parallel and DAE objectives."""
    training_cfg = cfg["training"]
    ibt_cfg = cfg["backtranslation"]
    batch_size = training_cfg["batch_size"]
    grad_acc = training_cfg.get("gradient_accumulation_steps", 1)
    dae_weight = ibt_cfg.get("dae_weight", 0.5)
    max_len = cfg["model"]["max_seq_len"]
    use_fp16 = training_cfg["fp16"] and device.type == "cuda"

    pad_id = sp.pad_id()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    tr_lang_id = sp.piece_to_id("<TR>")
    en_lang_id = sp.piece_to_id("<EN>")

    # Synthetic English -> real Turkish: request Turkish output.
    en_tr_dataset = SyntheticParallelDataset(synthetic_en, real_tr, sp)
    en_tr_loader = DataLoader(
        en_tr_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda batch: collate_parallel(
            batch, pad_id, tr_lang_id, bos_id, eos_id, max_len
        ),
    )

    # Synthetic Turkish -> real English: request English output.
    tr_en_dataset = SyntheticParallelDataset(synthetic_tr, real_en, sp)
    tr_en_loader = DataLoader(
        tr_en_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda batch: collate_parallel(
            batch, pad_id, en_lang_id, bos_id, eos_id, max_len
        ),
    )

    tr_dae_loader = build_dataloader(
        tr_dataset, batch_size=batch_size, shuffle=True
    )
    en_dae_loader = build_dataloader(
        en_dataset, batch_size=batch_size, shuffle=True
    )

    parallel_batches = min(len(en_tr_loader), len(tr_en_loader))
    if parallel_batches == 0:
        raise ValueError("Not enough synthetic data for one complete batch")

    en_tr_iter = iter(en_tr_loader)
    tr_en_iter = iter(tr_en_loader)
    tr_dae_iter = iter(tr_dae_loader)
    en_dae_iter = iter(en_dae_loader)

    model.train()
    optimizer.zero_grad()
    total_batches = parallel_batches * 2
    total_bt_loss = 0.0
    total_dae_loss = 0.0
    optimizer_steps = 0

    progress = tqdm(range(total_batches), desc="IBT Iter {}".format(iteration))
    for batch_index in progress:
        if batch_index % 2 == 0:
            src, dec_in, dec_tgt = next(en_tr_iter)
            dae_batch = next(tr_dae_iter)
            dae_lang_id = tr_lang_id
        else:
            src, dec_in, dec_tgt = next(tr_en_iter)
            dae_batch = next(en_dae_iter)
            dae_lang_id = en_lang_id

        src = src.to(device)
        dec_in = dec_in.to(device)
        dec_tgt = dec_tgt.to(device)
        dae_inputs = prepare_dae_batch(
            dae_batch,
            dae_lang_id,
            noise_fn,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            device=device,
        )
        dae_src, dae_tgt_in, dae_tgt_out, dae_src_mask, dae_tgt_mask = dae_inputs

        group_start = (batch_index // grad_acc) * grad_acc
        accumulation_divisor = min(grad_acc, total_batches - group_start)

        with autocast(enabled=use_fp16):
            bt_loss = translation_loss(model, src, dec_in, dec_tgt, pad_id)
            dae_logits = model(
                dae_src,
                dae_tgt_in,
                src_key_padding_mask=dae_src_mask,
                tgt_key_padding_mask=dae_tgt_mask,
            )
            dae_loss = F.cross_entropy(
                dae_logits.reshape(-1, dae_logits.size(-1)),
                dae_tgt_out.reshape(-1),
                ignore_index=pad_id,
            )
            combined_loss = (
                bt_loss + dae_weight * dae_loss
            ) / accumulation_divisor

        scaler.scale(combined_loss).backward()
        total_bt_loss += bt_loss.item()
        total_dae_loss += dae_loss.item()

        if (
            (batch_index + 1) % grad_acc == 0
            or batch_index + 1 == total_batches
        ):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            optimizer_steps += 1

        completed = batch_index + 1
        progress.set_postfix(
            bt="{:.3f}".format(total_bt_loss / completed),
            dae="{:.3f}".format(total_dae_loss / completed),
        )

    return {
        "bt_loss": total_bt_loss / total_batches,
        "dae_loss": total_dae_loss / total_batches,
        "optimizer_steps": optimizer_steps,
    }


def save_ibt_checkpoint(
    cfg,
    model,
    optimizer,
    scaler,
    iteration,
    metrics,
):
    checkpoint_dir = ensure_dir(
        os.path.join(
            cfg["paths"]["base_dir"],
            cfg["training"]["checkpoint_subdir"],
        )
    )
    state = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "metrics": metrics,
        "cache_version": cfg["backtranslation"].get(
            "cache_version", "target_lang_v2"
        ),
    }
    iteration_path = os.path.join(
        checkpoint_dir, "checkpoint_ibt_iter{}.pt".format(iteration)
    )
    latest_path = os.path.join(checkpoint_dir, "checkpoint_ibt_latest.pt")
    torch.save(state, iteration_path)
    torch.save(state, latest_path)
    log.info("Saved IBT checkpoint to %s", iteration_path)
    return iteration_path


def main():
    parser = argparse.ArgumentParser(
        description="UNMT Iterative Back-Translation"
    )
    add_base_args(parser)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="DAE checkpoint or checkpoint_ibt_iterN.pt to resume",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Final IBT iteration number",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Monolingual sentences sampled per language and iteration",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=None,
        help="Override automatic resume iteration",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Ignore compatible synthetic translation caches",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.base_dir)
    set_seed(cfg["project"]["seed"])
    device = get_device()
    ibt_cfg = cfg["backtranslation"]
    training_cfg = cfg["training"]

    final_iteration = args.iterations or ibt_cfg.get("num_iterations", 10)
    sample_size = args.sample_size or ibt_cfg.get("sample_size", 25000)
    cache_version = ibt_cfg.get("cache_version", "target_lang_v2")

    proc_dir = resolve_path(cfg, "data", "processed_subdir")
    synth_dir = ensure_dir(
        resolve_path(cfg, "backtranslation", "synthetic_subdir")
    )
    sp_model_path = os.path.join(
        proc_dir, "{}.model".format(cfg["vocab"]["model_prefix"])
    )

    log.info("Loading SentencePiece model and monolingual datasets...")
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    dataset_max_len = cfg["model"]["max_seq_len"] - 2
    max_len = cfg["model"]["max_seq_len"] - 2

    train_files = cfg.get("data", {}).get("train_files", {})

    def _resolve_train_file(value, default_name):
        name = value or default_name

        # Absolute path ise direkt kullan.
        if os.path.isabs(name):
            return name

        # Config içinde "data/processed/..." gibi path varsa base_dir ile çöz.
        if os.path.dirname(name):
            return os.path.join(cfg["paths"]["base_dir"], name)

        # Sadece dosya adıysa processed dir altında çöz.
        return os.path.join(proc_dir, name)

    tr_train_file = _resolve_train_file(train_files.get("tr"), "clean.tr.txt")
    en_train_file = _resolve_train_file(train_files.get("en"), "clean.en.txt")

    log.info("  TR file  : %s", tr_train_file)
    log.info("  EN file  : %s", en_train_file)

    tr_dataset = MonolingualDataset(
        tr_train_file,
        sp_model_path,
        max_len=max_len,
        cache_dir=proc_dir,
    )
    en_dataset = MonolingualDataset(
        en_train_file,
        sp_model_path,
        max_len=max_len,
        cache_dir=proc_dir,
    )

    model = build_model(
        cfg, vocab_size=sp.get_piece_size(), pad_id=sp.pad_id()
    ).to(device)
    log.info("Loading checkpoint %s", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    ibt_learning_rate = ibt_cfg.get(
        "learning_rate", training_cfg["learning_rate"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ibt_learning_rate,
        weight_decay=training_cfg["weight_decay"],
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    use_fp16 = training_cfg["fp16"] and device.type == "cuda"
    scaler = GradScaler(enabled=use_fp16)

    completed_iteration = int(checkpoint.get("iteration", 0))
    if completed_iteration > 0:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        log.info("Resuming after IBT iteration %d", completed_iteration)

    start_iteration = (
        args.start_iteration
        if args.start_iteration is not None
        else completed_iteration + 1
    )
    if start_iteration > final_iteration:
        log.info("Nothing to do: checkpoint already reached iteration %d", final_iteration)
        return

    pad_id = sp.pad_id()
    special_ids = {
        pad_id,
        sp.bos_id(),
        sp.eos_id(),
        sp.piece_to_id("<TR>"),
        sp.piece_to_id("<EN>"),
    }
    noise_fn = make_noise_fn(
        cfg["noise"], special_ids, sp.piece_to_id("<MASK>")
    )

    for iteration in range(start_iteration, final_iteration + 1):
        log.info("=" * 60)
        log.info("IBT ITERATION %d / %d", iteration, final_iteration)
        log.info("=" * 60)

        iteration_sample_size = min(
            sample_size, len(tr_dataset), len(en_dataset)
        )
        rng = np.random.default_rng(cfg["project"]["seed"] + iteration)
        tr_indices = rng.choice(
            len(tr_dataset), size=iteration_sample_size, replace=False
        )
        en_indices = rng.choice(
            len(en_dataset), size=iteration_sample_size, replace=False
        )
        real_tr = sample_lines(tr_dataset, sp, tr_indices)
        real_en = sample_lines(en_dataset, sp, en_indices)

        fingerprint = model_fingerprint(model)
        metadata = {
            "cache_version": cache_version,
            "iteration": iteration,
            "model_fingerprint": fingerprint,
            "sample_size": iteration_sample_size,
            "tr_indices": index_fingerprint(tr_indices),
            "en_indices": index_fingerprint(en_indices),
            "prefix_semantics": "target_language",
        }
        paths = cache_paths(synth_dir, cache_version, iteration)
        synthetic_en = synthetic_tr = None
        if not args.force_regenerate:
            synthetic_en, synthetic_tr = load_synthetic_cache(
                paths, metadata
            )

        if synthetic_en is None or synthetic_tr is None:
            log.info("Generating Turkish -> English synthetic data...")
            synthetic_en = translate_lines(
                real_tr,
                model,
                sp,
                src_lang="tr",
                device=device,
                strategy="greedy",
                batch_size=training_cfg["batch_size"],
                max_len=cfg["model"]["max_seq_len"],
            )
            log.info("Generating English -> Turkish synthetic data...")
            synthetic_tr = translate_lines(
                real_en,
                model,
                sp,
                src_lang="en",
                device=device,
                strategy="greedy",
                batch_size=training_cfg["batch_size"],
                max_len=cfg["model"]["max_seq_len"],
            )
            save_synthetic_cache(
                paths, metadata, synthetic_en, synthetic_tr
            )
        else:
            log.info("Using verified synthetic cache for iteration %d", iteration)

        metrics = train_iteration(
            cfg=cfg,
            model=model,
            sp=sp,
            device=device,
            tr_dataset=tr_dataset,
            en_dataset=en_dataset,
            synthetic_en=synthetic_en,
            real_tr=real_tr,
            synthetic_tr=synthetic_tr,
            real_en=real_en,
            noise_fn=noise_fn,
            optimizer=optimizer,
            scaler=scaler,
            iteration=iteration,
        )
        save_ibt_checkpoint(
            cfg, model, optimizer, scaler, iteration, metrics
        )
        log.info(
            "Iteration %d complete: BT loss %.4f, DAE loss %.4f",
            iteration,
            metrics["bt_loss"],
            metrics["dae_loss"],
        )

    log.info("IBT complete.")


if __name__ == "__main__":
    main()
