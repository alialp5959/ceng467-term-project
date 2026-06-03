#!/usr/bin/env python3
"""
CENG467 UNMT Project — Preprocessing Pipeline
==============================================
Full pipeline for downloading, cleaning, vocabulary training,
and tokenization of monolingual CC-100 data (Turkish + English).

Pipeline Steps
--------------
  download  — Stream CC-100 via HuggingFace and save 1M sentences per lang
  clean     — Unicode norm, dedup, length filter, Turkish I/ı handling,
              web-noise removal
  vocab     — Train a joint SentencePiece (Unigram / BPE) vocabulary
  tokenize  — Segment cleaned corpora with the trained SP model

Usage
-----
  # Full pipeline (recommended first run)
  python src/preprocess.py --step all

  # Individual steps
  python src/preprocess.py --step download
  python src/preprocess.py --step clean
  python src/preprocess.py --step vocab
  python src/preprocess.py --step tokenize

  # Local development (no Google Drive)
  python src/preprocess.py --step all --base-dir .

Google Colab
------------
  The script auto-mounts Google Drive if it detects a Colab runtime.
  All outputs go to the shared Drive folder defined in config.yaml
  so both teammates can access data & checkpoints instantly.

Outputs
-------
  {base_dir}/data/raw/cc100.{tr,en}.txt       — downloaded raw text
  {base_dir}/data/processed/clean.{tr,en}.txt  — cleaned text
  {base_dir}/data/processed/spm.model          — SentencePiece model
  {base_dir}/data/processed/spm.vocab          — SentencePiece vocab
  {base_dir}/data/processed/tokenized.{tr,en}.txt — tokenized text
"""

import os
import re
import sys
import argparse
import unicodedata
from typing import Optional

# ── Ensure project root is on sys.path ──────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import (
    load_config,
    resolve_path,
    ensure_dir,
    set_seed,
    setup_logging,
    mount_drive_if_colab,
    add_base_args,
)

log = setup_logging("preprocess")


# ════════════════════════════════════════════════════════════
#  TURKISH LANGUAGE UTILITIES
# ════════════════════════════════════════════════════════════

# Translation tables for Turkish-aware case conversion.
# Standard Python str.lower() maps  I → i  which is WRONG in Turkish.
# Correct:  İ → i   and   I → ı
_TR_LOWER_MAP = str.maketrans({
    "İ": "i",
    "I": "ı",
    "Ç": "ç",
    "Ğ": "ğ",
    "Ö": "ö",
    "Ş": "ş",
    "Ü": "ü",
})


def turkish_lower(text: str) -> str:
    """Lowercase *text* using Turkish locale rules.

    Critical difference from ``str.lower()``:
      - ``İ`` (U+0130 Latin Capital Letter I With Dot Above) → ``i``
      - ``I`` (U+0049 Latin Capital Letter I)                 → ``ı`` (dotless)

    All other characters fall through to Python's default ``str.lower()``.
    """
    return text.translate(_TR_LOWER_MAP).lower()


# ════════════════════════════════════════════════════════════
#  REGEX PATTERNS FOR NOISE FILTERING
# ════════════════════════════════════════════════════════════

_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_HTML = re.compile(r"<[^>]+>")
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
_RE_MULTI_SPACE = re.compile(r"[ \t]+")           # Tabs & spaces (not newlines)
_RE_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Characters considered "normal" for both Turkish and English text.
# Everything outside this set counts towards the special-char ratio.
_RE_NORMAL_CHAR = re.compile(
    r"[a-zA-ZçğıöşüÇĞİÖŞÜâîûêôÂÎÛÊÔ0-9"   # Letters + digits
    r"\s"                                       # Whitespace
    r".,;:!?'\"()\-–—/&%@#\+="                 # Common punctuation
    r"]"
)


# ════════════════════════════════════════════════════════════
#  STEP 1 — DOWNLOAD CC-100
# ════════════════════════════════════════════════════════════

def download_cc100(cfg: dict) -> None:
    """Stream CC-100 monolingual data and save 1M sentences per language.

    Uses the HuggingFace ``datasets`` library in streaming mode so we
    never need to download the full (100+ GB) corpus to disk.  Paragraphs
    are split on newlines; only lines with >10 characters are kept.

    The function is **resumable**: if an output file already has ≥ max_sentences
    lines it is skipped entirely.
    """
    from datasets import load_dataset

    raw_dir = ensure_dir(resolve_path(cfg, "data", "raw_subdir"))
    max_sents = cfg["data"]["max_sentences"]

    for lang in cfg["data"]["languages"]:
        out_path = os.path.join(raw_dir, f"cc100.{lang}.txt")

        # ── Resume check ────────────────────────────────────
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                existing = sum(1 for _ in f)
            if existing >= max_sents:
                log.info(
                    f"[{lang.upper()}] {out_path} already has {existing:,} "
                    f"sentences (need {max_sents:,}) → skipping download"
                )
                continue
            log.info(
                f"[{lang.upper()}] Found {existing:,} sentences in "
                f"{out_path}, need {max_sents:,} → re-downloading"
            )

        log.info(
            f"[{lang.upper()}] Streaming CC-100 "
            f"(target: {max_sents:,} sentences)…"
        )

        ds = load_dataset("cc100", lang=lang, split="train", streaming=True, trust_remote_code=True)

        count = 0
        with open(out_path, "w", encoding="utf-8") as fout:
            for example in ds:
                # CC-100 "text" may contain multiple paragraph lines
                for line in example["text"].split("\n"):
                    line = line.strip()
                    if len(line) > 10:
                        fout.write(line + "\n")
                        count += 1
                        if count >= max_sents:
                            break
                if count >= max_sents:
                    break
                if count > 0 and count % 100_000 == 0:
                    log.info(
                        f"  [{lang.upper()}] {count:,} / {max_sents:,}…"
                    )

        log.info(
            f"[{lang.upper()}] ✓ Downloaded {count:,} sentences → {out_path}"
        )


# ════════════════════════════════════════════════════════════
#  STEP 2 — CLEAN
# ════════════════════════════════════════════════════════════

def clean_line(line: str, cfg_pre: dict, lang: str) -> Optional[str]:
    """Apply all cleaning rules to a single line.

    Returns the cleaned string, or ``None`` if the line should be
    discarded.  The cleaning order follows the project report §2.3:

    1. Unicode normalisation (NFKC)
    2. Control-character removal
    3. URL / HTML / e-mail stripping
    4. Whitespace normalisation
    5. Special-character ratio filter
    6. Token-count length filter (3 – 100 whitespace tokens)
    7. Optional Turkish-aware lowercasing
    """
    # 1. Unicode normalisation
    norm_form = cfg_pre.get("unicode_norm", "NFKC")
    line = unicodedata.normalize(norm_form, line)

    # 2. Strip & reject empty
    line = line.strip()
    if not line:
        return None

    # 3a. Remove control characters
    line = _RE_CONTROL_CHARS.sub("", line)

    # 3b. Remove URLs
    if cfg_pre.get("remove_urls", True):
        line = _RE_URL.sub(" ", line)

    # 3c. Remove HTML tags
    if cfg_pre.get("remove_html", True):
        line = _RE_HTML.sub(" ", line)

    # 3d. Remove e-mail addresses
    line = _RE_EMAIL.sub(" ", line)

    # 4. Collapse whitespace
    line = _RE_MULTI_SPACE.sub(" ", line).strip()
    if not line:
        return None

    # 5. Special-character ratio filter
    max_ratio = cfg_pre.get("max_special_char_ratio", 0.3)
    if len(line) > 0:
        normal_count = len(_RE_NORMAL_CHAR.findall(line))
        special_ratio = 1.0 - (normal_count / len(line))
        if special_ratio > max_ratio:
            return None

    # 6. Token-count filter
    tokens = line.split()
    min_tok = cfg_pre.get("min_tokens", 3)
    max_tok = cfg_pre.get("max_tokens", 100)
    if len(tokens) < min_tok or len(tokens) > max_tok:
        return None

    # 7. Optional lowercasing
    if cfg_pre.get("lowercase", False):
        if lang == "tr" and cfg_pre.get("turkish_lowercase", True):
            line = turkish_lower(line)
        else:
            line = line.lower()

    return line


def clean_corpus(cfg: dict) -> None:
    """Clean the raw monolingual corpora for both languages.

    For each language file in ``data/raw/cc100.{lang}.txt``:
      • Apply ``clean_line()`` to every line.
      • Optionally deduplicate via a hash set.
      • Write surviving lines to ``data/processed/clean.{lang}.txt``.
      • Print detailed statistics at the end.
    """
    raw_dir = resolve_path(cfg, "data", "raw_subdir")
    proc_dir = ensure_dir(resolve_path(cfg, "data", "processed_subdir"))
    cfg_pre = cfg["preprocessing"]

    for lang in cfg["data"]["languages"]:
        in_path = os.path.join(raw_dir, f"cc100.{lang}.txt")
        out_path = os.path.join(proc_dir, f"clean.{lang}.txt")

        if not os.path.exists(in_path):
            log.warning(
                f"[{lang.upper()}] Raw file not found: {in_path} → skipping"
            )
            continue

        log.info(f"[{lang.upper()}] Cleaning {in_path} …")

        # Dedup set (hash-based for memory efficiency on 1M+ lines)
        seen = set() if cfg_pre.get("dedup", True) else None
        kept, dropped, dupes = 0, 0, 0

        with (
            open(in_path, "r", encoding="utf-8") as fin,
            open(out_path, "w", encoding="utf-8") as fout,
        ):
            for raw_line in fin:
                cleaned = clean_line(raw_line, cfg_pre, lang)

                if cleaned is None:
                    dropped += 1
                    continue

                # ── Deduplication ────────────────────────────
                if seen is not None:
                    h = hash(cleaned)
                    if h in seen:
                        dupes += 1
                        continue
                    seen.add(h)

                fout.write(cleaned + "\n")
                kept += 1

                if kept % 200_000 == 0:
                    log.info(
                        f"  [{lang.upper()}] kept={kept:,}  "
                        f"dropped={dropped:,}  dupes={dupes:,}"
                    )

        log.info(
            f"[{lang.upper()}] ✓ Cleaning complete — "
            f"kept={kept:,}  dropped={dropped:,}  dupes={dupes:,}"
        )
        log.info(f"  → {out_path}")


# ════════════════════════════════════════════════════════════
#  STEP 3 — TRAIN JOINT SENTENCEPIECE VOCABULARY
# ════════════════════════════════════════════════════════════

def train_vocab(cfg: dict) -> None:
    """Train a joint SentencePiece vocabulary on cleaned TR + EN data.

    Key design decisions (from the approved plan):
      • **Shared vocabulary** – a single SP model covers both languages
        so the encoder learns to map them into the same latent space.
      • **Unigram** model by default (performs well on agglutinative Turkish);
        switchable to BPE via ``config.yaml → vocab.type``.
      • **32 000** subword pieces (ablation: 16k / 64k).
      • Special tokens: ``<TR>``, ``<EN>``, ``<MASK>`` are reserved via
        ``user_defined_symbols`` so they always have a fixed ID.
      • ``byte_fallback=True`` — graceful handling of rare / unknown chars.
    """
    import sentencepiece as spm

    proc_dir = resolve_path(cfg, "data", "processed_subdir")
    cfg_vocab = cfg["vocab"]

    # ── Gather cleaned input files ───────────────────────────
    input_files = []
    for lang in cfg["data"]["languages"]:
        p = os.path.join(proc_dir, f"clean.{lang}.txt")
        if os.path.exists(p):
            input_files.append(p)
        else:
            log.warning(f"Cleaned file not found: {p}")

    if not input_files:
        log.error("No cleaned data found. Run the 'clean' step first.")
        return

    model_prefix = os.path.join(proc_dir, cfg_vocab.get("model_prefix", "spm"))

    # ── Collect special tokens ───────────────────────────────
    special_tokens = list(cfg["model"]["language_tokens"])  # <TR>, <EN>
    special_tokens.append(cfg["noise"]["mask_token"])       # <MASK>

    log.info("Training SentencePiece vocabulary …")
    log.info(f"  Type             : {cfg_vocab['type']}")
    log.info(f"  Vocab size       : {cfg_vocab['size']:,}")
    log.info(f"  Char coverage    : {cfg_vocab['character_coverage']}")
    log.info(f"  Special tokens   : {special_tokens}")
    log.info(f"  Input files      : {input_files}")
    log.info(f"  Output prefix    : {model_prefix}")

    spm.SentencePieceTrainer.train(
        input=",".join(input_files),
        model_prefix=model_prefix,
        vocab_size=cfg_vocab["size"],
        model_type=cfg_vocab["type"],
        character_coverage=cfg_vocab["character_coverage"],
        # ── Reserved token IDs ───────────────────────────────
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=special_tokens,
        # ── Training knobs ───────────────────────────────────
        input_sentence_size=2_000_000,      # Sample up to 2M lines
        shuffle_input_sentence=True,
        num_threads=os.cpu_count() or 4,
        byte_fallback=True,
        # ── Normalisation inside SP ──────────────────────────
        normalization_rule_name="nfkc",
        split_digits=True,                  # Treat each digit separately
        allow_whitespace_only_pieces=False,
    )

    # ── Verify the model ─────────────────────────────────────
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    log.info(f"✓ Vocabulary trained — {sp.get_piece_size():,} pieces")

    # Quick encode sanity-check
    test_sentences = {
        "TR": "Türkiye'nin başkenti Ankara'dır ve nüfusu 85 milyondur.",
        "EN": "The capital of Turkey is Ankara and its population is 85 million.",
    }
    for label, sent in test_sentences.items():
        pieces = sp.encode(sent, out_type=str)
        ids = sp.encode(sent, out_type=int)
        log.info(f"  [{label}] \"{sent}\"")
        log.info(f"       pieces : {pieces}")
        log.info(f"       ids    : {ids}")
        log.info(f"       #tokens: {len(pieces)}")


# ════════════════════════════════════════════════════════════
#  STEP 4 — TOKENIZE CLEANED CORPORA
# ════════════════════════════════════════════════════════════

def tokenize_corpus(cfg: dict) -> None:
    """Segment cleaned text files with the trained SentencePiece model.

    For each language, reads ``data/processed/clean.{lang}.txt`` and
    writes ``data/processed/tokenized.{lang}.txt`` where each line
    contains space-separated subword pieces.
    """
    import sentencepiece as spm

    proc_dir = resolve_path(cfg, "data", "processed_subdir")
    cfg_vocab = cfg["vocab"]
    model_path = os.path.join(
        proc_dir, f"{cfg_vocab['model_prefix']}.model"
    )

    if not os.path.exists(model_path):
        log.error(
            f"SentencePiece model not found: {model_path} — "
            "run the 'vocab' step first."
        )
        return

    sp = spm.SentencePieceProcessor(model_file=model_path)
    log.info(f"Loaded SentencePiece model ({sp.get_piece_size():,} pieces)")

    for lang in cfg["data"]["languages"]:
        in_path = os.path.join(proc_dir, f"clean.{lang}.txt")
        out_path = os.path.join(proc_dir, f"tokenized.{lang}.txt")

        if not os.path.exists(in_path):
            log.warning(f"Clean file not found: {in_path} → skipping")
            continue

        log.info(f"[{lang.upper()}] Tokenizing {in_path} …")

        count = 0
        total_pieces = 0
        with (
            open(in_path, "r", encoding="utf-8") as fin,
            open(out_path, "w", encoding="utf-8") as fout,
        ):
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                pieces = sp.encode(line, out_type=str)
                fout.write(" ".join(pieces) + "\n")
                count += 1
                total_pieces += len(pieces)

                if count % 200_000 == 0:
                    avg = total_pieces / count
                    log.info(
                        f"  [{lang.upper()}] {count:,} sentences "
                        f"(avg {avg:.1f} pieces/sent)"
                    )

        avg_pieces = total_pieces / max(count, 1)
        log.info(
            f"[{lang.upper()}] ✓ Tokenisation complete — "
            f"{count:,} sentences, avg {avg_pieces:.1f} pieces/sent"
        )
        log.info(f"  → {out_path}")


# ════════════════════════════════════════════════════════════
#  PIPELINE ORCHESTRATION
# ════════════════════════════════════════════════════════════

STEPS = {
    "download": download_cc100,
    "clean":    clean_corpus,
    "vocab":    train_vocab,
    "tokenize": tokenize_corpus,
}

_STEP_ORDER = ["download", "clean", "vocab", "tokenize"]


def main():
    parser = argparse.ArgumentParser(
        description="CENG467 UNMT — Preprocessing & Vocabulary Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_base_args(parser)
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["all"] + list(STEPS.keys()),
        help="Which pipeline step to run (default: all)",
    )
    args = parser.parse_args()

    # ── Google Colab auto-mount ──────────────────────────────
    mounted = mount_drive_if_colab()

    # ── Load config ──────────────────────────────────────────
    cfg = load_config(
        config_path=args.config,
        base_dir_override=args.base_dir,
    )
    set_seed(cfg["project"]["seed"])

    base_dir = cfg["paths"]["base_dir"]
    log.info("=" * 62)
    log.info("  CENG467 UNMT — Preprocessing Pipeline")
    log.info(f"  Base dir : {base_dir}")
    log.info(f"  Colab    : {'yes (Drive mounted)' if mounted else 'no'}")
    log.info(f"  Step     : {args.step}")
    log.info("=" * 62)

    # ── Ensure base directories exist ────────────────────────
    ensure_dir(resolve_path(cfg, "data", "raw_subdir"))
    ensure_dir(resolve_path(cfg, "data", "processed_subdir"))

    # ── Execute ──────────────────────────────────────────────
    steps_to_run = _STEP_ORDER if args.step == "all" else [args.step]

    for step_name in steps_to_run:
        log.info("")
        log.info(f"{'─' * 50}")
        log.info(f"  STEP: {step_name.upper()}")
        log.info(f"{'─' * 50}")
        STEPS[step_name](cfg)

    log.info("")
    log.info("═" * 62)
    log.info("  ✅  Pipeline complete.")
    log.info("═" * 62)


if __name__ == "__main__":
    main()
