#!/usr/bin/env python3
"""
CENG467 UNMT — Word-by-Word Dictionary Translation Baseline
=============================================================
Translates each source word independently using a bilingual dictionary
built from MUSE-aligned fastText embeddings (Facebook Research).

This is a **non-trivial unsupervised baseline** that does NOT use any
parallel data.  It sits between the trivial Copy Baseline and the full
UNMT system, providing a meaningful lower bound for comparison.

Characteristics:
  • No reordering — source word order is preserved.
  • No morphological processing.
  • No context — each word translated in isolation.
  • Unknown words copied as-is.

Pipeline:
  1. Download aligned fastText vectors for TR and EN  (~600 MB + ~2 GB)
  2. Build bilingual dictionaries via cosine-similarity NN search
  3. Cache dictionaries to JSON for instant reuse
  4. Translate FLORES-200 devtest sentences word by word
  5. Evaluate with BLEU and chrF (both directions)

Usage:
  python src/baseline_word_by_word.py                           # Full run
  python src/baseline_word_by_word.py --base-dir .              # Local dev
  python src/baseline_word_by_word.py --skip-download           # Use cached
  python src/baseline_word_by_word.py --max-vocab 30000         # Smaller dict

Outputs:
  {base_dir}/data/dictionaries/dict_tr_en.json    — TR→EN dictionary
  {base_dir}/data/dictionaries/dict_en_tr.json    — EN→TR dictionary
  {base_dir}/results/wbw_baseline_results.csv     — BLEU & chrF scores
  {base_dir}/results/wbw_translations_tr_en.csv   — Sample translations
  {base_dir}/results/wbw_translations_en_tr.csv   — Sample translations
"""

import os
import re
import sys
import json
import argparse
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sacrebleu
from tqdm import tqdm

# ── Project root on sys.path ────────────────────────────────
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

log = setup_logging("wbw_baseline")


# ════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════

# Pre-aligned cross-lingual fastText embeddings (Facebook/MUSE).
# These are projected into a common vector space so that
# cosine similarity across languages ≈ translation equivalence.
ALIGNED_EMBEDDING_URLS = {
    "tr": "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.tr.align.vec",
    "en": "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec",
}


# ════════════════════════════════════════════════════════════
#  TURKISH TEXT UTILITIES
# ════════════════════════════════════════════════════════════

_TR_LOWER_MAP = str.maketrans({
    "İ": "i", "I": "ı",
    "Ç": "ç", "Ğ": "ğ", "Ö": "ö", "Ş": "ş", "Ü": "ü",
})


def turkish_lower(text: str) -> str:
    """Lowercase with Turkish rules (İ→i, I→ı)."""
    return text.translate(_TR_LOWER_MAP).lower()


# ════════════════════════════════════════════════════════════
#  1. DOWNLOAD ALIGNED EMBEDDINGS
# ════════════════════════════════════════════════════════════

def _download_with_progress(url: str, dest: str) -> None:
    """Download *url* to *dest* with a tqdm progress bar.

    Uses a ``.tmp`` suffix during download and renames on completion
    so that interrupted downloads don't leave partial files.
    """
    tmp = dest + ".tmp"

    # Get content length for the progress bar
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0))
    except Exception:
        total = 0

    desc = os.path.basename(dest)
    with tqdm(total=total or None, unit="B", unit_scale=True, desc=desc) as pbar:
        last_b = [0]

        def _hook(block_num: int, block_size: int, total_size: int):
            downloaded = block_num * block_size
            pbar.update(downloaded - last_b[0])
            last_b[0] = downloaded

        urllib.request.urlretrieve(url, tmp, reporthook=_hook)

    os.replace(tmp, dest)


def download_embeddings(cfg: dict, skip: bool = False) -> Dict[str, str]:
    """Download aligned fastText vectors for each language.

    Returns a ``{lang: filepath}`` mapping.  Skips languages whose
    files already exist on disk.
    """
    emb_dir = ensure_dir(
        os.path.join(cfg["paths"]["base_dir"], cfg["baseline"]["embeddings_subdir"])
    )
    paths = {}

    for lang in cfg["data"]["languages"]:
        url = ALIGNED_EMBEDDING_URLS.get(lang)
        if url is None:
            log.warning(f"No embedding URL for language '{lang}'")
            continue

        dest = os.path.join(emb_dir, f"wiki.{lang}.align.vec")
        paths[lang] = dest

        if os.path.exists(dest):
            log.info(f"[{lang.upper()}] Embeddings already cached → {dest}")
            continue

        if skip:
            log.warning(
                f"[{lang.upper()}] Embeddings not found and --skip-download "
                f"is set. File expected at: {dest}"
            )
            continue

        log.info(f"[{lang.upper()}] Downloading aligned embeddings …")
        log.info(f"  URL  : {url}")
        log.info(f"  Dest : {dest}")
        _download_with_progress(url, dest)
        log.info(f"[{lang.upper()}] ✓ Download complete.")

    return paths


# ════════════════════════════════════════════════════════════
#  2. LOAD EMBEDDINGS
# ════════════════════════════════════════════════════════════

def load_vectors(
    path: str,
    max_words: int = 50_000,
) -> Tuple[List[str], np.ndarray]:
    """Load the first *max_words* vectors from a fastText ``.vec`` file.

    Returns ``(words, vectors)`` where *vectors* is an
    ``(N, dim)`` float32 array, **L2-normalised** row-wise so that
    dot product = cosine similarity.
    """
    log.info(f"Loading embeddings: {path}  (max {max_words:,} words)")

    words: List[str] = []
    vecs: List[List[float]] = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split()
        # Header format: "<num_words> <dim>"
        dim = int(header[-1])

        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue  # skip malformed lines
            word = parts[0]
            # Skip purely numeric or single-char "words"
            if len(word) <= 1 and not word.isalpha():
                continue
            try:
                vec = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            words.append(word)
            vecs.append(vec)
            if len(words) >= max_words:
                break

    vectors = np.array(vecs, dtype=np.float32)

    # L2 normalise so dot product = cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms

    log.info(f"  Loaded {len(words):,} words × {vectors.shape[1]}d")
    return words, vectors


# ════════════════════════════════════════════════════════════
#  3. BUILD BILINGUAL DICTIONARY
# ════════════════════════════════════════════════════════════

def build_dictionary(
    src_words: List[str],
    src_vecs: np.ndarray,
    tgt_words: List[str],
    tgt_vecs: np.ndarray,
    batch_size: int = 1024,
) -> Dict[str, str]:
    """Build a {src_word → tgt_word} dictionary via cosine NN search.

    For each source word, finds the target word whose aligned embedding
    has the highest cosine similarity (= dot product, since vectors
    are already L2-normalised).

    Processes in batches to keep peak memory usage manageable on Colab.
    """
    n_src = len(src_words)
    dictionary: Dict[str, str] = {}

    log.info(
        f"Building dictionary: {n_src:,} src × {len(tgt_words):,} tgt "
        f"(batch={batch_size})"
    )

    for start in tqdm(range(0, n_src, batch_size), desc="Dict NN search"):
        end = min(start + batch_size, n_src)
        # Cosine similarities: (batch, tgt_size)
        sims = src_vecs[start:end] @ tgt_vecs.T
        best_indices = sims.argmax(axis=1)

        for j, idx in enumerate(best_indices):
            src_w = src_words[start + j]
            tgt_w = tgt_words[idx]
            # Skip identity mappings for common loanwords / numbers
            # (keeps them for genuinely shared words)
            dictionary[src_w] = tgt_w

    log.info(f"  Dictionary size: {len(dictionary):,} entries")
    return dictionary


def save_dictionary(d: Dict[str, str], path: str) -> None:
    """Persist dictionary as JSON."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=0)
    log.info(f"  Dictionary saved → {path}")


def load_dictionary(path: str) -> Dict[str, str]:
    """Load a cached JSON dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    log.info(f"  Loaded cached dictionary ({len(d):,} entries) ← {path}")
    return d


def get_or_build_dictionary(
    cfg: dict,
    src_lang: str,
    tgt_lang: str,
    emb_paths: Dict[str, str],
) -> Dict[str, str]:
    """Load cached dictionary or build from embeddings.

    The dictionary is cached as JSON under ``data/dictionaries/``
    so subsequent runs are instantaneous.
    """
    cache_dir = ensure_dir(
        os.path.join(cfg["paths"]["base_dir"], cfg["baseline"]["dict_cache_subdir"])
    )
    cache_path = os.path.join(cache_dir, f"dict_{src_lang}_{tgt_lang}.json")

    # ── Try cache first ──────────────────────────────────────
    if os.path.exists(cache_path):
        return load_dictionary(cache_path)

    # ── Build from embeddings ────────────────────────────────
    max_vocab = cfg["baseline"]["max_vocab"]
    batch_size = cfg["baseline"]["dict_batch_size"]

    src_path = emb_paths.get(src_lang)
    tgt_path = emb_paths.get(tgt_lang)
    if not src_path or not tgt_path:
        log.error(f"Embedding files missing for {src_lang}→{tgt_lang}")
        return {}

    src_words, src_vecs = load_vectors(src_path, max_words=max_vocab)
    tgt_words, tgt_vecs = load_vectors(tgt_path, max_words=max_vocab)

    dictionary = build_dictionary(
        src_words, src_vecs, tgt_words, tgt_vecs, batch_size=batch_size
    )

    save_dictionary(dictionary, cache_path)

    # Free memory
    del src_vecs, tgt_vecs
    return dictionary


# ════════════════════════════════════════════════════════════
#  4. WORD-BY-WORD TRANSLATION
# ════════════════════════════════════════════════════════════

# Regex to split a token into (prefix_punct, core_word, suffix_punct)
_RE_PUNCT_SPLIT = re.compile(
    r"^([^\w]*)(.+?)([^\w]*)$", re.UNICODE
)


def translate_sentence(
    sentence: str,
    dictionary: Dict[str, str],
    src_lang: str = "tr",
) -> str:
    """Translate a sentence word by word.

    For each token:
      1. Strip leading / trailing punctuation.
      2. Look up the core word in the dictionary.
         - Try original form, lowercase, and (for Turkish) ``turkish_lower``.
      3. Re-attach punctuation to the translated word.
      4. Copy unknown words as-is.
    """
    tokens = sentence.split()
    result: List[str] = []

    for token in tokens:
        # Separate punctuation from core word
        m = _RE_PUNCT_SPLIT.match(token)
        if not m or not m.group(2):
            result.append(token)
            continue

        prefix, core, suffix = m.group(1), m.group(2), m.group(3)

        # Try multiple casing forms for lookup
        candidates = [core, core.lower()]
        if src_lang == "tr":
            candidates.append(turkish_lower(core))

        translated = None
        for form in candidates:
            if form in dictionary:
                translated = dictionary[form]
                break

        if translated is None:
            translated = core  # OOV → copy

        result.append(f"{prefix}{translated}{suffix}")

    return " ".join(result)


def translate_corpus(
    sentences: List[str],
    dictionary: Dict[str, str],
    src_lang: str = "tr",
) -> List[str]:
    """Translate a list of sentences word by word."""
    return [
        translate_sentence(s, dictionary, src_lang=src_lang)
        for s in tqdm(sentences, desc=f"Translating ({src_lang}→?)")
    ]


# ════════════════════════════════════════════════════════════
#  5. EVALUATION
# ════════════════════════════════════════════════════════════

def read_lines(path: str) -> List[str]:
    """Read a text file into a list of stripped lines."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def load_flores(cfg: dict) -> Tuple[List[str], List[str]]:
    """Load FLORES-200 devtest (Turkish + English).

    Looks for local files first (at two possible locations),
    then falls back to HuggingFace datasets.
    """
    base_dir = cfg["paths"]["base_dir"]
    flores_dir = os.path.join(base_dir, cfg["evaluation"]["flores_dir"])
    tr_file = cfg["evaluation"]["source_files"]["tr"]
    en_file = cfg["evaluation"]["source_files"]["en"]

    # ── Try 1: {base_dir}/flores200_dataset/devtest/ ─────────
    tr_path = os.path.join(flores_dir, tr_file)
    en_path = os.path.join(flores_dir, en_file)

    # ── Try 2: project root (legacy, checkpoint_baselines.py) ─
    if not os.path.exists(tr_path):
        alt_flores = os.path.join(_PROJECT_ROOT, cfg["evaluation"]["flores_dir"])
        tr_path = os.path.join(alt_flores, tr_file)
        en_path = os.path.join(alt_flores, en_file)

    if os.path.exists(tr_path) and os.path.exists(en_path):
        log.info(f"Loading FLORES-200 from local files:")
        log.info(f"  TR: {tr_path}")
        log.info(f"  EN: {en_path}")
        return read_lines(tr_path), read_lines(en_path)

    # ── Try 3: Direct Download from Meta AWS S3 ───────────────
    from src.utils import download_flores_if_needed
    tr_dest, en_dest = download_flores_if_needed(cfg)
    
    if os.path.exists(tr_dest) and os.path.exists(en_dest):
        log.info(f"Loading FLORES-200 from local files:")
        log.info(f"  TR: {tr_dest}")
        log.info(f"  EN: {en_dest}")
        return read_lines(tr_dest), read_lines(en_dest)
        
    log.error("Could not load FLORES-200.")
    sys.exit(1)


def compute_metrics(
    preds: List[str], refs: List[str]
) -> Tuple[float, float]:
    """Compute corpus-level BLEU and chrF."""
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return round(bleu, 2), round(chrf, 2)


def evaluate_baseline(cfg: dict, emb_paths: Dict[str, str]) -> None:
    """Run word-by-word translation on FLORES-200 and compute metrics.

    Evaluates both directions:  TR→EN  and  EN→TR.
    Saves numeric results and sample translations to the results dir.
    """
    base_dir = cfg["paths"]["base_dir"]
    results_dir = ensure_dir(
        os.path.join(base_dir, cfg["evaluation"]["results_subdir"])
    )

    # ── Load FLORES-200 ─────────────────────────────────────
    tr_lines, en_lines = load_flores(cfg)
    log.info(f"FLORES-200 devtest: {len(tr_lines):,} TR / {len(en_lines):,} EN sentences")

    # ── Evaluate both directions ─────────────────────────────
    directions = [
        ("tr", "en", tr_lines, en_lines),
        ("en", "tr", en_lines, tr_lines),
    ]

    rows = []
    for src_lang, tgt_lang, sources, references in directions:
        tag = f"{src_lang.upper()}→{tgt_lang.upper()}"
        log.info(f"\n{'─' * 50}")
        log.info(f"  Direction: {tag}")
        log.info(f"{'─' * 50}")

        # Get / build dictionary
        dictionary = get_or_build_dictionary(cfg, src_lang, tgt_lang, emb_paths)
        if not dictionary:
            log.error(f"  Empty dictionary for {tag} — skipping")
            continue

        # Translate
        predictions = translate_corpus(sources, dictionary, src_lang=src_lang)

        # Metrics
        bleu, chrf = compute_metrics(predictions, references)
        log.info(f"  [{tag}]  BLEU = {bleu:.2f}   chrF = {chrf:.2f}")

        rows.append({
            "model": "Word-by-Word Baseline (MUSE fastText)",
            "direction": f"{src_lang.upper()}->{tgt_lang.upper()}",
            "BLEU": bleu,
            "chrF": chrf,
            "status": "Complete",
        })

        # ── Save sample translations ────────────────────────
        n_samples = min(len(sources), len(references), len(predictions))
        sample_df = pd.DataFrame({
            f"source_{src_lang}": sources[:n_samples],
            f"reference_{tgt_lang}": references[:n_samples],
            "wbw_prediction": predictions[:n_samples],
        })
        sample_path = os.path.join(
            results_dir, f"wbw_translations_{src_lang}_{tgt_lang}.csv"
        )
        sample_df.to_csv(sample_path, index=False)
        log.info(f"  Translations saved → {sample_path}")

        # ── Show a few examples ─────────────────────────────
        log.info(f"\n  Sample translations ({tag}):")
        for i in range(min(3, n_samples)):
            log.info(f"    SRC : {sources[i][:100]}")
            log.info(f"    REF : {references[i][:100]}")
            log.info(f"    WBW : {predictions[i][:100]}")
            log.info("")

    # ── Save aggregate results ───────────────────────────────
    if rows:
        results_df = pd.DataFrame(rows)
        results_path = os.path.join(results_dir, "wbw_baseline_results.csv")
        results_df.to_csv(results_path, index=False)
        log.info(f"\n✓ Results saved → {results_path}")

        # Print comparison table
        log.info("\n  Results Summary:")
        log.info("  " + "─" * 60)
        log.info(f"  {'Model':<40} {'Dir':<8} {'BLEU':>6} {'chrF':>6}")
        log.info("  " + "─" * 60)
        for r in rows:
            log.info(
                f"  {r['model']:<40} {r['direction']:<8} "
                f"{r['BLEU']:>6.2f} {r['chrF']:>6.2f}"
            )
        log.info("  " + "─" * 60)


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CENG467 UNMT — Word-by-Word Dictionary Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_base_args(parser)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip embedding download (use cached files)",
    )
    parser.add_argument(
        "--max-vocab",
        type=int,
        default=None,
        help="Override max vocabulary size from config",
    )
    args = parser.parse_args()

    # ── Colab auto-mount ─────────────────────────────────────
    mounted = mount_drive_if_colab()

    # ── Config ───────────────────────────────────────────────
    cfg = load_config(
        config_path=args.config,
        base_dir_override=args.base_dir,
    )
    set_seed(cfg["project"]["seed"])

    if args.max_vocab:
        cfg["baseline"]["max_vocab"] = args.max_vocab

    base_dir = cfg["paths"]["base_dir"]

    log.info("=" * 62)
    log.info("  CENG467 UNMT — Word-by-Word Baseline")
    log.info(f"  Base dir   : {base_dir}")
    log.info(f"  Colab      : {'yes' if mounted else 'no'}")
    log.info(f"  Max vocab  : {cfg['baseline']['max_vocab']:,}")
    log.info("=" * 62)

    # ── Step 1: Download embeddings ──────────────────────────
    emb_paths = download_embeddings(cfg, skip=args.skip_download)

    # ── Steps 2–5: Build dicts → translate → evaluate ────────
    evaluate_baseline(cfg, emb_paths)

    log.info("")
    log.info("═" * 62)
    log.info("  ✅  Word-by-word baseline complete.")
    log.info("═" * 62)


if __name__ == "__main__":
    main()
