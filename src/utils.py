"""
CENG467 UNMT Project — Shared Utilities
========================================
Common helpers used by every script in the project:
  • load_config()          – YAML config loading with CLI / env overrides
  • resolve_path()         – Build absolute paths from base_dir
  • mount_drive_if_colab() – Auto-mount Google Drive on Colab
  • set_seed()             – Reproducibility
  • get_device()           – Best available torch device
  • setup_logging()        – Consistent log formatting
"""

import os
import sys
import yaml
import random
import logging

import numpy as np
import torch

# ── Default config location (relative to this file) ────────
_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "configs", "config.yaml"
)


# ────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────

def load_config(config_path=None, base_dir_override=None):
    """Load the YAML config and optionally override *base_dir*.

    Resolution order for base_dir:
      1. ``base_dir_override`` argument  (--base-dir CLI flag)
      2. ``CENG467_BASE_DIR`` environment variable
      3. Value in config.yaml
    """
    path = config_path or _DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if base_dir_override:
        cfg["paths"]["base_dir"] = base_dir_override
    elif os.environ.get("CENG467_BASE_DIR"):
        cfg["paths"]["base_dir"] = os.environ["CENG467_BASE_DIR"]

    return cfg


def resolve_path(cfg, *subkeys):
    """Resolve an absolute path by joining *base_dir* with a config value.

    Example::

        resolve_path(cfg, "data", "raw_subdir")
        # → /content/drive/MyDrive/CENG467_Project/data/raw
    """
    base = cfg["paths"]["base_dir"]
    node = cfg
    for k in subkeys:
        node = node[k]
    return os.path.join(base, node)


def ensure_dir(path):
    """Create *path* (and parents) if it does not exist, then return it."""
    os.makedirs(path, exist_ok=True)
    return path


# ────────────────────────────────────────────────────────────
#  Google Colab / Drive Integration
# ────────────────────────────────────────────────────────────

def mount_drive_if_colab():
    """Mount Google Drive when running inside Colab. No-op otherwise.

    Returns ``True`` if the mount was performed.
    """
    try:
        import google.colab          # noqa: F401
        from google.colab import drive
        import os
        # If already mounted, skip calling drive.mount()
        if os.path.exists("/content/drive/MyDrive"):
            return True
        drive.mount("/content/drive")
        return True
    except Exception:
        return False


# ────────────────────────────────────────────────────────────
#  Reproducibility & Device
# ────────────────────────────────────────────────────────────

def set_seed(seed):
    """Set random seed across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return ``torch.device('cuda')`` if a GPU is available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ────────────────────────────────────────────────────────────
#  Logging
# ────────────────────────────────────────────────────────────

def setup_logging(name="ceng467", level=logging.INFO):
    """Return a logger with a clean ``[HH:MM:SS] LEVEL — msg`` format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ────────────────────────────────────────────────────────────
#  CLI Helpers
# ────────────────────────────────────────────────────────────

def add_base_args(parser):
    """Attach ``--config`` and ``--base-dir`` to an *argparse* parser."""
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--base-dir", type=str, default=None,
        help="Override paths.base_dir from config (useful for local dev)",
    )
    return parser

# ────────────────────────────────────────────────────────────
#  FLORES-200 Direct Download (Bypassing HuggingFace)
# ────────────────────────────────────────────────────────────

def download_flores_if_needed(cfg):
    """Download FLORES-200 devtest directly from Meta if not present."""
    import urllib.request
    import tarfile
    
    base_dir = cfg["paths"]["base_dir"]
    flores_dir = os.path.join(base_dir, cfg["evaluation"]["flores_dir"])
    tr_file = cfg["evaluation"]["source_files"]["tr"]
    en_file = cfg["evaluation"]["source_files"]["en"]
    
    tr_dest = os.path.join(flores_dir, tr_file)
    en_dest = os.path.join(flores_dir, en_file)
    
    if os.path.exists(tr_dest) and os.path.exists(en_dest):
        return tr_dest, en_dest
        
    log = logging.getLogger("ceng467")
    log.info("FLORES-200 not found locally. Downloading from Meta AWS S3...")
    
    tar_url = "https://dl.fbaipublicfiles.com/flores200/dataset/flores200_dataset.tar.gz"
    tar_path = os.path.join(base_dir, "flores200_dataset.tar.gz")
    
    try:
        if not os.path.exists(tar_path):
            urllib.request.urlretrieve(tar_url, tar_path)
            log.info("Download complete. Extracting...")
            
        with tarfile.open(tar_path, "r:gz") as tar:
            # The tar has structure: flores200_dataset/devtest/tur_Latn.devtest
            for member in tar.getmembers():
                if member.name.endswith("tur_Latn.devtest") or member.name.endswith("eng_Latn.devtest"):
                    tar.extract(member, path=base_dir)
                    
        if os.path.exists(tar_path):
            os.remove(tar_path)
            
        log.info(f"FLORES-200 extracted successfully to {flores_dir}")
    except Exception as e:
        log.error(f"Failed to download FLORES-200: {e}")
        
    return tr_dest, en_dest
