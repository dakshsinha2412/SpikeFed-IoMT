"""
utils.py – SpikeFed-IoMT shared utilities
==========================================
Provides:
  - set_seed()           : reproducible runs
  - get_logger()         : structured console logger
  - save_json()          : persist dicts to JSON in results/
  - print_metrics_table(): pretty-print a metrics dict
"""

import json
import logging
import os
import random
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch

# ─────────────────────────────── reproducibility ────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Prefer CUDA when available, otherwise use CPU (including Raspberry Pi)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────── logging ───────────────────────────────

_LOG_FMT = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
_DATE_FMT = "%H:%M:%S"


def get_logger(name: str = "spikefed") -> logging.Logger:
    """Return (or create) a named logger with console output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# ───────────────────────────────── JSON results ──────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def save_json(data: Dict[str, Any], filename: str) -> str:
    """Serialize *data* to ``results/<filename>``.  Returns the full path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    return path


def load_json(filename: str) -> Dict[str, Any]:
    """Load a JSON file from the results directory."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ─────────────────────────────── display helpers ────────────────────────────

def print_metrics_table(metrics: Dict[str, Any], title: str = "Metrics") -> None:
    """Pretty-print a flat metrics dictionary as an aligned table."""
    if not metrics:
        return
    width = max(len(str(k)) for k in metrics) + 2
    bar = "─" * (width + 20)
    print(f"\n┌{bar}┐")
    print(f"│  {title:<{width + 18}}│")
    print(f"├{bar}┤")
    for key, val in metrics.items():
        if isinstance(val, float):
            val_str = f"{val:.4f}"
        else:
            val_str = str(val)
        print(f"│  {key:<{width}}{val_str:>18}  │")
    print(f"└{bar}┘\n")


def timestamp() -> str:
    """ISO-8601 timestamp string suitable for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
