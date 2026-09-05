"""Central configuration for mokuro's performance-related defaults.

This is the *single* place to tune how mokuro uses your hardware. Every value
in this file is a default: CLI flags (``--num_workers``, ``--ocr_batch_size``,
``--num_beams``) always override it, and library callers can pass the same
arguments directly to :class:`mokuro.mokuro_generator.MokuroGenerator`.

Edit the numbers below to change the out-of-the-box behaviour on your machine.
"""

import os
import platform

import torch

# ---------------------------------------------------------------------------
# Parallel page processing
# ---------------------------------------------------------------------------
# How many pages are OCR'd per chunk (the generator processes a chunk of
# pages, then runs batched OCR over all of their text-line crops at once).
OCR_CHUNK_SIZE = 8


def get_device() -> str:
    """Return the best available compute device: ``"cuda"``, ``"mps"`` or ``"cpu"``."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_apple_silicon() -> bool:
    """True when running on an Apple Silicon (arm64) machine."""
    return platform.machine() in ("arm64", "aarch64")


def get_default_num_workers() -> int:
    """
    Number of pages loaded (and detected) concurrently per chunk.

    Apple Silicon machines benefit from a high worker count thanks to their
    unified memory and many cores; NVIDIA GPUs are usually memory-bound, and
    plain CPUs prefer a modest thread count.
    """
    cpu_count = os.cpu_count() or 4

    if is_apple_silicon():
        # Cap at 8-10 to avoid excessive memory usage while staying very fast.
        return min(8, max(1, int(cpu_count * 0.75)))

    if torch.cuda.is_available():
        return min(4, cpu_count)

    return max(1, cpu_count // 2)


def get_default_ocr_batch_size() -> int:
    """
    Number of text-line crops fed to the OCR model per ``generate()`` call.

    Unified memory (Apple Silicon) tolerates larger batches without OOM;
    dedicated NVIDIA GPUs usually sit well in the 32-64 range; CPUs want small
    batches to keep latency per page low.
    """
    device = get_device()

    if device == "mps":
        return 64

    if device == "cuda":
        return 32

    return 16
