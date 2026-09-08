"""Central configuration for mokuro's performance-related defaults.

**This is the only file you normally need to edit** to tune how mokuro uses
your machine. Every value below is a *default*: command-line flags
(``--num_workers``, ``--ocr_batch_size``, ``--num_beams``) and library
arguments always take precedence over it, but if you never pass flags, the
values chosen here (or auto-detected from your hardware) apply everywhere —
CLI, mokuro-bridge, and library callers alike.

Quick guide to what matters on your machine:

* **Apple Silicon (M1–M4)** — unified memory lets you run a high worker count
  and a large OCR batch. Defaults: 8 workers, batch 64.
* **NVIDIA GPU (CUDA)** — the GPU is the bottleneck, so fewer workers (4) and
  a moderate batch (32) avoid memory pressure; ``--fp16`` gives the biggest
  win here (not exact, see ``USE_FP16``).
* **CPU only** — modest concurrency (cores / 2) and a small batch (16) keep
  latency per page low; fp16/fusion are irrelevant on CPU.
* Running out of memory? Lower ``OCR_BATCH_SIZE`` / ``NUM_WORKERS``.
* Want better OCR accuracy at the cost of speed? Raise ``NUM_BEAMS`` to 4.
"""

import os
import platform

import torch

# ===========================================================================
# ⚙️  EDIT ME — per-machine tuning knobs
# ===========================================================================
# Set a knob to a concrete value to force it everywhere; leave it ``None`` to
# keep the automatic, hardware-aware default (functions at the bottom of this
# file). CLI flags still override whichever choice you make here.

# -- concurrency ------------------------------------------------------------
# Pages loaded & detected concurrently per processing chunk.
#   Apple Silicon: 8 · NVIDIA: 4 · CPU: cores / 2      (None = auto)
NUM_WORKERS = None

# Text-line crops sent to the OCR model per batched generate() call.
# Bigger batches use the GPU better but need more memory.
#   Apple Silicon: 64 · NVIDIA: 32 · CPU: 16           (None = auto)
OCR_BATCH_SIZE = None

# Pages processed per chunk before a batched OCR pass runs.
# (The effective chunk is max(OCR_CHUNK_SIZE, num_workers).)
OCR_CHUNK_SIZE = 8

# Threads used to decode page images (disk I/O is the bottleneck, so more
# than ~4 rarely helps and can hurt on spinning disks / network mounts).
IMAGE_LOAD_THREADS = 4

# Page image decoder: "auto" decodes plain RGB/grayscale JPEGs with
# cv2.imread (pixel-identical to the PIL path, ~2x faster) and everything else
# with PIL; "pil" forces the PIL path for every file.
IMAGE_DECODER = "auto"

# -- OCR decoding quality vs. speed -----------------------------------------
# Beam width for the OCR transformer:
#   None -> use the model's own generation config (num_beams=4 — identical
#           output to upstream mokuro / manga-ocr, best accuracy)
#   1    -> greedy decoding (fastest; occasionally misreads ambiguous glyphs)
#   4    -> force beam search (matches upstream default quality)
# Anything other than the model default also disables USE_CUSTOM_BEAM's fast
# path (transformers' generate() is used instead).
NUM_BEAMS = None

# -- GPU feature toggles ----------------------------------------------------
# OCR transformer precision on CUDA/ROCm/MPS (ignored on CPU). Default fp32:
# byte-identical to upstream on every tested volume/tier. fp16 (``--fp16`` on
# the CLI, or ``USE_FP16 = True`` here) is 1.6x faster on an RTX 4090 and 4.9x
# on an RX 9070 XT in this pipeline, but is NOT exact: measured on 140 volumes
# (2.52 M characters) it changes 0.19% of the characters on 26 pages per 1000
# (mostly hallucination-prone lines; occasionally real text, e.g. a dropped
# bracket). Boxes are never affected. Details: README, "Precision policy".
USE_FP16 = False

# Fold batch-norm layers into the preceding conv layers of the text detector
# at load time. Measured: no speed gain on any tested GPU/CPU and the detector
# output changes on CUDA (boxes move by up to 10 px on ~2% of blocks), so it
# is OFF; opt in if you want to experiment.
FUSE_CONV_BN = False

# torch.compile the text detector + OCR encoder (CUDA/ROCm, inductor default
# mode). Measured SLOWER than eager in every mode on an RTX 4090 and an
# RX 9070 XT (torch 2.13), and the fork's original "reduce-overhead" mode
# crashed. Kept only as an off-by-default experiment knob.
USE_TORCH_COMPILE = False

# Let cuDNN use TF32 for the detector's fp32 convolutions on CUDA. TF32 makes
# the CUDA detector output drift from CPU/ROCm output (box edges by 1..10 px
# on some blocks, ~50 character edits per volume) for no measurable speed
# gain, so it is off. Only affects NVIDIA GPUs.
ALLOW_CUDNN_TF32 = False

# -- exact-parity optimisations (all on; each can be switched off to get the
#    reference code path back, e.g. when bisecting a problem) ---------------
# Run the text detector in channels_last (NHWC) memory format when it executes
# on the CPU (oneDNN keeps its blocked layout between conv layers). ~1.8-2x on
# the detector forward; no effect on GPU.
DETECTOR_CPU_CHANNELS_LAST = True

# Use mokuro/beam.py (a hand-rolled beam search with an in-place static KV
# cache and per-crop shared cross-attention K/V) instead of transformers'
# generic generate(). Same beam-search semantics and the same kernels, so the
# output is identical; it removes the per-step Python glue that dominates the
# decoder on CPU. Automatically falls back to generate() when the requested
# decoding differs from the model's default (e.g. NUM_BEAMS=1).
USE_CUSTOM_BEAM = True

# Number of decoder steps the host may run ahead of the GPU before checking
# the "all sequences finished" flag (CUDA/ROCm only). 0 = block on the flag
# every step; 1 = read the previous step's flag (overlaps CPU dispatch with GPU
# execution; at most one wasted step per batch, output unchanged).
BEAM_SYNC_LAG = 1

# When transformers' generate() is used (custom beam off or not applicable):
# skip the per-step re-gather of the cross-attention KV cache, which is a
# semantic no-op in beam search (beam indices never leave an item's group and
# every beam holds the same encoder K/V). Identical output.
SKIP_CROSS_ATTN_CACHE_REORDER = True

# ===========================================================================
# Automatic hardware detection — usually nothing to edit below this line.
# ===========================================================================


def get_device(force_cpu: bool = False) -> str:
    """Return the compute device: ``"cuda"`` (also ROCm), ``"mps"`` or ``"cpu"``."""
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_apple_silicon() -> bool:
    """True when running on an Apple Silicon (arm64) machine."""
    return platform.machine() in ("arm64", "aarch64")


def get_default_num_workers(force_cpu: bool = False) -> int:
    """
    Number of pages loaded (and detected) concurrently per chunk.

    Apple Silicon machines benefit from a high worker count thanks to their
    unified memory and many cores; NVIDIA GPUs are usually memory-bound, and
    plain CPUs prefer a modest thread count. Override by setting
    ``NUM_WORKERS`` above.
    """
    if NUM_WORKERS is not None:
        return NUM_WORKERS

    cpu_count = os.cpu_count() or 4

    if not force_cpu and is_apple_silicon():
        # Cap at 8-10 to avoid excessive memory usage while staying very fast.
        return min(8, max(1, int(cpu_count * 0.75)))

    if get_device(force_cpu) == "cuda":
        return min(4, cpu_count)

    return max(1, cpu_count // 2)


def get_default_ocr_batch_size(force_cpu: bool = False) -> int:
    """
    Number of text-line crops fed to the OCR model per ``generate()`` call.

    Unified memory (Apple Silicon) tolerates larger batches without OOM;
    dedicated NVIDIA GPUs usually sit well in the 32-64 range; CPUs want small
    batches to keep latency per page low. Override by setting
    ``OCR_BATCH_SIZE`` above.
    """
    if OCR_BATCH_SIZE is not None:
        return OCR_BATCH_SIZE

    device = get_device(force_cpu)

    if device == "mps":
        return 64

    if device == "cuda":
        return 32

    return 16
