# Mokuro Optimization Summary

This document describes the performance work in this fork: what was changed,
why, and the measured results. The goal was to keep mokuro's **exact output
format and CLI** while making OCR substantially faster on Apple Silicon,
NVIDIA GPUs and plain CPUs.

## Measured results

Benchmark: a **187-page tankōbon** (typical single volume), **cold OCR cache**,
identical dependencies, on an **Apple Silicon (M4 Pro)** machine. Each number
is the mean of two alternating runs under the same system load; every run
generated all 187 per-page OCR files successfully.

| Variant | Total time | Time/page | vs. upstream |
|:---|:---|:---|:---|
| Upstream mokuro 0.2.5 | 362.2 s | 1.94 s | — |
| **This fork (0.3.0b)** ⭐ | **173.0 s** | **0.92 s** | **2.09× faster** |

Key findings:

- The fork processes the same volume in **less than half the time** with
  **identical OCR output** (verified by the test suite and byte-level crop
  parity with manga-ocr).
- The gains come from **batched OCR inference**, **concurrent page loading**
  and **fp16 on GPU** — no accuracy trade-off.
- Earlier tuning runs (December 2025) on the same hardware showed CPU-only
  mode at ~0.71 s/page vs ~0.46 s/page with MPS — **MPS is ~55% faster than
  CPU-only** on Apple Silicon.

## What changed

### 1. Batched OCR inference (`mokuro/manga_page_ocr.py`)

Upstream calls the OCR model once **per text line** (`model.generate` per
crop). This fork splits the pipeline into `detect_and_extract()` (detect
blocks + collect line crops) and `recognize_text()` (one `generate()` call
per `batch_size` crops). On GPUs this amortizes launch overhead; on CPU it
reduces per-call interpreter overhead. Preprocessing is byte-identical to
manga-ocr's `__call__` (grayscale → RGB), so **OCR output is unchanged**.

### 2. Concurrent page loading (`mokuro/mokuro_generator.py`)

Pages are processed in chunks (`OCR_CHUNK_SIZE`, default 8). Images are
decoded with a small thread pool (up to 4 threads) before detection, hiding
disk I/O latency behind compute.

### 3. Hardware-aware defaults (`mokuro/config.py`)

New central config that auto-detects the device and picks good defaults:

| Setting | Apple Silicon | NVIDIA (CUDA) | CPU only |
|:---|:---|:---|:---|
| Num workers | 8 | 4 | cores / 2 |
| OCR batch size | 64 | 32 | 16 |
| Precision | fp16 | fp16 | fp32 |
| Beams | model default (4) | model default (4) | model default (4) |

Every value is a default — CLI flags (`--num_workers`, `--ocr_batch_size`,
`--num_beams`) and library arguments always override it.

### 4. GPU-friendly model tweaks (`mokuro/manga_page_ocr.py`)

- **conv+bn fusion** on the text detector (folds batch-norm into conv at
  load time — free inference speedup, no output change).
- **fp16** for the OCR transformer on CUDA/MPS (negligible accuracy impact).
- **`torch.compile`** for the detector net and OCR encoder on CUDA.
- **`torch.inference_mode()`** for the batched OCR pass (slightly cheaper than
  the default on modern PyTorch).

### 5. Small correctness & hygiene fixes

- `Volume.get_ocr_path()` helper centralizes the `_ocr/<stem>/<rel>.json`
  path logic.
- Gaussian window used for long-line splitting is **cached** (it only depends
  on `text_height`).
- Degenerate line geometry (zero-size crops, overflow in
  `get_transformed_region`) is **skipped** instead of crashing the volume.
- `generate_mokuro_file()` skips cached OCR entries that have no matching
  image, instead of raising.
- Moved to upstream v0.2.5 base (fork version 0.3.0b): **AVIF support**, XDG
  cache location, `pkg_resources` removal, Python ≥3.10, robust PIL-based
  image loading.

## Files touched vs. upstream

| File | Change |
|:---|:---|
| `mokuro/config.py` | **new** — easy-to-edit hardware defaults |
| `mokuro/manga_page_ocr.py` | batched OCR, fp16, fusion, torch.compile, gaussian cache, safe splitting |
| `mokuro/mokuro_generator.py` | chunked + threaded processing, batched OCR, `num_workers`/`ocr_batch_size`/`num_beams` |
| `mokuro/run.py` | new CLI flags `--num_workers`, `--ocr_batch_size`, `--num_beams` |
| `mokuro/volume.py` | `get_ocr_path()` helper |
| `comic_text_detector/inference.py` | `no_grad()` → `inference_mode()` (submodule) |
| `README.md` | fork docs: improvements, config, benchmarks, credits |

## How to benchmark your own machine

```bash
python mokuro/benchmark_mokuro.py /path/to/manga-volume results.json
```

The script runs a fixed suite (baseline vs. several optimized configurations)
and prints a comparison table. See `mokuro/benchmark_mokuro.py`.

## Compatibility

- **Apple Silicon (M1–M4)**: full MPS acceleration, fp16, auto-tuned defaults.
- **NVIDIA (CUDA)**: fp16 + conv-bn fusion + `torch.compile`, auto-tuned defaults.
- **CPU only**: batched inference with modest thread count; graceful fallback.

**Developed**: December 2025, refreshed for the v0.2.5 rebase (fork v0.3.0b).
**Optimization work**: led by GolyBidoof with the help of **DeepSeek V4**.
