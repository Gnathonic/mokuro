# mokuro — performance-optimized fork

Read Japanese manga with selectable text inside a browser — **optimized for
speed** on Apple Silicon (MPS), NVIDIA (CUDA) and CPU.

This is a fork of [kha-white/mokuro](https://github.com/kha-white/mokuro)
(rebased on upstream **v0.2.5**) that keeps the exact same CLI, output format
and workflow while making OCR significantly faster through batched inference,
hardware-aware defaults and GPU-friendly model tweaks.

**Version: 0.3.0b** — the `b` marks this fork's *bridge* lineage (it grew out
of the mokuro-bridge project) and distinguishes it from upstream releases.

**See demo: https://kha-white.github.io/manga-demo**

mokuro is aimed towards Japanese learners, who want to read manga in Japanese with a pop-up dictionary like [Yomitan](https://github.com/themoeway/yomitan).
It works like this:
1. Perform text detection and OCR for each page.
2. After processing a whole volume, generate a .mokuro file, which contains OCR results and metadata. All processing is done offline (before reading).
3. Load the .mokuro file together with manga images in [web reader](https://reader.mokuro.app/), which serves both as a manga reader and a catalog for processed series and volumes.

Alternatively, you can still use the old method from mokuro 0.1.*:
Instead of a .mokuro file, generate an HTML file, which you can open in a browser.
You can transfer the resulting HTML file together with manga images to another device (e.g. your mobile phone) and read there.
This method is still supported for backward compatibility, but it is recommended to use the new .mokuro format and the web reader.
For details, see [Legacy HTML vs. new .mokuro format](#legacy-html-vs-new-mokuro-format).

mokuro uses [comic-text-detector](https://github.com/dmMaze/comic-text-detector) for text detection
and [manga-ocr](https://github.com/kha-white/manga-ocr) for OCR.

---

## What's improved in this fork

| Feature | Upstream | This fork |
|---|---|---|
| OCR inference | one `generate()` call **per text line** | **batched** — one call per `ocr_batch_size` crops |
| Page loading | sequential | **concurrent** (thread pool) |
| Device selection | CUDA/MPS/CPU | CUDA/MPS/CPU + **fp16** on GPUs |
| Text detector | — | **conv+bn fusion** and **torch.compile** (CUDA) |
| Defaults | fixed | **hardware-aware** (`mokuro/config.py`) |
| Long-line splitting | gaussian rebuilt per line | **cached** gaussian window |
| Degenerate lines | crash on malformed geometry | **skipped gracefully** |

All output files (`.mokuro`, `.html`, `_ocr/` cache) are **byte-format
identical** to upstream — the optimizations change *how fast* pages are
processed, not *what* is produced.

### How it works

- Pages are processed in **chunks** (`OCR_CHUNK_SIZE`): images are loaded
  concurrently, text blocks are detected, and all text-line crops from the
  whole chunk go through **one batched OCR pass**.
- On NVIDIA GPUs the detector net is **conv+bn fused** and both models are
  **`torch.compile`d**; on Apple Silicon and CUDA the OCR transformer runs in
  **fp16**.
- **`mokuro/config.py`** auto-detects your hardware and picks sensible
  defaults (see below) — override any of them on the command line.

## Easy-to-edit parameters

**Everything is tuned in one file: [`mokuro/config.py`](mokuro/config.py).**
Open it and you'll find a clearly marked *"EDIT ME"* block at the top with a
comment on every knob telling you what it does and what values suit which
hardware. Edit it, save, and the new defaults apply everywhere — CLI,
mokuro-bridge and library callers. No code changes needed.

> Command-line flags always override the config file: `--num_workers`,
> `--ocr_batch_size` and `--num_beams` win for that one run.

### The knobs

| Config constant | What it controls | Recommended values | Set to `None` |
|---|---|---|---|
| `NUM_WORKERS` | Pages processed per chunk before a batched OCR pass | Apple Silicon: **8** · NVIDIA: **4** · CPU: **cores/2** | auto-detect |
| `OCR_BATCH_SIZE` | Text-line crops per batched OCR `generate()` call | MPS: **64** · CUDA: **32** · CPU: **16** | auto-detect |
| `OCR_CHUNK_SIZE` | Pages per processing chunk (floored to `NUM_WORKERS`) | **8** | — |
| `IMAGE_LOAD_THREADS` | Threads decoding page images (disk-I/O bound) | **4** | — |
| `NUM_BEAMS` | OCR beam width: `1` = greedy/fast, `4` = best accuracy | **`None`** (model default, 4) | model default (4) |
| `USE_FP16` | Half-precision inference on CUDA/MPS | **`True`** | — |
| `FUSE_CONV_BN` | Fold batch-norm into conv layers of the text detector | **`True`** | — |
| `USE_TORCH_COMPILE` | `torch.compile` the models (CUDA only) | **`True`** | — |

### How to choose values for your machine

- **Apple Silicon (M1–M4)** — unified memory likes big batches and lots of
  workers. Leave `None` and you get 8 workers / batch 64 / fp16. If you run
  long volumes and memory pressure builds, lower `OCR_BATCH_SIZE` to 48 or 32.
- **NVIDIA GPU (CUDA)** — auto: 4 workers / batch 32 / fp16 + fusion +
  `torch.compile`. If you hit "CUDA out of memory", drop `OCR_BATCH_SIZE` to
  16 first; if you have a high-VRAM card (12 GB+), raise it to 64.
- **CPU only** — auto: cores/2 workers / batch 16. `USE_FP16` and
  `USE_TORCH_COMPILE` do nothing on CPU; `FUSE_CONV_BN` still helps a little.
- **Accuracy vs. speed** — the default (`NUM_BEAMS = None`) matches upstream's
  beam search (4) for identical output. If you want the fastest possible OCR,
  set `NUM_BEAMS = 1` (greedy) — on ambiguous glyphs you may occasionally see
  a different character than beam search would pick.

### Examples

Trade a little speed for faster (greedy) OCR, just for one run:

```bash
mokuro --num_beams 1 /path/to/manga/vol1
```

Force a large batch on a beefy GPU, just for one run:

```bash
mokuro --ocr_batch_size 64 /path/to/manga/vol1
```

Make it permanent for every run — edit `mokuro/config.py`:

```python
NUM_BEAMS = 1  # greedy decoding everywhere
OCR_BATCH_SIZE = 64  # you have a 16 GB GPU
```

## Performance

Measured head-to-head on an **Apple Silicon (M4 Pro)** machine with a
**187-page tankōbon volume** (cold OCR cache, identical dependencies, all 187
OCR files generated successfully in every run):

| Variant | Total time | Per page | vs. upstream |
|---|---:|---:|---:|
| Upstream mokuro 0.2.5 | 362.2 s | 1.94 s | — |
| **This fork (0.3.0b)** ⭐ | **173.0 s** | **0.92 s** | **2.09× faster** |

*Each number is the mean of two alternating runs under the same system load.*

Key takeaways:

- The fork processes the same volume in **less than half the time** — ~2.1×
  faster than upstream 0.2.5 with **identical OCR output** (verified by the
  test suite and byte-level crop parity with manga-ocr).
- The speedup comes from **batched OCR inference**, **concurrent page
  loading** and **fp16 on GPU** — no accuracy trade-off.
- CUDA users additionally get **conv-bn fusion** and **torch.compile**
  (biggest wins on older GPUs).

Reproduce it on your own volumes with
[`benchmark_mokuro.py`](mokuro/benchmark_mokuro.py):

```bash
python mokuro/benchmark_mokuro.py /path/to/manga-volume results.json
```

## Installation

You need Python 3.10 or newer. Please note, that the newest Python release might not be supported due to a PyTorch dependency,
which often breaks with new Python releases and needs some time to catch up.
Refer to [PyTorch website](https://pytorch.org/get-started/locally/) for a list of supported Python versions.

Some users have reported problems with Python installed from Microsoft Store. If you see an error:
`ImportError: DLL load failed while importing fugashi: The specified module could not be found.`,
try installing Python from the [official site](https://www.python.org/downloads).

If you want to run with GPU, install PyTorch as described [here](https://pytorch.org/get-started/locally/#start-locally),
otherwise this step can be skipped.

Run in command line:

```commandline
pip3 install git+https://github.com/<your-fork>/mokuro.git
```

or from a local checkout:

```commandline
pip3 install -e .
```

## Replacing the pip-installed mokuro with this fork (no downloads)

The mokuro OCR engine is a **Python package installed via pip** (it has no
npm counterpart — see the note at the end). "Replacing mokuro" therefore
always means making `import mokuro` (and the `mokuro` CLI) resolve to this
local checkout instead of the PyPI wheel. All options below point Python at
the **local checkout** — the fork itself is never downloaded again.

**One-time clone** (skip if you already have a checkout):

```bash
git clone <your-fork-url> mokuro-fork
cd mokuro-fork
git submodule update --init --recursive   # checks out comic_text_detector
```

The fork's runtime dependencies (torch, manga-ocr, transformers, …) must
already be installed in the target environment — install them normally once.
The steps below only swap *which* mokuro code gets used.

### Option A — replace inside one virtualenv (recommended)

```bash
source /path/to/your-venv/bin/activate
pip uninstall -y mokuro
pip install -e /path/to/mokuro-fork --no-deps --no-build-isolation
```

`--no-deps` stops pip from fetching anything from the network, and
`--no-build-isolation` reuses the already-installed setuptools (fully offline
install). Because the fork keeps the same distribution name (`mokuro`), pip
cleanly supersedes the previous install — no leftover copies.

Verify from inside that venv:

```bash
mokuro --version          # → 0.3.0b
python -c "import mokuro; print(mokuro.__file__)"   # → .../mokuro-fork/mokuro/__init__.py
```

### Option B — replace machine-wide (your default `python3`)

Same idea, but for the interpreter your scripts use by default:

```bash
pip3 uninstall -y mokuro
pip3 install -e /path/to/mokuro-fork --no-deps --no-build-isolation
```

Every `python3` process on the machine now imports the fork, and the global
`mokuro` command runs it too. (If pip refuses with an
`externally-managed-environment` error, use Option A in a venv instead.)
Repeat Option A inside any other virtualenv that should use it — venvs do not
inherit global installs unless created with `--system-site-packages`.

### Option C — zero-install pointer file (library imports only)

If you only use mokuro as a **library** and never call the `mokuro` CLI, a
one-line `.pth` file in the interpreter's site-packages is enough — pip is
never involved:

```bash
python3 - <<'EOF'
import site
site_pkgs = site.getsitepackages()[0]
with open(f"{site_pkgs}/mokuro-fork.pth", "w") as f:
    f.write("/path/to/mokuro-fork\n")
print("wrote", f"{site_pkgs}/mokuro-fork.pth")
EOF
```

Any `import mokuro` under that interpreter now resolves to the fork. Caveats:

- It affects **imports only** — a previously installed `mokuro` console
  command still runs the old version (use Option B to replace the CLI too).
- Remove the `.pth` file before `pip install`-ing mokuro again; the old
  distribution is not uninstalled, so `pip list` shows both.

### Notes

- **mokuro-bridge**: the bridge already auto-uses a sibling `mokuro/`
  checkout and otherwise honours `MOKURO_REPO=/path/to/mokuro-fork` — no pip
  step is needed there at all.
- **npm**: there is no npm package for the mokuro OCR engine, so nothing to
  replace on that side. The web reader (reader.mokuro.app) is a hosted app
  and never runs OCR locally.

## Usage

## Run on one volume

```bash
mokuro /path/to/manga/vol1
```

This will generate `/path/to/manga/vol1.html` file, which you can open in a browser.

If your path contains spaces, enclose it in double quotes, like this:

```bash
mokuro "/path/to/manga/volume 1"
```

## Run on multiple volumes

```bash
mokuro /path/to/manga/vol1 /path/to/manga/vol2 /path/to/manga/vol3
```

For each volume, a separate HTML file will be generated.

## Run on a directory containing multiple volumes

If your directory structure looks somewhat like this:
```
manga_title/
├─vol1/
├─vol2/
├─vol3/
└─vol4/
```

You can process all volumes by running:

```bash
mokuro --parent_dir manga_title/
```

## Other options

```
--pretrained_model_name_or_path: Name or path of the manga-ocr model.
--force_cpu: Force the use of CPU even if CUDA/MPS is available.
--disable_confirmation: Disable confirmation prompt. If False, the user will be prompted to confirm the list of volumes to be processed.
--disable_ocr: Disable OCR processing. Generate mokuro/HTML files without OCR results.
--ignore_errors: Continue processing volumes even if an error occurs.
--no_cache: Do not use cached OCR results from previous runs (_ocr directories).
--unzip: Extract volumes in zip/cbz format in their original location.
--disable_html: Disable legacy HTML output. If True, acts as if --unzip is True.
--as_one_file: Applies only to legacy HTML. If False, generate separate CSS and JS files instead of embedding them in the HTML file.
--num_workers: Pages processed concurrently per chunk (default: auto-detected).
--ocr_batch_size: Text-line crops per batched OCR call (default: auto-detected).
--num_beams: Beam width for OCR decoding. 1 = fast/greedy, 4 = higher quality (default: 1).
--version: Print the version of mokuro and exit.
```

## Legacy HTML vs. new .mokuro format

Before version 0.2.0, mokuro generated a separate HTML file for each processed volume, which caused some usability issues:
- HTML files contained both the OCR results and the whole web reader GUI, so in order to update the GUI, all volumes needed to be updated with a new mokuro version
- images were stored separately and linked in HTML files, so any change in the directory structure could break the links
- transferring the manga to another device required transferring both the HTML files and the images
- there was no unified GUI for a whole catalog containing multiple volumes
- on some mobile devices, some workarounds were needed to open HTML files

Starting from version 0.2.0, a new .mokuro format is introduced, which is generated for each volume and contains only the OCR results and metadata necessary for the web reader GUI.
Web reader is now a separate web app, which can open manga volumes with their associated .mokuro files.

The old HTML format is still generated for backward compatibility, but it will not be developed further, and it is recommended to use the new .mokuro format and the web reader.

## Development

```bash
pip3 install -e ".[dev]"
python3 -m pytest tests/          # run the test suite (CPU)
python3 -m ruff check mokuro/     # lint
```

## Keeping in sync with upstream

This fork tracks [kha-white/mokuro](https://github.com/kha-white/mokuro).
To pull the latest upstream changes into your clone:

```bash
git remote add upstream https://github.com/kha-white/mokuro.git   # once
git fetch upstream
git merge upstream/master          # resolve conflicts, then commit
git submodule update --init --recursive
```

The fork's changes are deliberately confined to a handful of files
(`mokuro/config.py`, `mokuro/manga_page_ocr.py`, `mokuro/mokuro_generator.py`,
`mokuro/run.py`, `mokuro/volume.py` + docs), so merges stay small.

## License & credits

- **GPL-3.0** — see [LICENSE](LICENSE). This fork inherits upstream mokuro's
  license unmodified; any use must comply with GPL-3.0.
- Upstream: [kha-white/mokuro](https://github.com/kha-white/mokuro) by
  [Maciej Budyś](https://github.com/kha-white).
- Optimizations developed and refined with the help of **DeepSeek V4**,
  under the direction of **GolyBidoof** (this fork's maintainer).
- Text detection: [comic-text-detector](https://github.com/dmMaze/comic-text-detector);
  OCR: [manga-ocr](https://github.com/kha-white/manga-ocr);
  text segmentation: [Manga-Text-Segmentation](https://github.com/juvian/Manga-Text-Segmentation).
