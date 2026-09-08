import os

import cv2
import numpy as np
import torch
from loguru import logger
from manga_ocr import MangaOcr
from manga_ocr.ocr import post_process as ocr_post_process
from PIL import Image
from scipy.signal.windows import gaussian

from comic_text_detector.inference import TextDetector
from mokuro import __version__
from mokuro.beam import BeamSearchOCR
from mokuro.cache import cache
from mokuro.config import (
    ALLOW_CUDNN_TF32,
    BEAM_SYNC_LAG,
    DETECTOR_CPU_CHANNELS_LAST,
    FUSE_CONV_BN,
    NUM_BEAMS,
    SKIP_CROSS_ATTN_CACHE_REORDER,
    USE_CUSTOM_BEAM,
    USE_FP16,
    USE_TORCH_COMPILE,
    get_default_ocr_batch_size,
    get_device,
)
from mokuro.hf_patches import install_skip_cross_attn_cache_reorder
from mokuro.utils import imread

_log_once_seen: set = set()


def _log_once(msg: str) -> None:
    """Log a warning once per unique message (avoids flooding on batch errors)."""
    if msg in _log_once_seen:
        return
    _log_once_seen.add(msg)
    logger.warning(f"[mokuro] {msg}")


# Suppress noisy transformers warnings (e.g. "Some weights not used")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Cache gaussian windows keyed by (size, std) — they only depend on text_height.
_gaussian_cache = {}


class MangaPageOcr:
    def __init__(
        self,
        pretrained_model_name_or_path="kha-white/manga-ocr-base",
        force_cpu=False,
        detector_input_size=1024,
        text_height=64,
        max_ratio_vert=16,
        max_ratio_hor=8,
        anchor_window=2,
        disable_ocr=False,
    ):
        self.text_height = text_height
        self.max_ratio_vert = max_ratio_vert
        self.max_ratio_hor = max_ratio_hor
        self.anchor_window = anchor_window
        self.disable_ocr = disable_ocr

        if not self.disable_ocr:
            device = get_device(force_cpu)
            if device == "cuda":
                # cuDNN TF32 convolutions make the CUDA detector output drift
                # from CPU/ROCm output at no speed benefit (ALLOW_CUDNN_TF32).
                torch.backends.cudnn.allow_tf32 = bool(ALLOW_CUDNN_TF32)
            logger.info(f"Initializing text detector, using device {device}")
            self.text_detector = TextDetector(
                model_path=cache.comic_text_detector, input_size=detector_input_size, device=device, act="leaky"
            )

            # Opt-in: fold batch-norm into the preceding conv layers. Changes the
            # detector output slightly on CUDA and measured no faster (config.py).
            if FUSE_CONV_BN and hasattr(self.text_detector.net, "fuse"):
                try:
                    self.text_detector.net.fuse()
                    logger.warning(
                        "Fused conv+bn layers in text detector (FUSE_CONV_BN): output may differ from upstream"
                    )
                except Exception as e:  # noqa: BLE001 - opt-in fast path; fall back to the unfused net
                    logger.warning(f"FUSE_CONV_BN: fuse() failed ({e}); using the unfused detector")

            # CPU only: channels_last memory format keeps oneDNN's blocked
            # layout across conv layers (~2x on the detector forward).
            if device == "cpu" and DETECTOR_CPU_CHANNELS_LAST:
                try:
                    self.text_detector.net = self.text_detector.net.to(memory_format=torch.channels_last)
                    self.text_detector.channels_last = True
                except Exception as e:  # noqa: BLE001 - optional layout; keep the default memory format
                    logger.warning(f"channels_last for text detector skipped: {e}")

            self.mocr = MangaOcr(pretrained_model_name_or_path, force_cpu)

            # Move the OCR transformer to the active device and use half
            # precision on GPUs for faster inference with negligible accuracy
            # loss (the model was trained with fp32, fp16 is fine for OCR).
            # Toggle via USE_FP16 in mokuro/config.py.
            if device != "cpu" and USE_FP16:
                try:
                    self.mocr.model.to(device)
                    self.mocr.model.half()
                    logger.info(f"Moved MangaOcr model to {device} (half precision)")
                except Exception as e:  # noqa: BLE001 - run on the default device/precision instead
                    logger.warning(f"Could not move model to {device}: {e}. Falling back to default.")

            # Beam search: mokuro/beam.py by default; transformers' generate()
            # (with the cache-reorder patch) as the fallback for non-default
            # decoding settings.
            self._beam = BeamSearchOCR(self.mocr.model, sync_lag=BEAM_SYNC_LAG) if USE_CUSTOM_BEAM else None
            if SKIP_CROSS_ATTN_CACHE_REORDER:
                install_skip_cross_attn_cache_reorder(self.mocr.model)

            # Experimental (off by default; measured slower than eager on the
            # GPUs tested): torch.compile in the default inductor mode.
            if device == "cuda" and USE_TORCH_COMPILE and hasattr(torch, "compile"):
                try:
                    self.text_detector.net = torch.compile(self.text_detector.net)
                    self.mocr.model.encoder = torch.compile(self.mocr.model.encoder, dynamic=True)
                    logger.info("Compiled models with torch.compile (USE_TORCH_COMPILE)")
                except Exception as e:  # noqa: BLE001 - experimental; eager mode is the fallback
                    logger.debug(f"torch.compile skipped: {e}")

            self._device = device

    def __call__(self, img_path):
        """Process a single page image path and return the OCR result dict.

        Equivalent to upstream behaviour; internally it detects text blocks,
        extracts the text-line crops and runs batched OCR over them.
        """
        img = imread(img_path)
        result, all_crops, crop_metadata = self.detect_and_extract(img)

        if not self.disable_ocr and all_crops:
            all_texts = self.recognize_text(all_crops)
            for (blk_idx, line_idx), text in zip(crop_metadata, all_texts):
                result["blocks"][blk_idx]["lines"][line_idx] += text

        return result

    def detect_and_extract(self, img):
        """Run text detection on a decoded page image.

        Returns ``(result_dict, crops, crop_metadata)`` where ``crops`` is a
        list of PIL images (one per text line) and ``crop_metadata`` maps each
        crop back to its ``(block_idx, line_idx)`` in ``result_dict``.
        """
        H, W, *_ = img.shape
        if self.disable_ocr:
            return {"version": __version__, "img_width": W, "img_height": H, "blocks": []}, [], []

        _, mask_refined, blk_list = self.text_detector(img, refine_mode=1, keep_undetected_mask=True)
        return self._extract_crops(img, blk_list, mask_refined)

    def _extract_crops(self, img, blk_list, mask_refined):
        """Split detected blocks into text-line crops (shared by call/GPU paths)."""
        H, W, *_ = img.shape
        result = {"version": __version__, "img_width": W, "img_height": H, "blocks": []}
        all_crops = []
        crop_metadata = []

        for blk_idx, blk in enumerate(blk_list):
            result_blk = {
                "box": list(blk.xyxy),
                "vertical": blk.vertical,
                "font_size": blk.font_size,
                "lines_coords": [],
                "lines": [],
            }
            result["blocks"].append(result_blk)

            for line_idx, line in enumerate(blk.lines_array()):
                max_ratio = self.max_ratio_vert if blk.vertical else self.max_ratio_hor

                line_crops, _ = self.split_into_chunks(
                    img,
                    mask_refined,
                    blk,
                    line_idx,
                    textheight=self.text_height,
                    max_ratio=max_ratio,
                    anchor_window=self.anchor_window,
                )

                result_blk["lines_coords"].append(line.tolist())
                result_blk["lines"].append("")

                for line_crop in line_crops:
                    if blk.vertical:
                        line_crop = cv2.rotate(line_crop, cv2.ROTATE_90_CLOCKWISE)
                    all_crops.append(Image.fromarray(line_crop).convert("RGB"))
                    crop_metadata.append((blk_idx, line_idx))

        return result, all_crops, crop_metadata

    def recognize_text(self, crops, batch_size=None, **generation_kwargs):
        """Run batched OCR over a list of PIL crops, returning one text per crop.

        Batching turns many small per-line ``generate()`` calls into one call
        per ``batch_size`` crops, which is dramatically faster on GPUs and
        still a win on CPU.

        Decoding defaults match manga-ocr's behaviour exactly: the model's own
        generation config (beam search, ``num_beams=4``) is used unless
        overridden (e.g. ``num_beams=1`` for greedy/fast decoding). The
        machine-wide default for ``batch_size`` and ``num_beams`` lives in
        ``mokuro/config.py``.
        """
        all_texts = []

        if batch_size is None:
            batch_size = get_default_ocr_batch_size(self._device == "cpu")

        gen_config = self.mocr.model.generation_config
        gen_args = {
            "max_length": getattr(gen_config, "max_length", 300),
            "num_beams": getattr(gen_config, "num_beams", 1),
            "do_sample": getattr(gen_config, "do_sample", False),
            "use_cache": True,
        }
        # Explicit Nones fall back to the model's generation config
        # (manga-ocr's __call__ passes no decoding overrides at all).
        overrides = {k: v for k, v in generation_kwargs.items() if v is not None}
        # A machine-wide beam default set in mokuro/config.py wins over the
        # model config unless the caller passed num_beams explicitly.
        if NUM_BEAMS is not None and "num_beams" not in overrides:
            overrides["num_beams"] = NUM_BEAMS
        gen_args.update(overrides)

        device = self.mocr.model.device
        model_dtype = next(self.mocr.model.parameters()).dtype

        for i in range(0, len(crops), batch_size):
            batch_items = crops[i : i + batch_size]

            # Match manga-ocr's preprocessing exactly: grayscale -> RGB. The
            # ViT model was trained on that transform; skipping it shifts OCR
            # output on ambiguous glyphs.
            batch_items = [img.convert("L").convert("RGB") for img in batch_items]

            pixel_values = self.mocr.processor(batch_items, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device, non_blocking=True)

            if model_dtype == torch.float16:
                pixel_values = pixel_values.half()

            with torch.inference_mode():
                if self._beam is not None and self._beam.matches(gen_args):
                    generated_ids = self._beam.generate(pixel_values)
                else:
                    generated_ids = self.mocr.model.generate(pixel_values, **gen_args)

            texts = self.mocr.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for text in texts:
                # batch_decode can return bare token ids (int) for some inputs;
                # post_process (jaconv/tokenizer) calls str methods on the item
                # and would raise "'int' object has no attribute 'lower'".
                # Coerce defensively — identical to a str, and preserves stock
                # behaviour (stock manga-ocr's post_process only ever sees str).
                if not isinstance(text, str):
                    _log_once(
                        "non-str decode item in recognize_text batch "
                        f"(type={type(text).__name__}, value={text!r}); coercing to str"
                    )
                    text = str(text) if text is not None else ""
                all_texts.extend([ocr_post_process(text)])

        return all_texts

    @staticmethod
    def split_into_chunks(img, mask_refined, blk, line_idx, textheight, max_ratio=16, anchor_window=2):
        try:
            line_crop = blk.get_transformed_region(img, line_idx, textheight)
        except (OverflowError, ValueError, ZeroDivisionError):
            # Degenerate line geometry — skip it rather than crash the volume.
            return [], []

        h, w, *_ = line_crop.shape
        if h == 0 or w == 0:
            return [], []

        ratio = w / h

        if ratio <= max_ratio:
            return [line_crop], []

        cache_key = (textheight * 2, textheight / 8)
        if cache_key not in _gaussian_cache:
            _gaussian_cache[cache_key] = gaussian(cache_key[0], cache_key[1])
        k = _gaussian_cache[cache_key]

        line_mask = blk.get_transformed_region(mask_refined, line_idx, textheight)
        num_chunks = int(np.ceil(ratio / max_ratio))

        anchors = np.linspace(0, w, num_chunks + 1)[1:-1]

        line_density = line_mask.sum(axis=0)
        line_density = np.convolve(line_density, k, "same")
        line_density /= line_density.max()

        anchor_window *= textheight

        cut_points = []
        for anchor in anchors:
            anchor = int(anchor)

            n0 = np.clip(anchor - anchor_window // 2, 0, w)
            n1 = np.clip(anchor + anchor_window // 2, 0, w)

            p = line_density[n0:n1].argmin()
            p += n0

            cut_points.append(p)

        return np.split(line_crop, cut_points, axis=1), cut_points
