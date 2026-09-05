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
from mokuro.cache import cache
from mokuro.config import get_device
from mokuro.utils import imread

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
            device = "cpu" if force_cpu else get_device()
            logger.info(f"Initializing text detector, using device {device}")
            self.text_detector = TextDetector(
                model_path=cache.comic_text_detector, input_size=detector_input_size, device=device, act="leaky"
            )

            # Fold batch-norm into preceding conv layers: a free speedup at
            # inference time, no effect on output.
            if hasattr(self.text_detector.net, "fuse"):
                try:
                    self.text_detector.net.fuse()
                    logger.info("Fused conv+bn layers in text detector")
                except Exception:
                    pass

            self.mocr = MangaOcr(pretrained_model_name_or_path, force_cpu)

            # Move the OCR transformer to the active device and use half
            # precision on GPUs for faster inference with negligible accuracy
            # loss (the model was trained with fp32, fp16 is fine for OCR).
            if device != "cpu":
                try:
                    self.mocr.model.to(device)
                    self.mocr.model.half()
                    logger.info(f"Moved MangaOcr model to {device} (half precision)")
                except Exception as e:
                    logger.warning(f"Could not move model to {device}: {e}. Falling back to default.")

            # torch.compile on CUDA only — MPS support is unstable in PyTorch 2.x.
            if device == "cuda" and hasattr(torch, "compile"):
                try:
                    self.text_detector.net = torch.compile(self.text_detector.net, mode="reduce-overhead")
                    self.mocr.model.encoder = torch.compile(self.mocr.model.encoder, mode="reduce-overhead")
                    logger.info("Compiled models with torch.compile")
                except Exception as e:
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

    def recognize_text(self, crops, batch_size=64, **generation_kwargs):
        """Run batched OCR over a list of PIL crops, returning one text per crop.

        Batching turns many small per-line ``generate()`` calls into one call
        per ``batch_size`` crops, which is dramatically faster on GPUs and
        still a win on CPU.

        Decoding defaults match manga-ocr's behaviour exactly: the model's own
        generation config (beam search, ``num_beams=4``) is used unless
        overridden (e.g. ``num_beams=1`` for greedy/fast decoding).
        """
        all_texts = []

        gen_config = self.mocr.model.generation_config
        gen_args = {
            "max_length": getattr(gen_config, "max_length", 300),
            "num_beams": getattr(gen_config, "num_beams", 1),
            "do_sample": getattr(gen_config, "do_sample", False),
            "use_cache": True,
        }
        gen_args.update(generation_kwargs)
        # Drop explicit Nones so they fall back to the model's generation
        # config (manga-ocr's __call__ passes no decoding overrides at all).
        gen_args = {k: v for k, v in gen_args.items() if v is not None}

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
                generated_ids = self.mocr.model.generate(pixel_values, **gen_args)

            texts = self.mocr.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_texts.extend(ocr_post_process(text) for text in texts)

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
