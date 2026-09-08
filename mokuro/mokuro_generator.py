from concurrent.futures import ThreadPoolExecutor
from json import JSONDecodeError

from loguru import logger
from tqdm import tqdm

from mokuro import __version__
from mokuro.config import IMAGE_LOAD_THREADS, OCR_CHUNK_SIZE, get_default_num_workers, get_default_ocr_batch_size
from mokuro.manga_page_ocr import MangaPageOcr
from mokuro.utils import dump_json, imread, load_json
from mokuro.volume import Volume


class MokuroGenerator:
    def __init__(
        self,
        pretrained_model_name_or_path="kha-white/manga-ocr-base",
        force_cpu=False,
        disable_ocr=False,
        num_workers=None,
        ocr_batch_size=None,
        **kwargs,
    ):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.force_cpu = force_cpu
        self.disable_ocr = disable_ocr

        # None -> auto-detect from hardware (see mokuro/config.py)
        self.num_workers = num_workers if num_workers is not None else get_default_num_workers(force_cpu)
        self.ocr_batch_size = ocr_batch_size if ocr_batch_size is not None else get_default_ocr_batch_size(force_cpu)

        self.kwargs = kwargs
        self.mpocr = None

    def init_models(self):
        if self.mpocr is None:
            # num_beams is consumed by recognize_text(), not by MangaPageOcr
            mpocr_kwargs = {k: v for k, v in self.kwargs.items() if k != "num_beams"}
            self.mpocr = MangaPageOcr(
                self.pretrained_model_name_or_path,
                force_cpu=self.force_cpu,
                disable_ocr=self.disable_ocr,
                **mpocr_kwargs,
            )

    def process_volume(self, volume: Volume, ignore_errors=False, no_cache=False):
        volume.path_ocr_cache.mkdir(parents=True, exist_ok=True)

        if volume.mokuro_data is not None:
            for page in volume.mokuro_data["pages"]:
                json_path = volume.get_ocr_path(page["img_path"])
                if json_path.is_file():
                    continue
                json_path.parent.mkdir(parents=True, exist_ok=True)
                page = page.copy()
                page.pop("img_path")
                dump_json(page, json_path)

        img_paths = volume.get_img_paths()
        self.init_models()

        # Process pages in chunks: detect each page, collect its text-line
        # crops, then run OCR over the whole chunk's crops in one batched pass.
        chunk_size = max(OCR_CHUNK_SIZE, self.num_workers)
        img_paths_list = list(img_paths.items())

        with tqdm(total=len(img_paths_list), desc="Processing pages...") as pbar:
            for chunk_start in range(0, len(img_paths_list), chunk_size):
                chunk = img_paths_list[chunk_start : chunk_start + chunk_size]

                # Skip pages whose OCR cache entry is already valid.
                to_process = []
                for key, img_path_rel in chunk:
                    json_path = volume.get_ocr_path(img_path_rel)
                    if not no_cache and json_path.is_file():
                        try:
                            load_json(json_path)
                            continue
                        except (FileNotFoundError, JSONDecodeError, UnicodeDecodeError) as e:
                            logger.warning(f"Error loading cached OCR for {img_path_rel}: {e}. Re-processing.")
                    to_process.append((key, img_path_rel))

                if not to_process:
                    pbar.update(len(chunk))
                    continue

                # Load page images concurrently — the actual I/O bottleneck.
                def read_image(item):
                    key, img_path_rel = item
                    return key, img_path_rel, imread(volume.path_in / img_path_rel)

                with ThreadPoolExecutor(max_workers=min(IMAGE_LOAD_THREADS, len(to_process))) as executor:
                    loaded = list(executor.map(read_image, to_process))

                # Detect text blocks per page (sequential; detection is GPU-bound).
                page_results = {}  # key -> (result_dict, crop_metadata)
                all_crops = []
                crop_page_map = []  # (key, local_crop_idx) per global crop

                for key, img_path_rel, img in loaded:
                    if img is None:
                        logger.error(f"Could not load image: {volume.path_in / img_path_rel}")
                        continue

                    try:
                        result, crops, meta = self.mpocr.detect_and_extract(img)
                        page_results[key] = (result, meta)
                        for j in range(len(crops)):
                            all_crops.append(crops[j])
                            crop_page_map.append((key, j))
                    except Exception as e:
                        if ignore_errors:
                            logger.error(f"Error detecting {img_path_rel}: {e}")
                        else:
                            raise

                # One batched OCR pass over every crop in this chunk.
                if all_crops and not self.disable_ocr:
                    try:
                        # num_beams=None -> use the model's generation config
                        # (beam search, num_beams=4) for identical output to
                        # upstream; pass num_beams=1 for fast greedy decoding.
                        all_texts = self.mpocr.recognize_text(
                            all_crops,
                            batch_size=self.ocr_batch_size,
                            num_beams=self.kwargs.get("num_beams"),
                        )

                        for global_idx, text in enumerate(all_texts):
                            key, local_idx = crop_page_map[global_idx]
                            result, meta = page_results[key]
                            blk_idx, line_idx = meta[local_idx]
                            result["blocks"][blk_idx]["lines"][line_idx] += text
                    except Exception as e:
                        if ignore_errors:
                            logger.error(f"Error in OCR: {e}")
                        else:
                            raise

                # Write per-page OCR cache files.
                for key, (result, _) in page_results.items():
                    img_path_rel = img_paths[key]
                    json_path = volume.get_ocr_path(img_path_rel)
                    json_path.parent.mkdir(parents=True, exist_ok=True)
                    dump_json(result, json_path)

                pbar.update(len(chunk))

        self.generate_mokuro_file(volume, ignore_errors=ignore_errors)

    @staticmethod
    def generate_mokuro_file(volume: Volume, ignore_errors=False):
        json_paths = volume.get_json_paths()
        img_paths = volume.get_img_paths()

        out = {
            "version": __version__,
            "title": volume.title.name,
            "title_uuid": volume.title.uuid,
            "volume": volume.name,
            "volume_uuid": volume.uuid,
            "pages": [],
        }

        for key, json_path_rel in json_paths.items():
            try:
                if key not in img_paths:
                    logger.warning(f"No matching image found for cached OCR: {key}")
                    continue
                img_path_rel = img_paths[key]
                page_json = load_json(volume.path_ocr_cache / json_path_rel)
                page_json["img_path"] = str(img_path_rel).replace("\\", "/")
                out["pages"].append(page_json)
            except Exception as e:
                if ignore_errors:
                    logger.error(e)
                else:
                    raise

        dump_json(out, volume.path_mokuro)
