#!/usr/bin/env python3
"""
Benchmark mokuro performance across worker/batch-size configurations.

Usage:
    python benchmark_mokuro.py /path/to/manga-volume [output.json]

Runs a fixed suite of configurations (baseline vs optimized) over one volume
and prints a comparison table. Pass no volume path to see usage.
"""

import json
import shutil
import sys
import time
from pathlib import Path

from loguru import logger

from mokuro import MokuroGenerator
from mokuro.volume import Title, Volume


def benchmark_configuration(
    volume_path: Path,
    num_workers: int | None = None,
    ocr_batch_size: int | None = None,
    num_beams: int = 1,
    force_cpu: bool = False,
    no_cache: bool = True,  # Don't use cache for fair comparison
    description: str = "",
) -> dict[str, float]:
    """
    Benchmark a single configuration.
    
    Returns:
        Dictionary with timing information
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: {description}")
    logger.info(f"  num_workers: {num_workers}")
    logger.info(f"  ocr_batch_size: {ocr_batch_size}")
    logger.info(f"  num_beams: {num_beams}")
    logger.info(f"  force_cpu: {force_cpu}")
    logger.info(f"{'='*60}\n")
    
    # Create volume
    volume = Volume(volume_path)
    
    # Set up title (required for generate_mokuro_file)
    if volume.title is None:
        volume.title = Title(volume.path_title)
        volume.title.set_uuid()
    
    # Initialize generator
    mg = MokuroGenerator(
        force_cpu=force_cpu,
        num_workers=num_workers,
        ocr_batch_size=ocr_batch_size,
        num_beams=num_beams,
    )
    
    # Get image count before processing
    img_paths = volume.get_img_paths()
    num_images = len(img_paths)
    
    if num_images == 0:
        logger.error("No images found in volume!")
        return {
            "description": description,
            "num_workers": num_workers,
            "ocr_batch_size": ocr_batch_size,
            "num_beams": num_beams,
            "force_cpu": force_cpu,
            "elapsed_time": 0,
            "num_images": 0,
            "time_per_image": 0,
            "success": False,
        }
    
    # Count existing OCR files before processing
    ocr_cache = volume.path_ocr_cache
    existing_ocr_files = len(list(ocr_cache.glob("**/*.json"))) if ocr_cache.exists() else 0
    
    # Time the processing
    start_time = time.time()
    
    try:
        # Force processing by clearing cache if no_cache is True
        if no_cache and ocr_cache.exists():
            logger.info("Clearing OCR cache for fair benchmark...")
            shutil.rmtree(ocr_cache)
            ocr_cache.mkdir(parents=True, exist_ok=True)
        
        mg.process_volume(volume, ignore_errors=False, no_cache=no_cache)
        
        # Verify that OCR files were actually created
        final_ocr_files = len(list(ocr_cache.glob("**/*.json"))) if ocr_cache.exists() else 0
        files_created = final_ocr_files - existing_ocr_files
        
        # Success means we processed at least 80% of images (some might fail)
        success = files_created >= (num_images * 0.8)
        
        if not success:
            logger.warning(f"Only {files_created}/{num_images} OCR files created. Expected at least {int(num_images * 0.8)}")
        
    except Exception as e:
        logger.error(f"Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()
        success = False
        files_created = 0
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    result = {
        "description": description,
        "num_workers": num_workers,
        "ocr_batch_size": ocr_batch_size,
        "num_beams": num_beams,
        "force_cpu": force_cpu,
        "elapsed_time": elapsed_time,
        "num_images": num_images,
        "files_created": files_created,
        "time_per_image": elapsed_time / num_images if num_images > 0 else 0,
        "success": success,
    }
    
    logger.info("\nResults:")
    logger.info(f"  Total time: {elapsed_time:.2f} seconds")
    logger.info(f"  Images in volume: {num_images}")
    logger.info(f"  OCR files created: {result.get('files_created', 0)}")
    logger.info(f"  Time per image: {result['time_per_image']:.3f} seconds")
    logger.info(f"  Success: {success}")
    
    if elapsed_time < 10 and num_images > 50:
        logger.warning(f"⚠️  Processing time seems too fast ({elapsed_time:.2f}s for {num_images} images).")
        logger.warning("   This might indicate the benchmark didn't actually process images.")
    
    return result


def run_benchmark_suite(volume_path: Path) -> list[dict[str, float]]:
    """
    Run a suite of benchmarks with different configurations.
    """
    volume_path = Path(volume_path).expanduser().absolute()
    
    if not volume_path.exists():
        logger.error(f"Volume path does not exist: {volume_path}")
        return []
    
    logger.info(f"Benchmarking volume: {volume_path}")
    
    # Get image count for reference
    volume = Volume(volume_path)
    img_paths = volume.get_img_paths()
    num_images = len(img_paths)
    logger.info(f"Found {num_images} images to process\n")
    
    results = []
    
    # Configuration 1: Baseline (single worker, default batch size) - BEFORE optimizations
    results.append(benchmark_configuration(
        volume_path,
        num_workers=1,
        ocr_batch_size=32,
        num_beams=1,
        description="BEFORE: Baseline (1 worker, batch_size=32)",
    ))
    
    # Configuration 2: AFTER - Optimized workers, default batch
    results.append(benchmark_configuration(
        volume_path,
        num_workers=None,  # Auto-detect (optimized)
        ocr_batch_size=32,
        num_beams=1,
        description="AFTER: Auto workers (optimized), batch_size=32",
    ))
    
    # Configuration 3: AFTER - Fully optimized (auto workers + auto batch)
    results.append(benchmark_configuration(
        volume_path,
        num_workers=None,  # Auto-detect (optimized)
        ocr_batch_size=None,  # Auto-optimize (48 for Apple Silicon)
        num_beams=1,
        description="AFTER: Fully Optimized (auto workers + auto batch_size=48)",
    ))
    
    # Configuration 4: AFTER - High workers, high batch (aggressive)
    results.append(benchmark_configuration(
        volume_path,
        num_workers=12,
        ocr_batch_size=64,
        num_beams=1,
        description="AFTER: Aggressive (12 workers, batch_size=64)",
    ))
    
    # Configuration 5: AFTER - Moderate workers, optimized batch
    results.append(benchmark_configuration(
        volume_path,
        num_workers=8,
        ocr_batch_size=48,
        num_beams=1,
        description="AFTER: Balanced (8 workers, batch_size=48)",
    ))
    
    # Configuration 6: AFTER - Conservative (fewer workers, moderate batch)
    results.append(benchmark_configuration(
        volume_path,
        num_workers=6,
        ocr_batch_size=48,
        num_beams=1,
        description="AFTER: Conservative (6 workers, batch_size=48)",
    ))
    
    # Configuration 7: CPU-only baseline (for comparison)
    results.append(benchmark_configuration(
        volume_path,
        num_workers=4,
        ocr_batch_size=32,
        num_beams=1,
        force_cpu=True,
        description="CPU-only: 4 workers, batch_size=32",
    ))
    
    return results


def print_summary(results: list[dict[str, float]]):
    """
    Print a summary comparison of all benchmark results.
    """
    if not results:
        logger.error("No results to summarize")
        return
    
    # Filter successful results
    successful_results = [r for r in results if r.get("success", False)]
    
    if not successful_results:
        logger.error("No successful benchmarks to compare")
        return
    
    # Sort by elapsed time
    successful_results.sort(key=lambda x: x["elapsed_time"])
    
    baseline = successful_results[0]  # First one is baseline
    
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*80)
    
    # Find baseline (BEFORE) and best AFTER
    before_results = [r for r in successful_results if "BEFORE:" in r["description"]]
    after_results = [r for r in successful_results if "AFTER:" in r["description"]]
    
    if before_results:
        baseline = before_results[0]
        logger.info(f"\nBEFORE (Baseline): {baseline['description']}")
        logger.info(f"  Time: {baseline['elapsed_time']:.2f}s ({baseline['time_per_image']:.3f}s per image)")
    else:
        baseline = successful_results[0]
        logger.info(f"\nBaseline: {baseline['description']}")
        logger.info(f"  Time: {baseline['elapsed_time']:.2f}s ({baseline['time_per_image']:.3f}s per image)")
    
    logger.info("\nAll Configurations (sorted by speed):")
    logger.info("-" * 80)
    logger.info(f"{'Configuration':<55} {'Time':<12} {'Speedup':<10} {'Per Image':<12}")
    logger.info("-" * 80)
    
    for result in successful_results:
        speedup = baseline["elapsed_time"] / result["elapsed_time"]
        marker = " ⭐" if result == successful_results[0] else ""
        logger.info(
            f"{result['description']:<55} "
            f"{result['elapsed_time']:>8.2f}s  "
            f"{speedup:>6.2f}x  "
            f"{result['time_per_image']:>8.3f}s{marker}"
        )
    
    # Find best configuration
    best = successful_results[0]
    speedup_vs_baseline = baseline["elapsed_time"] / best["elapsed_time"]
    
    logger.info("\n" + "="*80)
    logger.info(f"🏆 BEST CONFIGURATION: {best['description']}")
    logger.info(f"  Total time: {best['elapsed_time']:.2f}s")
    logger.info(f"  Time per image: {best['time_per_image']:.3f}s")
    logger.info(f"  Speedup vs baseline: {speedup_vs_baseline:.2f}x")
    logger.info(f"  Time saved: {baseline['elapsed_time'] - best['elapsed_time']:.2f}s")
    
    if after_results:
        best_after = min(after_results, key=lambda x: x["elapsed_time"])
        if before_results:
            improvement = (baseline["elapsed_time"] - best_after["elapsed_time"]) / baseline["elapsed_time"] * 100
            logger.info("\n📊 OPTIMIZATION IMPACT:")
            logger.info(f"  Best AFTER config: {best_after['description']}")
            logger.info(f"  Improvement: {improvement:.1f}% faster than baseline")
            logger.info(f"  Time saved: {baseline['elapsed_time'] - best_after['elapsed_time']:.2f}s")
    
    logger.info("="*80 + "\n")


def save_results(results: list[dict[str, float]], output_path: Path):
    """
    Save benchmark results to a JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {output_path}")


def main():
    """
    Main benchmarking function.
    """
    if len(sys.argv) < 2:
        logger.error("Usage: python benchmark_mokuro.py <volume_path> [output_json]")
        logger.error("Example: python benchmark_mokuro.py /path/to/my-manga-volume")
        sys.exit(1)
    
    volume_path = Path(sys.argv[1])
    output_json = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    logger.info("Starting mokuro benchmark suite...")
    logger.info(f"Volume: {volume_path}")
    
    results = run_benchmark_suite(volume_path)
    
    print_summary(results)
    
    if output_json:
        save_results(results, output_json)
    else:
        # Auto-save to volume directory
        volume = Volume(volume_path)
        output_path = volume.path_in.parent / "benchmark_results.json"
        save_results(results, output_path)


if __name__ == "__main__":
    main()

