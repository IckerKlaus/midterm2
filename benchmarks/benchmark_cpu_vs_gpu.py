#!/usr/bin/env python3
"""
Benchmark Script: CPU vs GPU Performance Comparison

This script measures and compares the performance of different approaches
for Voronoi cell assignment (nearest neighbor search):

1. CPU Naive: Brute-force Euclidean distance computation
2. CPU KD-Tree: KD-tree based nearest neighbor search
3. GPU Accelerated: Batch distance computation on GPU

The benchmark tests various problem sizes to understand scaling behavior
and compute speedup factors.

Usage:
    python benchmarks/benchmark_cpu_vs_gpu.py
    python benchmarks/benchmark_cpu_vs_gpu.py --sizes 100,1000,10000 --trials 5

Output:
    - Console table with timing results
    - CSV file with detailed results
    - Analysis of speedup trends

Research Hypothesis:
    We expect GPU + KD-tree to achieve 3-5x speedup over naive CPU
    for realistic retail scenarios (1,000-10,000 products).
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.synthetic_data import generate_shelf_scenario
from src.geometry.kd_tree import KDTree, brute_force_nearest_neighbor_batch
from src.geometry.voronoi_cpu import assign_to_voronoi_cells
from src.geometry.voronoi_gpu import (
    gpu_assign_to_voronoi_cells,
    is_gpu_available,
    compute_distance_matrix
)
from src.hpc.timing import Timer, BenchmarkResult
from src.hpc.gpu_kernels import get_gpu_info


def benchmark_cpu_naive(
    planogram_points: np.ndarray,
    detection_points: np.ndarray,
    n_trials: int = 3
) -> BenchmarkResult:
    """
    Benchmark CPU brute-force nearest neighbor search.
    
    Complexity: O(n × m) where n = planogram, m = detections
    """
    result = BenchmarkResult("CPU Naive")
    
    for _ in range(n_trials):
        start = time.perf_counter()
        
        # Brute force: compute all distances for each detection
        for det in detection_points:
            dists = np.sqrt(np.sum((planogram_points - det) ** 2, axis=1))
            _ = np.argmin(dists)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        result.add_trial(elapsed_ms)
    
    return result


def benchmark_cpu_naive_vectorized(
    planogram_points: np.ndarray,
    detection_points: np.ndarray,
    n_trials: int = 3
) -> BenchmarkResult:
    """
    Benchmark CPU vectorized brute-force (batch distance matrix).
    
    Complexity: O(n × m) but with better memory access patterns
    """
    result = BenchmarkResult("CPU Naive Vectorized")
    
    for _ in range(n_trials):
        start = time.perf_counter()
        
        # Compute full distance matrix at once
        diff = detection_points[:, np.newaxis, :] - planogram_points[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        _ = np.argmin(distances, axis=1)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        result.add_trial(elapsed_ms)
    
    return result


def benchmark_cpu_kdtree(
    planogram_points: np.ndarray,
    detection_points: np.ndarray,
    n_trials: int = 3
) -> BenchmarkResult:
    """
    Benchmark CPU KD-tree nearest neighbor search.
    
    Complexity: O(n log n + m log n) where n = planogram, m = detections
    """
    result = BenchmarkResult("CPU KD-Tree")
    
    for _ in range(n_trials):
        start = time.perf_counter()
        
        # Build tree and query
        tree = KDTree(planogram_points)
        for det in detection_points:
            _ = tree.nearest_neighbor(det)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        result.add_trial(elapsed_ms)
    
    return result


def benchmark_cpu_kdtree_batch(
    planogram_points: np.ndarray,
    detection_points: np.ndarray,
    n_trials: int = 3
) -> BenchmarkResult:
    """
    Benchmark CPU KD-tree with batch query method.
    """
    result = BenchmarkResult("CPU KD-Tree Batch")
    
    for _ in range(n_trials):
        start = time.perf_counter()
        
        tree = KDTree(planogram_points)
        _, _ = tree.query_batch(detection_points)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        result.add_trial(elapsed_ms)
    
    return result


def benchmark_gpu(
    planogram_points: np.ndarray,
    detection_points: np.ndarray,
    n_trials: int = 3,
    warmup: int = 1
) -> BenchmarkResult:
    """
    Benchmark GPU-accelerated nearest neighbor search.
    
    Complexity: O(n × m / p) where p = GPU parallelism
    """
    result = BenchmarkResult("GPU")
    
    if not is_gpu_available():
        result.metadata['note'] = 'GPU not available, using CPU fallback'
    
    # Warmup runs
    for _ in range(warmup):
        _ = gpu_assign_to_voronoi_cells(planogram_points, detection_points)
    
    for _ in range(n_trials):
        start = time.perf_counter()
        
        _, _ = gpu_assign_to_voronoi_cells(planogram_points, detection_points)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        result.add_trial(elapsed_ms)
    
    return result


def run_benchmark_suite(
    sizes: List[int],
    n_trials: int = 3,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Run complete benchmark suite for multiple problem sizes.
    
    Args:
        sizes: List of problem sizes (number of products)
        n_trials: Number of timing trials per benchmark
        verbose: Print progress information
    
    Returns:
        List of result dictionaries
    """
    results = []
    
    for size in sizes:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Benchmarking size: {size}")
            print('='*60)
        
        # Calculate grid dimensions for approximately 'size' products
        cols = int(np.sqrt(size * 4))
        rows = max(1, size // cols)
        while rows * cols < size:
            cols += 1
        
        # Generate synthetic scenario
        scenario = generate_shelf_scenario(
            num_rows=rows,
            num_cols=cols,
            noise_std=0.6,
            anomaly_rate=0.15,
            seed=42 + size
        )
        
        planogram_points = scenario.planogram.get_points_array()
        detection_points = scenario.detections.get_points_array()
        
        actual_size = len(planogram_points)
        n_detections = len(detection_points)
        
        if verbose:
            print(f"  Grid: {rows}×{cols} = {actual_size} positions")
            print(f"  Detections: {n_detections}")
        
        # Run benchmarks
        result_entry = {
            'num_points': actual_size,
            'num_detections': n_detections,
            'grid_rows': rows,
            'grid_cols': cols
        }
        
        # CPU Naive (skip for very large sizes)
        if actual_size <= 5000:
            if verbose:
                print("  Running CPU Naive...")
            naive_result = benchmark_cpu_naive(planogram_points, detection_points, n_trials)
            result_entry['cpu_naive_ms'] = naive_result.mean_ms
            result_entry['cpu_naive_std'] = naive_result.std_ms
            if verbose:
                print(f"    {naive_result.mean_ms:.2f} ± {naive_result.std_ms:.2f} ms")
        else:
            result_entry['cpu_naive_ms'] = np.nan
            result_entry['cpu_naive_std'] = np.nan
            if verbose:
                print("  Skipping CPU Naive (too slow for this size)")
        
        # CPU Naive Vectorized
        if verbose:
            print("  Running CPU Naive Vectorized...")
        vec_result = benchmark_cpu_naive_vectorized(planogram_points, detection_points, n_trials)
        result_entry['cpu_vectorized_ms'] = vec_result.mean_ms
        result_entry['cpu_vectorized_std'] = vec_result.std_ms
        if verbose:
            print(f"    {vec_result.mean_ms:.2f} ± {vec_result.std_ms:.2f} ms")
        
        # CPU KD-Tree
        if verbose:
            print("  Running CPU KD-Tree...")
        kdtree_result = benchmark_cpu_kdtree(planogram_points, detection_points, n_trials)
        result_entry['cpu_kdtree_ms'] = kdtree_result.mean_ms
        result_entry['cpu_kdtree_std'] = kdtree_result.std_ms
        if verbose:
            print(f"    {kdtree_result.mean_ms:.2f} ± {kdtree_result.std_ms:.2f} ms")
        
        # CPU KD-Tree Batch
        if verbose:
            print("  Running CPU KD-Tree Batch...")
        kdtree_batch_result = benchmark_cpu_kdtree_batch(planogram_points, detection_points, n_trials)
        result_entry['cpu_kdtree_batch_ms'] = kdtree_batch_result.mean_ms
        result_entry['cpu_kdtree_batch_std'] = kdtree_batch_result.std_ms
        if verbose:
            print(f"    {kdtree_batch_result.mean_ms:.2f} ± {kdtree_batch_result.std_ms:.2f} ms")
        
        # GPU
        if verbose:
            print("  Running GPU...")
        gpu_result = benchmark_gpu(planogram_points, detection_points, n_trials)
        result_entry['gpu_ms'] = gpu_result.mean_ms
        result_entry['gpu_std'] = gpu_result.std_ms
        result_entry['gpu_available'] = is_gpu_available()
        if verbose:
            print(f"    {gpu_result.mean_ms:.2f} ± {gpu_result.std_ms:.2f} ms")
        
        # Calculate speedups
        baseline_ms = result_entry.get('cpu_naive_ms', result_entry['cpu_vectorized_ms'])
        if not np.isnan(baseline_ms):
            result_entry['speedup_kdtree_vs_naive'] = baseline_ms / result_entry['cpu_kdtree_ms']
            result_entry['speedup_gpu_vs_naive'] = baseline_ms / result_entry['gpu_ms']
        else:
            result_entry['speedup_kdtree_vs_naive'] = result_entry['cpu_vectorized_ms'] / result_entry['cpu_kdtree_ms']
            result_entry['speedup_gpu_vs_naive'] = result_entry['cpu_vectorized_ms'] / result_entry['gpu_ms']
        
        result_entry['speedup_gpu_vs_kdtree'] = result_entry['cpu_kdtree_ms'] / result_entry['gpu_ms']
        
        results.append(result_entry)
    
    return results


def save_results_csv(results: List[Dict[str, Any]], filepath: str):
    """Save benchmark results to CSV file."""
    if not results:
        return
    
    fieldnames = list(results[0].keys())
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {filepath}")


def print_results_table(results: List[Dict[str, Any]]):
    """Print formatted results table."""
    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 90)
    
    # Header
    print(f"{'Size':>8} {'Naive(ms)':>12} {'Vec(ms)':>10} {'KD-Tree':>10} "
          f"{'GPU(ms)':>10} {'Speedup':>10}")
    print("-" * 90)
    
    for r in results:
        naive_str = f"{r['cpu_naive_ms']:.2f}" if not np.isnan(r.get('cpu_naive_ms', np.nan)) else "N/A"
        print(f"{r['num_points']:>8} {naive_str:>12} "
              f"{r['cpu_vectorized_ms']:>10.2f} {r['cpu_kdtree_ms']:>10.2f} "
              f"{r['gpu_ms']:>10.2f} {r['speedup_gpu_vs_naive']:>10.2f}×")
    
    print("=" * 90)


def print_analysis(results: List[Dict[str, Any]]):
    """Print analysis of benchmark results."""
    print("\nPERFORMANCE ANALYSIS")
    print("-" * 60)
    
    # Check hypothesis (3-5x speedup for 1000-10000 range)
    medium_results = [r for r in results if 1000 <= r['num_points'] <= 10000]
    
    if medium_results:
        avg_speedup = np.mean([r['speedup_gpu_vs_naive'] for r in medium_results])
        print(f"\nAverage speedup (1K-10K range): {avg_speedup:.2f}×")
        
        if 3 <= avg_speedup <= 5:
            print("✓ Hypothesis CONFIRMED: Speedup in expected 3-5× range")
        elif avg_speedup > 5:
            print("✓ Hypothesis EXCEEDED: Speedup better than expected!")
        else:
            print("○ Hypothesis not met: Speedup below expected range")
            print("  (This may be due to GPU overhead or hardware limitations)")
    
    # Scaling analysis
    if len(results) >= 2:
        sizes = [r['num_points'] for r in results]
        kdtree_times = [r['cpu_kdtree_ms'] for r in results]
        gpu_times = [r['gpu_ms'] for r in results]
        
        print("\nScaling Behavior:")
        print(f"  Problem size range: {min(sizes)} to {max(sizes)}")
        print(f"  KD-Tree time range: {min(kdtree_times):.2f} to {max(kdtree_times):.2f} ms")
        print(f"  GPU time range: {gpu_times[0]:.2f} to {gpu_times[-1]:.2f} ms")
    
    # GPU status (unificado con get_gpu_info)
    gpu_info = get_gpu_info()
    print(f"\nGPU Status: {'Available' if gpu_info['gpu_available'] else 'Not available (using CPU fallback)'}")
    if gpu_info['device_name']:
        print(f"GPU Device: {gpu_info['device_name']}")
    if gpu_info.get("backend"):
        print(f"GPU Backend: {gpu_info['backend']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Benchmark CPU vs GPU performance for Voronoi assignment'
    )
    parser.add_argument(
        '--sizes', type=str, default='100,500,1000,2500,5000,10000',
        help='Comma-separated problem sizes (default: 100,500,1000,2500,5000,10000)'
    )
    parser.add_argument(
        '--trials', type=int, default=3,
        help='Number of timing trials per benchmark (default: 3)'
    )
    parser.add_argument(
        '--output', type=str, default='benchmarks/benchmark_results.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Minimal output'
    )
    
    args = parser.parse_args()
    
    # Parse sizes
    sizes = [int(s.strip()) for s in args.sizes.split(',')]
    
    # Print header
    if not args.quiet:
        print("=" * 60)
        print("  VORONOI ASSIGNMENT BENCHMARK")
        print("  CPU Naive vs KD-Tree vs GPU")
        print("=" * 60)
        print(f"\nProblem sizes: {sizes}")
        print(f"Trials per size: {args.trials}")
        
        gpu_info = get_gpu_info()
        print(f"\nGPU Available: {gpu_info['gpu_available']}")
        if gpu_info.get("backend"):
            print(f"GPU Backend: {gpu_info['backend']}")
        if gpu_info['device_name']:
            print(f"GPU Device: {gpu_info['device_name']}")
        total_mem = gpu_info.get('total_memory_gb')
        if total_mem is not None:
            print(f"GPU Memory: {total_mem:.1f} GB")
    
    # Run benchmarks
    results = run_benchmark_suite(sizes, args.trials, verbose=not args.quiet)
    
    # Print summary
    print_results_table(results)
    
    if not args.quiet:
        print_analysis(results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results_csv(results, str(output_path))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
