"""
Main Entry Point for Shelf Product Anomaly Detector

This script provides a command-line interface for running the planogram
compliance verification system. It orchestrates:

1. Synthetic data generation (planogram + detections)
2. Voronoi-based spatial assignment
3. Anomaly detection and classification
4. Compliance reporting

Usage:
    # Generate data and run analysis
    python -m src.main --generate-data --num-rows 3 --num-cols 12 --anomaly-rate 0.15

    # Run with existing data files
    python -m src.main --input data/scenario_example_small.json

    # Run benchmark comparison
    python -m src.main --benchmark --sizes 100,1000,10000

All data is synthetically generated - there is NO external dataset.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

import numpy as np

from .synthetic_data import (
    generate_shelf_scenario,
    save_scenario_to_json,
    load_scenario_from_json,
    visualize_scenario,
    SKUPattern
)
from .data_models import ShelfScenario, ComplianceReport
from .anomaly.anomaly_detector import (
    AnomalyDetector,
    detect_anomalies,
    generate_compliance_report,
    validate_against_ground_truth
)
from .geometry.voronoi_cpu import assign_to_voronoi_cells, compute_assignment_statistics
from .geometry.voronoi_gpu import is_gpu_available, gpu_assign_to_voronoi_cells
from .geometry.kd_tree import KDTree
from .hpc.timing import Timer, BenchmarkResult
from .hpc.gpu_kernels import get_gpu_info


def print_header():
    """Print application header."""
    print("=" * 70)
    print("  PARALLEL VORONOI-BASED SHELF PRODUCT ANOMALY DETECTOR")
    print("  GPU-Accelerated Spatial Search for Planogram Compliance")
    print("=" * 70)
    print()


def print_system_info():
    """Print system and GPU information."""
    print("System Configuration:")
    print("-" * 40)
    
    gpu_info = get_gpu_info()
    print(f"  GPU Available: {gpu_info['gpu_available']}")
    if gpu_info['gpu_available']:
        print(f"  CuPy Backend: {gpu_info['cupy_available']}")
        print(f"  Numba CUDA: {gpu_info['numba_cuda_available']}")
        if gpu_info['device_name']:
            print(f"  Device: {gpu_info['device_name']}")
            print(f"  Memory: {gpu_info['total_memory_gb']:.1f} GB total, "
                  f"{gpu_info['free_memory_gb']:.1f} GB free")
    print()


def generate_data(args) -> ShelfScenario:
    """
    Generate synthetic planogram and detection data.
    
    Args:
        args: Command line arguments
    
    Returns:
        Generated ShelfScenario
    """
    print("Generating Synthetic Data...")
    print("-" * 40)
    print(f"  Shelf dimensions: {args.num_rows} rows × {args.num_cols} columns")
    print(f"  Total positions: {args.num_rows * args.num_cols}")
    print(f"  Spacing: {args.spacing_x} × {args.spacing_y}")
    print(f"  Noise std: {args.noise_std}")
    print(f"  Anomaly rate: {args.anomaly_rate:.1%}")
    print(f"  Random seed: {args.seed}")
    print()
    
    # Map pattern string to enum
    pattern_map = {
        'unique': SKUPattern.UNIQUE,
        'blocks': SKUPattern.BLOCKS,
        'rows': SKUPattern.ROWS,
        'random': SKUPattern.RANDOM,
        'mixed': SKUPattern.MIXED
    }
    sku_pattern = pattern_map.get(args.sku_pattern, SKUPattern.UNIQUE)
    
    scenario = generate_shelf_scenario(
        num_rows=args.num_rows,
        num_cols=args.num_cols,
        spacing_x=args.spacing_x,
        spacing_y=args.spacing_y,
        noise_std=args.noise_std,
        anomaly_rate=args.anomaly_rate,
        sku_pattern=sku_pattern,
        seed=args.seed,
        scenario_id=f"generated_{args.num_rows}x{args.num_cols}"
    )
    
    print(f"Generated scenario: {scenario.scenario_id}")
    print(f"  Planogram entries: {len(scenario.planogram.entries)}")
    print(f"  Detection entries: {len(scenario.detections.entries)}")
    
    if scenario.ground_truth:
        print(f"  Ground truth anomalies: {len(scenario.ground_truth.anomaly_indices)}")
        print(f"  Missing SKUs: {len(scenario.ground_truth.missing_indices)}")
        print(f"  Swapped pairs: {len(scenario.ground_truth.swapped_pairs)}")
        print(f"  Foreign SKUs: {len(scenario.ground_truth.foreign_skus)}")
    print()
    
    # Save to data directory
    if args.save_data:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        planogram_path, detections_path = save_scenario_to_json(scenario, str(output_dir))
        print(f"Data saved to:")
        print(f"  {planogram_path}")
        print(f"  {detections_path}")
        
        # Save complete scenario
        scenario_path = output_dir / f"scenario_{scenario.scenario_id}.json"
        scenario.save_to_json(str(scenario_path))
        print(f"  {scenario_path}")
        print()
    
    return scenario


def load_data(args) -> ShelfScenario:
    """
    Load scenario from JSON file.
    
    Args:
        args: Command line arguments
    
    Returns:
        Loaded ShelfScenario
    """
    print(f"Loading data from: {args.input}")
    print("-" * 40)
    
    scenario = load_scenario_from_json(args.input)
    
    print(f"Loaded scenario: {scenario.scenario_id}")
    print(f"  Planogram entries: {len(scenario.planogram.entries)}")
    print(f"  Detection entries: {len(scenario.detections.entries)}")
    print()
    
    return scenario


def run_analysis(scenario: ShelfScenario, args) -> ComplianceReport:
    """
    Run the complete anomaly detection analysis.
    
    Args:
        scenario: Shelf scenario to analyze
        args: Command line arguments
    
    Returns:
        ComplianceReport with results
    """
    print("Running Anomaly Detection Analysis...")
    print("-" * 40)
    
    # Determine method
    use_gpu = args.use_gpu and is_gpu_available()
    method = "gpu" if use_gpu else args.method
    
    print(f"  Method: {method}")
    print(f"  GPU enabled: {use_gpu}")
    print()
    
    # Create detector and analyze
    detector = AnomalyDetector(
        use_gpu=use_gpu,
        distance_threshold=args.distance_threshold
    )
    
    with Timer("Anomaly Detection") as timer:
        report = detector.analyze(scenario, method=method)
    
    print(f"  Processing time: {report.processing_time_ms:.2f} ms")
    print()
    
    return report


def print_report(report: ComplianceReport, verbose: bool = True):
    """Print the compliance report."""
    print(report.summary())
    
    if verbose and report.anomalies:
        print("\nDetailed Anomalies (first 20):")
        print("-" * 40)
        
        for i, anomaly in enumerate(report.anomalies[:20]):
            print(f"  {i+1}. [{anomaly.anomaly_type.value.upper():10}] "
                  f"Cell {anomaly.planogram_index:4}: {anomaly.details}")
        
        if len(report.anomalies) > 20:
            print(f"  ... and {len(report.anomalies) - 20} more anomalies")
    print()


def run_validation(scenario: ShelfScenario, report: ComplianceReport):
    """Validate results against ground truth if available."""
    if scenario.ground_truth is None:
        print("No ground truth available for validation.")
        return
    
    print("Ground Truth Validation:")
    print("-" * 40)
    
    metrics = validate_against_ground_truth(scenario, use_gpu=False)
    
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print()
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  F1 Score:  {metrics['f1_score']:.2%}")
    print()


def run_benchmark(args):
    """Run performance benchmarks comparing different methods."""
    print("Running Performance Benchmarks...")
    print("-" * 40)
    
    # Parse sizes
    sizes = [int(s.strip()) for s in args.sizes.split(',')]
    
    print(f"  Problem sizes: {sizes}")
    print(f"  Trials per size: {args.trials}")
    print()
    
    results = []
    
    for size in sizes:
        print(f"\nBenchmarking size: {size}")
        
        # Calculate grid dimensions
        cols = int(np.sqrt(size * 4))
        rows = max(1, size // cols)
        while rows * cols < size:
            cols += 1
        
        # Generate scenario
        scenario = generate_shelf_scenario(
            num_rows=rows,
            num_cols=cols,
            anomaly_rate=0.15,
            seed=42 + size
        )
        
        actual_size = len(scenario.planogram.entries)
        print(f"  Actual size: {actual_size} ({rows}×{cols})")
        
        planogram_points = scenario.planogram.get_points_array()
        detection_points = scenario.detections.get_points_array()
        
        # Benchmark CPU brute force
        cpu_naive_times = []
        for _ in range(args.trials):
            with Timer() as t:
                for det in detection_points:
                    dists = np.sqrt(np.sum((planogram_points - det) ** 2, axis=1))
                    _ = np.argmin(dists)
            cpu_naive_times.append(t.elapsed_ms)
        cpu_naive_ms = np.mean(cpu_naive_times)
        
        # Benchmark CPU KD-tree
        kdtree_times = []
        for _ in range(args.trials):
            with Timer() as t:
                tree = KDTree(planogram_points)
                for det in detection_points:
                    _ = tree.nearest_neighbor(det)
            kdtree_times.append(t.elapsed_ms)
        cpu_kdtree_ms = np.mean(kdtree_times)
        
        # Benchmark GPU (if available)
        if is_gpu_available():
            # Warmup
            _ = gpu_assign_to_voronoi_cells(planogram_points, detection_points)
            
            gpu_times = []
            for _ in range(args.trials):
                with Timer() as t:
                    _ = gpu_assign_to_voronoi_cells(planogram_points, detection_points)
                gpu_times.append(t.elapsed_ms)
            gpu_ms = np.mean(gpu_times)
        else:
            gpu_ms = cpu_kdtree_ms  # Fallback
        
        # Calculate speedups
        speedup_naive_vs_kdtree = cpu_naive_ms / cpu_kdtree_ms
        speedup_naive_vs_gpu = cpu_naive_ms / gpu_ms
        speedup_kdtree_vs_gpu = cpu_kdtree_ms / gpu_ms
        
        result = {
            'num_points': actual_size,
            'cpu_naive_ms': cpu_naive_ms,
            'cpu_kdtree_ms': cpu_kdtree_ms,
            'gpu_ms': gpu_ms,
            'speedup_naive_vs_kdtree': speedup_naive_vs_kdtree,
            'speedup_naive_vs_gpu': speedup_naive_vs_gpu,
            'speedup_kdtree_vs_gpu': speedup_kdtree_vs_gpu
        }
        results.append(result)
        
        print(f"  CPU Naive:   {cpu_naive_ms:>10.2f} ms")
        print(f"  CPU KD-tree: {cpu_kdtree_ms:>10.2f} ms  ({speedup_naive_vs_kdtree:.2f}× vs naive)")
        print(f"  GPU:         {gpu_ms:>10.2f} ms  ({speedup_naive_vs_gpu:.2f}× vs naive)")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Size':>10} {'Naive(ms)':>12} {'KD-tree(ms)':>12} {'GPU(ms)':>12} {'Speedup':>10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['num_points']:>10} {r['cpu_naive_ms']:>12.2f} "
              f"{r['cpu_kdtree_ms']:>12.2f} {r['gpu_ms']:>12.2f} "
              f"{r['speedup_naive_vs_gpu']:>10.2f}×")
    
    # Save results to CSV
    if args.save_benchmark:
        import csv
        output_path = Path('benchmarks/benchmark_results.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nResults saved to: {output_path}")
    
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Parallel Voronoi-Based Shelf Product Anomaly Detector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate data and run analysis
  python -m src.main --generate-data --num-rows 3 --num-cols 12

  # Load existing data
  python -m src.main --input data/scenario_example_small.json

  # Run benchmarks
  python -m src.main --benchmark --sizes 100,1000,10000

  # Full analysis with visualization
  python -m src.main --generate-data --visualize --verbose
        """
    )
    
    # Data source options
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        '--generate-data', '-g',
        action='store_true',
        help='Generate synthetic data'
    )
    data_group.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input scenario JSON file'
    )
    data_group.add_argument(
        '--benchmark', '-b',
        action='store_true',
        help='Run performance benchmarks'
    )
    
    # Data generation parameters
    gen_group = parser.add_argument_group('Data Generation')
    gen_group.add_argument('--num-rows', type=int, default=3,
                          help='Number of shelf rows (default: 3)')
    gen_group.add_argument('--num-cols', type=int, default=12,
                          help='Number of products per row (default: 12)')
    gen_group.add_argument('--spacing-x', type=float, default=10.0,
                          help='Horizontal spacing (default: 10.0)')
    gen_group.add_argument('--spacing-y', type=float, default=5.0,
                          help='Vertical spacing (default: 5.0)')
    gen_group.add_argument('--noise-std', type=float, default=0.6,
                          help='Position noise std (default: 0.6)')
    gen_group.add_argument('--anomaly-rate', type=float, default=0.15,
                          help='Anomaly probability (default: 0.15)')
    gen_group.add_argument('--sku-pattern', type=str, default='unique',
                          choices=['unique', 'blocks', 'rows', 'random', 'mixed'],
                          help='SKU distribution pattern (default: unique)')
    gen_group.add_argument('--seed', type=int, default=42,
                          help='Random seed (default: 42)')
    
    # Processing options
    proc_group = parser.add_argument_group('Processing')
    proc_group.add_argument('--method', type=str, default='kdtree',
                           choices=['kdtree', 'brute_force', 'auto'],
                           help='Assignment method (default: kdtree)')
    proc_group.add_argument('--use-gpu', action='store_true', default=True,
                           help='Use GPU acceleration if available')
    proc_group.add_argument('--no-gpu', action='store_true',
                           help='Disable GPU acceleration')
    proc_group.add_argument('--distance-threshold', type=float, default=10.0,
                           help='Distance threshold for assignment (default: 10.0)')
    
    # Benchmark options
    bench_group = parser.add_argument_group('Benchmarking')
    bench_group.add_argument('--sizes', type=str, default='100,500,1000,5000',
                            help='Comma-separated problem sizes (default: 100,500,1000,5000)')
    bench_group.add_argument('--trials', type=int, default=3,
                            help='Number of timing trials (default: 3)')
    bench_group.add_argument('--save-benchmark', action='store_true',
                            help='Save benchmark results to CSV')
    
    # Output options
    out_group = parser.add_argument_group('Output')
    out_group.add_argument('--output-dir', type=str, default='data',
                          help='Output directory for generated data (default: data)')
    out_group.add_argument('--save-data', action='store_true', default=True,
                          help='Save generated data to files')
    out_group.add_argument('--no-save', action='store_true',
                          help='Do not save generated data')
    out_group.add_argument('--visualize', '-v', action='store_true',
                          help='Show visualization plot')
    out_group.add_argument('--verbose', action='store_true',
                          help='Verbose output with detailed anomalies')
    out_group.add_argument('--quiet', '-q', action='store_true',
                          help='Minimal output')
    
    args = parser.parse_args()
    
    # Handle flag conflicts
    if args.no_gpu:
        args.use_gpu = False
    if args.no_save:
        args.save_data = False
    
    # Print header
    if not args.quiet:
        print_header()
        print_system_info()
    
    # Run benchmark mode
    if args.benchmark:
        run_benchmark(args)
        return 0
    
    # Get or generate scenario
    if args.generate_data or args.input is None:
        scenario = generate_data(args)
    else:
        scenario = load_data(args)
    
    # Run analysis
    report = run_analysis(scenario, args)
    
    # Print report
    if not args.quiet:
        print_report(report, verbose=args.verbose)
    
    # Validate against ground truth
    if scenario.ground_truth and not args.quiet:
        run_validation(scenario, report)
    
    # Show visualization
    if args.visualize:
        try:
            visualize_scenario(scenario, show_connections=True)
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    # Print final summary
    if not args.quiet:
        print("=" * 70)
        print(f"Analysis complete. Compliance score: {report.compliance_score:.1%}")
        print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
