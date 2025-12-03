"""
GPU-Accelerated Voronoi Assignment for Shelf Product Analysis

This module provides GPU-accelerated spatial assignment using CuPy
or Numba CUDA. The GPU excels at batched distance computations where
we need to compute many distances in parallel.

Strategy:
    For n planogram points and m detections, we compute an m×n distance
    matrix in parallel on the GPU, then find the argmin for each row.
    This is embarrassingly parallel and maps well to GPU architecture.

Complexity Analysis:
    - Distance matrix: O(m × n) operations, fully parallelizable
    - Argmin reduction: O(m × n / p) where p = number of GPU cores
    - Memory: O(m × n) for distance matrix (can be problematic for very large inputs)

Fallback Behavior:
    If GPU is not available (no CUDA, no CuPy), the module gracefully
    falls back to CPU implementations using NumPy. This ensures the
    code runs on any machine while benefiting from GPU when available.

Performance Notes:
    - GPU overhead dominates for small inputs (< 1000 points)
    - GPU shines for large inputs (> 5000 points)
    - Memory transfer can be a bottleneck; reuse GPU arrays when possible
"""

from typing import Tuple, Optional
import numpy as np

# Try to import GPU libraries
_GPU_AVAILABLE = False
_GPU_BACKEND = None

try:
    import cupy as cp
    _GPU_AVAILABLE = True
    _GPU_BACKEND = "cupy"
except ImportError:
    cp = None

# Try Numba as alternative
if not _GPU_AVAILABLE:
    try:
        from numba import cuda
        if cuda.is_available():
            _GPU_AVAILABLE = True
            _GPU_BACKEND = "numba"
    except ImportError:
        pass


def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is available.
    
    Returns:
        True if CuPy or Numba CUDA is available and functional
    """
    return _GPU_AVAILABLE


def get_gpu_backend() -> Optional[str]:
    """
    Get the name of the available GPU backend.
    
    Returns:
        "cupy", "numba", or None if no GPU available
    """
    return _GPU_BACKEND


def _compute_distance_matrix_cupy(
    planogram_points: np.ndarray,
    detection_points: np.ndarray
) -> np.ndarray:
    """
    Compute distance matrix using CuPy (CUDA).
    
    Uses broadcasting to compute all pairwise distances in parallel.
    
    Args:
        planogram_points: Shape (n, 2)
        detection_points: Shape (m, 2)
    
    Returns:
        Distance matrix of shape (m, n)
    
    Complexity:
        Time: O(m × n / p) where p = GPU parallelism
        Space: O(m × n) on GPU
    """
    # Transfer to GPU
    plan_gpu = cp.asarray(planogram_points)
    det_gpu = cp.asarray(detection_points)
    
    # Compute squared distances using broadcasting
    # det_gpu: (m, 2) -> (m, 1, 2)
    # plan_gpu: (n, 2) -> (1, n, 2)
    # diff: (m, n, 2)
    diff = det_gpu[:, np.newaxis, :] - plan_gpu[np.newaxis, :, :]
    
    # Squared Euclidean distance
    sq_distances = cp.sum(diff ** 2, axis=2)
    
    # Return sqrt distances (transfer back to CPU)
    distances = cp.sqrt(sq_distances)
    return cp.asnumpy(distances)


def _compute_distance_matrix_numba(
    planogram_points: np.ndarray,
    detection_points: np.ndarray
) -> np.ndarray:
    """
    Compute distance matrix using Numba CUDA.
    
    Args:
        planogram_points: Shape (n, 2)
        detection_points: Shape (m, 2)
    
    Returns:
        Distance matrix of shape (m, n)
    """
    from numba import cuda
    import math
    
    @cuda.jit
    def distance_kernel(planogram, detections, result):
        """CUDA kernel for pairwise distance computation."""
        i, j = cuda.grid(2)
        m, n = result.shape
        
        if i < m and j < n:
            dx = detections[i, 0] - planogram[j, 0]
            dy = detections[i, 1] - planogram[j, 1]
            result[i, j] = math.sqrt(dx * dx + dy * dy)
    
    m = len(detection_points)
    n = len(planogram_points)
    
    # Allocate result array
    result = np.zeros((m, n), dtype=np.float64)
    
    # Configure grid
    threads_per_block = (16, 16)
    blocks_per_grid_x = (m + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (n + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # Copy to device and execute
    d_planogram = cuda.to_device(planogram_points)
    d_detections = cuda.to_device(detection_points)
    d_result = cuda.to_device(result)
    
    distance_kernel[blocks_per_grid, threads_per_block](
        d_planogram, d_detections, d_result
    )
    
    # Copy result back
    return d_result.copy_to_host()


def _compute_distance_matrix_cpu(
    planogram_points: np.ndarray,
    detection_points: np.ndarray
) -> np.ndarray:
    """
    Compute distance matrix using NumPy (CPU fallback).
    
    Uses vectorized operations for efficiency, but runs on CPU.
    
    Args:
        planogram_points: Shape (n, 2)
        detection_points: Shape (m, 2)
    
    Returns:
        Distance matrix of shape (m, n)
    
    Complexity:
        Time: O(m × n)
        Space: O(m × n)
    """
    # Broadcasting approach (memory intensive but fast)
    # det: (m, 2) -> (m, 1, 2)
    # plan: (n, 2) -> (1, n, 2)
    diff = detection_points[:, np.newaxis, :] - planogram_points[np.newaxis, :, :]
    
    # Squared distances and sqrt
    sq_distances = np.sum(diff ** 2, axis=2)
    distances = np.sqrt(sq_distances)
    
    return distances


def compute_distance_matrix(
    planogram_points: np.ndarray,
    detection_points: np.ndarray,
    force_cpu: bool = False
) -> np.ndarray:
    """
    Compute pairwise distance matrix, using GPU if available.
    
    This is the main entry point for batched distance computation.
    It automatically selects the best available backend.
    
    Args:
        planogram_points: Array of shape (n, 2) with planogram positions
        detection_points: Array of shape (m, 2) with detection positions
        force_cpu: If True, use CPU even if GPU is available
    
    Returns:
        Distance matrix of shape (m, n) where result[i, j] is the
        Euclidean distance from detection i to planogram point j
    
    Complexity:
        GPU: O(m × n / p) time, O(m × n) space
        CPU: O(m × n) time, O(m × n) space
    
    Example:
        >>> plan = np.array([[0, 0], [10, 0], [20, 0]])
        >>> det = np.array([[1, 1], [9, -1]])
        >>> dists = compute_distance_matrix(plan, det)
        >>> print(dists.shape)  # (2, 3)
    """
    planogram_points = np.asarray(planogram_points, dtype=np.float64)
    detection_points = np.asarray(detection_points, dtype=np.float64)
    
    if not force_cpu and _GPU_AVAILABLE:
        if _GPU_BACKEND == "cupy":
            return _compute_distance_matrix_cupy(planogram_points, detection_points)
        elif _GPU_BACKEND == "numba":
            return _compute_distance_matrix_numba(planogram_points, detection_points)
    
    return _compute_distance_matrix_cpu(planogram_points, detection_points)


def gpu_assign_to_voronoi_cells(
    planogram_points: np.ndarray,
    detection_points: np.ndarray,
    force_cpu: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated Voronoi cell assignment via nearest neighbor.
    
    Computes the distance matrix on GPU (if available), then finds
    the nearest planogram point for each detection.
    
    This is mathematically equivalent to determining which Voronoi
    cell each detection falls into, since Voronoi cells are defined
    by nearest-neighbor regions.
    
    Args:
        planogram_points: Array of shape (n, 2) with planogram positions
        detection_points: Array of shape (m, 2) with detection positions
        force_cpu: If True, use CPU even if GPU is available
    
    Returns:
        Tuple of:
        - assignments: Array of shape (m,) with planogram indices
        - distances: Array of shape (m,) with distances to assigned cells
    
    Complexity:
        Time: O(m × n / p) for distance matrix + O(m × n) for argmin
        Space: O(m × n) for distance matrix
        
        For m = n = 10,000 on GPU with 1000 cores:
        - ~100M distance computations / 1000 cores = 100K iterations
        - Much faster than sequential O(m × n) = 100M iterations
    
    Example:
        >>> plan = np.array([[0, 0], [10, 0], [20, 0]])
        >>> det = np.array([[1, 1], [9, -1], [21, 0.5]])
        >>> assignments, distances = gpu_assign_to_voronoi_cells(plan, det)
        >>> print(assignments)  # [0, 1, 2]
    """
    # Compute full distance matrix
    dist_matrix = compute_distance_matrix(
        planogram_points, detection_points, force_cpu=force_cpu
    )
    
    # Find nearest neighbor for each detection (argmin along axis 1)
    # This finds the planogram index with minimum distance for each detection
    if not force_cpu and _GPU_AVAILABLE and _GPU_BACKEND == "cupy":
        dist_gpu = cp.asarray(dist_matrix)
        assignments_gpu = cp.argmin(dist_gpu, axis=1)
        assignments = cp.asnumpy(assignments_gpu).astype(np.int32)
    else:
        assignments = np.argmin(dist_matrix, axis=1).astype(np.int32)
    
    # Extract minimum distances
    distances = dist_matrix[np.arange(len(detection_points)), assignments]
    
    return assignments, distances


def gpu_batch_nearest_neighbors(
    planogram_points: np.ndarray,
    detection_points: np.ndarray,
    k: int = 1,
    force_cpu: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated k-nearest neighbors for all detections.
    
    For each detection, find the k nearest planogram points.
    Useful for fuzzy assignment or when multiple candidates are needed.
    
    Args:
        planogram_points: Array of shape (n, 2)
        detection_points: Array of shape (m, 2)
        k: Number of nearest neighbors to find
        force_cpu: If True, use CPU even if GPU is available
    
    Returns:
        Tuple of:
        - indices: Array of shape (m, k) with k nearest planogram indices
        - distances: Array of shape (m, k) with corresponding distances
    
    Complexity:
        Time: O(m × n / p) for distances + O(m × k × n) for sorting
        Space: O(m × n) for distance matrix
    """
    dist_matrix = compute_distance_matrix(
        planogram_points, detection_points, force_cpu=force_cpu
    )
    
    m = len(detection_points)
    k = min(k, len(planogram_points))
    
    # Find k smallest distances per row
    if not force_cpu and _GPU_AVAILABLE and _GPU_BACKEND == "cupy":
        dist_gpu = cp.asarray(dist_matrix)
        # CuPy argpartition for efficient k-selection
        indices_gpu = cp.argpartition(dist_gpu, k, axis=1)[:, :k]
        # Sort the top-k
        row_indices = cp.arange(m)[:, np.newaxis]
        top_k_dists = dist_gpu[row_indices, indices_gpu]
        sort_order = cp.argsort(top_k_dists, axis=1)
        indices_gpu = cp.take_along_axis(indices_gpu, sort_order, axis=1)
        
        indices = cp.asnumpy(indices_gpu).astype(np.int32)
        distances = cp.asnumpy(top_k_dists[row_indices, sort_order])
    else:
        # NumPy version
        indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]
        row_indices = np.arange(m)[:, np.newaxis]
        top_k_dists = dist_matrix[row_indices, indices]
        sort_order = np.argsort(top_k_dists, axis=1)
        indices = np.take_along_axis(indices, sort_order, axis=1).astype(np.int32)
        distances = top_k_dists[row_indices, sort_order]
    
    return indices, distances


def benchmark_gpu_vs_cpu(
    n_planogram: int = 1000,
    n_detections: int = 1000,
    n_trials: int = 5
) -> dict:
    """
    Benchmark GPU vs CPU performance for distance computation.
    
    Args:
        n_planogram: Number of planogram points
        n_detections: Number of detection points
        n_trials: Number of timing trials
    
    Returns:
        Dictionary with timing results and speedup
    """
    import time
    
    # Generate random test data
    np.random.seed(42)
    planogram = np.random.randn(n_planogram, 2) * 100
    detections = np.random.randn(n_detections, 2) * 100
    
    results = {
        'n_planogram': n_planogram,
        'n_detections': n_detections,
        'gpu_available': is_gpu_available(),
        'gpu_backend': get_gpu_backend()
    }
    
    # Benchmark CPU
    cpu_times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        _ = compute_distance_matrix(planogram, detections, force_cpu=True)
        cpu_times.append(time.perf_counter() - start)
    
    results['cpu_mean_ms'] = np.mean(cpu_times) * 1000
    results['cpu_std_ms'] = np.std(cpu_times) * 1000
    
    # Benchmark GPU (if available)
    if is_gpu_available():
        # Warmup
        _ = compute_distance_matrix(planogram, detections, force_cpu=False)
        
        gpu_times = []
        for _ in range(n_trials):
            start = time.perf_counter()
            _ = compute_distance_matrix(planogram, detections, force_cpu=False)
            gpu_times.append(time.perf_counter() - start)
        
        results['gpu_mean_ms'] = np.mean(gpu_times) * 1000
        results['gpu_std_ms'] = np.std(gpu_times) * 1000
        results['speedup'] = results['cpu_mean_ms'] / results['gpu_mean_ms']
    else:
        results['gpu_mean_ms'] = None
        results['gpu_std_ms'] = None
        results['speedup'] = 1.0
    
    return results


def print_gpu_info() -> None:
    """Print information about GPU availability and configuration."""
    print("GPU Configuration")
    print("=" * 40)
    print(f"GPU Available: {is_gpu_available()}")
    print(f"Backend: {get_gpu_backend()}")
    
    if is_gpu_available() and _GPU_BACKEND == "cupy":
        try:
            device = cp.cuda.Device()
            print(f"Device: {device.id}")
            mem = device.mem_info
            print(f"Memory: {mem[1] / 1e9:.1f} GB total, {mem[0] / 1e9:.1f} GB free")
        except Exception as e:
            print(f"Could not get device info: {e}")


if __name__ == "__main__":
    print_gpu_info()
    print()
    
    # Run benchmark
    print("Running GPU vs CPU benchmark...")
    for size in [100, 500, 1000, 5000]:
        results = benchmark_gpu_vs_cpu(n_planogram=size, n_detections=size)
        print(f"\nSize: {size}×{size}")
        print(f"  CPU: {results['cpu_mean_ms']:.2f} ± {results['cpu_std_ms']:.2f} ms")
        if results['gpu_mean_ms'] is not None:
            print(f"  GPU: {results['gpu_mean_ms']:.2f} ± {results['gpu_std_ms']:.2f} ms")
            print(f"  Speedup: {results['speedup']:.2f}×")
        else:
            print("  GPU: Not available")
