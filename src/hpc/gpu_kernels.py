"""
GPU-Accelerated Computational Kernels

This module provides GPU-accelerated implementations of the core
computational operations needed for shelf product analysis:

1. Distance matrix computation
2. Nearest neighbor search
3. Batch operations for large datasets

The module automatically detects GPU availability and falls back
to optimized CPU implementations when no GPU is present.

Performance Characteristics:
- GPU excels for large batches (> 5000 points)
- CPU may be faster for small batches due to transfer overhead
- Memory usage is O(n × m) for distance matrices

Supported Backends:
- CuPy (preferred): Full GPU array operations
- Numba CUDA: Custom kernel support
- NumPy (fallback): Vectorized CPU operations
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np

# Try to import GPU libraries
_CUPY_AVAILABLE = False
_NUMBA_CUDA_AVAILABLE = False

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None

try:
    from numba import cuda
    if cuda.is_available():
        _NUMBA_CUDA_AVAILABLE = True
except ImportError:
    pass


def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is available.
    
    Returns:
        True if CuPy or Numba CUDA is available
    """
    return _CUPY_AVAILABLE or _NUMBA_CUDA_AVAILABLE


def get_gpu_info() -> Dict[str, Any]:
    """
    Get information about available GPU resources.
    
    Returns:
        Dictionary with GPU configuration details
    """
    info = {
        'gpu_available': is_gpu_available(),
        'cupy_available': _CUPY_AVAILABLE,
        'numba_cuda_available': _NUMBA_CUDA_AVAILABLE,
        'device_name': None,
        'total_memory_gb': None,
        'free_memory_gb': None
    }
    
    if _CUPY_AVAILABLE:
        try:
            device = cp.cuda.Device()
            info['device_name'] = f"CUDA Device {device.id}"
            mem = device.mem_info
            info['total_memory_gb'] = mem[1] / 1e9
            info['free_memory_gb'] = mem[0] / 1e9
        except Exception:
            pass
    
    return info


def gpu_distance_matrix(
    points_a: np.ndarray,
    points_b: np.ndarray,
    force_cpu: bool = False
) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix using GPU.
    
    For points_a of shape (m, d) and points_b of shape (n, d),
    computes an (m, n) matrix where result[i, j] = ||a_i - b_j||_2.
    
    Args:
        points_a: First point set, shape (m, d)
        points_b: Second point set, shape (n, d)
        force_cpu: Force CPU execution even if GPU available
    
    Returns:
        Distance matrix of shape (m, n)
    
    Complexity:
        Time: O(m × n × d / p) on GPU with p cores
        Space: O(m × n) for result matrix
    
    Example:
        >>> a = np.array([[0, 0], [1, 1]])
        >>> b = np.array([[0, 0], [2, 2], [3, 3]])
        >>> dists = gpu_distance_matrix(a, b)
        >>> print(dists.shape)  # (2, 3)
    """
    points_a = np.asarray(points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)
    
    if not force_cpu and _CUPY_AVAILABLE:
        return _distance_matrix_cupy(points_a, points_b)
    elif not force_cpu and _NUMBA_CUDA_AVAILABLE:
        return _distance_matrix_numba(points_a, points_b)
    else:
        return _distance_matrix_numpy(points_a, points_b)


def _distance_matrix_cupy(
    points_a: np.ndarray,
    points_b: np.ndarray
) -> np.ndarray:
    """CuPy implementation of distance matrix."""
    a_gpu = cp.asarray(points_a)
    b_gpu = cp.asarray(points_b)
    
    # Expand dimensions for broadcasting
    # a: (m, d) -> (m, 1, d)
    # b: (n, d) -> (1, n, d)
    diff = a_gpu[:, np.newaxis, :] - b_gpu[np.newaxis, :, :]
    
    # Sum of squared differences
    sq_dist = cp.sum(diff ** 2, axis=2)
    
    # Euclidean distance
    dist = cp.sqrt(sq_dist)
    
    return cp.asnumpy(dist)


def _distance_matrix_numba(
    points_a: np.ndarray,
    points_b: np.ndarray
) -> np.ndarray:
    """Numba CUDA implementation of distance matrix."""
    from numba import cuda
    import math
    
    @cuda.jit
    def dist_kernel(a, b, result):
        i, j = cuda.grid(2)
        if i < result.shape[0] and j < result.shape[1]:
            d = a.shape[1]
            sq_sum = 0.0
            for k in range(d):
                diff = a[i, k] - b[j, k]
                sq_sum += diff * diff
            result[i, j] = math.sqrt(sq_sum)
    
    m, d = points_a.shape
    n = points_b.shape[0]
    result = np.zeros((m, n), dtype=np.float64)
    
    # Configure grid
    threads = (16, 16)
    blocks = ((m + 15) // 16, (n + 15) // 16)
    
    # Execute kernel
    d_a = cuda.to_device(points_a)
    d_b = cuda.to_device(points_b)
    d_result = cuda.to_device(result)
    
    dist_kernel[blocks, threads](d_a, d_b, d_result)
    
    return d_result.copy_to_host()


def _distance_matrix_numpy(
    points_a: np.ndarray,
    points_b: np.ndarray
) -> np.ndarray:
    """NumPy (CPU) implementation of distance matrix."""
    # Broadcasting approach
    diff = points_a[:, np.newaxis, :] - points_b[np.newaxis, :, :]
    sq_dist = np.sum(diff ** 2, axis=2)
    return np.sqrt(sq_dist)


def gpu_nearest_neighbors(
    planogram_points: np.ndarray,
    detection_points: np.ndarray,
    force_cpu: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated batch nearest neighbor search.
    
    For each detection point, find the nearest planogram point.
    This is the core operation for Voronoi cell assignment.
    
    Args:
        planogram_points: Shape (n, 2), reference points
        detection_points: Shape (m, 2), query points
        force_cpu: Force CPU execution
    
    Returns:
        Tuple of:
        - indices: Shape (m,), index of nearest planogram point
        - distances: Shape (m,), distance to nearest point
    
    Complexity:
        Time: O(m × n / p) for distance + O(m × n) for argmin
        Space: O(m × n) for distance matrix
    
    Example:
        >>> plan = np.array([[0, 0], [10, 0], [20, 0]])
        >>> det = np.array([[1, 1], [9, -1]])
        >>> indices, distances = gpu_nearest_neighbors(plan, det)
        >>> print(indices)  # [0, 1]
    """
    # Compute distance matrix
    dist_matrix = gpu_distance_matrix(
        detection_points, planogram_points, force_cpu=force_cpu
    )
    
    # Find argmin for each row
    if not force_cpu and _CUPY_AVAILABLE:
        dist_gpu = cp.asarray(dist_matrix)
        indices = cp.argmin(dist_gpu, axis=1)
        indices = cp.asnumpy(indices).astype(np.int32)
    else:
        indices = np.argmin(dist_matrix, axis=1).astype(np.int32)
    
    # Extract minimum distances
    distances = dist_matrix[np.arange(len(detection_points)), indices]
    
    return indices, distances


def gpu_k_nearest_neighbors(
    planogram_points: np.ndarray,
    detection_points: np.ndarray,
    k: int = 5,
    force_cpu: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated k-nearest neighbors search.
    
    For each detection point, find the k nearest planogram points.
    
    Args:
        planogram_points: Shape (n, 2), reference points
        detection_points: Shape (m, 2), query points
        k: Number of neighbors to find
        force_cpu: Force CPU execution
    
    Returns:
        Tuple of:
        - indices: Shape (m, k), indices of k nearest points
        - distances: Shape (m, k), distances to k nearest points
    """
    dist_matrix = gpu_distance_matrix(
        detection_points, planogram_points, force_cpu=force_cpu
    )
    
    m = len(detection_points)
    n = len(planogram_points)
    k = min(k, n)
    
    if not force_cpu and _CUPY_AVAILABLE:
        dist_gpu = cp.asarray(dist_matrix)
        
        # Partition to get k smallest
        indices = cp.argpartition(dist_gpu, k, axis=1)[:, :k]
        
        # Get the actual distances for these indices
        row_idx = cp.arange(m)[:, np.newaxis]
        top_k_dist = dist_gpu[row_idx, indices]
        
        # Sort the top-k by distance
        sort_idx = cp.argsort(top_k_dist, axis=1)
        indices = cp.take_along_axis(indices, sort_idx, axis=1)
        top_k_dist = cp.take_along_axis(top_k_dist, sort_idx, axis=1)
        
        indices = cp.asnumpy(indices).astype(np.int32)
        distances = cp.asnumpy(top_k_dist)
    else:
        # NumPy implementation
        indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]
        row_idx = np.arange(m)[:, np.newaxis]
        top_k_dist = dist_matrix[row_idx, indices]
        
        sort_idx = np.argsort(top_k_dist, axis=1)
        indices = np.take_along_axis(indices, sort_idx, axis=1).astype(np.int32)
        distances = np.take_along_axis(top_k_dist, sort_idx, axis=1)
    
    return indices, distances


def gpu_radius_search(
    planogram_points: np.ndarray,
    detection_points: np.ndarray,
    radius: float,
    force_cpu: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated radius search.
    
    For each detection point, find all planogram points within radius.
    
    Args:
        planogram_points: Shape (n, 2), reference points
        detection_points: Shape (m, 2), query points
        radius: Search radius
        force_cpu: Force CPU execution
    
    Returns:
        Tuple of:
        - mask: Shape (m, n), boolean mask of points within radius
        - distances: Shape (m, n), distances (inf for points outside radius)
    """
    dist_matrix = gpu_distance_matrix(
        detection_points, planogram_points, force_cpu=force_cpu
    )
    
    mask = dist_matrix <= radius
    distances = np.where(mask, dist_matrix, np.inf)
    
    return mask, distances


def benchmark_gpu_kernel(
    sizes: list = [100, 500, 1000, 5000, 10000],
    n_trials: int = 3
) -> Dict[str, Any]:
    """
    Benchmark GPU kernels at various problem sizes.
    
    Args:
        sizes: List of problem sizes to test
        n_trials: Number of timing trials per size
    
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    results = {
        'sizes': sizes,
        'gpu_available': is_gpu_available(),
        'cpu_times_ms': [],
        'gpu_times_ms': [],
        'speedups': []
    }
    
    for size in sizes:
        np.random.seed(42)
        points_a = np.random.randn(size, 2) * 100
        points_b = np.random.randn(size, 2) * 100
        
        # CPU timing
        cpu_times = []
        for _ in range(n_trials):
            start = time.perf_counter()
            _ = gpu_distance_matrix(points_a, points_b, force_cpu=True)
            cpu_times.append(time.perf_counter() - start)
        cpu_time = np.mean(cpu_times) * 1000
        results['cpu_times_ms'].append(cpu_time)
        
        # GPU timing
        if is_gpu_available():
            # Warmup
            _ = gpu_distance_matrix(points_a, points_b, force_cpu=False)
            
            gpu_times = []
            for _ in range(n_trials):
                start = time.perf_counter()
                _ = gpu_distance_matrix(points_a, points_b, force_cpu=False)
                gpu_times.append(time.perf_counter() - start)
            gpu_time = np.mean(gpu_times) * 1000
            results['gpu_times_ms'].append(gpu_time)
            results['speedups'].append(cpu_time / gpu_time)
        else:
            results['gpu_times_ms'].append(cpu_time)
            results['speedups'].append(1.0)
    
    return results


if __name__ == "__main__":
    print("GPU Kernel Module Test")
    print("=" * 50)
    
    info = get_gpu_info()
    print(f"GPU Available: {info['gpu_available']}")
    print(f"CuPy: {info['cupy_available']}")
    print(f"Numba CUDA: {info['numba_cuda_available']}")
    if info['device_name']:
        print(f"Device: {info['device_name']}")
        print(f"Memory: {info['total_memory_gb']:.1f} GB total")
    
    print("\nRunning benchmark...")
    results = benchmark_gpu_kernel(sizes=[100, 500, 1000, 2000])
    
    print("\nResults:")
    print(f"{'Size':>8} {'CPU (ms)':>12} {'GPU (ms)':>12} {'Speedup':>10}")
    print("-" * 44)
    for i, size in enumerate(results['sizes']):
        print(f"{size:>8} {results['cpu_times_ms'][i]:>12.2f} "
              f"{results['gpu_times_ms'][i]:>12.2f} "
              f"{results['speedups'][i]:>10.2f}x")
