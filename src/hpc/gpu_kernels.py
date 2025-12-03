"""
GPU-Accelerated Computational Kernels

This module provides GPU-accelerated implementations of the core
computational operations needed for shelf product analysis:

1. Distance matrix computation
2. Nearest neighbor search
3. Batch operations for large datasets

The module automatically detects GPU availability and falls back
to optimized CPU implementations when no GPU is present.

Supported Backends (in order of preference):
- Apple Silicon (MLX): For M1/M2/M3/M4 Macs
- Apple Silicon (PyTorch MPS): Alternative for Apple GPUs
- CuPy (CUDA): For NVIDIA GPUs
- Numba CUDA: Custom kernel support for NVIDIA
- NumPy (fallback): Vectorized CPU operations
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np

# ============================================================
# GPU Backend Detection
# ============================================================

_MLX_AVAILABLE = False
_MPS_AVAILABLE = False
_CUPY_AVAILABLE = False
_NUMBA_CUDA_AVAILABLE = False
_GPU_BACKEND = None

# Try Apple MLX first (best for Apple Silicon)
try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
    _GPU_BACKEND = "mlx"
except ImportError:
    mx = None

# Try PyTorch MPS (Apple Metal Performance Shaders)
if not _MLX_AVAILABLE:
    try:
        import torch
        if torch.backends.mps.is_available():
            _MPS_AVAILABLE = True
            _GPU_BACKEND = "mps"
    except ImportError:
        torch = None

# Try CuPy (NVIDIA CUDA)
if not _MLX_AVAILABLE and not _MPS_AVAILABLE:
    try:
        import cupy as cp
        _CUPY_AVAILABLE = True
        _GPU_BACKEND = "cupy"
    except ImportError:
        cp = None

# Try Numba CUDA
if not _MLX_AVAILABLE and not _MPS_AVAILABLE and not _CUPY_AVAILABLE:
    try:
        from numba import cuda
        if cuda.is_available():
            _NUMBA_CUDA_AVAILABLE = True
            _GPU_BACKEND = "numba"
    except ImportError:
        pass


def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is available.
    
    Returns:
        True if any GPU backend is available (MLX, MPS, CuPy, or Numba CUDA)
    """
    return _MLX_AVAILABLE or _MPS_AVAILABLE or _CUPY_AVAILABLE or _NUMBA_CUDA_AVAILABLE


def get_gpu_info() -> Dict[str, Any]:
    """
    Get information about available GPU resources.
    
    Returns:
        Dictionary with GPU configuration details
    """
    info = {
        'gpu_available': is_gpu_available(),
        'backend': _GPU_BACKEND,
        'mlx_available': _MLX_AVAILABLE,
        'mps_available': _MPS_AVAILABLE,
        'cupy_available': _CUPY_AVAILABLE,
        'numba_cuda_available': _NUMBA_CUDA_AVAILABLE,
        'device_name': None,
        'total_memory_gb': None,
        'free_memory_gb': None
    }
    
    if _MLX_AVAILABLE:
        info['device_name'] = "Apple Silicon (MLX)"
        # MLX doesn't expose memory info directly
        
    elif _MPS_AVAILABLE:
        info['device_name'] = "Apple Silicon (MPS/Metal)"
        
    elif _CUPY_AVAILABLE:
        try:
            device = cp.cuda.Device()
            info['device_name'] = f"NVIDIA CUDA Device {device.id}"
            mem = device.mem_info
            info['total_memory_gb'] = mem[1] / 1e9
            info['free_memory_gb'] = mem[0] / 1e9
        except Exception:
            pass
    
    return info


# ============================================================
# MLX Implementation (Apple Silicon - BEST)
# ============================================================

def _distance_matrix_mlx(
    points_a: np.ndarray,
    points_b: np.ndarray
) -> np.ndarray:
    """MLX implementation of distance matrix for Apple Silicon."""
    # Convert to MLX arrays
    a_mlx = mx.array(points_a)
    b_mlx = mx.array(points_b)
    
    # Compute squared distances using broadcasting
    # a: (m, d) -> (m, 1, d)
    # b: (n, d) -> (1, n, d)
    diff = mx.expand_dims(a_mlx, axis=1) - mx.expand_dims(b_mlx, axis=0)
    
    # Sum of squared differences
    sq_dist = mx.sum(diff ** 2, axis=2)
    
    # Euclidean distance
    dist = mx.sqrt(sq_dist)
    
    # Evaluate and convert back to numpy
    mx.eval(dist)
    return np.array(dist)


def _nearest_neighbors_mlx(
    planogram_points: np.ndarray,
    detection_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """MLX nearest neighbor search for Apple Silicon."""
    # Convert to MLX arrays
    plan_mlx = mx.array(planogram_points)
    det_mlx = mx.array(detection_points)
    
    # Compute distance matrix
    diff = mx.expand_dims(det_mlx, axis=1) - mx.expand_dims(plan_mlx, axis=0)
    sq_dist = mx.sum(diff ** 2, axis=2)
    dist = mx.sqrt(sq_dist)
    
    # Find argmin for each detection
    indices = mx.argmin(dist, axis=1)
    
    # Get minimum distances
    min_dists = mx.min(dist, axis=1)
    
    # Evaluate and convert back
    mx.eval(indices, min_dists)
    
    return np.array(indices).astype(np.int32), np.array(min_dists)


# ============================================================
# PyTorch MPS Implementation (Apple Silicon - Alternative)
# ============================================================

def _distance_matrix_mps(
    points_a: np.ndarray,
    points_b: np.ndarray
) -> np.ndarray:
    """PyTorch MPS implementation for Apple Silicon."""
    import torch
    
    device = torch.device("mps")
    
    # Convert to tensors on MPS
    a_tensor = torch.from_numpy(points_a).float().to(device)
    b_tensor = torch.from_numpy(points_b).float().to(device)
    
    # Compute distances using broadcasting
    diff = a_tensor.unsqueeze(1) - b_tensor.unsqueeze(0)
    sq_dist = torch.sum(diff ** 2, dim=2)
    dist = torch.sqrt(sq_dist)
    
    # Return as numpy
    return dist.cpu().numpy()


def _nearest_neighbors_mps(
    planogram_points: np.ndarray,
    detection_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """PyTorch MPS nearest neighbor search."""
    import torch
    
    device = torch.device("mps")
    
    plan_tensor = torch.from_numpy(planogram_points).float().to(device)
    det_tensor = torch.from_numpy(detection_points).float().to(device)
    
    # Compute distance matrix
    diff = det_tensor.unsqueeze(1) - plan_tensor.unsqueeze(0)
    sq_dist = torch.sum(diff ** 2, dim=2)
    dist = torch.sqrt(sq_dist)
    
    # Find nearest
    min_dists, indices = torch.min(dist, dim=1)
    
    return indices.cpu().numpy().astype(np.int32), min_dists.cpu().numpy()


# ============================================================
# CuPy Implementation (NVIDIA CUDA)
# ============================================================

def _distance_matrix_cupy(
    points_a: np.ndarray,
    points_b: np.ndarray
) -> np.ndarray:
    """CuPy implementation of distance matrix."""
    a_gpu = cp.asarray(points_a)
    b_gpu = cp.asarray(points_b)
    
    diff = a_gpu[:, np.newaxis, :] - b_gpu[np.newaxis, :, :]
    sq_dist = cp.sum(diff ** 2, axis=2)
    dist = cp.sqrt(sq_dist)
    
    return cp.asnumpy(dist)


def _nearest_neighbors_cupy(
    planogram_points: np.ndarray,
    detection_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """CuPy nearest neighbor search."""
    plan_gpu = cp.asarray(planogram_points)
    det_gpu = cp.asarray(detection_points)
    
    diff = det_gpu[:, np.newaxis, :] - plan_gpu[np.newaxis, :, :]
    sq_dist = cp.sum(diff ** 2, axis=2)
    dist = cp.sqrt(sq_dist)
    
    indices = cp.argmin(dist, axis=1)
    min_dists = cp.min(dist, axis=1)
    
    return cp.asnumpy(indices).astype(np.int32), cp.asnumpy(min_dists)


# ============================================================
# NumPy Implementation (CPU Fallback)
# ============================================================

def _distance_matrix_numpy(
    points_a: np.ndarray,
    points_b: np.ndarray
) -> np.ndarray:
    """NumPy (CPU) implementation of distance matrix."""
    diff = points_a[:, np.newaxis, :] - points_b[np.newaxis, :, :]
    sq_dist = np.sum(diff ** 2, axis=2)
    return np.sqrt(sq_dist)


def _nearest_neighbors_numpy(
    planogram_points: np.ndarray,
    detection_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """NumPy nearest neighbor search (CPU fallback)."""
    diff = detection_points[:, np.newaxis, :] - planogram_points[np.newaxis, :, :]
    sq_dist = np.sum(diff ** 2, axis=2)
    dist = np.sqrt(sq_dist)
    
    indices = np.argmin(dist, axis=1).astype(np.int32)
    min_dists = np.min(dist, axis=1)
    
    return indices, min_dists


# ============================================================
# Public API
# ============================================================

def gpu_distance_matrix(
    points_a: np.ndarray,
    points_b: np.ndarray,
    force_cpu: bool = False
) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix using GPU.
    
    Automatically selects the best available backend:
    - Apple Silicon: MLX or MPS
    - NVIDIA: CuPy or Numba
    - Fallback: NumPy (CPU)
    
    Args:
        points_a: First point set, shape (m, d)
        points_b: Second point set, shape (n, d)
        force_cpu: Force CPU execution even if GPU available
    
    Returns:
        Distance matrix of shape (m, n)
    """
    points_a = np.asarray(points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)
    
    if force_cpu:
        return _distance_matrix_numpy(points_a, points_b)
    
    if _MLX_AVAILABLE:
        return _distance_matrix_mlx(points_a, points_b)
    elif _MPS_AVAILABLE:
        return _distance_matrix_mps(points_a, points_b)
    elif _CUPY_AVAILABLE:
        return _distance_matrix_cupy(points_a, points_b)
    else:
        return _distance_matrix_numpy(points_a, points_b)


def gpu_nearest_neighbors(
    planogram_points: np.ndarray,
    detection_points: np.ndarray,
    force_cpu: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated batch nearest neighbor search.
    
    Args:
        planogram_points: Shape (n, 2), reference points
        detection_points: Shape (m, 2), query points
        force_cpu: Force CPU execution
    
    Returns:
        Tuple of:
        - indices: Shape (m,), index of nearest planogram point
        - distances: Shape (m,), distance to nearest point
    """
    planogram_points = np.asarray(planogram_points, dtype=np.float64)
    detection_points = np.asarray(detection_points, dtype=np.float64)
    
    if force_cpu:
        return _nearest_neighbors_numpy(planogram_points, detection_points)
    
    if _MLX_AVAILABLE:
        return _nearest_neighbors_mlx(planogram_points, detection_points)
    elif _MPS_AVAILABLE:
        return _nearest_neighbors_mps(planogram_points, detection_points)
    elif _CUPY_AVAILABLE:
        return _nearest_neighbors_cupy(planogram_points, detection_points)
    else:
        return _nearest_neighbors_numpy(planogram_points, detection_points)


def gpu_k_nearest_neighbors(
    planogram_points: np.ndarray,
    detection_points: np.ndarray,
    k: int = 5,
    force_cpu: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated k-nearest neighbors search.
    
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
    
    # Use numpy for top-k selection (efficient enough)
    indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]
    row_idx = np.arange(m)[:, np.newaxis]
    top_k_dist = dist_matrix[row_idx, indices]
    
    sort_idx = np.argsort(top_k_dist, axis=1)
    indices = np.take_along_axis(indices, sort_idx, axis=1).astype(np.int32)
    distances = np.take_along_axis(top_k_dist, sort_idx, axis=1)
    
    return indices, distances


def benchmark_gpu_kernel(
    sizes: list = [100, 500, 1000, 5000, 10000],
    n_trials: int = 3
) -> Dict[str, Any]:
    """
    Benchmark GPU kernels at various problem sizes.
    """
    import time
    
    results = {
        'sizes': sizes,
        'gpu_available': is_gpu_available(),
        'backend': _GPU_BACKEND,
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
    print(f"Backend: {info['backend']}")
    print(f"  MLX (Apple Silicon): {info['mlx_available']}")
    print(f"  MPS (Apple Metal): {info['mps_available']}")
    print(f"  CuPy (NVIDIA CUDA): {info['cupy_available']}")
    print(f"  Numba CUDA: {info['numba_cuda_available']}")
    
    if info['device_name']:
        print(f"Device: {info['device_name']}")
    
    print("\nRunning benchmark...")
    results = benchmark_gpu_kernel(sizes=[100, 500, 1000, 2000])
    
    print("\nResults:")
    print(f"{'Size':>8} {'CPU (ms)':>12} {'GPU (ms)':>12} {'Speedup':>10}")
    print("-" * 44)
    for i, size in enumerate(results['sizes']):
        print(f"{size:>8} {results['cpu_times_ms'][i]:>12.2f} "
              f"{results['gpu_times_ms'][i]:>12.2f} "
              f"{results['speedups'][i]:>10.2f}x")