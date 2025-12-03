"""
High-Performance Computing Module

This module provides utilities for GPU acceleration and performance
measurement in the shelf product anomaly detection system.

Components:
- gpu_kernels: GPU-accelerated computational kernels
- timing: Benchmarking and profiling utilities
"""

from .gpu_kernels import (
    gpu_nearest_neighbors,
    gpu_distance_matrix,
    is_gpu_available,
    get_gpu_info
)
from .timing import (
    Timer,
    time_function,
    compute_speedup,
    BenchmarkResult
)

__all__ = [
    'gpu_nearest_neighbors',
    'gpu_distance_matrix',
    'is_gpu_available',
    'get_gpu_info',
    'Timer',
    'time_function',
    'compute_speedup',
    'BenchmarkResult'
]
