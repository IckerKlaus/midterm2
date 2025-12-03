"""
Geometry Module for Shelf Product Anomaly Detection

This module provides computational geometry algorithms:
- KD-tree for efficient nearest-neighbor search
- Voronoi-based cell assignment (CPU and GPU implementations)

The geometric operations form the core of the spatial analysis pipeline,
enabling efficient assignment of detected products to planogram cells.
"""

from .kd_tree import KDTree, brute_force_nearest_neighbor
from .voronoi_cpu import (
    assign_to_voronoi_cells,
    compute_voronoi_diagram,
    VoronoiAssignment
)
from .voronoi_gpu import (
    gpu_assign_to_voronoi_cells,
    is_gpu_available
)

__all__ = [
    'KDTree',
    'brute_force_nearest_neighbor',
    'assign_to_voronoi_cells',
    'compute_voronoi_diagram',
    'VoronoiAssignment',
    'gpu_assign_to_voronoi_cells',
    'is_gpu_available'
]
