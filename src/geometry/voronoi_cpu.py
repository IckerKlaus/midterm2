"""
CPU-Based Voronoi Assignment for Shelf Product Analysis

This module implements Voronoi-based spatial partitioning for assigning
detected products to their expected planogram cells.

Key Insight:
    In 2D Euclidean space, assigning a point to its Voronoi cell is
    mathematically equivalent to finding its nearest neighbor among the
    generating points (planogram positions). This equivalence allows us
    to use efficient nearest-neighbor search (KD-tree) instead of
    explicitly constructing Voronoi diagrams.

Mathematical Background:
    A Voronoi diagram V(P) for a set of points P = {p₁, p₂, ..., pₙ}
    partitions the plane into regions R₁, R₂, ..., Rₙ such that:
    
    Rᵢ = {x ∈ ℝ² : ||x - pᵢ|| ≤ ||x - pⱼ|| for all j ≠ i}
    
    This means point x belongs to region Rᵢ if and only if pᵢ is its
    nearest neighbor among all points in P.

Implementation Strategy:
    1. For small datasets (< 1000 points): Use brute-force distance computation
    2. For medium datasets: Use KD-tree for O(log n) queries
    3. For large datasets: Use GPU-accelerated batch computation (see voronoi_gpu.py)

Reference:
    Aurenhammer, F. (1991). Voronoi diagrams—a survey of a fundamental
    geometric data structure. ACM Computing Surveys, 23(3), 345-405.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np

from .kd_tree import KDTree, brute_force_nearest_neighbor


@dataclass
class VoronoiAssignment:
    """
    Result of assigning detections to Voronoi cells.
    
    Attributes:
        assignments: Array mapping each detection index to a planogram index
        distances: Distance from each detection to its assigned cell center
        detection_skus: SKU IDs of detections
        planogram_skus: SKU IDs of assigned planogram positions
    
    Usage:
        Given a VoronoiAssignment result, you can check if detection i
        was assigned to the correct position:
        
        >>> is_correct = assignment.detection_skus[i] == assignment.planogram_skus[i]
    """
    assignments: np.ndarray  # Shape: (num_detections,) -> planogram indices
    distances: np.ndarray    # Shape: (num_detections,) -> distances to cells
    detection_skus: List[str]
    planogram_skus: List[str]
    
    @property
    def num_detections(self) -> int:
        return len(self.assignments)
    
    def get_cell_contents(self) -> Dict[int, List[int]]:
        """
        Group detections by their assigned Voronoi cell.
        
        Returns:
            Dict mapping planogram_index -> list of detection indices
        
        Complexity: O(n) where n is number of detections
        """
        contents: Dict[int, List[int]] = {}
        for det_idx, plan_idx in enumerate(self.assignments):
            if plan_idx not in contents:
                contents[plan_idx] = []
            contents[plan_idx].append(det_idx)
        return contents
    
    def get_mismatches(self) -> List[Tuple[int, str, str]]:
        """
        Find detections where the SKU doesn't match the assigned cell.
        
        Returns:
            List of (detection_index, detected_sku, expected_sku) tuples
        """
        mismatches = []
        for i in range(self.num_detections):
            if self.detection_skus[i] != self.planogram_skus[i]:
                mismatches.append((
                    i,
                    self.detection_skus[i],
                    self.planogram_skus[i]
                ))
        return mismatches


def assign_to_voronoi_cells(
    planogram_points: np.ndarray,
    detection_points: np.ndarray,
    planogram_skus: Optional[List[str]] = None,
    detection_skus: Optional[List[str]] = None,
    method: str = "kdtree"
) -> VoronoiAssignment:
    """
    Assign each detection to its nearest planogram position (Voronoi cell).
    
    This is the core spatial assignment function. Each detection point is
    assigned to the Voronoi cell of its nearest planogram position, which
    corresponds to finding the planogram entry that "owns" that detection.
    
    Args:
        planogram_points: Array of shape (n, 2) with planogram (x, y) positions
        detection_points: Array of shape (m, 2) with detection (x, y) positions
        planogram_skus: Optional list of n SKU IDs for planogram entries
        detection_skus: Optional list of m SKU IDs for detections
        method: Assignment method - "kdtree", "brute_force", or "auto"
    
    Returns:
        VoronoiAssignment: Complete assignment result with indices and distances
    
    Complexity:
        - brute_force: O(n × m) time, O(n) space
        - kdtree: O(n log n + m log n) time, O(n) space
        
        For m queries on n planogram points:
        - brute_force: 10,000 × 10,000 = 100M distance computations
        - kdtree: ~10,000 × 13 = 130K node visits (assuming balanced tree)
    
    Example:
        >>> planogram = np.array([[0, 0], [10, 0], [20, 0]])
        >>> detections = np.array([[1, 1], [9, -1], [21, 0.5]])
        >>> result = assign_to_voronoi_cells(planogram, detections)
        >>> print(result.assignments)  # [0, 1, 2] - each detection to nearest planogram
    
    Note on Voronoi Equivalence:
        Finding the nearest neighbor in Euclidean space is mathematically
        equivalent to determining which Voronoi cell contains a query point.
        This is because Voronoi cells are defined as the set of all points
        closer to one generator than to any other.
    """
    planogram_points = np.asarray(planogram_points, dtype=np.float64)
    detection_points = np.asarray(detection_points, dtype=np.float64)
    
    n_planogram = len(planogram_points)
    m_detections = len(detection_points)
    
    # Validate inputs
    if n_planogram == 0:
        raise ValueError("Planogram cannot be empty")
    if m_detections == 0:
        return VoronoiAssignment(
            assignments=np.array([], dtype=np.int32),
            distances=np.array([], dtype=np.float64),
            detection_skus=[],
            planogram_skus=[]
        )
    
    # Default SKU lists
    if planogram_skus is None:
        planogram_skus = [f"P_{i}" for i in range(n_planogram)]
    if detection_skus is None:
        detection_skus = [f"D_{i}" for i in range(m_detections)]
    
    # Choose method
    if method == "auto":
        # Heuristic: KD-tree is faster when m * log(n) < m * n
        # which is always true for n > 1, so prefer KD-tree for n > 100
        method = "kdtree" if n_planogram > 100 else "brute_force"
    
    # Perform assignment
    assignments = np.zeros(m_detections, dtype=np.int32)
    distances = np.zeros(m_detections, dtype=np.float64)
    
    if method == "kdtree":
        # Build KD-tree on planogram points
        # Time: O(n log n)
        tree = KDTree(planogram_points)
        
        # Query nearest neighbor for each detection
        # Time: O(m log n) average
        for i in range(m_detections):
            idx, dist = tree.nearest_neighbor(detection_points[i])
            assignments[i] = idx
            distances[i] = dist
    
    elif method == "brute_force":
        # Simple nested loop approach
        # Time: O(n × m)
        for i in range(m_detections):
            idx, dist = brute_force_nearest_neighbor(planogram_points, detection_points[i])
            assignments[i] = idx
            distances[i] = dist
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Build assigned planogram SKUs list
    assigned_planogram_skus = [planogram_skus[idx] for idx in assignments]
    
    return VoronoiAssignment(
        assignments=assignments,
        distances=distances,
        detection_skus=detection_skus,
        planogram_skus=assigned_planogram_skus
    )


def assign_batch_kdtree(
    planogram_points: np.ndarray,
    detection_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized batch assignment using KD-tree.
    
    This function is the main computational kernel for CPU-based
    Voronoi assignment. It builds a KD-tree once and queries it
    for all detections.
    
    Args:
        planogram_points: Array of shape (n, 2)
        detection_points: Array of shape (m, 2)
    
    Returns:
        Tuple of (assignments array, distances array)
    
    Complexity:
        Time: O(n log n + m log n)
        Space: O(n) for tree + O(m) for results
    """
    tree = KDTree(planogram_points)
    return tree.query_batch(detection_points)


def compute_voronoi_diagram(
    points: np.ndarray,
    bounding_box: Optional[Tuple[float, float, float, float]] = None
) -> Optional[object]:
    """
    Compute the actual Voronoi diagram for visualization purposes.
    
    This function uses scipy.spatial.Voronoi if available. Note that
    for the assignment task, we don't actually need the diagram itself -
    nearest neighbor search is sufficient and more efficient.
    
    Args:
        points: Array of shape (n, 2) with generating points
        bounding_box: Optional (x_min, y_min, x_max, y_max) for clipping
    
    Returns:
        scipy.spatial.Voronoi object if scipy is available, else None
    
    Note:
        This is for visualization and debugging only. The actual
        assignment is done via nearest-neighbor search which is
        mathematically equivalent but computationally more efficient.
    """
    try:
        from scipy.spatial import Voronoi
        
        points = np.asarray(points, dtype=np.float64)
        
        if len(points) < 4:
            # Voronoi needs at least 4 points for a meaningful diagram
            return None
        
        vor = Voronoi(points)
        return vor
    
    except ImportError:
        print("scipy not available for Voronoi diagram computation")
        return None


def visualize_voronoi_assignment(
    planogram_points: np.ndarray,
    detection_points: np.ndarray,
    assignment: VoronoiAssignment,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize the Voronoi assignment with cells and point assignments.
    
    Creates a plot showing:
    - Voronoi cells for planogram points (if scipy available)
    - Planogram points (blue squares)
    - Detection points (red circles)
    - Lines connecting detections to assigned planogram points
    - Color coding for correct/incorrect assignments
    
    Args:
        planogram_points: Array of planogram positions
        detection_points: Array of detection positions
        assignment: VoronoiAssignment result
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ImportError:
        print("matplotlib not available for visualization")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Try to plot Voronoi cells
    vor = compute_voronoi_diagram(planogram_points)
    if vor is not None:
        try:
            from scipy.spatial import voronoi_plot_2d
            voronoi_plot_2d(vor, ax=ax, show_vertices=False, 
                          line_colors='lightgray', line_width=0.5)
        except Exception:
            pass  # Skip if visualization fails
    
    # Plot planogram points
    ax.scatter(
        planogram_points[:, 0], planogram_points[:, 1],
        c='blue', s=100, marker='s', label='Planogram', zorder=3
    )
    
    # Identify correct and incorrect assignments
    correct_mask = np.array([
        assignment.detection_skus[i] == assignment.planogram_skus[i]
        for i in range(len(detection_points))
    ])
    
    # Plot correct detections (green)
    if np.any(correct_mask):
        ax.scatter(
            detection_points[correct_mask, 0],
            detection_points[correct_mask, 1],
            c='green', s=60, marker='o', label='Correct', zorder=4
        )
    
    # Plot incorrect detections (red)
    if np.any(~correct_mask):
        ax.scatter(
            detection_points[~correct_mask, 0],
            detection_points[~correct_mask, 1],
            c='red', s=60, marker='o', label='Incorrect', zorder=4
        )
    
    # Draw assignment lines
    lines = []
    colors = []
    for i in range(len(detection_points)):
        plan_idx = assignment.assignments[i]
        line = [detection_points[i], planogram_points[plan_idx]]
        lines.append(line)
        colors.append('green' if correct_mask[i] else 'red')
    
    lc = LineCollection(lines, colors=colors, alpha=0.3, linewidths=1)
    ax.add_collection(lc)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Voronoi Cell Assignment')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def compute_assignment_statistics(
    assignment: VoronoiAssignment,
    distance_threshold: float = 5.0
) -> Dict[str, float]:
    """
    Compute statistics about the Voronoi assignment quality.
    
    Args:
        assignment: VoronoiAssignment result
        distance_threshold: Threshold for considering a match "close"
    
    Returns:
        Dictionary with statistics:
        - mean_distance: Average distance to assigned cell
        - max_distance: Maximum distance to assigned cell
        - std_distance: Standard deviation of distances
        - close_match_rate: Fraction of assignments within threshold
        - sku_match_rate: Fraction where SKU matches assigned cell
    """
    if assignment.num_detections == 0:
        return {
            'mean_distance': 0.0,
            'max_distance': 0.0,
            'std_distance': 0.0,
            'close_match_rate': 0.0,
            'sku_match_rate': 0.0
        }
    
    # Distance statistics
    mean_dist = float(np.mean(assignment.distances))
    max_dist = float(np.max(assignment.distances))
    std_dist = float(np.std(assignment.distances))
    
    # Close match rate
    close_matches = np.sum(assignment.distances <= distance_threshold)
    close_rate = close_matches / assignment.num_detections
    
    # SKU match rate
    sku_matches = sum(
        1 for i in range(assignment.num_detections)
        if assignment.detection_skus[i] == assignment.planogram_skus[i]
    )
    sku_rate = sku_matches / assignment.num_detections
    
    return {
        'mean_distance': mean_dist,
        'max_distance': max_dist,
        'std_distance': std_dist,
        'close_match_rate': close_rate,
        'sku_match_rate': sku_rate
    }


if __name__ == "__main__":
    # Demo and validation
    print("Voronoi Assignment Demo")
    print("=" * 50)
    
    # Create a simple test case
    planogram = np.array([
        [0, 0], [10, 0], [20, 0],
        [0, -5], [10, -5], [20, -5]
    ], dtype=np.float64)
    
    # Detections with some noise
    detections = np.array([
        [1, 0.5],    # Should match [0, 0]
        [9, -0.5],   # Should match [10, 0]
        [21, 0.2],   # Should match [20, 0]
        [0.5, -4.5], # Should match [0, -5]
        [11, -5.5],  # Should match [10, -5]
        [19, -4.8]   # Should match [20, -5]
    ], dtype=np.float64)
    
    planogram_skus = ['A', 'B', 'C', 'D', 'E', 'F']
    detection_skus = ['A', 'B', 'C', 'D', 'E', 'F']  # All correct
    
    # Test both methods
    for method in ["brute_force", "kdtree"]:
        print(f"\nMethod: {method}")
        result = assign_to_voronoi_cells(
            planogram, detections,
            planogram_skus, detection_skus,
            method=method
        )
        print(f"Assignments: {result.assignments}")
        print(f"Distances: {np.round(result.distances, 2)}")
        
        stats = compute_assignment_statistics(result)
        print(f"Mean distance: {stats['mean_distance']:.2f}")
        print(f"SKU match rate: {stats['sku_match_rate']:.2%}")
    
    print("\n✓ Demo complete!")
