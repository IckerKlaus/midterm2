"""
KD-Tree Implementation for Fast Nearest-Neighbor Search

This module provides a custom KD-tree implementation optimized for 2D points,
as commonly needed in planogram analysis. The implementation demonstrates
the algorithmic principles rather than relying on black-box library calls.

Key Features:
- Build tree from point set
- Nearest neighbor query
- K-nearest neighbors query
- Radius (range) search
- Brute-force baseline for comparison

Complexity Analysis:
- Build: O(n log n) average, O(n²) worst case (presorted data)
- Nearest Neighbor Query: O(log n) average, O(n) worst case
- K-Nearest Neighbors: O(k log n) average
- Radius Search: O(√n + k) average where k is result size
- Space: O(n)

Reference:
    Bentley, J. L. (1975). Multidimensional binary search trees used for
    associative searching. Communications of the ACM, 18(9), 509-517.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
import numpy as np
from heapq import heappush, heappop


@dataclass
class KDNode:
    """
    A node in the KD-tree.
    
    Attributes:
        point: The 2D point stored at this node
        index: Original index of the point in the input array
        split_dim: Dimension used for splitting (0 for x, 1 for y)
        left: Left subtree (points with smaller split dimension value)
        right: Right subtree (points with larger split dimension value)
    """
    point: np.ndarray
    index: int
    split_dim: int
    left: Optional['KDNode'] = None
    right: Optional['KDNode'] = None


class KDTree:
    """
    KD-Tree for efficient 2D nearest neighbor queries.
    
    This is a binary space partitioning tree that recursively divides
    the point set by alternating dimensions. For 2D data, it alternates
    between x (dimension 0) and y (dimension 1) at each level.
    
    Example:
        >>> points = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> tree = KDTree(points)
        >>> nearest_idx, nearest_dist = tree.nearest_neighbor([4, 5])
        >>> print(f"Nearest point index: {nearest_idx}, distance: {nearest_dist:.2f}")
    
    Attributes:
        points: Original array of points
        root: Root node of the tree
        n_points: Number of points in the tree
        n_dimensions: Number of dimensions (always 2 for this implementation)
    """
    
    def __init__(self, points: np.ndarray):
        """
        Build a KD-tree from an array of 2D points.
        
        Args:
            points: NumPy array of shape (n, 2) containing 2D points
        
        Complexity:
            Time: O(n log n) average case
            Space: O(n) for storing points and tree structure
        """
        self.points = np.asarray(points, dtype=np.float64)
        self.n_points = len(points)
        self.n_dimensions = 2  # Fixed for 2D
        
        if self.n_points == 0:
            self.root = None
        else:
            # Create indices array
            indices = list(range(self.n_points))
            self.root = self._build(indices, depth=0)
    
    def _build(self, indices: List[int], depth: int) -> Optional[KDNode]:
        """
        Recursively build the KD-tree.
        
        At each level, we:
        1. Choose split dimension based on depth (alternating x/y)
        2. Find median point along that dimension
        3. Create node with median
        4. Recursively build left and right subtrees
        
        Args:
            indices: List of point indices to include in this subtree
            depth: Current depth in the tree (determines split dimension)
        
        Returns:
            Root node of the subtree
        
        Complexity:
            Time: O(n log n) for n points
            Space: O(n) call stack depth in worst case
        """
        if not indices:
            return None
        
        # Alternate splitting dimension: x at even depths, y at odd depths
        split_dim = depth % self.n_dimensions
        
        # Sort indices by the splitting dimension
        indices.sort(key=lambda i: self.points[i, split_dim])
        
        # Find median index
        median_pos = len(indices) // 2
        median_idx = indices[median_pos]
        
        # Create node
        node = KDNode(
            point=self.points[median_idx],
            index=median_idx,
            split_dim=split_dim
        )
        
        # Recursively build subtrees
        node.left = self._build(indices[:median_pos], depth + 1)
        node.right = self._build(indices[median_pos + 1:], depth + 1)
        
        return node
    
    def nearest_neighbor(self, query: np.ndarray) -> Tuple[int, float]:
        """
        Find the nearest neighbor to a query point.
        
        Uses branch-and-bound pruning: if the closest point found so far
        is closer than the distance to the splitting plane, we can skip
        the other branch entirely.
        
        Args:
            query: Query point as [x, y] array
        
        Returns:
            Tuple of (index of nearest point, distance to nearest point)
        
        Complexity:
            Time: O(log n) average, O(n) worst case
            Space: O(log n) call stack
        """
        query = np.asarray(query, dtype=np.float64)
        
        if self.root is None:
            raise ValueError("Cannot query empty tree")
        
        # Track best candidate
        self._best_idx = -1
        self._best_dist = float('inf')
        
        self._nn_search(self.root, query)
        
        return self._best_idx, self._best_dist
    
    def _nn_search(self, node: Optional[KDNode], query: np.ndarray) -> None:
        """
        Recursive nearest neighbor search with pruning.
        
        The algorithm:
        1. If current node is closer than best, update best
        2. Determine which child to search first (based on query position)
        3. Search the nearer child
        4. If splitting plane could contain closer points, search other child
        
        Args:
            node: Current node in traversal
            query: Query point
        """
        if node is None:
            return
        
        # Compute distance to current node
        dist = np.sqrt(np.sum((node.point - query) ** 2))
        
        if dist < self._best_dist:
            self._best_dist = dist
            self._best_idx = node.index
        
        # Determine which side of the splitting plane the query is on
        split_dim = node.split_dim
        diff = query[split_dim] - node.point[split_dim]
        
        # Search nearer side first
        if diff < 0:
            near_child = node.left
            far_child = node.right
        else:
            near_child = node.right
            far_child = node.left
        
        # Search the near side
        self._nn_search(near_child, query)
        
        # Check if we need to search the far side
        # Only if the splitting plane is closer than current best
        if abs(diff) < self._best_dist:
            self._nn_search(far_child, query)
    
    def k_nearest_neighbors(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """
        Find the k nearest neighbors to a query point.
        
        Uses a max-heap to track the k closest points found so far.
        The heap is keyed by negative distance so that we can efficiently
        remove the farthest point when a closer one is found.
        
        Args:
            query: Query point as [x, y] array
            k: Number of neighbors to find
        
        Returns:
            List of (index, distance) tuples, sorted by distance
        
        Complexity:
            Time: O(k log k log n) average
            Space: O(k) for the heap
        """
        query = np.asarray(query, dtype=np.float64)
        
        if self.root is None:
            return []
        
        k = min(k, self.n_points)
        
        # Max-heap using negative distances
        # (negative_distance, index)
        self._knn_heap: List[Tuple[float, int]] = []
        self._k = k
        
        self._knn_search(self.root, query)
        
        # Extract results and sort by distance
        results = [(-neg_dist, idx) for neg_dist, idx in self._knn_heap]
        results.sort(key=lambda x: x[0])
        
        return [(idx, dist) for dist, idx in results]
    
    def _knn_search(self, node: Optional[KDNode], query: np.ndarray) -> None:
        """
        Recursive k-nearest neighbors search.
        
        Similar to nearest neighbor search, but maintains a heap of
        the k best candidates found so far.
        
        Args:
            node: Current node in traversal
            query: Query point
        """
        if node is None:
            return
        
        # Compute distance to current node
        dist = np.sqrt(np.sum((node.point - query) ** 2))
        
        # Add to heap if heap not full or closer than farthest in heap
        if len(self._knn_heap) < self._k:
            heappush(self._knn_heap, (-dist, node.index))
        elif dist < -self._knn_heap[0][0]:
            heappop(self._knn_heap)
            heappush(self._knn_heap, (-dist, node.index))
        
        # Get current radius (distance to farthest candidate)
        if len(self._knn_heap) < self._k:
            radius = float('inf')
        else:
            radius = -self._knn_heap[0][0]
        
        # Determine which side to search first
        split_dim = node.split_dim
        diff = query[split_dim] - node.point[split_dim]
        
        if diff < 0:
            near_child = node.left
            far_child = node.right
        else:
            near_child = node.right
            far_child = node.left
        
        # Search near side first
        self._knn_search(near_child, query)
        
        # Update radius after searching near side
        if len(self._knn_heap) < self._k:
            radius = float('inf')
        else:
            radius = -self._knn_heap[0][0]
        
        # Search far side if necessary
        if abs(diff) < radius:
            self._knn_search(far_child, query)
    
    def radius_search(self, query: np.ndarray, radius: float) -> List[Tuple[int, float]]:
        """
        Find all points within a given radius of the query point.
        
        Args:
            query: Query point as [x, y] array
            radius: Search radius
        
        Returns:
            List of (index, distance) tuples for points within radius
        
        Complexity:
            Time: O(√n + k) average where k is the number of results
            Space: O(k) for results
        """
        query = np.asarray(query, dtype=np.float64)
        
        if self.root is None:
            return []
        
        self._radius = radius
        self._radius_results: List[Tuple[int, float]] = []
        
        self._radius_search(self.root, query)
        
        return sorted(self._radius_results, key=lambda x: x[1])
    
    def _radius_search(self, node: Optional[KDNode], query: np.ndarray) -> None:
        """
        Recursive radius search.
        
        Args:
            node: Current node in traversal
            query: Query point
        """
        if node is None:
            return
        
        # Check if current node is within radius
        dist = np.sqrt(np.sum((node.point - query) ** 2))
        if dist <= self._radius:
            self._radius_results.append((node.index, dist))
        
        # Determine pruning
        split_dim = node.split_dim
        diff = query[split_dim] - node.point[split_dim]
        
        if diff < 0:
            near_child = node.left
            far_child = node.right
        else:
            near_child = node.right
            far_child = node.left
        
        # Always search near side
        self._radius_search(near_child, query)
        
        # Search far side only if splitting plane intersects search sphere
        if abs(diff) <= self._radius:
            self._radius_search(far_child, query)
    
    def query_batch(self, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find nearest neighbors for multiple query points.
        
        Args:
            queries: Array of shape (m, 2) containing query points
        
        Returns:
            Tuple of (indices array, distances array) both of shape (m,)
        
        Complexity:
            Time: O(m log n) average where m is number of queries
        """
        queries = np.asarray(queries, dtype=np.float64)
        m = len(queries)
        
        indices = np.zeros(m, dtype=np.int32)
        distances = np.zeros(m, dtype=np.float64)
        
        for i in range(m):
            idx, dist = self.nearest_neighbor(queries[i])
            indices[i] = idx
            distances[i] = dist
        
        return indices, distances


def brute_force_nearest_neighbor(
    points: np.ndarray,
    query: np.ndarray
) -> Tuple[int, float]:
    """
    Brute-force nearest neighbor search (baseline).
    
    Computes distance to all points and returns the minimum.
    Used for correctness testing and benchmarking against KD-tree.
    
    Args:
        points: Array of shape (n, 2) containing data points
        query: Query point as [x, y] array
    
    Returns:
        Tuple of (index of nearest point, distance)
    
    Complexity:
        Time: O(n) - must check all points
        Space: O(n) for distance array (could be O(1) with streaming min)
    """
    points = np.asarray(points, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    
    # Compute all distances at once (vectorized)
    distances = np.sqrt(np.sum((points - query) ** 2, axis=1))
    
    # Find minimum
    nearest_idx = np.argmin(distances)
    nearest_dist = distances[nearest_idx]
    
    return int(nearest_idx), float(nearest_dist)


def brute_force_nearest_neighbor_batch(
    points: np.ndarray,
    queries: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Brute-force nearest neighbor for multiple queries.
    
    This is the baseline against which we compare KD-tree and GPU approaches.
    
    Args:
        points: Array of shape (n, 2) containing data points
        queries: Array of shape (m, 2) containing query points
    
    Returns:
        Tuple of (indices array shape (m,), distances array shape (m,))
    
    Complexity:
        Time: O(n × m) - nested loops over all points and queries
        Space: O(n × m) if computing full distance matrix, O(m) for results
    """
    points = np.asarray(points, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    
    n = len(points)
    m = len(queries)
    
    indices = np.zeros(m, dtype=np.int32)
    distances = np.zeros(m, dtype=np.float64)
    
    for i in range(m):
        # Compute distances from query i to all points
        dists = np.sqrt(np.sum((points - queries[i]) ** 2, axis=1))
        nearest_idx = np.argmin(dists)
        indices[i] = nearest_idx
        distances[i] = dists[nearest_idx]
    
    return indices, distances


def validate_kdtree(n_points: int = 1000, n_queries: int = 100, seed: int = 42) -> bool:
    """
    Validate KD-tree correctness against brute force.
    
    Generates random points and queries, then verifies that KD-tree
    returns the same nearest neighbors as brute force.
    
    Args:
        n_points: Number of random data points
        n_queries: Number of random query points
        seed: Random seed for reproducibility
    
    Returns:
        True if all queries match, False otherwise
    
    Example:
        >>> assert validate_kdtree(1000, 100, seed=42)
    """
    np.random.seed(seed)
    
    # Generate random points
    points = np.random.randn(n_points, 2) * 100
    queries = np.random.randn(n_queries, 2) * 100
    
    # Build KD-tree
    tree = KDTree(points)
    
    # Compare results
    all_match = True
    for query in queries:
        kd_idx, kd_dist = tree.nearest_neighbor(query)
        bf_idx, bf_dist = brute_force_nearest_neighbor(points, query)
        
        # Check if distances match (indices might differ for equidistant points)
        if not np.isclose(kd_dist, bf_dist, rtol=1e-10):
            print(f"Mismatch: KD-tree dist={kd_dist}, brute force dist={bf_dist}")
            all_match = False
    
    return all_match


if __name__ == "__main__":
    # Run validation test
    print("Validating KD-tree implementation...")
    if validate_kdtree():
        print("✓ KD-tree validation passed!")
    else:
        print("✗ KD-tree validation failed!")
    
    # Simple demo
    print("\nDemo:")
    points = np.array([
        [0, 0], [10, 0], [5, 5], [2, 8], [8, 3],
        [1, 4], [6, 7], [3, 2], [9, 9], [4, 6]
    ], dtype=np.float64)
    
    tree = KDTree(points)
    query = np.array([5, 4])
    
    idx, dist = tree.nearest_neighbor(query)
    print(f"Query: {query}")
    print(f"Nearest neighbor: index={idx}, point={points[idx]}, distance={dist:.2f}")
    
    k_neighbors = tree.k_nearest_neighbors(query, k=3)
    print(f"3-nearest neighbors: {k_neighbors}")
    
    radius_results = tree.radius_search(query, radius=5.0)
    print(f"Points within radius 5: {radius_results}")
