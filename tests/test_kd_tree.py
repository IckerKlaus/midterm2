"""
Tests for KD-Tree Implementation

This module tests the custom KD-tree implementation against brute-force
nearest neighbor search to verify correctness.

Test Categories:
1. Basic operations (build, query)
2. Correctness vs brute force
3. Edge cases (empty, single point, duplicates)
4. K-nearest neighbors
5. Radius search

Run with: pytest tests/test_kd_tree.py -v
"""

import pytest
import numpy as np
from typing import List, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry.kd_tree import (
    KDTree,
    KDNode,
    brute_force_nearest_neighbor,
    brute_force_nearest_neighbor_batch,
    validate_kdtree
)


class TestKDTreeConstruction:
    """Tests for KD-tree construction."""
    
    def test_build_empty(self):
        """Test building tree with empty array."""
        points = np.array([]).reshape(0, 2)
        tree = KDTree(points)
        assert tree.root is None
        assert tree.n_points == 0
    
    def test_build_single_point(self):
        """Test building tree with single point."""
        points = np.array([[5.0, 3.0]])
        tree = KDTree(points)
        
        assert tree.root is not None
        assert tree.n_points == 1
        assert np.allclose(tree.root.point, [5.0, 3.0])
        assert tree.root.left is None
        assert tree.root.right is None
    
    def test_build_two_points(self):
        """Test building tree with two points."""
        points = np.array([[0.0, 0.0], [10.0, 10.0]])
        tree = KDTree(points)
        
        assert tree.root is not None
        assert tree.n_points == 2
        # One should be in left or right subtree
        assert tree.root.left is not None or tree.root.right is not None
    
    def test_build_grid(self):
        """Test building tree with grid of points."""
        # Create 3x3 grid
        x = np.arange(3)
        y = np.arange(3)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
        
        tree = KDTree(points)
        
        assert tree.n_points == 9
        assert tree.root is not None
    
    def test_build_random_points(self):
        """Test building tree with random points."""
        np.random.seed(42)
        points = np.random.randn(100, 2)
        
        tree = KDTree(points)
        
        assert tree.n_points == 100
        assert tree.root is not None


class TestNearestNeighbor:
    """Tests for nearest neighbor queries."""
    
    def test_nn_single_point(self):
        """Test NN with single point in tree."""
        points = np.array([[5.0, 5.0]])
        tree = KDTree(points)
        
        idx, dist = tree.nearest_neighbor([0.0, 0.0])
        
        assert idx == 0
        assert np.isclose(dist, np.sqrt(50))
    
    def test_nn_exact_match(self):
        """Test NN query that exactly matches a point."""
        points = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 5.0]])
        tree = KDTree(points)
        
        idx, dist = tree.nearest_neighbor([10.0, 0.0])
        
        assert idx == 1
        assert np.isclose(dist, 0.0)
    
    def test_nn_vs_brute_force_small(self):
        """Test NN matches brute force on small dataset."""
        np.random.seed(42)
        points = np.random.randn(20, 2) * 10
        tree = KDTree(points)
        
        for _ in range(10):
            query = np.random.randn(2) * 10
            
            kd_idx, kd_dist = tree.nearest_neighbor(query)
            bf_idx, bf_dist = brute_force_nearest_neighbor(points, query)
            
            # Distances should match (indices might differ for equidistant points)
            assert np.isclose(kd_dist, bf_dist, rtol=1e-10)
    
    def test_nn_vs_brute_force_medium(self):
        """Test NN matches brute force on medium dataset."""
        np.random.seed(123)
        points = np.random.randn(500, 2) * 100
        tree = KDTree(points)
        
        queries = np.random.randn(50, 2) * 100
        
        for query in queries:
            kd_idx, kd_dist = tree.nearest_neighbor(query)
            bf_idx, bf_dist = brute_force_nearest_neighbor(points, query)
            
            assert np.isclose(kd_dist, bf_dist, rtol=1e-10), \
                f"Mismatch: KD={kd_dist}, BF={bf_dist}"
    
    def test_nn_vs_brute_force_large(self):
        """Test NN matches brute force on larger dataset."""
        np.random.seed(456)
        points = np.random.randn(2000, 2) * 100
        tree = KDTree(points)
        
        queries = np.random.randn(20, 2) * 100
        
        for query in queries:
            kd_idx, kd_dist = tree.nearest_neighbor(query)
            bf_idx, bf_dist = brute_force_nearest_neighbor(points, query)
            
            assert np.isclose(kd_dist, bf_dist, rtol=1e-10)
    
    def test_nn_grid_points(self):
        """Test NN on regular grid."""
        # Create 10x10 grid with spacing 10
        points = []
        for i in range(10):
            for j in range(10):
                points.append([i * 10, j * 10])
        points = np.array(points, dtype=np.float64)
        
        tree = KDTree(points)
        
        # Query at grid center
        idx, dist = tree.nearest_neighbor([45.0, 45.0])
        
        # Should find one of the adjacent grid points
        assert dist < 10  # Should be within one grid cell
    
    def test_nn_empty_raises(self):
        """Test that NN on empty tree raises error."""
        points = np.array([]).reshape(0, 2)
        tree = KDTree(points)
        
        with pytest.raises(ValueError):
            tree.nearest_neighbor([0.0, 0.0])


class TestKNearestNeighbors:
    """Tests for k-nearest neighbors queries."""
    
    def test_knn_k_equals_1(self):
        """Test KNN with k=1 matches NN."""
        np.random.seed(42)
        points = np.random.randn(100, 2) * 10
        tree = KDTree(points)
        
        query = np.array([0.0, 0.0])
        
        nn_idx, nn_dist = tree.nearest_neighbor(query)
        knn_results = tree.k_nearest_neighbors(query, k=1)
        
        assert len(knn_results) == 1
        assert knn_results[0][0] == nn_idx
        assert np.isclose(knn_results[0][1], nn_dist)
    
    def test_knn_k_equals_n(self):
        """Test KNN with k=n returns all points."""
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])
        tree = KDTree(points)
        
        results = tree.k_nearest_neighbors([0.5, 0.5], k=4)
        
        assert len(results) == 4
        # All indices should be present
        indices = {r[0] for r in results}
        assert indices == {0, 1, 2, 3}
    
    def test_knn_k_greater_than_n(self):
        """Test KNN with k > n returns n results."""
        points = np.array([[0.0, 0.0], [1.0, 1.0]])
        tree = KDTree(points)
        
        results = tree.k_nearest_neighbors([0.0, 0.0], k=10)
        
        assert len(results) == 2
    
    def test_knn_sorted_by_distance(self):
        """Test KNN results are sorted by distance."""
        np.random.seed(42)
        points = np.random.randn(50, 2) * 10
        tree = KDTree(points)
        
        results = tree.k_nearest_neighbors([0.0, 0.0], k=10)
        
        distances = [r[1] for r in results]
        assert distances == sorted(distances)
    
    def test_knn_correctness(self):
        """Test KNN correctness against brute force."""
        np.random.seed(42)
        points = np.random.randn(100, 2) * 10
        tree = KDTree(points)
        
        query = np.array([5.0, 5.0])
        k = 5
        
        # Get KNN results
        knn_results = tree.k_nearest_neighbors(query, k=k)
        knn_distances = sorted([r[1] for r in knn_results])
        
        # Brute force: compute all distances and sort
        all_distances = [
            np.sqrt(np.sum((p - query) ** 2))
            for p in points
        ]
        bf_distances = sorted(all_distances)[:k]
        
        # Compare
        for kd, bf in zip(knn_distances, bf_distances):
            assert np.isclose(kd, bf, rtol=1e-10)


class TestRadiusSearch:
    """Tests for radius search queries."""
    
    def test_radius_search_empty_result(self):
        """Test radius search with no points in range."""
        points = np.array([[100.0, 100.0], [200.0, 200.0]])
        tree = KDTree(points)
        
        results = tree.radius_search([0.0, 0.0], radius=5.0)
        
        assert len(results) == 0
    
    def test_radius_search_single_result(self):
        """Test radius search with one point in range."""
        points = np.array([[0.0, 0.0], [100.0, 100.0]])
        tree = KDTree(points)
        
        results = tree.radius_search([1.0, 1.0], radius=5.0)
        
        assert len(results) == 1
        assert results[0][0] == 0  # Index of [0, 0]
    
    def test_radius_search_multiple_results(self):
        """Test radius search with multiple points."""
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0]
        ])
        tree = KDTree(points)
        
        results = tree.radius_search([0.5, 0.5], radius=2.0)
        
        assert len(results) == 3  # First three points
        indices = {r[0] for r in results}
        assert indices == {0, 1, 2}
    
    def test_radius_search_exact_boundary(self):
        """Test radius search with point exactly on boundary."""
        points = np.array([[5.0, 0.0]])
        tree = KDTree(points)
        
        results = tree.radius_search([0.0, 0.0], radius=5.0)
        
        assert len(results) == 1  # Should include boundary
    
    def test_radius_search_sorted(self):
        """Test radius search results are sorted by distance."""
        np.random.seed(42)
        points = np.random.randn(50, 2)
        tree = KDTree(points)
        
        results = tree.radius_search([0.0, 0.0], radius=2.0)
        
        if len(results) > 1:
            distances = [r[1] for r in results]
            assert distances == sorted(distances)


class TestBatchQueries:
    """Tests for batch query operations."""
    
    def test_batch_query(self):
        """Test batch nearest neighbor queries."""
        np.random.seed(42)
        points = np.random.randn(100, 2) * 10
        tree = KDTree(points)
        
        queries = np.random.randn(20, 2) * 10
        
        indices, distances = tree.query_batch(queries)
        
        assert len(indices) == 20
        assert len(distances) == 20
        
        # Verify each result
        for i, query in enumerate(queries):
            idx, dist = tree.nearest_neighbor(query)
            assert indices[i] == idx
            assert np.isclose(distances[i], dist)
    
    def test_batch_vs_brute_force(self):
        """Test batch query matches brute force batch."""
        np.random.seed(42)
        points = np.random.randn(200, 2) * 10
        queries = np.random.randn(50, 2) * 10
        
        tree = KDTree(points)
        kd_indices, kd_distances = tree.query_batch(queries)
        bf_indices, bf_distances = brute_force_nearest_neighbor_batch(points, queries)
        
        # Distances should match
        assert np.allclose(kd_distances, bf_distances, rtol=1e-10)


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""
    
    def test_duplicate_points(self):
        """Test handling of duplicate points."""
        points = np.array([
            [5.0, 5.0],
            [5.0, 5.0],
            [5.0, 5.0],
            [0.0, 0.0]
        ])
        tree = KDTree(points)
        
        idx, dist = tree.nearest_neighbor([5.0, 5.0])
        
        assert dist == 0.0
        assert idx in [0, 1, 2]  # Any of the duplicate points
    
    def test_collinear_points(self):
        """Test with collinear points."""
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0]
        ])
        tree = KDTree(points)
        
        idx, dist = tree.nearest_neighbor([2.5, 0.0])
        
        assert idx in [2, 3]  # Either [2, 0] or [3, 0]
        assert np.isclose(dist, 0.5)
    
    def test_very_close_points(self):
        """Test with very close points (numerical precision)."""
        eps = 1e-10
        points = np.array([
            [0.0, 0.0],
            [eps, 0.0],
            [0.0, eps]
        ])
        tree = KDTree(points)
        
        idx, dist = tree.nearest_neighbor([0.0, 0.0])
        
        assert idx == 0
        assert np.isclose(dist, 0.0)
    
    def test_large_coordinates(self):
        """Test with large coordinate values."""
        points = np.array([
            [1e6, 1e6],
            [1e6 + 1, 1e6],
            [1e6, 1e6 + 1]
        ])
        tree = KDTree(points)
        
        idx, dist = tree.nearest_neighbor([1e6, 1e6])
        
        assert idx == 0
        assert np.isclose(dist, 0.0)
    
    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        points = np.array([
            [-10.0, -10.0],
            [-5.0, -5.0],
            [0.0, 0.0]
        ])
        tree = KDTree(points)
        
        idx, dist = tree.nearest_neighbor([-4.0, -4.0])
        
        assert idx == 1  # Closest to [-5, -5]


class TestValidation:
    """Integration tests using the validation function."""
    
    def test_validate_small(self):
        """Validate KD-tree with small dataset."""
        assert validate_kdtree(n_points=50, n_queries=20, seed=42)
    
    def test_validate_medium(self):
        """Validate KD-tree with medium dataset."""
        assert validate_kdtree(n_points=500, n_queries=50, seed=123)
    
    def test_validate_large(self):
        """Validate KD-tree with larger dataset."""
        assert validate_kdtree(n_points=2000, n_queries=100, seed=456)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
