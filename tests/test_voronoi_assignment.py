"""
Tests for Voronoi Cell Assignment

This module tests the Voronoi-based spatial assignment functionality,
verifying that detections are correctly assigned to their nearest
planogram positions (Voronoi cells).

Test Categories:
1. Basic assignment correctness
2. Perfect alignment (zero noise)
3. CPU vs GPU consistency
4. Edge cases
5. Integration with synthetic data

The key property: In 2D Euclidean space, Voronoi cell assignment
is equivalent to nearest-neighbor search.

Run with: pytest tests/test_voronoi_assignment.py -v
"""

import pytest
import numpy as np
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry.voronoi_cpu import (
    assign_to_voronoi_cells,
    VoronoiAssignment,
    compute_assignment_statistics
)
from src.geometry.voronoi_gpu import (
    gpu_assign_to_voronoi_cells,
    is_gpu_available
)
from src.geometry.kd_tree import brute_force_nearest_neighbor
from src.synthetic_data import (
    generate_planogram,
    generate_detections,
    generate_shelf_scenario,
    SKUPattern
)


class TestBasicAssignment:
    """Tests for basic Voronoi assignment functionality."""
    
    def test_assignment_simple(self):
        """Test assignment with simple 3-point planogram."""
        planogram = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0]
        ])
        detections = np.array([
            [1.0, 0.5],   # Closest to [0, 0]
            [9.0, -0.5],  # Closest to [10, 0]
            [21.0, 0.2]   # Closest to [20, 0]
        ])
        
        result = assign_to_voronoi_cells(planogram, detections)
        
        assert len(result.assignments) == 3
        assert result.assignments[0] == 0
        assert result.assignments[1] == 1
        assert result.assignments[2] == 2
    
    def test_assignment_grid(self):
        """Test assignment with grid planogram."""
        # 2x3 grid
        planogram = np.array([
            [0.0, 0.0], [10.0, 0.0], [20.0, 0.0],
            [0.0, -10.0], [10.0, -10.0], [20.0, -10.0]
        ])
        
        # Detections near each grid point
        detections = np.array([
            [1.0, 1.0],      # Near [0, 0]
            [11.0, -1.0],    # Near [10, 0]
            [19.0, 0.5],     # Near [20, 0]
            [0.5, -9.0],     # Near [0, -10]
            [10.5, -10.5],   # Near [10, -10]
            [20.5, -9.5]     # Near [20, -10]
        ])
        
        result = assign_to_voronoi_cells(planogram, detections)
        
        expected = [0, 1, 2, 3, 4, 5]
        assert list(result.assignments) == expected
    
    def test_assignment_with_skus(self):
        """Test assignment preserves SKU information."""
        planogram = np.array([[0.0, 0.0], [10.0, 0.0]])
        detections = np.array([[1.0, 0.0], [9.0, 0.0]])
        planogram_skus = ['SKU_A', 'SKU_B']
        detection_skus = ['SKU_A', 'SKU_B']
        
        result = assign_to_voronoi_cells(
            planogram, detections,
            planogram_skus, detection_skus
        )
        
        assert result.detection_skus == ['SKU_A', 'SKU_B']
        assert result.planogram_skus == ['SKU_A', 'SKU_B']
    
    def test_assignment_empty_detections(self):
        """Test assignment with no detections."""
        planogram = np.array([[0.0, 0.0], [10.0, 0.0]])
        detections = np.array([]).reshape(0, 2)
        
        result = assign_to_voronoi_cells(planogram, detections)
        
        assert len(result.assignments) == 0
        assert len(result.distances) == 0
    
    def test_assignment_single_planogram_point(self):
        """Test assignment with single planogram point."""
        planogram = np.array([[5.0, 5.0]])
        detections = np.array([
            [0.0, 0.0],
            [10.0, 10.0],
            [5.0, 5.0]
        ])
        
        result = assign_to_voronoi_cells(planogram, detections)
        
        # All should be assigned to the only point
        assert all(result.assignments == 0)


class TestMethodComparison:
    """Tests comparing different assignment methods."""
    
    def test_kdtree_vs_brute_force(self):
        """Test KD-tree matches brute force."""
        np.random.seed(42)
        planogram = np.random.randn(50, 2) * 10
        detections = np.random.randn(30, 2) * 10
        
        result_kd = assign_to_voronoi_cells(
            planogram, detections, method="kdtree"
        )
        result_bf = assign_to_voronoi_cells(
            planogram, detections, method="brute_force"
        )
        
        # Assignments should be identical
        assert np.array_equal(result_kd.assignments, result_bf.assignments)
        # Distances should be very close
        assert np.allclose(result_kd.distances, result_bf.distances, rtol=1e-10)
    
    @pytest.mark.skipif(not is_gpu_available(), reason="GPU not available")
    def test_gpu_vs_cpu(self):
        """Test GPU matches CPU results."""
        np.random.seed(42)
        planogram = np.random.randn(100, 2) * 10
        detections = np.random.randn(50, 2) * 10
        
        # CPU result
        result_cpu = assign_to_voronoi_cells(
            planogram, detections, method="kdtree"
        )
        
        # GPU result
        gpu_assignments, gpu_distances = gpu_assign_to_voronoi_cells(
            planogram, detections
        )
        
        # Should match
        assert np.array_equal(result_cpu.assignments, gpu_assignments)
        assert np.allclose(result_cpu.distances, gpu_distances, rtol=1e-10)
    
    def test_auto_method_selection(self):
        """Test automatic method selection."""
        planogram = np.random.randn(200, 2)
        detections = np.random.randn(100, 2)
        
        # Should work regardless of which method is selected
        result = assign_to_voronoi_cells(
            planogram, detections, method="auto"
        )
        
        assert len(result.assignments) == 100


class TestPerfectAlignment:
    """Tests with perfectly aligned detections (zero noise)."""
    
    def test_perfect_alignment_small(self):
        """Test perfect alignment on small grid with no anomaly injection."""
        # Create planogram manually to ensure no anomalies
        from src.synthetic_data import generate_planogram
        
        planogram = generate_planogram(
            num_rows=2,
            num_cols=5,
            spacing_x=10.0,
            spacing_y=5.0,
            seed=42
        )
        
        planogram_points = planogram.get_points_array()
        # Create perfect detections at exact planogram positions
        detection_points = planogram_points.copy()
        planogram_skus = planogram.get_sku_list()
        detection_skus = planogram_skus.copy()
        
        result = assign_to_voronoi_cells(
            planogram_points, detection_points,
            planogram_skus, detection_skus
        )
        
        # With zero noise and perfect positions, each detection should match its cell
        assert np.all(result.distances < 1e-10)
        
        # SKUs should match
        for i in range(len(detection_skus)):
            assert result.detection_skus[i] == result.planogram_skus[i]
    
    def test_perfect_alignment_large(self):
        """Test perfect alignment on larger grid."""
        from src.synthetic_data import generate_planogram
        
        planogram = generate_planogram(
            num_rows=10,
            num_cols=20,
            spacing_x=10.0,
            spacing_y=5.0,
            seed=123
        )
        
        planogram_points = planogram.get_points_array()
        # Create perfect detections
        detection_points = planogram_points.copy()
        
        result = assign_to_voronoi_cells(planogram_points, detection_points)
        
        # All distances should be essentially zero
        assert np.all(result.distances < 1e-10)
        
        # Assignments should be 0, 1, 2, ..., n-1
        expected = np.arange(len(planogram_points))
        assert np.array_equal(result.assignments, expected)


class TestNoisyData:
    """Tests with noisy synthetic data."""
    
    def test_noisy_assignment_preserves_nearest(self):
        """Test noisy detections are assigned to nearest planogram point."""
        np.random.seed(42)
        
        # Create planogram
        planogram = generate_planogram(
            num_rows=3, num_cols=5,
            spacing_x=10, spacing_y=5
        )
        
        # Create noisy detections
        detections, _ = generate_detections(
            planogram,
            noise_std=1.0,
            anomaly_rate=0.0,
            seed=42
        )
        
        planogram_points = planogram.get_points_array()
        detection_points = detections.get_points_array()
        
        result = assign_to_voronoi_cells(planogram_points, detection_points)
        
        # Verify each assignment is actually the nearest
        for i, det_point in enumerate(detection_points):
            assigned_idx = result.assignments[i]
            assigned_dist = result.distances[i]
            
            # Check against brute force
            bf_idx, bf_dist = brute_force_nearest_neighbor(planogram_points, det_point)
            
            assert np.isclose(assigned_dist, bf_dist, rtol=1e-10)
    
    def test_high_noise_still_valid(self):
        """Test high noise still produces valid assignments."""
        scenario = generate_shelf_scenario(
            num_rows=3, num_cols=10,
            noise_std=5.0,  # High noise
            anomaly_rate=0.0,
            seed=42
        )
        
        planogram_points = scenario.planogram.get_points_array()
        detection_points = scenario.detections.get_points_array()
        
        result = assign_to_voronoi_cells(planogram_points, detection_points)
        
        # All assignments should be valid indices
        assert np.all(result.assignments >= 0)
        assert np.all(result.assignments < len(planogram_points))
        
        # All distances should be positive
        assert np.all(result.distances >= 0)


class TestCellContents:
    """Tests for cell content grouping."""
    
    def test_get_cell_contents(self):
        """Test grouping detections by cell."""
        planogram = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        detections = np.array([
            [0.5, 0.0],   # Cell 0
            [1.0, 0.0],   # Cell 0
            [10.5, 0.0],  # Cell 1
            [20.1, 0.0]   # Cell 2
        ])
        
        result = assign_to_voronoi_cells(planogram, detections)
        contents = result.get_cell_contents()
        
        assert 0 in contents
        assert 1 in contents
        assert 2 in contents
        assert len(contents[0]) == 2  # Two detections in cell 0
        assert len(contents[1]) == 1
        assert len(contents[2]) == 1
    
    def test_empty_cells(self):
        """Test cells with no detections."""
        planogram = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        detections = np.array([
            [0.5, 0.0],   # Cell 0 only
        ])
        
        result = assign_to_voronoi_cells(planogram, detections)
        contents = result.get_cell_contents()
        
        assert 0 in contents
        assert 1 not in contents  # Empty
        assert 2 not in contents  # Empty


class TestMismatches:
    """Tests for SKU mismatch detection."""
    
    def test_get_mismatches(self):
        """Test finding SKU mismatches."""
        planogram = np.array([[0.0, 0.0], [10.0, 0.0]])
        detections = np.array([[0.5, 0.0], [10.5, 0.0]])
        planogram_skus = ['SKU_A', 'SKU_B']
        detection_skus = ['SKU_A', 'SKU_C']  # Second is wrong
        
        result = assign_to_voronoi_cells(
            planogram, detections,
            planogram_skus, detection_skus
        )
        
        mismatches = result.get_mismatches()
        
        assert len(mismatches) == 1
        assert mismatches[0] == (1, 'SKU_C', 'SKU_B')
    
    def test_no_mismatches(self):
        """Test when all SKUs match."""
        planogram = np.array([[0.0, 0.0], [10.0, 0.0]])
        detections = np.array([[0.5, 0.0], [10.5, 0.0]])
        planogram_skus = ['SKU_A', 'SKU_B']
        detection_skus = ['SKU_A', 'SKU_B']
        
        result = assign_to_voronoi_cells(
            planogram, detections,
            planogram_skus, detection_skus
        )
        
        mismatches = result.get_mismatches()
        
        assert len(mismatches) == 0


class TestAssignmentStatistics:
    """Tests for assignment statistics computation."""
    
    def test_statistics_perfect(self):
        """Test statistics with perfect alignment."""
        planogram = np.array([[0.0, 0.0], [10.0, 0.0]])
        detections = np.array([[0.0, 0.0], [10.0, 0.0]])
        planogram_skus = ['A', 'B']
        detection_skus = ['A', 'B']
        
        result = assign_to_voronoi_cells(
            planogram, detections,
            planogram_skus, detection_skus
        )
        
        stats = compute_assignment_statistics(result)
        
        assert stats['mean_distance'] == 0.0
        assert stats['max_distance'] == 0.0
        assert stats['sku_match_rate'] == 1.0
        assert stats['close_match_rate'] == 1.0
    
    def test_statistics_with_noise(self):
        """Test statistics with noisy data."""
        np.random.seed(42)
        planogram = np.random.randn(20, 2) * 10
        detections = planogram + np.random.randn(20, 2) * 0.5  # Small noise
        
        result = assign_to_voronoi_cells(planogram, detections)
        stats = compute_assignment_statistics(result, distance_threshold=2.0)
        
        assert stats['mean_distance'] > 0
        assert stats['close_match_rate'] > 0.9  # Most should be close


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_detection_single_planogram(self):
        """Test with single point each."""
        planogram = np.array([[5.0, 5.0]])
        detections = np.array([[6.0, 6.0]])
        
        result = assign_to_voronoi_cells(planogram, detections)
        
        assert result.assignments[0] == 0
        assert np.isclose(result.distances[0], np.sqrt(2))
    
    def test_many_detections_few_planogram(self):
        """Test many detections assigned to few planogram points."""
        planogram = np.array([[0.0, 0.0], [100.0, 0.0]])
        
        # 10 detections clustered around first point
        detections = np.random.randn(10, 2) * 5
        
        result = assign_to_voronoi_cells(planogram, detections)
        
        # All should be assigned to first point (closer to origin)
        assert np.all(result.assignments == 0)
    
    def test_equidistant_points(self):
        """Test with equidistant planogram points."""
        planogram = np.array([[-5.0, 0.0], [5.0, 0.0]])
        detections = np.array([[0.0, 0.0]])  # Equidistant
        
        result = assign_to_voronoi_cells(planogram, detections)
        
        # Should assign to one of them (implementation dependent)
        assert result.assignments[0] in [0, 1]
        assert np.isclose(result.distances[0], 5.0)


class TestIntegrationWithSyntheticData:
    """Integration tests using synthetic data generators."""
    
    def test_full_scenario_assignment(self):
        """Test assignment on full synthetic scenario."""
        scenario = generate_shelf_scenario(
            num_rows=3,
            num_cols=10,
            noise_std=0.5,
            anomaly_rate=0.1,
            seed=42
        )
        
        planogram_points = scenario.planogram.get_points_array()
        detection_points = scenario.detections.get_points_array()
        planogram_skus = scenario.planogram.get_sku_list()
        detection_skus = scenario.detections.get_sku_list()
        
        result = assign_to_voronoi_cells(
            planogram_points, detection_points,
            planogram_skus, detection_skus,
            method="kdtree"
        )
        
        # Basic sanity checks
        assert len(result.assignments) == len(detection_points)
        assert np.all(result.assignments >= 0)
        assert np.all(result.assignments < len(planogram_points))
    
    def test_different_sku_patterns(self):
        """Test assignment with different SKU patterns."""
        for pattern in [SKUPattern.UNIQUE, SKUPattern.BLOCKS, SKUPattern.ROWS]:
            planogram = generate_planogram(
                num_rows=3, num_cols=5,
                sku_pattern=pattern,
                seed=42
            )
            
            detections, _ = generate_detections(
                planogram,
                noise_std=0.5,
                anomaly_rate=0.0,
                seed=42
            )
            
            planogram_points = planogram.get_points_array()
            detection_points = detections.get_points_array()
            
            result = assign_to_voronoi_cells(planogram_points, detection_points)
            
            # Should produce valid assignments
            assert len(result.assignments) == len(detection_points)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])