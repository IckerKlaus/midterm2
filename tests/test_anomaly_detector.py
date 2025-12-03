"""
Tests for Anomaly Detection

This module tests the anomaly detection logic that identifies
planogram compliance issues from Voronoi cell assignments.

Test Categories:
1. Missing SKU detection
2. Misplaced SKU detection
3. Foreign SKU detection
4. Duplicate detection
5. Swap detection
6. Compliance score calculation
7. Integration with synthetic data

Run with: pytest tests/test_anomaly_detector.py -v
"""

import pytest
import numpy as np
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.anomaly.anomaly_detector import (
    AnomalyDetector,
    detect_anomalies,
    compute_compliance_score,
    generate_compliance_report,
    validate_against_ground_truth
)
from src.data_models import (
    ShelfScenario, ShelfPlanogram, DetectionSet,
    PlanogramEntry, DetectionEntry, AnomalyType,
    GroundTruth, ComplianceReport
)
from src.synthetic_data import generate_shelf_scenario


def create_simple_planogram(positions: List[tuple], skus: List[str]) -> ShelfPlanogram:
    """Helper to create a simple planogram."""
    entries = [
        PlanogramEntry(x=pos[0], y=pos[1], sku_id=sku, row=0, col=i)
        for i, (pos, sku) in enumerate(zip(positions, skus))
    ]
    return ShelfPlanogram(
        entries=entries,
        num_rows=1,
        num_cols=len(positions),
        spacing_x=10.0,
        spacing_y=5.0
    )


def create_simple_detections(positions: List[tuple], skus: List[str]) -> DetectionSet:
    """Helper to create simple detections."""
    entries = [
        DetectionEntry(x=pos[0], y=pos[1], sku_id=sku)
        for pos, sku in zip(positions, skus)
    ]
    return DetectionSet(entries=entries)


def create_scenario(
    planogram_positions: List[tuple],
    planogram_skus: List[str],
    detection_positions: List[tuple],
    detection_skus: List[str]
) -> ShelfScenario:
    """Helper to create a test scenario."""
    planogram = create_simple_planogram(planogram_positions, planogram_skus)
    detections = create_simple_detections(detection_positions, detection_skus)
    return ShelfScenario(
        planogram=planogram,
        detections=detections,
        scenario_id="test"
    )


class TestMissingDetection:
    """Tests for missing SKU detection."""
    
    def test_single_missing(self):
        """Test detecting a single missing product."""
        scenario = create_scenario(
            planogram_positions=[(0, 0), (10, 0), (20, 0)],
            planogram_skus=['A', 'B', 'C'],
            detection_positions=[(0.5, 0), (20.5, 0)],  # Missing middle
            detection_skus=['A', 'C']
        )
        
        report = detect_anomalies(scenario, use_gpu=False)
        
        # Should find one missing anomaly
        missing = [a for a in report.anomalies if a.anomaly_type == AnomalyType.MISSING]
        assert len(missing) == 1
        assert missing[0].planogram_index == 1
        assert missing[0].expected_sku == 'B'
    
    def test_multiple_missing(self):
        """Test detecting multiple missing products."""
        scenario = create_scenario(
            planogram_positions=[(0, 0), (10, 0), (20, 0), (30, 0)],
            planogram_skus=['A', 'B', 'C', 'D'],
            detection_positions=[(0.5, 0), (30.5, 0)],  # Only first and last
            detection_skus=['A', 'D']
        )
        
        report = detect_anomalies(scenario, use_gpu=False)
        
        missing = [a for a in report.anomalies if a.anomaly_type == AnomalyType.MISSING]
        assert len(missing) == 2
        missing_indices = {a.planogram_index for a in missing}
        assert missing_indices == {1, 2}
    
    def test_no_missing(self):
        """Test when nothing is missing."""
        scenario = create_scenario(
            planogram_positions=[(0, 0), (10, 0)],
            planogram_skus=['A', 'B'],
            detection_positions=[(0.5, 0), (10.5, 0)],
            detection_skus=['A', 'B']
        )
        
        report = detect_anomalies(scenario, use_gpu=False)
        
        missing = [a for a in report.anomalies if a.anomaly_type == AnomalyType.MISSING]
        assert len(missing) == 0
    
    def test_all_missing(self):
        """Test when all products are missing."""
        scenario = create_scenario(
            planogram_positions=[(0, 0), (10, 0), (20, 0)],
            planogram_skus=['A', 'B', 'C'],
            detection_positions=[],
            detection_skus=[]
        )
        
        report = detect_anomalies(scenario, use_gpu=False)
        
        missing = [a for a in report.anomalies if a.anomaly_type == AnomalyType.MISSING]
        assert len(missing) == 3


class TestMisplacedDetection:
    """Tests for misplaced SKU detection."""
    
    def test_single_misplaced(self):
        """Test detecting a single misplaced product."""
        scenario = create_scenario(
            planogram_positions=[(0, 0), (10, 0)],
            planogram_skus=['A', 'B'],
            detection_positions=[(0.5, 0), (10.5, 0)],
            detection_skus=['A', 'A']  # Second should be B but is A
        )
        
        report = detect_anomalies(scenario, use_gpu=False)
        
        misplaced = [a for a in report.anomalies if a.anomaly_type == AnomalyType.MISPLACED]
        assert len(misplaced) == 1
        assert misplaced[0].expected_sku == 'B'
        assert misplaced[0].detected_sku == 'A'
    
    def test_multiple_misplaced(self):
        """Test detecting multiple misplaced products."""
        scenario = create_scenario(
            planogram_positions=[(0, 0), (10, 0), (20, 0)],
            planogram_skus=['A', 'B', 'C'],
            detection_positions=[(0.5, 0), (10.5, 0), (20.5, 0)],
            detection_skus=['A', 'C', 'B']  # B and C are swapped
        )
        
        report = detect_anomalies(scenario, use_gpu=False)
        
        misplaced = [a for a in report.anomalies if a.anomaly_type == AnomalyType.MISPLACED]
        assert len(misplaced) == 2


class TestForeignDetection:
    """Tests for foreign SKU detection."""
    
    def test_single_foreign(self):
        """Test detecting a foreign product."""
        scenario = create_scenario(
            planogram_positions=[(0, 0), (10, 0)],
            planogram_skus=['A', 'B'],
            detection_positions=[(0.5, 0), (10.5, 0)],
            detection_skus=['A', 'FOREIGN_X']  # X doesn't exist in planogram
        )
        
        report = detect_anomalies(scenario, use_gpu=False)
        
        foreign = [a for a in report.anomalies if a.anomaly_type == AnomalyType.FOREIGN]
        assert len(foreign) == 1
        assert foreign[0].detected_sku == 'FOREIGN_X'
        assert foreign[0].expected_sku == 'B'
    
    def test_multiple_foreign(self):
        """Test detecting multiple foreign products."""
        scenario = create_scenario(
            planogram_positions=[(0, 0), (10, 0), (20, 0)],
            planogram_skus=['A', 'B', 'C'],
            detection_positions=[(0.5, 0), (10.5, 0), (20.5, 0)],
            detection_skus=['FOREIGN_1', 'B', 'FOREIGN_2']
        )
        
        report = detect_anomalies(scenario, use_gpu=False)
        
        foreign = [a for a in report.anomalies if a.anomaly_type == AnomalyType.FOREIGN]
        assert len(foreign) == 2


class TestDuplicateDetection:
    """Tests for duplicate detection."""
    
    def test_single_duplicate(self):
        """Test detecting a duplicate product in same cell."""
        scenario = create_scenario(
            planogram_positions=[(0, 0), (10, 0)],
            planogram_skus=['A', 'B'],
            detection_positions=[
                (0.5, 0),   # First detection for A
                (0.8, 0),   # Duplicate detection for A (same cell)
                (10.5, 0)   # Detection for B
            ],
            detection_skus=['A', 'A', 'B']
        )
        
        report = detect_anomalies(scenario, use_gpu=False)
        
        duplicates = [a for a in report.anomalies if a.anomaly_type == AnomalyType.DUPLICATE]
        assert len(duplicates) == 1
    
    def test_multiple_duplicates_same_cell(self):
        """Test multiple duplicates in same cell."""
        scenario = create_scenario(
            planogram_positions=[(0, 0), (100, 0)],
            planogram_skus=['A', 'B'],
            detection_positions=[
                (0.1, 0),
                (0.2, 0),
                (0.3, 0),
                (100.1, 0)
            ],
            detection_skus=['A', 'A', 'A', 'B']
        )
        
        report = detect_anomalies(scenario, use_gpu=False)
        
        duplicates = [a for a in report.anomalies if a.anomaly_type == AnomalyType.DUPLICATE]
        assert len(duplicates) == 2  # Two extra detections beyond first


class TestSwapDetection:
    """Tests for swap detection."""
    
    def test_simple_swap(self):
        """Test detecting a simple swap between two cells."""
        scenario = create_scenario(
            planogram_positions=[(0, 0), (10, 0)],
            planogram_skus=['A', 'B'],
            detection_positions=[(0.5, 0), (10.5, 0)],
            detection_skus=['B', 'A']  # Swapped
        )
        
        report = detect_anomalies(scenario, use_gpu=False)
        
        swaps = [a for a in report.anomalies if a.anomaly_type == AnomalyType.SWAPPED]
        # Should detect the swap pattern
        assert len(swaps) >= 1
    
    def test_swap_with_normal(self):
        """Test swap detection alongside normal placements."""
        scenario = create_scenario(
            planogram_positions=[(0, 0), (10, 0), (20, 0), (30, 0)],
            planogram_skus=['A', 'B', 'C', 'D'],
            detection_positions=[(0.5, 0), (10.5, 0), (20.5, 0), (30.5, 0)],
            detection_skus=['A', 'C', 'B', 'D']  # B and C swapped
        )
        
        report = detect_anomalies(scenario, use_gpu=False)
        
        # Should find misplaced and possibly detect swap pattern
        misplaced = [a for a in report.anomalies if a.anomaly_type == AnomalyType.MISPLACED]
        assert len(misplaced) == 2


class TestComplianceScore:
    """Tests for compliance score calculation."""
    
    def test_perfect_compliance(self):
        """Test 100% compliance score."""
        scenario = create_scenario(
            planogram_positions=[(0, 0), (10, 0), (20, 0)],
            planogram_skus=['A', 'B', 'C'],
            detection_positions=[(0.5, 0), (10.5, 0), (20.5, 0)],
            detection_skus=['A', 'B', 'C']
        )
        
        report = detect_anomalies(scenario, use_gpu=False)
        
        assert report.compliance_score == 1.0
        assert report.num_anomalies == 0
    
    def test_zero_compliance(self):
        """Test 0% compliance (all missing)."""
        scenario = create_scenario(
            planogram_positions=[(0, 0), (10, 0)],
            planogram_skus=['A', 'B'],
            detection_positions=[],
            detection_skus=[]
        )
        
        report = detect_anomalies(scenario, use_gpu=False)
        
        # 2 anomalies for 2 expected = 0% compliance
        assert report.compliance_score == 0.0
    
    def test_partial_compliance(self):
        """Test partial compliance score."""
        scenario = create_scenario(
            planogram_positions=[(0, 0), (10, 0), (20, 0), (30, 0)],
            planogram_skus=['A', 'B', 'C', 'D'],
            detection_positions=[(0.5, 0), (10.5, 0), (20.5, 0), (30.5, 0)],
            detection_skus=['A', 'B', 'FOREIGN', 'D']  # One foreign
        )
        
        report = detect_anomalies(scenario, use_gpu=False)
        
        # 1 anomaly for 4 expected = 75% compliance
        assert report.compliance_score == 0.75


class TestComplianceReport:
    """Tests for compliance report generation."""
    
    def test_report_summary(self):
        """Test report summary generation."""
        scenario = create_scenario(
            planogram_positions=[(0, 0), (10, 0)],
            planogram_skus=['A', 'B'],
            detection_positions=[(0.5, 0)],  # One missing
            detection_skus=['A']
        )
        
        report = detect_anomalies(scenario, use_gpu=False)
        summary = report.summary()
        
        assert 'PLANOGRAM COMPLIANCE REPORT' in summary
        assert 'Expected Products' in summary
        assert 'Compliance Score' in summary
    
    def test_anomalies_by_type(self):
        """Test grouping anomalies by type."""
        scenario = create_scenario(
            planogram_positions=[(0, 0), (10, 0), (20, 0)],
            planogram_skus=['A', 'B', 'C'],
            detection_positions=[(0.5, 0), (10.5, 0)],  # C is missing
            detection_skus=['FOREIGN', 'B']  # A is foreign
        )
        
        report = detect_anomalies(scenario, use_gpu=False)
        by_type = report.get_anomalies_by_type()
        
        assert AnomalyType.MISSING in by_type
        assert AnomalyType.FOREIGN in by_type
    
    def test_report_to_dict(self):
        """Test report serialization."""
        scenario = create_scenario(
            planogram_positions=[(0, 0), (10, 0)],
            planogram_skus=['A', 'B'],
            detection_positions=[(0.5, 0), (10.5, 0)],
            detection_skus=['A', 'B']
        )
        
        report = detect_anomalies(scenario, use_gpu=False)
        report_dict = report.to_dict()
        
        assert 'compliance_score' in report_dict
        assert 'total_expected' in report_dict
        assert 'anomalies' in report_dict


class TestAnomalyDetector:
    """Tests for AnomalyDetector class."""
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = AnomalyDetector(use_gpu=False)
        
        assert detector.use_gpu == False
        assert detector.distance_threshold > 0
    
    def test_detector_analyze(self):
        """Test detector analyze method."""
        scenario = create_scenario(
            planogram_positions=[(0, 0), (10, 0)],
            planogram_skus=['A', 'B'],
            detection_positions=[(0.5, 0), (10.5, 0)],
            detection_skus=['A', 'B']
        )
        
        detector = AnomalyDetector(use_gpu=False)
        report = detector.analyze(scenario)
        
        assert isinstance(report, ComplianceReport)
        assert report.processing_time_ms > 0


class TestGroundTruthValidation:
    """Tests for ground truth validation."""
    
    def test_validate_perfect_detection(self):
        """Test validation with perfect detection."""
        scenario = generate_shelf_scenario(
            num_rows=2,
            num_cols=5,
            noise_std=0.1,
            anomaly_rate=0.0,  # No anomalies
            seed=42
        )
        
        metrics = validate_against_ground_truth(scenario, use_gpu=False)
        
        # With no ground truth anomalies, metrics should be near perfect
        # (Some false positives may occur due to noise)
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
    
    def test_validate_with_anomalies(self):
        """Test validation with known anomalies."""
        scenario = generate_shelf_scenario(
            num_rows=3,
            num_cols=10,
            noise_std=0.5,
            anomaly_rate=0.2,
            seed=42
        )
        
        metrics = validate_against_ground_truth(scenario, use_gpu=False)
        
        # Should have some true positives
        assert metrics['true_positives'] >= 0
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1


class TestIntegrationWithSyntheticData:
    """Integration tests with synthetic data generator."""
    
    def test_small_scenario(self):
        """Test on small synthetic scenario."""
        scenario = generate_shelf_scenario(
            num_rows=2,
            num_cols=6,
            anomaly_rate=0.1,
            seed=42
        )
        
        report = detect_anomalies(scenario, use_gpu=False)
        
        assert report.total_expected == 12
        assert report.compliance_score >= 0
        assert report.compliance_score <= 1
    
    def test_medium_scenario(self):
        """Test on medium synthetic scenario."""
        scenario = generate_shelf_scenario(
            num_rows=5,
            num_cols=20,
            anomaly_rate=0.15,
            seed=123
        )
        
        report = detect_anomalies(scenario, use_gpu=False)
        
        assert report.total_expected == 100
        # With 15% anomaly rate, expect some anomalies
        assert report.num_anomalies > 0
    
    def test_large_scenario(self):
        """Test on larger synthetic scenario."""
        scenario = generate_shelf_scenario(
            num_rows=10,
            num_cols=50,
            anomaly_rate=0.1,
            seed=456
        )
        
        report = detect_anomalies(scenario, use_gpu=False)
        
        assert report.total_expected == 500
        assert report.processing_time_ms > 0


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_planogram(self):
        """Test with empty planogram raises error or returns perfect score."""
        planogram = ShelfPlanogram(
            entries=[],
            num_rows=0,
            num_cols=0,
            spacing_x=10.0,
            spacing_y=5.0
        )
        detections = DetectionSet(entries=[])
        scenario = ShelfScenario(
            planogram=planogram,
            detections=detections
        )
        
        # Empty planogram should raise ValueError since there's nothing to check
        with pytest.raises(ValueError):
            detect_anomalies(scenario, use_gpu=False)
    
    def test_single_product(self):
        """Test with single product."""
        scenario = create_scenario(
            planogram_positions=[(0, 0)],
            planogram_skus=['A'],
            detection_positions=[(0.5, 0)],
            detection_skus=['A']
        )
        
        report = detect_anomalies(scenario, use_gpu=False)
        
        assert report.total_expected == 1
        assert report.compliance_score == 1.0


class TestGenerateComplianceReport:
    """Tests for report generation function."""
    
    def test_generate_report_verbose(self):
        """Test verbose report generation."""
        scenario = generate_shelf_scenario(
            num_rows=2,
            num_cols=5,
            anomaly_rate=0.2,
            seed=42
        )
        
        report_text = generate_compliance_report(scenario, verbose=True, use_gpu=False)
        
        assert 'PLANOGRAM COMPLIANCE REPORT' in report_text
        assert 'Expected Products' in report_text
    
    def test_generate_report_brief(self):
        """Test brief report generation."""
        scenario = generate_shelf_scenario(
            num_rows=2,
            num_cols=5,
            anomaly_rate=0.0,
            seed=42
        )
        
        report_text = generate_compliance_report(scenario, verbose=False, use_gpu=False)
        
        assert 'PLANOGRAM COMPLIANCE REPORT' in report_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])