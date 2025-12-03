"""
Anomaly Detection for Planogram Compliance

This module implements the core anomaly detection logic that compares
expected planogram positions with actual detected product positions.

Detection Strategy:
    1. Use Voronoi assignment to map each detection to a planogram cell
    2. For each planogram cell, check what was detected there
    3. Classify anomalies based on mismatches between expected and actual

Anomaly Types:
    - MISSING: A planogram cell has no detection assigned to it
    - MISPLACED: A detection's SKU doesn't match its assigned cell's expected SKU
    - FOREIGN: A detected SKU doesn't exist anywhere in the planogram
    - DUPLICATE: Multiple detections assigned to the same cell
    - SWAPPED: Two cells have each other's expected SKUs (detected via pattern)

Complexity Analysis:
    The detection algorithm is O(n + m) where n = planogram size, m = detections,
    after the O(n log n + m log n) Voronoi assignment is complete.

Integration with Course Topics:
    - Computational Geometry: Voronoi-based region assignment
    - Advanced Search: KD-tree for efficient nearest-neighbor queries
    - HPC: GPU acceleration for large-scale distance computations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
import time
import numpy as np

from ..data_models import (
    ShelfScenario, ShelfPlanogram, DetectionSet,
    AnomalyRecord, AnomalyType, ComplianceReport
)
from ..geometry.voronoi_cpu import assign_to_voronoi_cells, VoronoiAssignment
from ..geometry.voronoi_gpu import gpu_assign_to_voronoi_cells, is_gpu_available


class AnomalyDetector:
    """
    Detects anomalies in shelf product placement by comparing
    detected positions against the expected planogram.
    
    The detector uses Voronoi cell assignment to determine which
    planogram position "owns" each detection, then identifies
    mismatches between expected and actual SKUs.
    
    Attributes:
        use_gpu: Whether to use GPU acceleration for assignment
        distance_threshold: Maximum distance for valid assignment
        duplicate_threshold: Distance threshold for duplicate detection
    
    Example:
        >>> detector = AnomalyDetector(use_gpu=False)
        >>> report = detector.analyze(scenario)
        >>> print(report.summary())
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        distance_threshold: float = 10.0,
        duplicate_threshold: float = 2.0
    ):
        """
        Initialize the anomaly detector.
        
        Args:
            use_gpu: Use GPU acceleration if available
            distance_threshold: Max distance for a valid cell assignment
            duplicate_threshold: Min distance to consider detections as duplicates
        """
        self.use_gpu = use_gpu and is_gpu_available()
        self.distance_threshold = distance_threshold
        self.duplicate_threshold = duplicate_threshold
    
    def analyze(
        self,
        scenario: ShelfScenario,
        method: str = "auto"
    ) -> ComplianceReport:
        """
        Analyze a shelf scenario and detect all anomalies.
        
        This is the main entry point for anomaly detection. It:
        1. Performs Voronoi assignment of detections to planogram cells
        2. Identifies all anomaly types
        3. Computes compliance metrics
        4. Generates a comprehensive report
        
        Args:
            scenario: Complete shelf scenario with planogram and detections
            method: Assignment method ("auto", "kdtree", "gpu", "brute_force")
        
        Returns:
            ComplianceReport with all detected anomalies and metrics
        
        Complexity:
            Time: O(n log n + m log n) for assignment + O(n + m) for detection
            Space: O(n + m) for storing assignments and results
        """
        start_time = time.perf_counter()
        
        # Extract data arrays
        planogram_points = scenario.planogram.get_points_array()
        detection_points = scenario.detections.get_points_array()
        planogram_skus = scenario.planogram.get_sku_list()
        detection_skus = scenario.detections.get_sku_list()
        
        # Perform Voronoi assignment
        if method == "auto":
            method = "gpu" if self.use_gpu else "kdtree"
        
        if method == "gpu" and self.use_gpu:
            assignments, distances = gpu_assign_to_voronoi_cells(
                planogram_points, detection_points
            )
            assignment_result = VoronoiAssignment(
                assignments=assignments,
                distances=distances,
                detection_skus=detection_skus,
                planogram_skus=[planogram_skus[i] for i in assignments]
            )
        else:
            assignment_result = assign_to_voronoi_cells(
                planogram_points, detection_points,
                planogram_skus, detection_skus,
                method="kdtree" if method in ["auto", "kdtree"] else method
            )
        
        # Detect anomalies
        anomalies = self._detect_all_anomalies(
            scenario.planogram,
            scenario.detections,
            assignment_result,
            planogram_skus,
            detection_skus
        )
        
        # Compute compliance score
        compliance_score = self._compute_compliance_score(
            len(planogram_skus),
            len(anomalies)
        )
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return ComplianceReport(
            anomalies=anomalies,
            total_expected=len(planogram_skus),
            total_detected=len(detection_skus),
            compliance_score=compliance_score,
            processing_time_ms=processing_time
        )
    
    def _detect_all_anomalies(
        self,
        planogram: ShelfPlanogram,
        detections: DetectionSet,
        assignment: VoronoiAssignment,
        planogram_skus: List[str],
        detection_skus: List[str]
    ) -> List[AnomalyRecord]:
        """
        Detect all types of anomalies from the assignment result.
        
        Args:
            planogram: Reference planogram
            detections: Detected products
            assignment: Voronoi cell assignment result
            planogram_skus: List of expected SKUs
            detection_skus: List of detected SKUs
        
        Returns:
            List of all detected anomalies
        
        Complexity: O(n + m) where n = planogram size, m = detections
        """
        anomalies: List[AnomalyRecord] = []
        
        # Build set of all planogram SKUs for foreign detection
        planogram_sku_set = set(planogram_skus)
        
        # Group detections by assigned cell
        cell_contents: Dict[int, List[int]] = defaultdict(list)
        for det_idx, plan_idx in enumerate(assignment.assignments):
            cell_contents[int(plan_idx)].append(det_idx)
        
        # Check each planogram cell
        for plan_idx in range(len(planogram_skus)):
            expected_sku = planogram_skus[plan_idx]
            detections_in_cell = cell_contents.get(plan_idx, [])
            
            if len(detections_in_cell) == 0:
                # MISSING: No detection in this cell
                anomalies.append(AnomalyRecord(
                    anomaly_type=AnomalyType.MISSING,
                    planogram_index=plan_idx,
                    expected_sku=expected_sku,
                    position=(planogram.entries[plan_idx].x, 
                             planogram.entries[plan_idx].y),
                    details=f"Expected {expected_sku} but no detection found"
                ))
            
            elif len(detections_in_cell) == 1:
                # Single detection - check if SKU matches
                det_idx = detections_in_cell[0]
                detected_sku = detection_skus[det_idx]
                
                if detected_sku != expected_sku:
                    # Check if it's a foreign SKU
                    if detected_sku not in planogram_sku_set:
                        anomalies.append(AnomalyRecord(
                            anomaly_type=AnomalyType.FOREIGN,
                            planogram_index=plan_idx,
                            detection_index=det_idx,
                            expected_sku=expected_sku,
                            detected_sku=detected_sku,
                            position=(detections.entries[det_idx].x,
                                    detections.entries[det_idx].y),
                            details=f"Foreign SKU {detected_sku} in cell expecting {expected_sku}"
                        ))
                    else:
                        # MISPLACED: SKU exists in planogram but in wrong cell
                        anomalies.append(AnomalyRecord(
                            anomaly_type=AnomalyType.MISPLACED,
                            planogram_index=plan_idx,
                            detection_index=det_idx,
                            expected_sku=expected_sku,
                            detected_sku=detected_sku,
                            position=(detections.entries[det_idx].x,
                                    detections.entries[det_idx].y),
                            details=f"Expected {expected_sku}, detected {detected_sku}"
                        ))
            
            else:
                # Multiple detections - DUPLICATE case
                # First detection is considered primary, rest are duplicates
                for i, det_idx in enumerate(detections_in_cell):
                    detected_sku = detection_skus[det_idx]
                    
                    if i == 0:
                        # Check if primary detection matches
                        if detected_sku != expected_sku:
                            if detected_sku not in planogram_sku_set:
                                anomalies.append(AnomalyRecord(
                                    anomaly_type=AnomalyType.FOREIGN,
                                    planogram_index=plan_idx,
                                    detection_index=det_idx,
                                    expected_sku=expected_sku,
                                    detected_sku=detected_sku,
                                    position=(detections.entries[det_idx].x,
                                            detections.entries[det_idx].y),
                                    details=f"Foreign SKU {detected_sku}"
                                ))
                            else:
                                anomalies.append(AnomalyRecord(
                                    anomaly_type=AnomalyType.MISPLACED,
                                    planogram_index=plan_idx,
                                    detection_index=det_idx,
                                    expected_sku=expected_sku,
                                    detected_sku=detected_sku,
                                    position=(detections.entries[det_idx].x,
                                            detections.entries[det_idx].y),
                                    details=f"Expected {expected_sku}, detected {detected_sku}"
                                ))
                    else:
                        # Additional detections are duplicates
                        anomalies.append(AnomalyRecord(
                            anomaly_type=AnomalyType.DUPLICATE,
                            planogram_index=plan_idx,
                            detection_index=det_idx,
                            expected_sku=expected_sku,
                            detected_sku=detected_sku,
                            position=(detections.entries[det_idx].x,
                                    detections.entries[det_idx].y),
                            details=f"Duplicate detection of {detected_sku} in cell {plan_idx}"
                        ))
        
        # Detect swaps (post-processing)
        swap_anomalies = self._detect_swaps(anomalies, planogram_skus)
        
        return anomalies + swap_anomalies
    
    def _detect_swaps(
        self,
        anomalies: List[AnomalyRecord],
        planogram_skus: List[str]
    ) -> List[AnomalyRecord]:
        """
        Detect swap patterns from misplaced anomalies.
        
        A swap occurs when cell A has SKU of cell B and vice versa.
        
        Args:
            anomalies: Existing anomaly list
            planogram_skus: List of expected SKUs
        
        Returns:
            List of additional swap anomaly records
        """
        swap_anomalies: List[AnomalyRecord] = []
        
        # Build mapping of cell -> detected SKU for misplaced items
        misplaced_map: Dict[int, str] = {}
        for anomaly in anomalies:
            if anomaly.anomaly_type == AnomalyType.MISPLACED:
                if anomaly.planogram_index >= 0 and anomaly.detected_sku:
                    misplaced_map[anomaly.planogram_index] = anomaly.detected_sku
        
        # Find swap pairs
        checked: Set[Tuple[int, int]] = set()
        for cell_a, detected_a in misplaced_map.items():
            expected_a = planogram_skus[cell_a]
            
            # Find if any cell has expected_a as detected and detected_a as expected
            for cell_b, detected_b in misplaced_map.items():
                if cell_a >= cell_b:
                    continue
                if (cell_a, cell_b) in checked:
                    continue
                
                expected_b = planogram_skus[cell_b]
                
                # Check if it's a swap: A has B's SKU, B has A's SKU
                if detected_a == expected_b and detected_b == expected_a:
                    checked.add((cell_a, cell_b))
                    swap_anomalies.append(AnomalyRecord(
                        anomaly_type=AnomalyType.SWAPPED,
                        planogram_index=cell_a,
                        detection_index=cell_b,  # Using detection_index to store second cell
                        expected_sku=expected_a,
                        detected_sku=detected_a,
                        details=f"Cells {cell_a} and {cell_b} have swapped SKUs"
                    ))
        
        return swap_anomalies
    
    def _compute_compliance_score(
        self,
        total_expected: int,
        num_anomalies: int
    ) -> float:
        """
        Compute compliance score (0-1) based on anomaly count.
        
        Score = 1 - (anomalies / expected)
        
        A score of 1.0 means perfect compliance (no anomalies).
        A score of 0.0 means every position has an anomaly.
        
        Args:
            total_expected: Number of expected product positions
            num_anomalies: Number of detected anomalies
        
        Returns:
            Compliance score between 0 and 1
        """
        if total_expected == 0:
            return 1.0
        
        # Clamp to [0, 1] in case anomalies > expected (duplicates)
        score = max(0.0, 1.0 - (num_anomalies / total_expected))
        return score


def detect_anomalies(
    scenario: ShelfScenario,
    use_gpu: bool = True,
    method: str = "auto"
) -> ComplianceReport:
    """
    Convenience function for anomaly detection.
    
    Creates an AnomalyDetector and analyzes the scenario.
    
    Args:
        scenario: Shelf scenario to analyze
        use_gpu: Whether to use GPU acceleration
        method: Assignment method
    
    Returns:
        ComplianceReport with detected anomalies
    
    Example:
        >>> from src.synthetic_data import generate_shelf_scenario
        >>> scenario = generate_shelf_scenario(3, 12, anomaly_rate=0.15)
        >>> report = detect_anomalies(scenario)
        >>> print(report.summary())
    """
    detector = AnomalyDetector(use_gpu=use_gpu)
    return detector.analyze(scenario, method=method)


def compute_compliance_score(
    scenario: ShelfScenario,
    use_gpu: bool = True
) -> float:
    """
    Quick compliance score computation.
    
    Args:
        scenario: Shelf scenario to evaluate
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Compliance score between 0 and 1
    """
    report = detect_anomalies(scenario, use_gpu=use_gpu)
    return report.compliance_score


def generate_compliance_report(
    scenario: ShelfScenario,
    verbose: bool = True,
    use_gpu: bool = True
) -> str:
    """
    Generate a human-readable compliance report.
    
    Args:
        scenario: Shelf scenario to analyze
        verbose: Include detailed anomaly list
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Formatted string report
    """
    report = detect_anomalies(scenario, use_gpu=use_gpu)
    
    lines = [report.summary()]
    
    if verbose and report.anomalies:
        lines.append("\nDETAILED ANOMALIES:")
        lines.append("-" * 40)
        
        for i, anomaly in enumerate(report.anomalies[:20]):  # Limit to first 20
            lines.append(
                f"{i+1}. [{anomaly.anomaly_type.value.upper()}] "
                f"Cell {anomaly.planogram_index}: {anomaly.details}"
            )
        
        if len(report.anomalies) > 20:
            lines.append(f"... and {len(report.anomalies) - 20} more anomalies")
    
    return "\n".join(lines)


def validate_against_ground_truth(
    scenario: ShelfScenario,
    use_gpu: bool = True
) -> Dict[str, float]:
    """
    Validate detection results against ground truth labels.
    
    Only works for synthetic scenarios that have ground truth.
    
    Args:
        scenario: Scenario with ground_truth populated
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Dictionary with precision, recall, and F1 score
    """
    if scenario.ground_truth is None:
        raise ValueError("Scenario has no ground truth labels")
    
    report = detect_anomalies(scenario, use_gpu=use_gpu)
    
    # Extract detected anomaly positions
    detected_anomaly_positions = set()
    for anomaly in report.anomalies:
        if anomaly.planogram_index >= 0:
            detected_anomaly_positions.add(anomaly.planogram_index)
    
    # Ground truth anomaly positions
    ground_truth_positions = set(scenario.ground_truth.anomaly_indices)
    
    # Compute metrics
    true_positives = len(detected_anomaly_positions & ground_truth_positions)
    false_positives = len(detected_anomaly_positions - ground_truth_positions)
    false_negatives = len(ground_truth_positions - detected_anomaly_positions)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


if __name__ == "__main__":
    # Demo with synthetic data
    from ..synthetic_data import generate_shelf_scenario
    
    print("Anomaly Detection Demo")
    print("=" * 60)
    
    # Generate test scenario
    scenario = generate_shelf_scenario(
        num_rows=3,
        num_cols=12,
        anomaly_rate=0.15,
        seed=42
    )
    
    print(f"Planogram: {len(scenario.planogram.entries)} positions")
    print(f"Detections: {len(scenario.detections.entries)} detections")
    print()
    
    # Run detection
    report_text = generate_compliance_report(scenario, verbose=True, use_gpu=False)
    print(report_text)
    
    # Validate against ground truth
    if scenario.ground_truth:
        print("\nGround Truth Validation:")
        metrics = validate_against_ground_truth(scenario, use_gpu=False)
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  Recall: {metrics['recall']:.2%}")
        print(f"  F1 Score: {metrics['f1_score']:.2%}")