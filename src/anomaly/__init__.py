"""
Anomaly Detection Module for Shelf Product Analysis

This module provides the core logic for identifying planogram compliance
anomalies based on Voronoi cell assignments.

Anomaly Types:
- MISSING: Expected product not detected
- MISPLACED: Product detected at wrong location
- FOREIGN: Unexpected product appears
- DUPLICATE: Same product appears multiple times in one cell
- SWAPPED: Two products exchanged positions
"""

from .anomaly_detector import (
    AnomalyDetector,
    detect_anomalies,
    compute_compliance_score,
    generate_compliance_report
)

__all__ = [
    'AnomalyDetector',
    'detect_anomalies',
    'compute_compliance_score',
    'generate_compliance_report'
]
