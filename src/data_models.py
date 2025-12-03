"""
Data Models for the Shelf Product Anomaly Detector

This module defines the core data structures used throughout the system.
Uses Python dataclasses for clean, type-hinted data containers.

Data Flow:
    PlanogramEntry → ShelfPlanogram → ShelfScenario
    DetectionEntry →                → ShelfScenario
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import json
import numpy as np


class AnomalyType(Enum):
    """Types of anomalies that can occur in shelf product placement."""
    MISSING = "missing"           # Expected product not detected
    MISPLACED = "misplaced"       # Product detected in wrong location
    FOREIGN = "foreign"           # Unexpected product detected
    DUPLICATE = "duplicate"       # Same product detected multiple times in one cell
    SWAPPED = "swapped"           # Two products exchanged positions


@dataclass
class PlanogramEntry:
    """
    Represents a single expected product position in the planogram.
    
    Attributes:
        x: Horizontal position (in shelf coordinate units)
        y: Vertical position (in shelf coordinate units)
        sku_id: Unique identifier for the product (e.g., "SKU_001")
        row: Optional row index in the shelf grid
        col: Optional column index in the shelf grid
        metadata: Additional product information (name, category, etc.)
    
    Complexity: O(1) space per entry
    """
    x: float
    y: float
    sku_id: str
    row: Optional[int] = None
    col: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "x": self.x,
            "y": self.y,
            "sku_id": self.sku_id,
            "row": self.row,
            "col": self.col,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanogramEntry":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            sku_id=str(data["sku_id"]),
            row=data.get("row"),
            col=data.get("col"),
            metadata=data.get("metadata")
        )
    
    def as_point(self) -> np.ndarray:
        """Return coordinates as numpy array for geometric operations."""
        return np.array([self.x, self.y], dtype=np.float64)


@dataclass
class DetectionEntry:
    """
    Represents a detected product position from the vision system.
    
    Attributes:
        x: Detected horizontal position
        y: Detected vertical position
        sku_id: Identified product SKU (from vision/classifier)
        confidence: Optional detection confidence score (0-1)
        bbox: Optional bounding box [x_min, y_min, x_max, y_max]
    
    Complexity: O(1) space per entry
    """
    x: float
    y: float
    sku_id: str
    confidence: Optional[float] = None
    bbox: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "x": self.x,
            "y": self.y,
            "sku_id": self.sku_id,
            "confidence": self.confidence,
            "bbox": self.bbox
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionEntry":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            sku_id=str(data["sku_id"]),
            confidence=data.get("confidence"),
            bbox=data.get("bbox")
        )
    
    def as_point(self) -> np.ndarray:
        """Return coordinates as numpy array for geometric operations."""
        return np.array([self.x, self.y], dtype=np.float64)


@dataclass
class AnomalyRecord:
    """
    Records a single detected anomaly.
    
    Attributes:
        anomaly_type: Type of anomaly (from AnomalyType enum)
        planogram_index: Index of affected planogram entry (-1 if N/A)
        detection_index: Index of involved detection (-1 if N/A)
        expected_sku: SKU that was expected
        detected_sku: SKU that was actually detected
        position: Position where anomaly occurred
        details: Additional context about the anomaly
    """
    anomaly_type: AnomalyType
    planogram_index: int = -1
    detection_index: int = -1
    expected_sku: Optional[str] = None
    detected_sku: Optional[str] = None
    position: Optional[tuple] = None
    details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "anomaly_type": self.anomaly_type.value,
            "planogram_index": self.planogram_index,
            "detection_index": self.detection_index,
            "expected_sku": self.expected_sku,
            "detected_sku": self.detected_sku,
            "position": self.position,
            "details": self.details
        }


@dataclass
class ShelfPlanogram:
    """
    Container for a complete shelf planogram.
    
    Attributes:
        entries: List of all product positions in the planogram
        num_rows: Number of rows in the shelf
        num_cols: Number of columns in the shelf
        spacing_x: Horizontal spacing between products
        spacing_y: Vertical spacing between rows
        metadata: Additional shelf information
    
    Space Complexity: O(n) where n = num_rows × num_cols
    """
    entries: List[PlanogramEntry]
    num_rows: int
    num_cols: int
    spacing_x: float
    spacing_y: float
    metadata: Optional[Dict[str, Any]] = None
    
    def get_points_array(self) -> np.ndarray:
        """
        Extract all positions as a NumPy array for vectorized operations.
        
        Returns:
            np.ndarray: Shape (n, 2) array of (x, y) coordinates
        
        Complexity: O(n) time, O(n) space
        """
        return np.array([[e.x, e.y] for e in self.entries], dtype=np.float64)
    
    def get_sku_list(self) -> List[str]:
        """Get list of SKU IDs in order."""
        return [e.sku_id for e in self.entries]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "entries": [e.to_dict() for e in self.entries],
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "spacing_x": self.spacing_x,
            "spacing_y": self.spacing_y,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShelfPlanogram":
        """Create from dictionary (JSON deserialization)."""
        entries = [PlanogramEntry.from_dict(e) for e in data["entries"]]
        return cls(
            entries=entries,
            num_rows=data["num_rows"],
            num_cols=data["num_cols"],
            spacing_x=data["spacing_x"],
            spacing_y=data["spacing_y"],
            metadata=data.get("metadata")
        )


@dataclass
class DetectionSet:
    """
    Container for all detections from a shelf image.
    
    Attributes:
        entries: List of all detected products
        image_id: Optional identifier for source image
        timestamp: Optional capture timestamp
        metadata: Additional detection context
    """
    entries: List[DetectionEntry]
    image_id: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def get_points_array(self) -> np.ndarray:
        """
        Extract all positions as a NumPy array for vectorized operations.
        
        Returns:
            np.ndarray: Shape (m, 2) array of (x, y) coordinates
        
        Complexity: O(m) time, O(m) space
        """
        return np.array([[e.x, e.y] for e in self.entries], dtype=np.float64)
    
    def get_sku_list(self) -> List[str]:
        """Get list of detected SKU IDs in order."""
        return [e.sku_id for e in self.entries]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "entries": [e.to_dict() for e in self.entries],
            "image_id": self.image_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionSet":
        """Create from dictionary (JSON deserialization)."""
        entries = [DetectionEntry.from_dict(e) for e in data["entries"]]
        return cls(
            entries=entries,
            image_id=data.get("image_id"),
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata")
        )


@dataclass
class GroundTruth:
    """
    Ground truth labels for a synthetic scenario (for testing/validation).
    
    Attributes:
        anomaly_indices: Indices of detections that are anomalies
        anomaly_types: Type of each anomaly
        missing_indices: Planogram indices with no detection
        swapped_pairs: List of (idx1, idx2) pairs that were swapped
    """
    anomaly_indices: List[int] = field(default_factory=list)
    anomaly_types: List[AnomalyType] = field(default_factory=list)
    missing_indices: List[int] = field(default_factory=list)
    swapped_pairs: List[tuple] = field(default_factory=list)
    foreign_skus: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "anomaly_indices": self.anomaly_indices,
            "anomaly_types": [t.value for t in self.anomaly_types],
            "missing_indices": self.missing_indices,
            "swapped_pairs": self.swapped_pairs,
            "foreign_skus": self.foreign_skus
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundTruth":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            anomaly_indices=data.get("anomaly_indices", []),
            anomaly_types=[AnomalyType(t) for t in data.get("anomaly_types", [])],
            missing_indices=data.get("missing_indices", []),
            swapped_pairs=[tuple(p) for p in data.get("swapped_pairs", [])],
            foreign_skus=data.get("foreign_skus", [])
        )


@dataclass
class ShelfScenario:
    """
    Complete scenario containing planogram, detections, and ground truth.
    
    This is the main data container passed through the pipeline.
    
    Attributes:
        planogram: Expected product layout
        detections: Observed product positions
        ground_truth: Optional labels for synthetic data validation
        scenario_id: Unique identifier for this scenario
    
    Space Complexity: O(n + m) where n = planogram size, m = detections count
    """
    planogram: ShelfPlanogram
    detections: DetectionSet
    ground_truth: Optional[GroundTruth] = None
    scenario_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "planogram": self.planogram.to_dict(),
            "detections": self.detections.to_dict(),
            "ground_truth": self.ground_truth.to_dict() if self.ground_truth else None,
            "scenario_id": self.scenario_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShelfScenario":
        """Create from dictionary (JSON deserialization)."""
        planogram = ShelfPlanogram.from_dict(data["planogram"])
        detections = DetectionSet.from_dict(data["detections"])
        ground_truth = None
        if data.get("ground_truth"):
            ground_truth = GroundTruth.from_dict(data["ground_truth"])
        return cls(
            planogram=planogram,
            detections=detections,
            ground_truth=ground_truth,
            scenario_id=data.get("scenario_id")
        )
    
    def save_to_json(self, filepath: str) -> None:
        """Save scenario to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> "ShelfScenario":
        """Load scenario from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class ComplianceReport:
    """
    Final compliance report after anomaly detection.
    
    Attributes:
        anomalies: List of detected anomalies
        total_expected: Number of expected products
        total_detected: Number of detected products
        compliance_score: Ratio of compliant positions (0-1)
        processing_time_ms: Time taken for analysis
    """
    anomalies: List[AnomalyRecord]
    total_expected: int
    total_detected: int
    compliance_score: float
    processing_time_ms: float = 0.0
    
    @property
    def num_anomalies(self) -> int:
        """Total number of anomalies detected."""
        return len(self.anomalies)
    
    @property
    def anomaly_rate(self) -> float:
        """Proportion of positions with anomalies."""
        if self.total_expected == 0:
            return 0.0
        return self.num_anomalies / self.total_expected
    
    def get_anomalies_by_type(self) -> Dict[AnomalyType, List[AnomalyRecord]]:
        """Group anomalies by their type."""
        by_type: Dict[AnomalyType, List[AnomalyRecord]] = {}
        for anomaly in self.anomalies:
            if anomaly.anomaly_type not in by_type:
                by_type[anomaly.anomaly_type] = []
            by_type[anomaly.anomaly_type].append(anomaly)
        return by_type
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "PLANOGRAM COMPLIANCE REPORT",
            "=" * 60,
            f"Expected Products:    {self.total_expected}",
            f"Detected Products:    {self.total_detected}",
            f"Total Anomalies:      {self.num_anomalies}",
            f"Compliance Score:     {self.compliance_score:.2%}",
            f"Processing Time:      {self.processing_time_ms:.2f} ms",
            "-" * 60,
            "ANOMALIES BY TYPE:",
        ]
        
        by_type = self.get_anomalies_by_type()
        for anomaly_type in AnomalyType:
            count = len(by_type.get(anomaly_type, []))
            lines.append(f"  {anomaly_type.value.upper():12}: {count}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "anomalies": [a.to_dict() for a in self.anomalies],
            "total_expected": self.total_expected,
            "total_detected": self.total_detected,
            "compliance_score": self.compliance_score,
            "processing_time_ms": self.processing_time_ms,
            "num_anomalies": self.num_anomalies,
            "anomaly_rate": self.anomaly_rate
        }
