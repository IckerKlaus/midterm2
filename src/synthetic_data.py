"""
Synthetic Data Generator for Shelf Product Anomaly Detection

This module generates all planogram and detection data synthetically.
There is NO external dataset - all data is created programmatically.

The generator is inspired by a simple grid-based shelf simulation and has been
generalized to support various shelf configurations, noise models, and anomaly types.

Key Features:
- Configurable shelf layouts (rows, columns, spacing)
- Multiple SKU distribution patterns (unique, blocks, mixed)
- Realistic detection noise (Gaussian)
- Explicit anomaly injection (missing, swapped, foreign, duplicate)
- Reproducible results via random seed control
- JSON/CSV export for persistence

Example Usage:
    >>> from src.synthetic_data import generate_shelf_scenario
    >>> scenario = generate_shelf_scenario(
    ...     num_rows=3, num_cols=12,
    ...     anomaly_rate=0.15,
    ...     seed=42
    ... )
    >>> scenario.save_to_json("data/scenario.json")
"""

import json
import csv
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import numpy as np

from .data_models import (
    PlanogramEntry, DetectionEntry, ShelfPlanogram, DetectionSet,
    GroundTruth, ShelfScenario, AnomalyType
)


class SKUPattern(Enum):
    """Patterns for distributing SKUs across shelf positions."""
    UNIQUE = "unique"       # Each position has a unique SKU
    BLOCKS = "blocks"       # Same SKU in horizontal blocks
    ROWS = "rows"           # Same SKU per row
    RANDOM = "random"       # Random SKU assignment from a pool
    MIXED = "mixed"         # Combination of patterns


def generate_planogram(
    num_rows: int = 3,
    num_cols: int = 12,
    spacing_x: float = 10.0,
    spacing_y: float = 5.0,
    sku_pattern: SKUPattern = SKUPattern.UNIQUE,
    sku_pool_size: int = 20,
    block_width: int = 3,
    seed: Optional[int] = None
) -> ShelfPlanogram:
    """
    Generate a synthetic planogram (ideal product layout).
    
    This function creates a grid-based shelf layout with configurable
    spacing and SKU distribution patterns.
    
    Args:
        num_rows: Number of shelf rows (vertical levels)
        num_cols: Number of product positions per row
        spacing_x: Horizontal distance between product centers
        spacing_y: Vertical distance between rows
        sku_pattern: How SKUs are distributed across positions
        sku_pool_size: Number of unique SKUs for RANDOM pattern
        block_width: Width of blocks for BLOCKS pattern
        seed: Random seed for reproducibility
    
    Returns:
        ShelfPlanogram: Complete planogram with all entries
    
    Complexity:
        Time: O(num_rows × num_cols)
        Space: O(num_rows × num_cols)
    
    Example:
        >>> planogram = generate_planogram(3, 12, spacing_x=10, spacing_y=5)
        >>> print(f"Generated {len(planogram.entries)} positions")
        Generated 36 positions
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    entries: List[PlanogramEntry] = []
    sku_counter = 1
    
    # Pre-generate SKU pool for RANDOM pattern
    sku_pool = [f"SKU_{i:04d}" for i in range(1, sku_pool_size + 1)]
    
    for row in range(num_rows):
        for col in range(num_cols):
            # Calculate position
            # Rows go downward (negative y), columns go rightward (positive x)
            x = col * spacing_x
            y = -row * spacing_y
            
            # Determine SKU based on pattern
            if sku_pattern == SKUPattern.UNIQUE:
                sku_id = f"SKU_{sku_counter:04d}"
                sku_counter += 1
            elif sku_pattern == SKUPattern.BLOCKS:
                block_index = col // block_width
                sku_id = f"SKU_BLK{row:02d}_{block_index:02d}"
            elif sku_pattern == SKUPattern.ROWS:
                sku_id = f"SKU_ROW{row:02d}"
            elif sku_pattern == SKUPattern.RANDOM:
                sku_id = random.choice(sku_pool)
            elif sku_pattern == SKUPattern.MIXED:
                # Mix of unique and repeating
                if random.random() < 0.7:
                    sku_id = f"SKU_{sku_counter:04d}"
                    sku_counter += 1
                else:
                    sku_id = random.choice(sku_pool[:5])
            else:
                sku_id = f"SKU_{sku_counter:04d}"
                sku_counter += 1
            
            entry = PlanogramEntry(
                x=x,
                y=y,
                sku_id=sku_id,
                row=row,
                col=col,
                metadata={"category": f"CAT_{row % 3}"}
            )
            entries.append(entry)
    
    return ShelfPlanogram(
        entries=entries,
        num_rows=num_rows,
        num_cols=num_cols,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        metadata={"pattern": sku_pattern.value}
    )


def generate_detections(
    planogram: ShelfPlanogram,
    noise_std: float = 0.6,
    anomaly_rate: float = 0.15,
    anomaly_types: Optional[List[AnomalyType]] = None,
    missing_rate: float = 0.05,
    foreign_rate: float = 0.03,
    swap_rate: float = 0.02,
    duplicate_rate: float = 0.02,
    seed: Optional[int] = None
) -> Tuple[DetectionSet, GroundTruth]:
    """
    Generate synthetic detections with noise and anomalies.
    
    This function simulates what a vision system would detect on a real shelf,
    including positional noise and various types of placement anomalies.
    
    Args:
        planogram: The reference planogram to base detections on
        noise_std: Standard deviation of Gaussian noise for positions
        anomaly_rate: Overall probability of any anomaly (legacy param)
        anomaly_types: List of anomaly types to inject (if None, use all)
        missing_rate: Probability of missing product
        foreign_rate: Probability of foreign SKU
        swap_rate: Probability of swap (per pair)
        duplicate_rate: Probability of duplicate detection
        seed: Random seed for reproducibility
    
    Returns:
        Tuple[DetectionSet, GroundTruth]: Detections and ground truth labels
    
    Complexity:
        Time: O(n) where n = number of planogram entries
        Space: O(n) for detections + O(k) for ground truth labels
    
    Anomaly Types:
        - MISSING: Product expected but not detected
        - MISPLACED: Product detected far from expected position
        - FOREIGN: Unexpected SKU appears in detection
        - DUPLICATE: Same position has multiple detections
        - SWAPPED: Two products exchange positions
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if anomaly_types is None:
        anomaly_types = list(AnomalyType)
    
    entries: List[DetectionEntry] = []
    ground_truth = GroundTruth()
    
    # Track indices for swap pairs
    n_entries = len(planogram.entries)
    swap_indices: List[Tuple[int, int]] = []
    
    # First pass: determine swaps
    if AnomalyType.SWAPPED in anomaly_types:
        available_for_swap = list(range(n_entries))
        while len(available_for_swap) >= 2 and random.random() < swap_rate:
            idx1 = random.choice(available_for_swap)
            available_for_swap.remove(idx1)
            idx2 = random.choice(available_for_swap)
            available_for_swap.remove(idx2)
            swap_indices.append((idx1, idx2))
            ground_truth.swapped_pairs.append((idx1, idx2))
    
    # Build swap mapping
    swap_map = {}
    for idx1, idx2 in swap_indices:
        swap_map[idx1] = idx2
        swap_map[idx2] = idx1
    
    # Second pass: generate detections
    for i, pentry in enumerate(planogram.entries):
        # Check if this product is missing
        if AnomalyType.MISSING in anomaly_types and random.random() < missing_rate:
            ground_truth.missing_indices.append(i)
            ground_truth.anomaly_indices.append(i)
            ground_truth.anomaly_types.append(AnomalyType.MISSING)
            continue  # No detection for missing product
        
        # Base position from planogram
        x = pentry.x
        y = pentry.y
        sku_id = pentry.sku_id
        
        # Apply swap if this index is in a swap pair
        if i in swap_map:
            partner_idx = swap_map[i]
            partner = planogram.entries[partner_idx]
            sku_id = partner.sku_id  # Take partner's SKU
            # Note: position stays the same, SKU changes
        
        # Add Gaussian noise to position
        x_noisy = x + np.random.normal(0, noise_std)
        y_noisy = y + np.random.normal(0, noise_std)
        
        # Check for misplacement (larger displacement)
        if AnomalyType.MISPLACED in anomaly_types:
            if random.random() < anomaly_rate * 0.3:  # Subset of anomaly rate
                # Large displacement
                x_noisy += np.random.uniform(-10, 10)
                y_noisy += np.random.uniform(-5, 5)
                if i not in ground_truth.anomaly_indices:
                    ground_truth.anomaly_indices.append(i)
                    ground_truth.anomaly_types.append(AnomalyType.MISPLACED)
        
        # Check for foreign SKU
        if AnomalyType.FOREIGN in anomaly_types and random.random() < foreign_rate:
            # Replace with foreign SKU
            foreign_sku = f"FOREIGN_{random.randint(1000, 9999)}"
            sku_id = foreign_sku
            ground_truth.foreign_skus.append(foreign_sku)
            if i not in ground_truth.anomaly_indices:
                ground_truth.anomaly_indices.append(i)
                ground_truth.anomaly_types.append(AnomalyType.FOREIGN)
        
        # Create detection entry
        detection = DetectionEntry(
            x=x_noisy,
            y=y_noisy,
            sku_id=sku_id,
            confidence=random.uniform(0.85, 0.99)
        )
        entries.append(detection)
        
        # Check for duplicate (add extra detection)
        if AnomalyType.DUPLICATE in anomaly_types and random.random() < duplicate_rate:
            dup_x = x_noisy + np.random.normal(0, noise_std * 0.5)
            dup_y = y_noisy + np.random.normal(0, noise_std * 0.5)
            duplicate = DetectionEntry(
                x=dup_x,
                y=dup_y,
                sku_id=sku_id,
                confidence=random.uniform(0.7, 0.9)
            )
            entries.append(duplicate)
            # Mark as duplicate anomaly
            dup_idx = len(entries) - 1
            ground_truth.anomaly_indices.append(dup_idx)
            ground_truth.anomaly_types.append(AnomalyType.DUPLICATE)
    
    detection_set = DetectionSet(
        entries=entries,
        image_id=f"synthetic_{random.randint(1000, 9999)}",
        metadata={"noise_std": noise_std, "anomaly_rate": anomaly_rate}
    )
    
    return detection_set, ground_truth


def generate_shelf_scenario(
    num_rows: int = 3,
    num_cols: int = 12,
    spacing_x: float = 10.0,
    spacing_y: float = 5.0,
    noise_std: float = 0.6,
    anomaly_rate: float = 0.15,
    sku_pattern: SKUPattern = SKUPattern.UNIQUE,
    seed: Optional[int] = None,
    scenario_id: Optional[str] = None
) -> ShelfScenario:
    """
    Generate a complete shelf scenario (planogram + detections + ground truth).
    
    This is the main entry point for synthetic data generation. It creates
    a complete scenario ready for anomaly detection testing.
    
    Args:
        num_rows: Number of shelf rows
        num_cols: Number of products per row
        spacing_x: Horizontal spacing
        spacing_y: Vertical spacing
        noise_std: Position noise standard deviation
        anomaly_rate: Overall anomaly probability
        sku_pattern: SKU distribution pattern
        seed: Random seed for reproducibility
        scenario_id: Optional unique identifier
    
    Returns:
        ShelfScenario: Complete scenario with all data
    
    Example:
        >>> scenario = generate_shelf_scenario(3, 12, anomaly_rate=0.15, seed=42)
        >>> print(f"Planogram: {len(scenario.planogram.entries)} products")
        >>> print(f"Detections: {len(scenario.detections.entries)} detections")
        >>> print(f"Anomalies: {len(scenario.ground_truth.anomaly_indices)}")
    """
    if scenario_id is None:
        scenario_id = f"scenario_{num_rows}x{num_cols}_{random.randint(1000, 9999)}"
    
    # Generate planogram
    planogram = generate_planogram(
        num_rows=num_rows,
        num_cols=num_cols,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        sku_pattern=sku_pattern,
        seed=seed
    )
    
    # Generate detections with anomalies
    # Use different seed for detections to maintain independence
    detection_seed = seed + 1000 if seed is not None else None
    detections, ground_truth = generate_detections(
        planogram=planogram,
        noise_std=noise_std,
        anomaly_rate=anomaly_rate,
        seed=detection_seed
    )
    
    return ShelfScenario(
        planogram=planogram,
        detections=detections,
        ground_truth=ground_truth,
        scenario_id=scenario_id
    )


def generate_multiple_scenarios(
    num_scenarios: int = 10,
    base_seed: int = 42,
    **kwargs
) -> List[ShelfScenario]:
    """
    Generate multiple scenarios for batch testing.
    
    Args:
        num_scenarios: Number of scenarios to generate
        base_seed: Starting seed for reproducibility
        **kwargs: Arguments passed to generate_shelf_scenario
    
    Returns:
        List[ShelfScenario]: List of generated scenarios
    
    Complexity:
        Time: O(num_scenarios × num_rows × num_cols)
        Space: O(num_scenarios × num_rows × num_cols)
    """
    scenarios = []
    for i in range(num_scenarios):
        seed = base_seed + i * 100
        scenario_id = f"batch_{i:04d}"
        scenario = generate_shelf_scenario(
            seed=seed,
            scenario_id=scenario_id,
            **kwargs
        )
        scenarios.append(scenario)
    return scenarios


def save_scenario_to_json(
    scenario: ShelfScenario,
    output_dir: str = "data"
) -> Tuple[str, str]:
    """
    Save scenario components to JSON files.
    
    Args:
        scenario: Scenario to save
        output_dir: Directory for output files
    
    Returns:
        Tuple[str, str]: Paths to planogram and detection files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    scenario_id = scenario.scenario_id or "default"
    
    # Save complete scenario
    scenario_file = output_path / f"scenario_{scenario_id}.json"
    scenario.save_to_json(str(scenario_file))
    
    # Also save separate files for convenience
    planogram_file = output_path / f"planogram_{scenario_id}.json"
    with open(planogram_file, 'w') as f:
        json.dump(scenario.planogram.to_dict(), f, indent=2)
    
    detections_file = output_path / f"detections_{scenario_id}.json"
    with open(detections_file, 'w') as f:
        json.dump(scenario.detections.to_dict(), f, indent=2)
    
    return str(planogram_file), str(detections_file)


def load_scenario_from_json(filepath: str) -> ShelfScenario:
    """
    Load a scenario from JSON file.
    
    Args:
        filepath: Path to scenario JSON file
    
    Returns:
        ShelfScenario: Loaded scenario
    """
    return ShelfScenario.load_from_json(filepath)


def save_planogram_to_csv(planogram: ShelfPlanogram, filepath: str) -> None:
    """
    Export planogram to CSV format for external tools.
    
    Args:
        planogram: Planogram to export
        filepath: Output CSV path
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'sku_id', 'row', 'col'])
        for entry in planogram.entries:
            writer.writerow([entry.x, entry.y, entry.sku_id, entry.row, entry.col])


def save_detections_to_csv(detections: DetectionSet, filepath: str) -> None:
    """
    Export detections to CSV format for external tools.
    
    Args:
        detections: Detections to export
        filepath: Output CSV path
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'sku_id', 'confidence'])
        for entry in detections.entries:
            writer.writerow([entry.x, entry.y, entry.sku_id, entry.confidence])


def visualize_scenario(
    scenario: ShelfScenario,
    show_connections: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    Create a 2D visualization of planogram vs detections.
    
    This function creates a matplotlib plot showing:
    - Blue circles: Expected planogram positions
    - Red circles: Actual detected positions
    - Gray lines: Connections between expected and detected
    
    Args:
        scenario: Scenario to visualize
        show_connections: Whether to draw lines between pairs
        save_path: If provided, save figure to this path
    
    Note:
        Requires matplotlib. Import error is caught gracefully.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return
    
    planogram_points = scenario.planogram.get_points_array()
    detection_points = scenario.detections.get_points_array()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot planogram positions (blue)
    ax.scatter(
        planogram_points[:, 0],
        planogram_points[:, 1],
        c='blue',
        s=100,
        label='Planogram (expected)',
        alpha=0.7,
        marker='s'
    )
    
    # Plot detected positions (red)
    ax.scatter(
        detection_points[:, 0],
        detection_points[:, 1],
        c='red',
        s=60,
        label='Detections (actual)',
        alpha=0.7,
        marker='o'
    )
    
    # Draw connections if enabled
    if show_connections and len(planogram_points) == len(detection_points):
        for i in range(len(planogram_points)):
            ax.plot(
                [planogram_points[i, 0], detection_points[i, 0]],
                [planogram_points[i, 1], detection_points[i, 1]],
                'gray',
                alpha=0.3,
                linewidth=0.5
            )
    
    # Mark anomaly positions
    if scenario.ground_truth:
        for idx in scenario.ground_truth.missing_indices:
            if idx < len(planogram_points):
                ax.scatter(
                    planogram_points[idx, 0],
                    planogram_points[idx, 1],
                    c='orange',
                    s=200,
                    marker='x',
                    linewidths=3,
                    label='Missing' if idx == scenario.ground_truth.missing_indices[0] else ''
                )
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Shelf Scenario: {scenario.scenario_id}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_benchmark_scenarios(
    sizes: List[int] = [100, 500, 1000, 5000, 10000],
    seed: int = 42
) -> Dict[int, ShelfScenario]:
    """
    Generate scenarios of various sizes for benchmarking.
    
    Args:
        sizes: List of total product counts to generate
        seed: Base random seed
    
    Returns:
        Dict mapping size to ShelfScenario
    
    Example:
        >>> scenarios = generate_benchmark_scenarios([100, 1000, 10000])
        >>> for size, scenario in scenarios.items():
        ...     print(f"Size {size}: {len(scenario.planogram.entries)} products")
    """
    scenarios = {}
    
    for size in sizes:
        # Calculate reasonable grid dimensions
        cols = int(np.sqrt(size * 4))  # Wider shelves
        rows = max(1, size // cols)
        
        # Adjust to get exact size (approximately)
        while rows * cols < size:
            cols += 1
        
        scenario = generate_shelf_scenario(
            num_rows=rows,
            num_cols=cols,
            spacing_x=10.0,
            spacing_y=5.0,
            noise_std=0.6,
            anomaly_rate=0.15,
            seed=seed + size,
            scenario_id=f"benchmark_{size}"
        )
        scenarios[size] = scenario
    
    return scenarios


# Generate sample data files on module import for convenience
def create_sample_data_files(output_dir: str = "data") -> None:
    """
    Create sample data files for demos and testing.
    
    This function generates example planogram and detection files
    that can be used immediately for testing the system.
    
    Args:
        output_dir: Directory to save sample files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Small example scenario
    small_scenario = generate_shelf_scenario(
        num_rows=3,
        num_cols=12,
        anomaly_rate=0.15,
        seed=42,
        scenario_id="example_small"
    )
    save_scenario_to_json(small_scenario, output_dir)
    
    # Medium example scenario
    medium_scenario = generate_shelf_scenario(
        num_rows=5,
        num_cols=20,
        anomaly_rate=0.10,
        seed=123,
        scenario_id="example_medium"
    )
    save_scenario_to_json(medium_scenario, output_dir)
    
    print(f"Sample data files created in {output_dir}/")


if __name__ == "__main__":
    # When run directly, create sample data files
    create_sample_data_files()
    print("Synthetic data generation complete!")
