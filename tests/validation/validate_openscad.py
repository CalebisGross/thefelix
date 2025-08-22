#!/usr/bin/env python3
"""
Validation script to compare Python implementation against OpenSCAD model.

This script recreates key calculations from thefelix.md to ensure 
our Python implementation matches the geometric prototype.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import math
from src.core.helix_geometry import HelixGeometry


def openscad_get_position(step, turns, segments_per_turn, height, top_radius, bottom_radius):
    """
    Recreate the OpenSCAD get_position function for comparison.
    
    From thefelix.md:
    function get_position(step, p_turns, p_segs, p_h, p_t_rad, p_b_rad) = let(
        total_steps = p_turns * p_segs,
        angle = step / total_steps * p_turns * 360,
        z = step / total_steps * p_h,
        r = p_b_rad * pow(p_t_rad / p_b_rad, z / p_h),
        x = r * cos(angle),
        y = r * sin(angle)
    ) [x, y, z];
    """
    total_steps = turns * segments_per_turn
    angle = step / total_steps * turns * 360  # degrees
    z = step / total_steps * height
    r = bottom_radius * pow(top_radius / bottom_radius, z / height)
    x = r * math.cos(math.radians(angle))
    y = r * math.sin(math.radians(angle))
    return (x, y, z)


def validate_implementation():
    """Compare Python implementation against OpenSCAD calculations."""
    
    # Parameters from thefelix.md
    top_radius = 33.0
    bottom_radius = 0.001
    height = 33.0
    turns = 33
    segments_per_turn = 33
    
    # Create Python helix
    helix = HelixGeometry(top_radius, bottom_radius, height, turns)
    
    print("=== OpenSCAD vs Python Implementation Validation ===\n")
    
    # Test specific positions
    total_steps = turns * segments_per_turn
    test_steps = [0, total_steps // 4, total_steps // 2, 3 * total_steps // 4, total_steps - 1]
    
    print("Comparing positions at key points:")
    print("Step | t    | OpenSCAD (x, y, z) | Python (x, y, z) | Max Diff | Debug")
    print("-" * 90)
    
    max_overall_diff = 0.0
    
    for step in test_steps:
        # OpenSCAD calculation
        openscad_pos = openscad_get_position(
            step, turns, segments_per_turn, height, top_radius, bottom_radius
        )
        
        # Python calculation - match the exact OpenSCAD conversion
        # OpenSCAD: t = step / total_steps, but we need step / (total_steps - 1) for discrete steps
        t_openscad = step / total_steps  # This is what OpenSCAD actually uses
        python_pos = helix.get_position(t_openscad)
        
        # Calculate differences
        diff_x = abs(openscad_pos[0] - python_pos[0])
        diff_y = abs(openscad_pos[1] - python_pos[1])
        diff_z = abs(openscad_pos[2] - python_pos[2])
        max_diff = max(diff_x, diff_y, diff_z)
        max_overall_diff = max(max_overall_diff, max_diff)
        
        # Debug info
        angle_openscad = step / total_steps * turns * 360
        angle_python = t_openscad * turns * 360
        
        print(f"{step:4d} | {t_openscad:.3f} | ({openscad_pos[0]:8.3f}, {openscad_pos[1]:8.3f}, {openscad_pos[2]:8.3f}) | "
              f"({python_pos[0]:8.3f}, {python_pos[1]:8.3f}, {python_pos[2]:8.3f}) | {max_diff:.2e} | "
              f"ang={angle_openscad:.1f}")
    
    print(f"\nMaximum overall difference: {max_overall_diff:.2e}")
    
    # Validation threshold - allow for floating point precision
    if max_overall_diff < 1e-6:
        print("✅ VALIDATION PASSED: Python implementation matches OpenSCAD model")
        return True
    else:
        print("❌ VALIDATION FAILED: Significant differences detected")
        return False


def performance_baseline():
    """Establish performance baseline for helix calculations."""
    import time
    
    helix = HelixGeometry(33.0, 0.001, 33.0, 33)
    
    print("\n=== Performance Baseline ===")
    
    # Time single position calculation
    start_time = time.perf_counter()
    for i in range(10000):
        t = i / 9999.0
        pos = helix.get_position(t)
    end_time = time.perf_counter()
    
    time_per_calc = (end_time - start_time) / 10000
    print(f"Position calculation: {time_per_calc:.2e} seconds per call")
    print(f"Throughput: {1.0/time_per_calc:.0f} calculations per second")
    
    # Time arc length approximation
    start_time = time.perf_counter()
    arc_length = helix.approximate_arc_length(segments=1000)
    end_time = time.perf_counter()
    
    print(f"Arc length calculation (1000 segments): {end_time - start_time:.3f} seconds")
    print(f"Total helix arc length: {arc_length:.1f} units")


if __name__ == "__main__":
    # Run validation
    validation_passed = validate_implementation()
    
    # Run performance baseline
    performance_baseline()
    
    # Summary
    print(f"\n=== Validation Summary ===")
    if validation_passed:
        print("Core helix mathematics implementation is VALIDATED ✅")
        print("Ready to proceed with agent system implementation.")
    else:
        print("Validation FAILED - implementation needs correction ❌")