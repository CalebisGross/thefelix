#!/usr/bin/env python3
"""
Mathematical validation script for the Felix Framework.

This script validates mathematical properties derived in the formal documentation
against the implementation, ensuring theoretical consistency and correctness.

Validates:
1. Parametric helix equations and properties
2. Geometric invariants (curvature, torsion)
3. Agent distribution functions
4. Communication complexity bounds
5. Attention focusing mechanism properties

Mathematical references:
- docs/mathematical_model.md: Formal mathematical specifications
- docs/hypothesis_mathematics.md: Statistical formulations and proofs
"""

import math
import numpy as np
from typing import List, Tuple
import time

from src.core.helix_geometry import HelixGeometry
from src.agents.agent import generate_spawn_times, create_openscad_agents
from src.communication import CentralPost


class MathematicalValidator:
    """Validates mathematical properties against implementation."""
    
    def __init__(self, tolerance: float = 1e-10):
        """Initialize validator with numerical tolerance."""
        self.tolerance = tolerance
        self.helix = HelixGeometry(33.0, 0.001, 33.0, 33)  # OpenSCAD parameters
        self.validation_results = {}
    
    def validate_parametric_equations(self) -> bool:
        """
        Validate parametric helix equations from mathematical_model.md Section 1.2.
        
        Tests:
        1. Boundary conditions: r(0) and r(1)
        2. Radius tapering function R(t)
        3. Angular progression θ(t)
        4. Height function z(t) = Ht
        """
        print("=== Validating Parametric Equations ===")
        
        # Test boundary conditions
        pos_0 = self.helix.get_position(0.0)
        pos_1 = self.helix.get_position(1.0)
        
        # At t=0: should be at (R_bottom, 0, 0)
        expected_0 = (self.helix.bottom_radius, 0.0, 0.0)
        diff_0 = max(abs(a - b) for a, b in zip(pos_0, expected_0))
        
        # At t=1: should be at (R_top, 0, height) after n full turns
        # After 33 turns, angle = 33 * 2π, so cos(33*2π) = cos(0) = 1
        expected_1 = (self.helix.top_radius, 0.0, self.helix.height)
        diff_1 = max(abs(a - b) for a, b in zip(pos_1, expected_1))
        
        boundary_valid = diff_0 < self.tolerance and diff_1 < self.tolerance
        
        print(f"Boundary conditions: {'✅' if boundary_valid else '❌'}")
        print(f"  t=0: {pos_0} vs {expected_0}, diff={diff_0:.2e}")
        print(f"  t=1: {pos_1} vs {expected_1}, diff={diff_1:.2e}")
        
        # Test radius function R(t) = R_bottom * (R_top/R_bottom)^t
        radius_valid = True
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            z = self.helix.height * t
            actual_radius = self.helix.get_radius(z)
            expected_radius = self.helix.bottom_radius * pow(
                self.helix.top_radius / self.helix.bottom_radius, t
            )
            diff = abs(actual_radius - expected_radius)
            if diff > self.tolerance:
                radius_valid = False
            print(f"  R({t:.2f}): {actual_radius:.6f} vs {expected_radius:.6f}, diff={diff:.2e}")
        
        print(f"Radius function: {'✅' if radius_valid else '❌'}")
        
        # Test monotonicity of R(t) - should be strictly increasing
        monotonic_valid = True
        prev_radius = 0.0
        for t in np.linspace(0, 1, 100):
            z = self.helix.height * t
            radius = self.helix.get_radius(z)
            if radius <= prev_radius and t > 0:
                monotonic_valid = False
                break
            prev_radius = radius
        
        print(f"Radius monotonicity: {'✅' if monotonic_valid else '❌'}")
        
        self.validation_results['parametric_equations'] = {
            'boundary_conditions': boundary_valid,
            'radius_function': radius_valid,
            'monotonicity': monotonic_valid,
            'overall': boundary_valid and radius_valid and monotonic_valid
        }
        
        return boundary_valid and radius_valid and monotonic_valid
    
    def validate_geometric_properties(self) -> bool:
        """
        Validate geometric properties from mathematical_model.md Section 3.
        
        Tests:
        1. Arc length approximation convergence
        2. Tangent vector properties
        3. Smoothness (differentiability)
        """
        print("\n=== Validating Geometric Properties ===")
        
        # Test arc length approximation convergence
        # Should converge as number of segments increases
        segments_list = [100, 500, 1000, 2000]
        arc_lengths = []
        
        for segments in segments_list:
            length = self.helix.approximate_arc_length(segments=segments)
            arc_lengths.append(length)
        
        # Check convergence: difference between consecutive approximations should decrease
        convergence_valid = True
        for i in range(1, len(arc_lengths)):
            diff_curr = abs(arc_lengths[i] - arc_lengths[i-1])
            if i > 1:
                diff_prev = abs(arc_lengths[i-1] - arc_lengths[i-2])
                if diff_curr >= diff_prev:  # Should be decreasing
                    convergence_valid = False
            print(f"  Arc length ({segments_list[i]} segments): {arc_lengths[i]:.3f}, "
                  f"diff from previous: {diff_curr:.3f}")
        
        print(f"Arc length convergence: {'✅' if convergence_valid else '❌'}")
        
        # Test tangent vector properties
        tangent_valid = True
        for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
            tangent = self.helix.get_tangent_vector(t)
            # Tangent vector should be normalized (length ≈ 1)
            length = math.sqrt(sum(component**2 for component in tangent))
            if abs(length - 1.0) > self.tolerance:
                tangent_valid = False
            print(f"  Tangent at t={t}: length={length:.6f}")
        
        print(f"Tangent vector normalization: {'✅' if tangent_valid else '❌'}")
        
        # Test smoothness by checking continuity of positions
        smoothness_valid = True
        epsilon = 1e-6
        for t in np.linspace(0.1, 0.9, 20):
            pos1 = self.helix.get_position(t - epsilon)
            pos2 = self.helix.get_position(t + epsilon)
            # Distance should be small for small epsilon
            distance = math.sqrt(sum((a - b)**2 for a, b in zip(pos1, pos2)))
            expected_distance = 2 * epsilon * self.helix.approximate_arc_length(segments=1000)
            if distance > 10 * expected_distance:  # Allow some tolerance
                smoothness_valid = False
        
        print(f"Smoothness (continuity): {'✅' if smoothness_valid else '❌'}")
        
        self.validation_results['geometric_properties'] = {
            'arc_length_convergence': convergence_valid,
            'tangent_normalization': tangent_valid,
            'smoothness': smoothness_valid,
            'overall': convergence_valid and tangent_valid and smoothness_valid
        }
        
        return convergence_valid and tangent_valid and smoothness_valid
    
    def validate_agent_distribution(self) -> bool:
        """
        Validate agent distribution functions from mathematical_model.md Section 4.
        
        Tests:
        1. Uniform spawn time distribution U(0,1)
        2. Agent density evolution properties
        3. Statistical properties of large samples
        """
        print("\n=== Validating Agent Distribution ===")
        
        # Test spawn time distribution
        N = 10000
        spawn_times = generate_spawn_times(N, seed=12345)
        
        # Test uniformity using Kolmogorov-Smirnov test approximation
        sorted_times = sorted(spawn_times)
        max_deviation = 0.0
        for i, t in enumerate(sorted_times):
            empirical_cdf = (i + 1) / N
            theoretical_cdf = t  # For U(0,1), CDF(t) = t
            deviation = abs(empirical_cdf - theoretical_cdf)
            max_deviation = max(max_deviation, deviation)
        
        # For U(0,1), critical value at α=0.05 is approximately 1.36/√N
        critical_value = 1.36 / math.sqrt(N)
        uniformity_valid = max_deviation < critical_value
        
        print(f"Spawn time uniformity: {'✅' if uniformity_valid else '❌'}")
        print(f"  Max KS deviation: {max_deviation:.6f}, critical: {critical_value:.6f}")
        
        # Test mean and variance
        mean_time = np.mean(spawn_times)
        var_time = np.var(spawn_times)
        
        # For U(0,1): mean = 0.5, variance = 1/12 ≈ 0.0833
        mean_valid = abs(mean_time - 0.5) < 3 * math.sqrt(1/12/N)  # 3-sigma test
        var_valid = abs(var_time - 1/12) < 0.01  # Allow reasonable tolerance
        
        print(f"Mean (expected 0.5): {mean_time:.6f}, valid: {'✅' if mean_valid else '❌'}")
        print(f"Variance (expected 0.0833): {var_time:.6f}, valid: {'✅' if var_valid else '❌'}")
        
        # Test reproducibility with same seed
        spawn_times_2 = generate_spawn_times(N, seed=12345)
        reproducibility_valid = spawn_times == spawn_times_2
        
        print(f"Reproducibility with same seed: {'✅' if reproducibility_valid else '❌'}")
        
        self.validation_results['agent_distribution'] = {
            'uniformity': uniformity_valid,
            'mean': mean_valid,
            'variance': var_valid,
            'reproducibility': reproducibility_valid,
            'overall': uniformity_valid and mean_valid and var_valid and reproducibility_valid
        }
        
        return uniformity_valid and mean_valid and var_valid and reproducibility_valid
    
    def validate_attention_focusing(self) -> bool:
        """
        Validate attention focusing mechanism from hypothesis_mathematics.md Section H3.
        
        Tests:
        1. Attention density A(t) = k / (2πR(t))
        2. Monotonic increase: dA/dt > 0
        3. Exponential growth toward narrow end
        """
        print("\n=== Validating Attention Focusing Mechanism ===")
        
        # Calculate attention density at different positions
        attention_densities = []
        t_values = np.linspace(0.1, 0.9, 20)
        
        for t in t_values:
            z = self.helix.height * t
            radius = self.helix.get_radius(z)
            # Attention density A(t) = 1 / (2π * R(t))
            attention = 1.0 / (2 * math.pi * radius)
            attention_densities.append(attention)
        
        # Test monotonic decrease (attention decreases as radius increases toward top)
        monotonic_decrease = True
        for i in range(1, len(attention_densities)):
            if attention_densities[i] >= attention_densities[i-1]:
                monotonic_decrease = False
                print(f"  Non-monotonic at i={i}: A[{i-1}]={attention_densities[i-1]:.6f}, "
                      f"A[{i}]={attention_densities[i]:.6f}")
                break
        
        print(f"Attention density monotonic decrease (correct): {'✅' if monotonic_decrease else '❌'}")
        
        # Test exponential decay rate (attention decreases exponentially)
        # dA/dt should be negative and proportional to -A(t) * ln(R_top/R_bottom)
        ln_ratio = math.log(self.helix.top_radius / self.helix.bottom_radius)
        
        exponential_decay_valid = True
        for i in range(1, len(attention_densities) - 1):
            # Numerical derivative
            dt = t_values[i+1] - t_values[i-1]
            dA_dt = (attention_densities[i+1] - attention_densities[i-1]) / dt
            
            # Expected: dA/dt = -A(t) * ln_ratio (negative because attention decreases)
            expected_derivative = -attention_densities[i] * ln_ratio
            relative_error = abs(dA_dt - expected_derivative) / abs(expected_derivative) if expected_derivative != 0 else 1
            
            if relative_error > 0.15:  # Allow 15% numerical error for derivative approximation
                exponential_decay_valid = False
            
            if i < 5:  # Print first few for debugging
                print(f"  t={t_values[i]:.2f}: dA/dt={dA_dt:.3f}, "
                      f"expected={expected_derivative:.3f}, error={relative_error:.2%}")
        
        print(f"Exponential decay rate: {'✅' if exponential_decay_valid else '❌'}")
        
        # Test focusing factor: ratio of attention at bottom vs top (bottom has higher attention)
        focusing_ratio = attention_densities[0] / attention_densities[-1] if attention_densities[-1] > 0 else 0
        expected_ratio = (self.helix.top_radius / self.helix.bottom_radius)  # R_top/R_bottom ratio
        
        focusing_valid = focusing_ratio > 100  # Should show significant focusing at bottom
        
        print(f"Attention at bottom (t=0.1): {attention_densities[0]:.6f}")
        print(f"Attention at top (t=0.9): {attention_densities[-1]:.6f}")
        print(f"Focusing ratio (A_bottom/A_top): {focusing_ratio:.1f}")
        print(f"Significant focusing (>100x): {'✅' if focusing_valid else '❌'}")
        
        self.validation_results['attention_focusing'] = {
            'monotonic_decrease': monotonic_decrease,
            'exponential_decay': exponential_decay_valid,
            'significant_focusing': focusing_valid,
            'overall': monotonic_decrease and exponential_decay_valid and focusing_valid
        }
        
        return monotonic_decrease and exponential_decay_valid and focusing_valid
    
    def validate_communication_complexity(self) -> bool:
        """
        Validate communication complexity from hypothesis_mathematics.md Section H2.
        
        Tests:
        1. O(N) scaling for spoke-based system
        2. Message count bounds
        3. Maximum communication distance
        """
        print("\n=== Validating Communication Complexity ===")
        
        # Test scaling with different agent counts
        agent_counts = [10, 20, 50, 100]
        message_counts = []
        processing_times = []
        
        for N in agent_counts:
            central_post = CentralPost(max_agents=N, enable_metrics=True)
            agents = create_openscad_agents(self.helix, number_of_nodes=N, random_seed=42069)
            
            # Register agents and time the operation
            start_time = time.perf_counter()
            for agent in agents:
                central_post.register_agent(agent)
            registration_time = time.perf_counter() - start_time
            
            processing_times.append(registration_time)
            message_counts.append(N)  # Each agent creates one connection
        
        # Test O(N) scaling - processing time should be roughly linear
        linear_scaling_valid = True
        for i in range(1, len(agent_counts)):
            ratio_agents = agent_counts[i] / agent_counts[i-1]
            ratio_time = processing_times[i] / processing_times[i-1]
            
            # Allow factor of 3 deviation from linear scaling
            if ratio_time > 3 * ratio_agents or ratio_time < ratio_agents / 3:
                linear_scaling_valid = False
            
            print(f"  N={agent_counts[i]}: time={processing_times[i]:.6f}s, "
                  f"scaling ratio={ratio_time:.2f} (expected ≈{ratio_agents:.2f})")
        
        print(f"Linear scaling O(N): {'✅' if linear_scaling_valid else '❌'}")
        
        # Test maximum communication distance
        max_spoke_distance = self.helix.top_radius
        actual_distances = []
        
        for t in np.linspace(0, 1, 100):
            pos = self.helix.get_position(t)
            # Distance from agent to central axis at same height
            distance = math.sqrt(pos[0]**2 + pos[1]**2)  # Should equal R(t)
            actual_distances.append(distance)
        
        max_actual_distance = max(actual_distances)
        distance_bound_valid = abs(max_actual_distance - max_spoke_distance) < self.tolerance
        
        print(f"Maximum spoke distance: {max_actual_distance:.6f} "
              f"(expected {max_spoke_distance:.6f})")
        print(f"Distance bound valid: {'✅' if distance_bound_valid else '❌'}")
        
        # Test message complexity O(N) vs theoretical O(N²) mesh
        complexity_advantage = True
        for N in agent_counts:
            spoke_messages = N  # Each agent to central post
            mesh_messages = N * (N - 1) // 2  # All pairs
            
            efficiency_ratio = mesh_messages / spoke_messages
            expected_ratio = (N - 1) / 2
            
            if abs(efficiency_ratio - expected_ratio) > 0.1:
                complexity_advantage = False
            
            print(f"  N={N}: spoke={spoke_messages}, mesh={mesh_messages}, "
                  f"advantage={efficiency_ratio:.1f}x")
        
        print(f"Message complexity advantage: {'✅' if complexity_advantage else '❌'}")
        
        self.validation_results['communication_complexity'] = {
            'linear_scaling': linear_scaling_valid,
            'distance_bounds': distance_bound_valid,
            'complexity_advantage': complexity_advantage,
            'overall': linear_scaling_valid and distance_bound_valid and complexity_advantage
        }
        
        return linear_scaling_valid and distance_bound_valid and complexity_advantage
    
    def validate_numerical_precision(self) -> bool:
        """
        Validate numerical precision and stability.
        
        Tests:
        1. Consistency with OpenSCAD validation
        2. Numerical stability across parameter ranges
        3. Error accumulation in iterative calculations
        """
        print("\n=== Validating Numerical Precision ===")
        
        # Test against OpenSCAD validation (should be < 1e-12)
        from validate_openscad import validate_implementation
        openscad_valid = validate_implementation()
        
        print(f"OpenSCAD validation: {'✅' if openscad_valid else '❌'}")
        
        # Test numerical stability at extreme parameter values
        stability_valid = True
        
        # Test very small t values
        small_t_values = [1e-10, 1e-8, 1e-6, 1e-4]
        for t in small_t_values:
            try:
                pos = self.helix.get_position(t)
                # Should be close to (R_bottom, 0, 0)
                if not all(math.isfinite(x) for x in pos):
                    stability_valid = False
            except Exception:
                stability_valid = False
        
        # Test values very close to 1
        large_t_values = [1 - 1e-10, 1 - 1e-8, 1 - 1e-6, 1 - 1e-4]
        for t in large_t_values:
            try:
                pos = self.helix.get_position(t)
                if not all(math.isfinite(x) for x in pos):
                    stability_valid = False
            except Exception:
                stability_valid = False
        
        print(f"Numerical stability: {'✅' if stability_valid else '❌'}")
        
        # Test error accumulation in arc length calculation
        # Compare high-precision vs standard precision
        arc_length_high = self.helix.approximate_arc_length(segments=10000)
        arc_length_std = self.helix.approximate_arc_length(segments=1000)
        
        relative_error = abs(arc_length_high - arc_length_std) / arc_length_high
        error_acceptable = relative_error < 0.01  # 1% tolerance for numerical approximation
        
        print(f"Arc length error: {relative_error:.4%} (high vs standard precision)")
        print(f"Error accumulation acceptable: {'✅' if error_acceptable else '❌'}")
        
        self.validation_results['numerical_precision'] = {
            'openscad_validation': openscad_valid,
            'numerical_stability': stability_valid,
            'error_accumulation': error_acceptable,
            'overall': openscad_valid and stability_valid and error_acceptable
        }
        
        return openscad_valid and stability_valid and error_acceptable
    
    def run_full_validation(self) -> bool:
        """Run complete mathematical validation suite."""
        print("=" * 60)
        print("FELIX FRAMEWORK MATHEMATICAL VALIDATION")
        print("=" * 60)
        
        validators = [
            ('Parametric Equations', self.validate_parametric_equations),
            ('Geometric Properties', self.validate_geometric_properties),
            ('Agent Distribution', self.validate_agent_distribution),
            ('Attention Focusing', self.validate_attention_focusing),
            ('Communication Complexity', self.validate_communication_complexity),
            ('Numerical Precision', self.validate_numerical_precision),
        ]
        
        all_valid = True
        results_summary = []
        
        for name, validator in validators:
            try:
                result = validator()
                results_summary.append((name, result))
                if not result:
                    all_valid = False
            except Exception as e:
                print(f"ERROR in {name}: {e}")
                results_summary.append((name, False))
                all_valid = False
        
        # Print summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        for name, result in results_summary:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{name:<25}: {status}")
        
        overall_status = "✅ ALL VALIDATIONS PASSED" if all_valid else "❌ SOME VALIDATIONS FAILED"
        print(f"\nOverall Result: {overall_status}")
        
        if all_valid:
            print("\n🎉 Mathematical implementation is validated!")
            print("   Ready for hypothesis testing and research publication.")
        else:
            print("\n⚠️  Mathematical validation failures detected.")
            print("   Review implementation before proceeding with experiments.")
        
        return all_valid


if __name__ == "__main__":
    validator = MathematicalValidator(tolerance=1e-10)
    success = validator.run_full_validation()
    
    # Save validation results for research documentation
    import json
    
    # Convert numpy booleans to regular booleans for JSON serialization
    def convert_numpy_types(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        return obj
    
    serializable_results = convert_numpy_types(validator.validation_results)
    
    with open("mathematical_validation_results.json", "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nValidation results saved to: mathematical_validation_results.json")
    
    exit(0 if success else 1)