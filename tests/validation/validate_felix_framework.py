#!/usr/bin/env python3
"""
Comprehensive validation of the Felix Framework.

This script runs a complete validation of the Felix Framework including:
- All three architectures (helix, linear, mesh)
- Hypothesis validation (H1, H2, H3)
- Performance benchmarking
- Statistical analysis and research conclusions

This serves as the primary validation script for the research project,
providing publication-ready results and analysis.
"""

import time
import json
from typing import Dict, Any

from src.core.helix_geometry import HelixGeometry
from src.comparison.architecture_comparison import ArchitectureComparison, ExperimentalConfig
from src.comparison.statistical_analysis import HypothesisValidator
from src.comparison.experimental_protocol import ExperimentalProtocol


def main():
    """Run comprehensive Felix Framework validation."""
    print("=" * 60)
    print("FELIX FRAMEWORK COMPREHENSIVE VALIDATION")
    print("=" * 60)
    print()
    
    # Initialize components
    print("Initializing framework components...")
    helix = HelixGeometry(
        top_radius=33.0,
        bottom_radius=0.001,
        height=33.0,
        turns=33
    )
    
    comparison = ArchitectureComparison(
        helix=helix,
        max_agents=50,  # Reduced for faster testing
        enable_detailed_metrics=True
    )
    
    validator = comparison.get_hypothesis_validator()
    
    protocol = ExperimentalProtocol(
        helix=helix,
        control_variables=["agent_count", "architecture", "task_load"],
        response_variables=["throughput", "completion_time", "communication_overhead"]
    )
    
    print("✓ Components initialized successfully")
    print()
    
    # Phase 1: Architecture Performance Comparison
    print("Phase 1: Architecture Performance Comparison")
    print("-" * 40)
    
    config = ExperimentalConfig(
        agent_count=20,
        simulation_time=1.0,
        task_load=100,
        random_seed=42069
    )
    
    print(f"Running comparative experiment with {config.agent_count} agents...")
    start_time = time.time()
    
    try:
        comparison_results = comparison.run_comparative_experiment(config)
        
        print("✓ Comparative experiment completed")
        print(f"  Duration: {time.time() - start_time:.2f} seconds")
        print(f"  Architectures tested: {len(comparison_results.performance_metrics)}")
        
        # Display performance summary
        print("\nPerformance Summary:")
        for metrics in comparison_results.performance_metrics:
            print(f"  {metrics.architecture_name}:")
            print(f"    Throughput: {metrics.throughput:.2f}")
            print(f"    Completion Time: {metrics.task_completion_time:.4f}s")
            print(f"    Communication Overhead: {metrics.communication_overhead:.4f}")
            print(f"    Memory Usage: {metrics.memory_usage:.0f}")
        
        print("\nArchitecture Rankings:")
        for rank, (arch, score) in enumerate(comparison_results.performance_rankings, 1):
            print(f"  {rank}. {arch} (score: {score:.3f})")
        
    except Exception as e:
        print(f"✗ Error in comparative experiment: {e}")
        return False
    
    print()
    
    # Phase 2: Hypothesis Validation
    print("Phase 2: Hypothesis Validation")
    print("-" * 30)
    
    print("Validating research hypotheses...")
    hypothesis_start = time.time()
    
    try:
        # Validate all hypotheses
        h1_results = validator.validate_hypothesis_h1(config)
        print(f"✓ H1 validation completed: {h1_results.conclusion}")
        
        h2_results = validator.validate_hypothesis_h2(config)
        print(f"✓ H2 validation completed: {h2_results.conclusion}")
        
        h3_results = validator.validate_hypothesis_h3(config)
        print(f"✓ H3 validation completed: {h3_results.conclusion}")
        
        print(f"  Duration: {time.time() - hypothesis_start:.2f} seconds")
        
        # Generate research summary
        all_results = [h1_results, h2_results, h3_results]
        research_summary = validator.generate_research_summary(all_results)
        
        print(f"\nResearch Summary:")
        print(f"  Overall Conclusion: {research_summary['overall_conclusion']}")
        print(f"  Framework Validation: {research_summary['felix_framework_validation']}")
        
        print(f"\nHypothesis Results:")
        for hypothesis in ['H1', 'H2', 'H3']:
            result = research_summary['hypothesis_validation_summary'][hypothesis]
            print(f"  {hypothesis}: {'SIGNIFICANT' if result['significant'] else 'NOT SIGNIFICANT'} "
                  f"(p={result['p_value']:.4f}, effect={result['effect_size']:.3f})")
        
    except Exception as e:
        print(f"✗ Error in hypothesis validation: {e}")
        return False
    
    print()
    
    # Phase 3: Experimental Protocol Validation
    print("Phase 3: Experimental Protocol Validation")
    print("-" * 38)
    
    print("Running validation protocol...")
    protocol_start = time.time()
    
    try:
        validation_results = protocol.run_validation_protocol(
            architectures=["helix_spoke", "linear_pipeline", "mesh_communication"],
            agent_counts=[5, 10, 15],  # Reduced for faster testing
            replications=2,
            random_seed=42069
        )
        
        print(f"✓ Validation protocol completed")
        print(f"  Duration: {time.time() - protocol_start:.2f} seconds")
        
        summary = validation_results["validation_summary"]
        print(f"  Experiments conducted: {summary['experiment_count']}")
        print(f"  Architectures tested: {summary['architectures_tested']}")
        print(f"  Significant metrics: {summary['significant_metrics']}")
        print(f"  Validation strength: {summary['validation_strength']}")
        print(f"  Research recommendation: {summary['research_recommendation']}")
        
    except Exception as e:
        print(f"✗ Error in validation protocol: {e}")
        return False
    
    print()
    
    # Phase 4: Performance Analysis
    print("Phase 4: Performance Analysis")
    print("-" * 26)
    
    print("Analyzing performance characteristics...")
    
    try:
        # Throughput analysis
        throughput_analysis = comparison.analyze_throughput_characteristics(comparison_results)
        print("✓ Throughput analysis completed")
        
        best_throughput = max(throughput_analysis["architecture_throughputs"].items(), 
                            key=lambda x: x[1])
        print(f"  Best throughput: {best_throughput[0]} ({best_throughput[1]:.2f})")
        
        # Memory analysis
        memory_analysis = comparison.analyze_memory_usage(comparison_results)
        print("✓ Memory usage analysis completed")
        
        most_efficient = memory_analysis["memory_efficiency_rankings"][0]
        print(f"  Most memory efficient: {most_efficient[0]} ({most_efficient[1]:.0f} units)")
        
        # Latency analysis
        latency_analysis = comparison.analyze_latency_distribution(comparison_results)
        print("✓ Latency distribution analysis completed")
        
        if latency_analysis["mean_latencies"]:
            lowest_latency = min(latency_analysis["mean_latencies"].items(), 
                               key=lambda x: x[1])
            print(f"  Lowest latency: {lowest_latency[0]} ({lowest_latency[1]:.6f}s)")
        
    except Exception as e:
        print(f"✗ Error in performance analysis: {e}")
        return False
    
    print()
    
    # Final Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    total_time = time.time() - start_time
    print(f"Total validation time: {total_time:.2f} seconds")
    print()
    
    print("Architecture Performance:")
    winner = comparison_results.performance_rankings[0]
    print(f"  Best overall performance: {winner[0]} (score: {winner[1]:.3f})")
    print()
    
    print("Hypothesis Validation:")
    supported_count = sum(1 for h in ['H1', 'H2', 'H3'] 
                         if research_summary['hypothesis_validation_summary'][h]['significant'])
    print(f"  Hypotheses supported: {supported_count}/3")
    
    if supported_count >= 2:
        print("  ✓ FELIX FRAMEWORK VALIDATION: SUCCESSFUL")
        print("    Sufficient evidence to support core research claims")
    else:
        print("  ⚠ FELIX FRAMEWORK VALIDATION: PARTIAL")
        print("    Limited evidence for research claims - additional research needed")
    
    print()
    print("Research Readiness:")
    if summary['research_recommendation'] == "Publication ready":
        print("  ✓ PUBLICATION READY: Results suitable for peer review")
    else:
        print("  ⚠ ADDITIONAL RESEARCH NEEDED: Strengthen evidence before publication")
    
    print()
    print("Felix Framework comprehensive validation completed successfully!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)