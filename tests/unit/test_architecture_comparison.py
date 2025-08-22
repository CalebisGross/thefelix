#!/usr/bin/env python3
"""
Test suite for architecture comparison framework - Felix Framework Phase 6.

This module provides comprehensive testing for the unified comparison system
that evaluates helix, linear pipeline, and mesh communication architectures
against each other for hypothesis validation.

Mathematical Foundation:
- H1 testing: Task distribution efficiency using coefficient of variation analysis
- H2 testing: Communication overhead comparison O(N) vs O(N×M) vs O(N²)
- H3 testing: Attention focusing mechanism validation through agent density analysis

Key Features:
- Unified comparison framework supporting all three architectures
- Statistical validation infrastructure for hypothesis testing
- Performance benchmarking with controlled experimental conditions
- Automated data collection and analysis for research validation

This supports comprehensive hypothesis testing by providing measurable
performance characteristics across all architectural approaches with
statistical rigor for publication-quality research validation.

Mathematical reference: docs/architecture/core/hypothesis_mathematics.md, Sections H1, H2, H3
"""

import pytest
import time
import numpy as np
import statistics
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock

from src.comparison.architecture_comparison import (
    ArchitectureComparison, ComparisonResults, ExperimentalConfig,
    PerformanceMetrics
)
from src.comparison.statistical_analysis import StatisticalResults, HypothesisValidator
from src.core.helix_geometry import HelixGeometry
from src.agents.agent import Agent, create_openscad_agents
from src.communication.central_post import CentralPost
from src.communication.spoke import SpokeManager
from src.communication.mesh import MeshCommunication
from src.pipeline.linear_pipeline import LinearPipeline


class TestArchitectureComparison:
    """Test unified architecture comparison framework."""
    
    @pytest.fixture
    def standard_helix(self):
        """Create helix with OpenSCAD model parameters."""
        return HelixGeometry(
            top_radius=33.0,
            bottom_radius=0.001,
            height=33.0,
            turns=33
        )
    
    @pytest.fixture
    def comparison_framework(self, standard_helix):
        """Create architecture comparison framework."""
        return ArchitectureComparison(
            helix=standard_helix,
            max_agents=50,  # Smaller for testing
            enable_detailed_metrics=True
        )
    
    def test_comparison_framework_initialization(self, comparison_framework):
        """Test comparison framework can be initialized properly."""
        assert comparison_framework.max_agents == 50
        assert comparison_framework.helix is not None
        assert comparison_framework.detailed_metrics_enabled is True
        assert len(comparison_framework.architectures) == 3
        
        # Should have all three architectures configured
        arch_names = {arch["name"] for arch in comparison_framework.architectures}
        expected_names = {"helix_spoke", "linear_pipeline", "mesh_communication"}
        assert arch_names == expected_names
    
    def test_experimental_config_validation(self):
        """Test experimental configuration validation."""
        # Valid configuration
        config = ExperimentalConfig(
            agent_count=20,
            simulation_time=1.0,
            task_load=100,
            random_seed=42069
        )
        
        assert config.agent_count == 20
        assert config.simulation_time == 1.0
        assert config.task_load == 100
        assert config.random_seed == 42069
        
        # Invalid configuration
        with pytest.raises(ValueError, match="agent_count must be positive"):
            ExperimentalConfig(
                agent_count=0,
                simulation_time=1.0,
                task_load=100,
                random_seed=42069
            )
        
        with pytest.raises(ValueError, match="simulation_time must be positive"):
            ExperimentalConfig(
                agent_count=20,
                simulation_time=0.0,
                task_load=100,
                random_seed=42069
            )
    
    def test_helix_architecture_performance_measurement(self, comparison_framework):
        """Test performance measurement for helix architecture."""
        config = ExperimentalConfig(
            agent_count=10,
            simulation_time=1.0,
            task_load=50,
            random_seed=42069
        )
        
        # Run helix architecture test
        results = comparison_framework.run_helix_experiment(config)
        
        assert isinstance(results, PerformanceMetrics)
        assert results.architecture_name == "helix_spoke"
        assert results.agent_count == 10
        assert results.task_completion_time > 0
        assert results.communication_overhead >= 0
        assert results.throughput > 0
        assert results.memory_usage > 0
        
        # Should have O(N) communication complexity
        assert results.communication_complexity_order == "O(N)"
    
    def test_linear_pipeline_performance_measurement(self, comparison_framework):
        """Test performance measurement for linear pipeline architecture."""
        config = ExperimentalConfig(
            agent_count=15,
            simulation_time=1.0,
            task_load=75,
            random_seed=42069
        )
        
        # Run linear pipeline test
        results = comparison_framework.run_linear_experiment(config)
        
        assert isinstance(results, PerformanceMetrics)
        assert results.architecture_name == "linear_pipeline"
        assert results.agent_count == 15
        assert results.task_completion_time > 0
        assert results.throughput > 0
        
        # Should have O(N×M) processing complexity
        assert results.communication_complexity_order == "O(N×M)"
        
        # Should have stage-specific metrics
        assert "stage_utilization" in results.architecture_specific_metrics
        assert "bottleneck_stages" in results.architecture_specific_metrics
    
    def test_mesh_communication_performance_measurement(self, comparison_framework):
        """Test performance measurement for mesh communication architecture."""
        config = ExperimentalConfig(
            agent_count=8,  # Smaller for O(N²) testing
            simulation_time=1.0,
            task_load=40,
            random_seed=42069
        )
        
        # Run mesh communication test
        results = comparison_framework.run_mesh_experiment(config)
        
        assert isinstance(results, PerformanceMetrics)
        assert results.architecture_name == "mesh_communication"
        assert results.agent_count == 8
        assert results.communication_overhead > 0
        assert results.throughput > 0
        
        # Should have O(N²) communication complexity
        assert results.communication_complexity_order == "O(N²)"
        
        # Should have mesh-specific metrics
        assert "connection_count" in results.architecture_specific_metrics
        assert "average_distance" in results.architecture_specific_metrics
    
    def test_comparative_experiment_execution(self, comparison_framework):
        """Test running comparative experiments across all architectures."""
        config = ExperimentalConfig(
            agent_count=12,
            simulation_time=1.0,
            task_load=60,
            random_seed=42069
        )
        
        # Run comparative experiment
        comparison_results = comparison_framework.run_comparative_experiment(config)
        
        assert isinstance(comparison_results, ComparisonResults)
        assert len(comparison_results.performance_metrics) == 3
        
        # Should have results for all three architectures
        arch_names = {metrics.architecture_name for metrics in comparison_results.performance_metrics}
        expected_names = {"helix_spoke", "linear_pipeline", "mesh_communication"}
        assert arch_names == expected_names
        
        # Should have comparative analysis
        assert comparison_results.statistical_analysis is not None
        assert len(comparison_results.performance_rankings) == 3


class TestHypothesisValidation:
    """Test statistical validation of research hypotheses."""
    
    @pytest.fixture
    def standard_helix(self):
        """Create helix with OpenSCAD model parameters."""
        return HelixGeometry(
            top_radius=33.0,
            bottom_radius=0.001,
            height=33.0,
            turns=33
        )
    
    @pytest.fixture
    def hypothesis_validator(self, standard_helix):
        """Create hypothesis validation framework."""
        comparison = ArchitectureComparison(
            helix=standard_helix,
            max_agents=30,
            enable_detailed_metrics=True
        )
        return comparison.get_hypothesis_validator()
    
    def test_hypothesis_h1_task_distribution_validation(self, hypothesis_validator):
        """Test H1: Helical paths improve task distribution efficiency."""
        config = ExperimentalConfig(
            agent_count=20,
            simulation_time=1.0,
            task_load=100,
            random_seed=42069
        )
        
        # Run H1 validation experiment
        h1_results = hypothesis_validator.validate_hypothesis_h1(config)
        
        assert isinstance(h1_results, StatisticalResults)
        assert h1_results.hypothesis == "H1"
        assert h1_results.test_statistic is not None
        assert h1_results.p_value is not None
        assert h1_results.effect_size is not None
        assert h1_results.confidence_interval is not None
        
        # Should have coefficient of variation analysis
        assert "coefficient_of_variation" in h1_results.statistical_metrics
        assert "f_test_statistic" in h1_results.statistical_metrics
        
        # Should compare helix vs other architectures
        assert len(h1_results.comparison_data) >= 2
    
    def test_hypothesis_h2_communication_overhead_validation(self, hypothesis_validator):
        """Test H2: Spoke communication reduces coordination overhead."""
        config = ExperimentalConfig(
            agent_count=15,
            simulation_time=1.0,
            task_load=75,
            random_seed=42069
        )
        
        # Run H2 validation experiment
        h2_results = hypothesis_validator.validate_hypothesis_h2(config)
        
        assert isinstance(h2_results, StatisticalResults)
        assert h2_results.hypothesis == "H2"
        assert h2_results.test_statistic is not None
        assert h2_results.p_value is not None
        
        # Should have communication complexity analysis
        assert "communication_overhead_ratio" in h2_results.statistical_metrics
        assert "scaling_factor" in h2_results.statistical_metrics
        assert "throughput_comparison" in h2_results.statistical_metrics
        
        # Should compare O(N) vs O(N²) scaling
        overhead_ratios = h2_results.comparison_data["communication_overhead"]
        helix_overhead = next(ratio for arch, ratio in overhead_ratios if arch == "helix_spoke")
        mesh_overhead = next(ratio for arch, ratio in overhead_ratios if arch == "mesh_communication")
        
        # Helix should have lower communication overhead than mesh
        assert helix_overhead < mesh_overhead
    
    def test_hypothesis_h3_attention_focusing_validation(self, hypothesis_validator):
        """Test H3: Geometric tapering provides natural attention focusing."""
        config = ExperimentalConfig(
            agent_count=25,
            simulation_time=1.0,
            task_load=125,
            random_seed=42069
        )
        
        # Run H3 validation experiment
        h3_results = hypothesis_validator.validate_hypothesis_h3(config)
        
        assert isinstance(h3_results, StatisticalResults)
        assert h3_results.hypothesis == "H3"
        assert h3_results.test_statistic is not None
        assert h3_results.p_value is not None
        
        # Should have attention density analysis
        assert "attention_concentration_ratio" in h3_results.statistical_metrics
        assert "agent_density_evolution" in h3_results.statistical_metrics
        assert "focusing_effectiveness" in h3_results.statistical_metrics
        
        # Should demonstrate exponential concentration
        concentration_ratio = h3_results.statistical_metrics["attention_concentration_ratio"]
        assert concentration_ratio > 1000  # Should be > 4,119x from mathematical analysis
    
    def test_comprehensive_hypothesis_validation(self, hypothesis_validator):
        """Test comprehensive validation of all three hypotheses."""
        config = ExperimentalConfig(
            agent_count=20,
            simulation_time=1.0,
            task_load=100,
            random_seed=42069
        )
        
        # Run comprehensive validation
        all_results = hypothesis_validator.validate_all_hypotheses(config)
        
        assert len(all_results) == 3
        hypothesis_names = {result.hypothesis for result in all_results}
        assert hypothesis_names == {"H1", "H2", "H3"}
        
        # All should have statistical results
        for result in all_results:
            assert result.test_statistic is not None
            assert result.p_value is not None
            assert result.effect_size is not None
            assert len(result.statistical_metrics) > 0
        
        # Should provide overall research conclusions
        research_summary = hypothesis_validator.generate_research_summary(all_results)
        assert "hypothesis_validation_summary" in research_summary
        assert "statistical_significance" in research_summary
        assert "effect_sizes" in research_summary
        assert "research_conclusions" in research_summary


class TestPerformanceBenchmarking:
    """Test performance benchmarking across architectures."""
    
    @pytest.fixture
    def standard_helix(self):
        """Create helix with OpenSCAD model parameters."""
        return HelixGeometry(
            top_radius=33.0,
            bottom_radius=0.001,
            height=33.0,
            turns=33
        )
    
    def test_scalability_benchmarking(self, standard_helix):
        """Test scalability benchmarking across different agent counts."""
        comparison = ArchitectureComparison(
            helix=standard_helix,
            max_agents=100,
            enable_detailed_metrics=True
        )
        
        # Test different agent counts
        agent_counts = [5, 10, 20, 40]
        benchmark_results = []
        
        for count in agent_counts:
            config = ExperimentalConfig(
                agent_count=count,
                simulation_time=1.0,
                task_load=count * 5,  # Scale task load with agent count
                random_seed=42069
            )
            
            results = comparison.run_comparative_experiment(config)
            benchmark_results.append(results)
        
        # Analyze scaling characteristics
        helix_times = []
        linear_times = []
        mesh_times = []
        
        for results in benchmark_results:
            for metrics in results.performance_metrics:
                if metrics.architecture_name == "helix_spoke":
                    helix_times.append(metrics.task_completion_time)
                elif metrics.architecture_name == "linear_pipeline":
                    linear_times.append(metrics.task_completion_time)
                elif metrics.architecture_name == "mesh_communication":
                    mesh_times.append(metrics.task_completion_time)
        
        # Validate scaling expectations
        assert len(helix_times) == len(agent_counts)
        assert len(linear_times) == len(agent_counts)
        assert len(mesh_times) == len(agent_counts)
        
        # Helix should scale better than mesh for larger agent counts
        if len(agent_counts) >= 3:
            helix_scaling = helix_times[-1] / helix_times[0]  # Ratio of largest to smallest
            mesh_scaling = mesh_times[-1] / mesh_times[0]
            
            # Helix should have better scaling characteristics
            assert helix_scaling <= mesh_scaling * 1.5  # Allow some variance
    
    def test_throughput_comparison(self, standard_helix):
        """Test throughput comparison across architectures."""
        comparison = ArchitectureComparison(
            helix=standard_helix,
            max_agents=25,
            enable_detailed_metrics=True
        )
        
        config = ExperimentalConfig(
            agent_count=15,
            simulation_time=1.0,
            task_load=150,
            random_seed=42069
        )
        
        # Run throughput benchmark
        results = comparison.run_comparative_experiment(config)
        throughput_data = comparison.analyze_throughput_characteristics(results)
        
        assert "architecture_throughputs" in throughput_data
        assert "relative_performance" in throughput_data
        assert "bottleneck_analysis" in throughput_data
        
        # Should have throughput data for all architectures
        throughputs = throughput_data["architecture_throughputs"]
        assert "helix_spoke" in throughputs
        assert "linear_pipeline" in throughputs
        assert "mesh_communication" in throughputs
        
        # All throughputs should be positive
        for arch, throughput in throughputs.items():
            assert throughput > 0
    
    def test_memory_usage_analysis(self, standard_helix):
        """Test memory usage analysis across architectures."""
        comparison = ArchitectureComparison(
            helix=standard_helix,
            max_agents=30,
            enable_detailed_metrics=True
        )
        
        config = ExperimentalConfig(
            agent_count=20,
            simulation_time=1.0,
            task_load=100,
            random_seed=42069
        )
        
        # Run memory usage analysis
        results = comparison.run_comparative_experiment(config)
        memory_analysis = comparison.analyze_memory_usage(results)
        
        assert "architecture_memory_usage" in memory_analysis
        assert "memory_scaling_factors" in memory_analysis
        assert "memory_efficiency_rankings" in memory_analysis
        
        # Mesh should have higher memory usage due to O(N²) connections
        memory_usage = memory_analysis["architecture_memory_usage"]
        helix_memory = memory_usage.get("helix_spoke", 0)
        mesh_memory = memory_usage.get("mesh_communication", 0)
        
        # Mesh should use more memory than helix
        assert mesh_memory >= helix_memory
    
    def test_latency_distribution_analysis(self, standard_helix):
        """Test latency distribution analysis for communication systems."""
        comparison = ArchitectureComparison(
            helix=standard_helix,
            max_agents=25,
            enable_detailed_metrics=True
        )
        
        config = ExperimentalConfig(
            agent_count=15,
            simulation_time=1.0,
            task_load=75,
            random_seed=42069
        )
        
        # Run latency analysis
        results = comparison.run_comparative_experiment(config)
        latency_analysis = comparison.analyze_latency_distribution(results)
        
        assert "mean_latencies" in latency_analysis
        assert "latency_variance" in latency_analysis
        assert "latency_percentiles" in latency_analysis
        
        # Should have latency data for communication architectures
        mean_latencies = latency_analysis["mean_latencies"]
        assert "helix_spoke" in mean_latencies
        assert "mesh_communication" in mean_latencies
        
        # All latencies should be non-negative
        for arch, latency in mean_latencies.items():
            assert latency >= 0


class TestStatisticalValidation:
    """Test statistical validation methods and analysis."""
    
    def test_statistical_significance_testing(self):
        """Test statistical significance testing methods."""
        # Create sample data for testing
        helix_performance = [0.85, 0.82, 0.87, 0.84, 0.86, 0.83, 0.88, 0.85]
        linear_performance = [0.76, 0.78, 0.74, 0.77, 0.75, 0.79, 0.73, 0.76]
        mesh_performance = [0.68, 0.71, 0.69, 0.67, 0.70, 0.66, 0.72, 0.68]
        
        from src.comparison.statistical_analysis import StatisticalAnalyzer
        analyzer = StatisticalAnalyzer()
        
        # Test t-test comparison
        t_stat, p_value = analyzer.two_sample_t_test(helix_performance, linear_performance)
        
        assert t_stat is not None
        assert p_value is not None
        assert 0 <= p_value <= 1
        
        # Helix should be significantly better than linear
        assert t_stat > 0  # Assuming helix has higher performance
        
        # Test ANOVA for multiple group comparison
        f_stat, p_value_anova = analyzer.one_way_anova([
            helix_performance, linear_performance, mesh_performance
        ])
        
        assert f_stat is not None
        assert p_value_anova is not None
        assert f_stat > 0
        assert 0 <= p_value_anova <= 1
    
    def test_effect_size_calculation(self):
        """Test effect size calculations for practical significance."""
        from src.comparison.statistical_analysis import StatisticalAnalyzer
        analyzer = StatisticalAnalyzer()
        
        # Sample performance data
        group1 = [85, 82, 87, 84, 86, 83, 88, 85]
        group2 = [76, 78, 74, 77, 75, 79, 73, 76]
        
        # Calculate Cohen's d
        cohens_d = analyzer.calculate_cohens_d(group1, group2)
        
        assert cohens_d is not None
        assert cohens_d > 0  # group1 should perform better
        
        # Large effect size expected (d > 0.8)
        assert cohens_d > 0.8
        
        # Calculate eta-squared for ANOVA
        eta_squared = analyzer.calculate_eta_squared([group1, group2])
        
        assert eta_squared is not None
        assert 0 <= eta_squared <= 1
    
    def test_confidence_intervals(self):
        """Test confidence interval calculations."""
        from src.comparison.statistical_analysis import StatisticalAnalyzer
        analyzer = StatisticalAnalyzer()
        
        sample_data = [0.85, 0.82, 0.87, 0.84, 0.86, 0.83, 0.88, 0.85]
        
        # Calculate 95% confidence interval
        ci_lower, ci_upper = analyzer.confidence_interval(sample_data, confidence_level=0.95)
        
        assert ci_lower is not None
        assert ci_upper is not None
        assert ci_lower < ci_upper
        
        # CI should contain the sample mean
        sample_mean = statistics.mean(sample_data)
        assert ci_lower <= sample_mean <= ci_upper
    
    def test_power_analysis(self):
        """Test statistical power analysis."""
        from src.comparison.statistical_analysis import StatisticalAnalyzer
        analyzer = StatisticalAnalyzer()
        
        # Test power calculation for t-test
        effect_size = 0.8  # Large effect size
        alpha = 0.05
        sample_size = 20
        
        power = analyzer.calculate_power_t_test(effect_size, sample_size, alpha)
        
        assert power is not None
        assert 0 <= power <= 1
        
        # With large effect size and reasonable sample, power should be moderately high
        assert power > 0.65
    
    def test_multiple_comparison_correction(self):
        """Test multiple comparison correction methods."""
        from src.comparison.statistical_analysis import StatisticalAnalyzer
        analyzer = StatisticalAnalyzer()
        
        # Sample p-values from multiple tests
        p_values = [0.01, 0.03, 0.045, 0.08, 0.12]
        
        # Bonferroni correction
        bonferroni_corrected = analyzer.bonferroni_correction(p_values, alpha=0.05)
        
        assert len(bonferroni_corrected) == len(p_values)
        for corrected in bonferroni_corrected:
            assert corrected in [True, False]
        
        # FDR correction (Benjamini-Hochberg)
        fdr_corrected = analyzer.fdr_correction(p_values, alpha=0.05)
        
        assert len(fdr_corrected) == len(p_values)
        for corrected in fdr_corrected:
            assert corrected in [True, False]


class TestExperimentalProtocols:
    """Test experimental protocol design and execution."""
    
    @pytest.fixture
    def standard_helix(self):
        """Create helix with OpenSCAD model parameters."""
        return HelixGeometry(
            top_radius=33.0,
            bottom_radius=0.001,
            height=33.0,
            turns=33
        )
    
    def test_controlled_experiment_design(self, standard_helix):
        """Test controlled experiment design for hypothesis testing."""
        from src.comparison.experimental_protocol import ExperimentalProtocol
        
        protocol = ExperimentalProtocol(
            helix=standard_helix,
            control_variables=["agent_count", "task_complexity", "random_seed"],
            response_variables=["completion_time", "throughput", "communication_overhead"]
        )
        
        # Design factorial experiment
        experiment_design = protocol.design_factorial_experiment(
            factors={
                "agent_count": [10, 20, 30],
                "task_complexity": ["low", "medium", "high"]
            },
            replications=3
        )
        
        assert len(experiment_design) == 3 * 3 * 3  # factors × replications
        
        # Each experiment should have all required factors
        for experiment in experiment_design:
            assert "agent_count" in experiment
            assert "task_complexity" in experiment
            assert "replication" in experiment
    
    def test_randomization_and_blocking(self, standard_helix):
        """Test experimental randomization and blocking procedures."""
        from src.comparison.experimental_protocol import ExperimentalProtocol
        
        protocol = ExperimentalProtocol(
            helix=standard_helix,
            control_variables=["agent_count", "architecture"],
            response_variables=["performance"]
        )
        
        # Create randomized block design
        blocked_design = protocol.randomized_block_design(
            treatments=["helix_spoke", "linear_pipeline", "mesh_communication"],
            blocks=["small_scale", "medium_scale", "large_scale"],
            replications=2
        )
        
        # Should have proper randomization
        assert len(blocked_design) == 3 * 3 * 2  # treatments × blocks × replications
        
        # Check randomization worked
        treatment_orders = []
        for block in ["small_scale", "medium_scale", "large_scale"]:
            block_treatments = [exp["treatment"] for exp in blocked_design 
                              if exp["block"] == block]
            treatment_orders.append(block_treatments)
        
        # Different blocks should potentially have different orders (randomized)
        # This is probabilistic, so we just check structure is correct
        for order in treatment_orders:
            assert len(order) == 6  # 3 treatments × 2 replications
    
    def test_experimental_validation_protocol(self, standard_helix):
        """Test comprehensive experimental validation protocol."""
        from src.comparison.experimental_protocol import ExperimentalProtocol
        
        protocol = ExperimentalProtocol(
            helix=standard_helix,
            control_variables=["agent_count", "architecture", "task_load"],
            response_variables=["completion_time", "throughput", "communication_overhead"]
        )
        
        # Run validation protocol
        validation_results = protocol.run_validation_protocol(
            architectures=["helix_spoke", "linear_pipeline", "mesh_communication"],
            agent_counts=[10, 20],
            replications=3,
            random_seed=42069
        )
        
        assert "experimental_data" in validation_results
        assert "statistical_analysis" in validation_results
        assert "hypothesis_tests" in validation_results
        
        # Should have data for all architecture-agent count combinations
        experimental_data = validation_results["experimental_data"]
        expected_conditions = 3 * 2 * 3  # 3 architectures × 2 agent counts × 3 replications
        assert len(experimental_data) == expected_conditions
        
        # Statistical analysis should include results for all metrics
        statistical_analysis = validation_results["statistical_analysis"]
        expected_metrics = ["completion_time", "throughput", "communication_overhead"]
        for metric in expected_metrics:
            assert metric in statistical_analysis
            # Each metric should have statistical test results
            metric_result = statistical_analysis[metric]
            assert "p_value" in metric_result
            assert "significant" in metric_result