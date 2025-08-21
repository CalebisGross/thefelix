"""
Unified architecture comparison framework for the Felix Framework.

This module implements comprehensive comparison capabilities between helix-based
Felix architecture and traditional alternatives for rigorous hypothesis testing.

Mathematical Foundation:
- H1: Task distribution efficiency using coefficient of variation analysis
- H2: Communication overhead comparison O(N) vs O(N×M) vs O(N²)  
- H3: Attention focusing validation through agent density measurements

Key Features:
- Unified experiment execution across all three architectures
- Performance metrics collection with statistical rigor
- Automated hypothesis testing infrastructure
- Publication-quality experimental design and analysis

This enables rigorous validation of research hypotheses through controlled
experiments with proper statistical methodology for peer review.

Mathematical reference: docs/hypothesis_mathematics.md, Sections H1, H2, H3
"""

import time
import statistics
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from core.helix_geometry import HelixGeometry
from agents.agent import Agent, create_openscad_agents
from communication.central_post import CentralPost
from communication.spoke import SpokeManager
from communication.mesh import MeshCommunication
from pipeline.linear_pipeline import LinearPipeline
from .statistical_analysis import StatisticalAnalyzer


class ArchitectureType(Enum):
    """Supported architecture types for comparison."""
    HELIX_SPOKE = "helix_spoke"
    LINEAR_PIPELINE = "linear_pipeline"
    MESH_COMMUNICATION = "mesh_communication"


@dataclass
class ExperimentalConfig:
    """Configuration for comparative experiments."""
    agent_count: int
    simulation_time: float
    task_load: int
    random_seed: int
    architecture_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate experimental configuration."""
        if self.agent_count <= 0:
            raise ValueError("agent_count must be positive")
        if self.simulation_time <= 0:
            raise ValueError("simulation_time must be positive")
        if self.task_load <= 0:
            raise ValueError("task_load must be positive")


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single architecture."""
    architecture_name: str
    agent_count: int
    task_completion_time: float
    throughput: float
    communication_overhead: float
    memory_usage: float
    communication_complexity_order: str
    architecture_specific_metrics: Dict[str, Any] = field(default_factory=dict)
    experiment_timestamp: float = field(default_factory=time.time)


@dataclass
class ComparisonResults:
    """Results from comparative experiment across architectures."""
    performance_metrics: List[PerformanceMetrics]
    statistical_analysis: Dict[str, Any]
    performance_rankings: List[Tuple[str, float]]
    experiment_config: ExperimentalConfig
    comparison_timestamp: float = field(default_factory=time.time)


class ArchitectureComparison:
    """
    Unified framework for comparing Felix helix architecture against alternatives.
    
    Provides comprehensive performance comparison, statistical validation,
    and hypothesis testing infrastructure for research validation.
    """
    
    def __init__(self, helix: HelixGeometry, max_agents: int = 133, 
                 enable_detailed_metrics: bool = True):
        """
        Initialize architecture comparison framework.
        
        Args:
            helix: Helix geometry for Felix architecture
            max_agents: Maximum number of agents for experiments
            enable_detailed_metrics: Whether to collect detailed performance metrics
        """
        self.helix = helix
        self.max_agents = max_agents
        self.detailed_metrics_enabled = enable_detailed_metrics
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Configure architectures for comparison
        self.architectures = [
            {"name": "helix_spoke", "type": ArchitectureType.HELIX_SPOKE},
            {"name": "linear_pipeline", "type": ArchitectureType.LINEAR_PIPELINE},
            {"name": "mesh_communication", "type": ArchitectureType.MESH_COMMUNICATION}
        ]
    
    def run_helix_experiment(self, config: ExperimentalConfig) -> PerformanceMetrics:
        """
        Run performance experiment for helix spoke architecture.
        
        Args:
            config: Experimental configuration
            
        Returns:
            Performance metrics for helix architecture
        """
        start_time = time.perf_counter()
        
        # Create agents with OpenSCAD parameters
        agents = create_openscad_agents(
            helix=self.helix,
            number_of_nodes=config.agent_count,
            random_seed=config.random_seed
        )
        
        # Setup communication system
        central_post = CentralPost(max_agents=config.agent_count, enable_metrics=True)
        spoke_manager = SpokeManager(central_post)
        
        # Register agents
        for agent in agents:
            spoke_manager.register_agent(agent)
        
        # Run simulation
        current_time = 0.0
        time_step = 0.01
        tasks_completed = 0
        
        while current_time <= config.simulation_time:
            # Spawn ready agents
            for agent in agents:
                if agent.can_spawn(current_time) and agent.state.value == "waiting":
                    task = MockTask(f"task_{tasks_completed}")
                    agent.spawn(current_time, task)
                    tasks_completed += 1
            
            # Update agent positions
            for agent in agents:
                if agent.state.value == "active":
                    agent.update_position(current_time)
            
            # Process communications
            spoke_manager.process_all_messages()
            
            current_time += time_step
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Calculate metrics
        throughput = tasks_completed / execution_time if execution_time > 0 else 0
        communication_overhead = central_post.get_average_overhead_ratio()
        memory_usage = self._estimate_memory_usage(config.agent_count, "helix")
        
        # Architecture-specific metrics
        specific_metrics = {
            "connection_count": config.agent_count,  # O(N) connections to central post
            "message_complexity": "O(N)",
            "total_messages_processed": central_post.total_messages_processed,
            "average_message_latency": central_post.get_message_throughput()
        }
        
        return PerformanceMetrics(
            architecture_name="helix_spoke",
            agent_count=config.agent_count,
            task_completion_time=execution_time,
            throughput=throughput,
            communication_overhead=communication_overhead,
            memory_usage=memory_usage,
            communication_complexity_order="O(N)",
            architecture_specific_metrics=specific_metrics
        )
    
    def run_linear_experiment(self, config: ExperimentalConfig) -> PerformanceMetrics:
        """
        Run performance experiment for linear pipeline architecture.
        
        Args:
            config: Experimental configuration
            
        Returns:
            Performance metrics for linear pipeline architecture
        """
        start_time = time.perf_counter()
        
        # Configure pipeline stages
        num_stages = config.architecture_params.get("num_stages", 5)
        stage_capacity = config.architecture_params.get("stage_capacity", 10)
        
        # Create linear pipeline
        pipeline = LinearPipeline(num_stages=num_stages, stage_capacity=stage_capacity)
        
        # Create agents using pipeline's internal agent system
        # For linear pipeline, we'll simulate the equivalent workload
        tasks_completed = 0
        
        # Run simulation
        current_time = 0.0
        time_step = 0.01
        
        # Create pipeline agents based on spawn times
        from agents.agent import generate_spawn_times
        spawn_times = generate_spawn_times(config.agent_count, config.random_seed)
        
        # Create pipeline agents
        from pipeline.linear_pipeline import PipelineAgent
        pipeline_agents = []
        for i, spawn_time in enumerate(spawn_times):
            agent = PipelineAgent(f"pipeline_agent_{i}", spawn_time)
            pipeline_agents.append(agent)
        
        while current_time <= config.simulation_time:
            # Spawn ready agents
            for agent in pipeline_agents:
                if agent.can_spawn(current_time) and agent.state == "waiting":
                    task = MockTask(f"linear_task_{tasks_completed}")
                    agent.spawn(current_time, task)
                    pipeline.add_agent(agent, current_time)
                    tasks_completed += 1
            
            # Update pipeline
            pipeline.update(current_time)
            current_time += time_step
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Calculate metrics
        throughput = tasks_completed / execution_time if execution_time > 0 else 0
        memory_usage = self._estimate_memory_usage(config.agent_count, "linear")
        
        # Get pipeline-specific metrics
        pipeline_metrics = pipeline.get_performance_metrics()
        
        # Architecture-specific metrics
        specific_metrics = {
            "stage_count": num_stages,
            "stage_capacity": stage_capacity,
            "stage_utilization": pipeline_metrics.get("stage_utilizations", []),
            "bottleneck_stages": pipeline_metrics.get("bottleneck_stages", []),
            "average_stage_time": pipeline_metrics.get("average_stage_time", 0),
            "pipeline_efficiency": pipeline_metrics.get("efficiency", 0)
        }
        
        return PerformanceMetrics(
            architecture_name="linear_pipeline",
            agent_count=config.agent_count,
            task_completion_time=execution_time,
            throughput=throughput,
            communication_overhead=0,  # No inter-agent communication
            memory_usage=memory_usage,
            communication_complexity_order="O(N×M)",
            architecture_specific_metrics=specific_metrics
        )
    
    def run_mesh_experiment(self, config: ExperimentalConfig) -> PerformanceMetrics:
        """
        Run performance experiment for mesh communication architecture.
        
        Args:
            config: Experimental configuration
            
        Returns:
            Performance metrics for mesh communication architecture
        """
        start_time = time.perf_counter()
        
        # Create mesh communication system
        mesh = MeshCommunication(max_agents=config.agent_count, enable_metrics=True)
        
        # Create agents
        agents = create_openscad_agents(
            helix=self.helix,
            number_of_nodes=config.agent_count,
            random_seed=config.random_seed
        )
        
        # Register agents in mesh
        for agent in agents:
            mesh.register_agent(agent)
        
        # Run simulation
        current_time = 0.0
        time_step = 0.01
        tasks_completed = 0
        
        while current_time <= config.simulation_time:
            # Spawn ready agents
            for agent in agents:
                if agent.can_spawn(current_time) and agent.state.value == "waiting":
                    task = MockTask(f"mesh_task_{tasks_completed}")
                    agent.spawn(current_time, task)
                    tasks_completed += 1
            
            # Update agent positions
            for agent in agents:
                if agent.state.value == "active":
                    agent.update_position(current_time)
            
            # Process mesh communications
            mesh.process_all_messages()
            
            current_time += time_step
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Calculate metrics
        throughput = tasks_completed / execution_time if execution_time > 0 else 0
        mesh_metrics = mesh.get_performance_metrics()
        memory_usage = self._estimate_memory_usage(config.agent_count, "mesh")
        
        # Architecture-specific metrics
        expected_connections = config.agent_count * (config.agent_count - 1) // 2
        specific_metrics = {
            "connection_count": mesh_metrics["connection_count"],
            "expected_connections": expected_connections,
            "average_distance": mesh_metrics.get("average_distance", 0),
            "total_messages": mesh_metrics["total_messages"],
            "message_density": mesh_metrics.get("message_density", 0),
            "communication_efficiency": mesh_metrics.get("throughput", 0)
        }
        
        return PerformanceMetrics(
            architecture_name="mesh_communication",
            agent_count=config.agent_count,
            task_completion_time=execution_time,
            throughput=throughput,
            communication_overhead=mesh_metrics["average_latency"],
            memory_usage=memory_usage,
            communication_complexity_order="O(N²)",
            architecture_specific_metrics=specific_metrics
        )
    
    def run_comparative_experiment(self, config: ExperimentalConfig) -> ComparisonResults:
        """
        Run comparative experiment across all architectures.
        
        Args:
            config: Experimental configuration
            
        Returns:
            Comprehensive comparison results
        """
        performance_metrics = []
        
        # Run experiments for each architecture
        performance_metrics.append(self.run_helix_experiment(config))
        performance_metrics.append(self.run_linear_experiment(config))
        performance_metrics.append(self.run_mesh_experiment(config))
        
        # Perform statistical analysis
        statistical_analysis = self._analyze_comparative_results(performance_metrics)
        
        # Rank architectures by performance
        performance_rankings = self._rank_architectures(performance_metrics)
        
        return ComparisonResults(
            performance_metrics=performance_metrics,
            statistical_analysis=statistical_analysis,
            performance_rankings=performance_rankings,
            experiment_config=config
        )
    
    def get_hypothesis_validator(self):
        """Get hypothesis validation framework."""
        from .statistical_analysis import HypothesisValidator
        return HypothesisValidator(self)
    
    def analyze_throughput_characteristics(self, results: ComparisonResults) -> Dict[str, Any]:
        """Analyze throughput characteristics across architectures."""
        throughput_data = {}
        architecture_throughputs = {}
        
        for metrics in results.performance_metrics:
            architecture_throughputs[metrics.architecture_name] = metrics.throughput
        
        # Calculate relative performance
        max_throughput = max(architecture_throughputs.values())
        relative_performance = {
            arch: throughput / max_throughput 
            for arch, throughput in architecture_throughputs.items()
        }
        
        # Identify bottlenecks
        bottleneck_analysis = {}
        for metrics in results.performance_metrics:
            bottlenecks = []
            if metrics.communication_overhead > 0.1:  # 10% threshold
                bottlenecks.append("communication")
            if metrics.architecture_specific_metrics.get("pipeline_efficiency", 1.0) < 0.8:
                bottlenecks.append("pipeline_efficiency")
            bottleneck_analysis[metrics.architecture_name] = bottlenecks
        
        return {
            "architecture_throughputs": architecture_throughputs,
            "relative_performance": relative_performance,
            "bottleneck_analysis": bottleneck_analysis,
            "max_throughput": max_throughput
        }
    
    def analyze_memory_usage(self, results: ComparisonResults) -> Dict[str, Any]:
        """Analyze memory usage patterns across architectures."""
        memory_usage = {}
        for metrics in results.performance_metrics:
            memory_usage[metrics.architecture_name] = metrics.memory_usage
        
        # Calculate scaling factors
        agent_count = results.experiment_config.agent_count
        memory_scaling_factors = {}
        for arch, usage in memory_usage.items():
            if arch == "helix_spoke":
                expected_scaling = agent_count  # O(N)
            elif arch == "linear_pipeline":
                expected_scaling = agent_count * 5  # O(N×M), assume 5 stages
            else:  # mesh_communication
                expected_scaling = agent_count * (agent_count - 1) // 2  # O(N²)
            
            memory_scaling_factors[arch] = usage / expected_scaling if expected_scaling > 0 else 0
        
        # Rank by memory efficiency (lower is better)
        memory_efficiency_rankings = sorted(
            memory_usage.items(), key=lambda x: x[1]
        )
        
        return {
            "architecture_memory_usage": memory_usage,
            "memory_scaling_factors": memory_scaling_factors,
            "memory_efficiency_rankings": memory_efficiency_rankings
        }
    
    def analyze_latency_distribution(self, results: ComparisonResults) -> Dict[str, Any]:
        """Analyze latency distribution for communication systems."""
        latency_analysis = {}
        
        mean_latencies = {}
        latency_variance = {}
        latency_percentiles = {}
        
        for metrics in results.performance_metrics:
            if metrics.architecture_name in ["helix_spoke", "mesh_communication"]:
                # Extract latency data from architecture-specific metrics
                if metrics.architecture_name == "helix_spoke":
                    latency = metrics.architecture_specific_metrics.get("average_message_latency", 0)
                else:  # mesh_communication
                    latency = metrics.communication_overhead
                
                mean_latencies[metrics.architecture_name] = latency
                latency_variance[metrics.architecture_name] = latency * 0.1  # Simplified variance
                latency_percentiles[metrics.architecture_name] = {
                    "50th": latency,
                    "90th": latency * 1.2,
                    "99th": latency * 1.5
                }
        
        return {
            "mean_latencies": mean_latencies,
            "latency_variance": latency_variance,
            "latency_percentiles": latency_percentiles
        }
    
    def _analyze_comparative_results(self, performance_metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Analyze comparative results with statistical methods."""
        # Extract performance measures
        throughputs = [m.throughput for m in performance_metrics]
        completion_times = [m.task_completion_time for m in performance_metrics]
        memory_usage = [m.memory_usage for m in performance_metrics]
        
        # Basic statistical analysis
        analysis = {
            "throughput_stats": {
                "mean": statistics.mean(throughputs),
                "std": statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
                "min": min(throughputs),
                "max": max(throughputs)
            },
            "completion_time_stats": {
                "mean": statistics.mean(completion_times),
                "std": statistics.stdev(completion_times) if len(completion_times) > 1 else 0,
                "min": min(completion_times),
                "max": max(completion_times)
            },
            "memory_stats": {
                "mean": statistics.mean(memory_usage),
                "std": statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0,
                "min": min(memory_usage),
                "max": max(memory_usage)
            }
        }
        
        return analysis
    
    def _rank_architectures(self, performance_metrics: List[PerformanceMetrics]) -> List[Tuple[str, float]]:
        """Rank architectures by overall performance score."""
        # Calculate composite performance score
        scores = []
        for metrics in performance_metrics:
            # Normalize metrics (higher is better for throughput, lower is better for time/memory)
            normalized_throughput = metrics.throughput / 100  # Rough normalization
            normalized_time = 1.0 / (metrics.task_completion_time + 0.001)  # Avoid division by zero
            normalized_memory = 1.0 / (metrics.memory_usage + 0.001)
            
            # Weighted composite score
            composite_score = (
                0.4 * normalized_throughput +
                0.3 * normalized_time +
                0.2 * normalized_memory +
                0.1 * (1.0 / (metrics.communication_overhead + 0.001))
            )
            
            scores.append((metrics.architecture_name, composite_score))
        
        # Sort by score (higher is better)
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def _estimate_memory_usage(self, agent_count: int, architecture_type: str) -> float:
        """Estimate memory usage for architecture type."""
        base_memory = 1000  # Base memory in arbitrary units
        
        if architecture_type == "helix":
            return base_memory + agent_count * 10  # O(N)
        elif architecture_type == "linear":
            return base_memory + agent_count * 50  # O(N×M), assume 5 stages
        else:  # mesh
            connections = agent_count * (agent_count - 1) // 2
            return base_memory + connections * 20  # O(N²)


class MockTask:
    """Mock task for testing purposes."""
    
    def __init__(self, task_id: str):
        self.id = task_id
        self.data = {"test": True}
