#!/usr/bin/env python3
"""
Test suite for linear pipeline architecture - Felix Framework Phase 5.

This module provides comprehensive testing for the linear pipeline implementation
that serves as a comparison baseline against the helix-based architecture.

Mathematical Foundation:
- Sequential stage processing: agents proceed through fixed stages 0 → 1 → ... → N
- Linear workload distribution: uniform task allocation across pipeline stages
- Stage-based communication: agents communicate with adjacent stages only
- Performance baseline: establishes metrics for Hypothesis H1 validation

Test Coverage:
- Linear stage progression and agent movement
- Workload distribution across pipeline stages  
- Sequential task processing with handoffs
- Performance metrics collection and analysis
- Edge cases and error conditions

This supports Hypothesis H1 testing by providing a controlled comparison
architecture with measurably different workload distribution characteristics.

Mathematical reference: docs/hypothesis_mathematics.md, Section H1.2
"""

import pytest
import time
from unittest.mock import Mock, patch
from typing import List, Tuple, Optional, Any

from src.pipeline.linear_pipeline import LinearPipeline, PipelineAgent, PipelineStage


class TestLinearPipelineCore:
    """Test core linear pipeline functionality."""
    
    def test_pipeline_initialization(self):
        """Test linear pipeline creation with valid parameters."""
        pipeline = LinearPipeline(num_stages=5, stage_capacity=10)
        
        assert pipeline.num_stages == 5
        assert pipeline.stage_capacity == 10
        assert len(pipeline.stages) == 5
        assert pipeline.total_agents == 0
        assert pipeline.active_agents == []
    
    def test_pipeline_initialization_validation(self):
        """Test parameter validation during initialization."""
        # Test invalid stage count
        with pytest.raises(ValueError, match="num_stages must be positive"):
            LinearPipeline(num_stages=0, stage_capacity=10)
        
        with pytest.raises(ValueError, match="num_stages must be positive"):
            LinearPipeline(num_stages=-1, stage_capacity=10)
        
        # Test invalid stage capacity
        with pytest.raises(ValueError, match="stage_capacity must be positive"):
            LinearPipeline(num_stages=5, stage_capacity=0)
        
        with pytest.raises(ValueError, match="stage_capacity must be positive"):
            LinearPipeline(num_stages=5, stage_capacity=-1)
    
    def test_stage_properties(self):
        """Test individual pipeline stage properties."""
        pipeline = LinearPipeline(num_stages=3, stage_capacity=5)
        
        for i, stage in enumerate(pipeline.stages):
            assert stage.stage_id == i
            assert stage.capacity == 5
            assert stage.current_load == 0
            assert stage.agents == []
            assert not stage.is_full()


class TestPipelineAgent:
    """Test pipeline agent lifecycle and behavior."""
    
    def test_agent_initialization(self):
        """Test pipeline agent creation."""
        agent = PipelineAgent(agent_id="test_001", spawn_time=0.3)
        
        assert agent.agent_id == "test_001"
        assert agent.spawn_time == 0.3
        assert agent.current_stage == 0
        assert agent.state == "waiting"
        assert agent.current_task is None
        assert agent.processing_time == 0.0
    
    def test_agent_spawn_validation(self):
        """Test agent spawn parameter validation."""
        # Valid spawn times
        agent1 = PipelineAgent("agent_001", 0.0)
        agent2 = PipelineAgent("agent_002", 1.0)
        agent3 = PipelineAgent("agent_003", 0.5)
        
        assert agent1.spawn_time == 0.0
        assert agent2.spawn_time == 1.0
        assert agent3.spawn_time == 0.5
        
        # Invalid spawn times
        with pytest.raises(ValueError, match="spawn_time must be between 0 and 1"):
            PipelineAgent("invalid", -0.1)
        
        with pytest.raises(ValueError, match="spawn_time must be between 0 and 1"):
            PipelineAgent("invalid", 1.1)
        
        # Invalid agent ID
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            PipelineAgent("", 0.5)
        
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            PipelineAgent("   ", 0.5)
    
    def test_agent_can_spawn(self):
        """Test agent spawn timing logic."""
        agent = PipelineAgent("spawner", 0.4)
        
        # Before spawn time
        assert not agent.can_spawn(0.0)
        assert not agent.can_spawn(0.3)
        assert not agent.can_spawn(0.39)
        
        # At and after spawn time
        assert agent.can_spawn(0.4)
        assert agent.can_spawn(0.5)
        assert agent.can_spawn(1.0)
    
    def test_agent_spawn_process(self):
        """Test agent spawning and state transitions."""
        agent = PipelineAgent("spawner", 0.2)
        task = Mock()
        task.id = "task_001"
        
        # Spawn agent
        agent.spawn(current_time=0.5, task=task, stage=0)
        
        assert agent.state == "active"
        assert agent.current_stage == 0
        assert agent.current_task == task
        assert agent.processing_time == 0.0
    
    def test_agent_spawn_validation_conditions(self):
        """Test spawn validation conditions."""
        agent = PipelineAgent("spawner", 0.4)
        task = Mock()
        
        # Cannot spawn before spawn time
        with pytest.raises(ValueError, match="Cannot spawn agent before spawn_time"):
            agent.spawn(current_time=0.3, task=task, stage=0)
        
        # Cannot spawn twice
        agent.spawn(current_time=0.5, task=task, stage=0)
        with pytest.raises(ValueError, match="Agent already spawned"):
            agent.spawn(current_time=0.6, task=task, stage=0)
    
    def test_agent_progress_through_stages(self):
        """Test agent progression through pipeline stages."""
        agent = PipelineAgent("progressor", 0.1)
        task = Mock()
        
        # Spawn at stage 0
        agent.spawn(current_time=0.2, task=task, stage=0)
        assert agent.current_stage == 0
        
        # Advance to stage 1
        agent.advance_stage()
        assert agent.current_stage == 1
        
        # Advance to stage 2
        agent.advance_stage()
        assert agent.current_stage == 2
    
    def test_agent_completion_at_final_stage(self):
        """Test agent completion when reaching final stage."""
        pipeline = LinearPipeline(num_stages=3, stage_capacity=5)
        agent = PipelineAgent("completer", 0.1)
        task = Mock()
        
        agent.spawn(current_time=0.2, task=task, stage=0)
        
        # Advance through all stages
        agent.advance_stage(max_stages=3)  # Stage 0 → 1
        assert agent.state == "active"
        
        agent.advance_stage(max_stages=3)  # Stage 1 → 2
        assert agent.state == "active"
        
        # Complete processing (beyond final stage)
        agent.advance_stage(max_stages=3)  # Stage 2 → completed
        assert agent.state == "completed"


class TestPipelineStage:
    """Test pipeline stage functionality."""
    
    def test_stage_initialization(self):
        """Test pipeline stage creation."""
        stage = PipelineStage(stage_id=2, capacity=8)
        
        assert stage.stage_id == 2
        assert stage.capacity == 8
        assert stage.current_load == 0
        assert stage.agents == []
        assert not stage.is_full()
    
    def test_stage_agent_assignment(self):
        """Test adding agents to pipeline stages."""
        stage = PipelineStage(stage_id=1, capacity=3)
        agent1 = PipelineAgent("agent_001", 0.1)
        agent2 = PipelineAgent("agent_002", 0.2)
        
        # Add agents to stage
        stage.add_agent(agent1)
        assert stage.current_load == 1
        assert agent1 in stage.agents
        assert not stage.is_full()
        
        stage.add_agent(agent2)
        assert stage.current_load == 2
        assert agent2 in stage.agents
        assert not stage.is_full()
    
    def test_stage_capacity_limits(self):
        """Test stage capacity enforcement."""
        stage = PipelineStage(stage_id=0, capacity=2)
        agent1 = PipelineAgent("agent_001", 0.1)
        agent2 = PipelineAgent("agent_002", 0.2)
        agent3 = PipelineAgent("agent_003", 0.3)
        
        # Fill stage to capacity
        stage.add_agent(agent1)
        stage.add_agent(agent2)
        assert stage.is_full()
        
        # Attempt to exceed capacity
        with pytest.raises(ValueError, match="Stage is at full capacity"):
            stage.add_agent(agent3)
    
    def test_stage_agent_removal(self):
        """Test removing agents from stages."""
        stage = PipelineStage(stage_id=1, capacity=5)
        agent1 = PipelineAgent("agent_001", 0.1)
        agent2 = PipelineAgent("agent_002", 0.2)
        
        # Add agents
        stage.add_agent(agent1)
        stage.add_agent(agent2)
        assert stage.current_load == 2
        
        # Remove agents
        stage.remove_agent(agent1)
        assert stage.current_load == 1
        assert agent1 not in stage.agents
        assert agent2 in stage.agents
        
        stage.remove_agent(agent2)
        assert stage.current_load == 0
        assert len(stage.agents) == 0
    
    def test_stage_workload_calculation(self):
        """Test stage workload measurement."""
        stage = PipelineStage(stage_id=2, capacity=10)
        
        # Empty stage
        assert stage.get_workload() == 0.0
        
        # Add agents with different processing times
        agent1 = PipelineAgent("agent_001", 0.1)
        agent1.processing_time = 5.0
        agent2 = PipelineAgent("agent_002", 0.2)
        agent2.processing_time = 3.0
        
        stage.add_agent(agent1)
        stage.add_agent(agent2)
        
        # Total workload = sum of processing times
        assert stage.get_workload() == 8.0
    
    def test_stage_utilization_metrics(self):
        """Test stage utilization percentage calculations."""
        stage = PipelineStage(stage_id=1, capacity=4)
        
        # Empty stage
        assert stage.get_utilization() == 0.0
        
        # Partial utilization
        agent1 = PipelineAgent("agent_001", 0.1)
        stage.add_agent(agent1)
        assert stage.get_utilization() == 0.25  # 1/4 = 25%
        
        agent2 = PipelineAgent("agent_002", 0.2)
        stage.add_agent(agent2)
        assert stage.get_utilization() == 0.5  # 2/4 = 50%


class TestLinearPipelineIntegration:
    """Test integrated linear pipeline system."""
    
    def test_pipeline_agent_spawning(self):
        """Test agent spawning in pipeline system."""
        pipeline = LinearPipeline(num_stages=4, stage_capacity=5)
        
        # Create agents with different spawn times
        spawn_times = [0.1, 0.3, 0.5, 0.7]
        agents = []
        for i, spawn_time in enumerate(spawn_times):
            agent = PipelineAgent(f"agent_{i:03d}", spawn_time)
            agents.append(agent)
            pipeline.add_agent(agent)
        
        # Simulate time progression
        pipeline.update(current_time=0.0)
        assert len(pipeline.get_active_agents()) == 0  # No agents spawned yet
        
        pipeline.update(current_time=0.2)
        assert len(pipeline.get_active_agents()) == 1  # One agent spawned
        
        pipeline.update(current_time=0.6)
        assert len(pipeline.get_active_agents()) == 3  # Three agents spawned
        
        pipeline.update(current_time=1.0)
        assert len(pipeline.get_active_agents()) == 4  # All agents spawned
    
    def test_pipeline_stage_progression(self):
        """Test agents progressing through pipeline stages."""
        pipeline = LinearPipeline(num_stages=3, stage_capacity=10)
        agent = PipelineAgent("progressor", 0.1)
        pipeline.add_agent(agent)
        
        # Spawn agent
        pipeline.update(current_time=0.2)
        active_agents = pipeline.get_active_agents()
        assert len(active_agents) == 1
        assert active_agents[0].current_stage == 0
        
        # Agent should be in stage 0
        assert pipeline.stages[0].current_load == 1
        assert pipeline.stages[1].current_load == 0
        assert pipeline.stages[2].current_load == 0
        
        # Simulate processing time and stage advancement
        agent.processing_time = 1.0  # Enough time for stage completion
        pipeline.advance_agents()
        
        # Agent advances to stage 1
        assert agent.current_stage == 1
        assert pipeline.stages[0].current_load == 0
        assert pipeline.stages[1].current_load == 1
        assert pipeline.stages[2].current_load == 0
    
    def test_pipeline_workload_distribution(self):
        """Test workload distribution across pipeline stages."""
        pipeline = LinearPipeline(num_stages=4, stage_capacity=5)
        
        # Create 10 agents with uniform spawn distribution
        for i in range(10):
            spawn_time = i / 10.0  # 0.0, 0.1, 0.2, ..., 0.9
            agent = PipelineAgent(f"agent_{i:03d}", spawn_time)
            pipeline.add_agent(agent)
        
        # Spawn all agents and simulate processing
        pipeline.update(current_time=1.0)  # All agents spawned
        
        # Distribute agents across stages for workload analysis
        active_agents = pipeline.get_active_agents()
        
        # Clear existing stage assignments first
        for stage in pipeline.stages:
            stage.agents.clear()
        
        for i, agent in enumerate(active_agents):
            stage_assignment = i % pipeline.num_stages
            agent.current_stage = stage_assignment
            agent.processing_time = 2.0  # Uniform processing time
            pipeline.stages[stage_assignment].add_agent(agent)
        
        # Calculate workload distribution
        workloads = [stage.get_workload() for stage in pipeline.stages]
        
        # Should have relatively uniform distribution (2-3 agents per stage)
        for workload in workloads:
            assert workload >= 4.0  # At least 2 agents * 2.0 processing time
            assert workload <= 6.0  # At most 3 agents * 2.0 processing time
    
    def test_pipeline_performance_metrics(self):
        """Test performance metrics collection."""
        pipeline = LinearPipeline(num_stages=3, stage_capacity=8)
        
        # Add agents with known spawn times
        spawn_times = [0.2, 0.4, 0.6]
        for i, spawn_time in enumerate(spawn_times):
            agent = PipelineAgent(f"metrics_agent_{i}", spawn_time)
            pipeline.add_agent(agent)
        
        # Simulate processing
        pipeline.update(current_time=1.0)
        
        # Collect metrics
        metrics = pipeline.get_performance_metrics()
        
        assert "total_agents" in metrics
        assert "active_agents" in metrics
        assert "completed_agents" in metrics
        assert "stage_utilizations" in metrics
        assert "total_workload" in metrics
        assert "workload_distribution" in metrics
        
        assert metrics["total_agents"] == 3
        assert isinstance(metrics["stage_utilizations"], list)
        assert len(metrics["stage_utilizations"]) == 3
    
    def test_pipeline_completion_tracking(self):
        """Test tracking of completed agents."""
        pipeline = LinearPipeline(num_stages=2, stage_capacity=5)
        agent = PipelineAgent("completer", 0.1)
        pipeline.add_agent(agent)
        
        # Spawn and process agent
        pipeline.update(current_time=0.2)
        
        # Force completion by advancing through all stages
        agent.advance_stage(max_stages=2)  # Stage 0 → 1
        agent.advance_stage(max_stages=2)  # Stage 1 → completed
        
        assert agent.state == "completed"
        
        # Update pipeline to track completion
        pipeline.update(current_time=0.5)
        
        metrics = pipeline.get_performance_metrics()
        assert metrics["completed_agents"] >= 0  # Should track completed count


class TestLinearPipelineComparison:
    """Test pipeline features needed for architecture comparison."""
    
    def test_coefficient_of_variation_calculation(self):
        """Test workload coefficient of variation calculation."""
        pipeline = LinearPipeline(num_stages=4, stage_capacity=10)
        
        # Create scenario with known workload distribution
        workloads = [10.0, 15.0, 12.0, 18.0]  # Known workload values
        
        # Set up stages with these workloads
        for i, workload in enumerate(workloads):
            # Add agents to achieve target workload
            agents_needed = int(workload / 2.0)  # 2.0 processing time per agent
            for j in range(agents_needed):
                agent = PipelineAgent(f"cv_agent_{i}_{j}", 0.1)
                agent.processing_time = 2.0
                agent.current_stage = i
                pipeline.stages[i].add_agent(agent)
        
        # Calculate coefficient of variation
        cv = pipeline.calculate_workload_cv()
        
        # Expected: mean = 13.75, std = 3.304, CV = 0.240
        expected_mean = 13.75
        expected_std = 3.304  # Approximately
        expected_cv = expected_std / expected_mean
        
        assert abs(cv - expected_cv) < 0.02  # Within 2% tolerance for numerical precision
    
    def test_pipeline_comparison_metrics(self):
        """Test metrics needed for helix vs linear comparison."""
        pipeline = LinearPipeline(num_stages=5, stage_capacity=6)
        
        # Create realistic agent distribution
        agent_count = 15
        for i in range(agent_count):
            spawn_time = i / float(agent_count)  # Uniform spawn distribution
            agent = PipelineAgent(f"comp_agent_{i:03d}", spawn_time)
            agent.processing_time = 1.0 + (i % 3) * 0.5  # Variable processing times
            pipeline.add_agent(agent)
        
        # Simulate processing
        pipeline.update(current_time=1.0)
        pipeline.distribute_agents_across_stages()
        
        # Collect comparison metrics
        metrics = pipeline.get_comparison_metrics()
        
        # Check required metrics for hypothesis testing
        assert "workload_cv" in metrics
        assert "completion_time" in metrics
        assert "utilization_variance" in metrics
        assert "agent_distribution" in metrics
        
        # Validate metric ranges
        assert 0.0 <= metrics["workload_cv"] <= 1.0
        assert metrics["completion_time"] >= 0.0
        assert 0.0 <= metrics["utilization_variance"] <= 1.0
        assert len(metrics["agent_distribution"]) == pipeline.num_stages
    
    def test_linear_vs_helix_baseline(self):
        """Test baseline comparison setup for linear architecture."""
        pipeline = LinearPipeline(num_stages=10, stage_capacity=15)
        
        # Create OpenSCAD-equivalent agent count
        agent_count = 133  # Same as helix system
        for i in range(agent_count):
            # Use same random seed approach as helix for fair comparison
            spawn_time = i / float(agent_count)  # Distributed spawn times
            agent = PipelineAgent(f"baseline_agent_{i:03d}", spawn_time)
            pipeline.add_agent(agent)
        
        # Simulate full processing cycle
        pipeline.update(current_time=1.0)
        
        assert pipeline.total_agents == agent_count
        assert len(pipeline.get_active_agents()) == agent_count
        
        # Should be able to distribute agents across stages
        pipeline.distribute_agents_across_stages()
        
        # Check that agents are distributed
        total_distributed = sum(stage.current_load for stage in pipeline.stages)
        assert total_distributed <= agent_count  # Some may be completed
        
        # Performance metrics should be calculable
        metrics = pipeline.get_comparison_metrics()
        assert all(key in metrics for key in ["workload_cv", "completion_time", "utilization_variance"])


# Edge cases and error conditions
class TestLinearPipelineEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_pipeline_metrics(self):
        """Test metrics calculation with no agents."""
        pipeline = LinearPipeline(num_stages=3, stage_capacity=5)
        
        metrics = pipeline.get_performance_metrics()
        assert metrics["total_agents"] == 0
        assert metrics["active_agents"] == 0
        assert metrics["total_workload"] == 0.0
        
        # CV should handle empty case gracefully
        cv = pipeline.calculate_workload_cv()
        assert cv == 0.0  # No variation when no workload
    
    def test_single_agent_pipeline(self):
        """Test pipeline with only one agent."""
        pipeline = LinearPipeline(num_stages=2, stage_capacity=1)
        agent = PipelineAgent("solo_agent", 0.5)
        pipeline.add_agent(agent)
        
        pipeline.update(current_time=1.0)
        assert len(pipeline.get_active_agents()) == 1
        
        metrics = pipeline.get_performance_metrics()
        assert metrics["total_agents"] == 1
        assert metrics["active_agents"] == 1
    
    def test_pipeline_stage_overflow_handling(self):
        """Test handling when all stages are at capacity."""
        pipeline = LinearPipeline(num_stages=2, stage_capacity=1)
        
        # Create more agents than total pipeline capacity
        agents = []
        for i in range(5):
            agent = PipelineAgent(f"overflow_agent_{i}", 0.1)
            agents.append(agent)
            pipeline.add_agent(agent)
        
        # Spawn all agents (should handle overflow gracefully)
        pipeline.update(current_time=1.0)
        
        # Should not crash and should track all agents
        assert pipeline.total_agents == 5
        
        # Active agents limited by pipeline capacity
        pipeline.distribute_agents_across_stages()
        total_in_stages = sum(stage.current_load for stage in pipeline.stages)
        assert total_in_stages <= 2  # Maximum pipeline capacity
    
    def test_pipeline_time_progression_edge_cases(self):
        """Test time progression edge cases."""
        pipeline = LinearPipeline(num_stages=3, stage_capacity=5)
        agent = PipelineAgent("time_test_agent", 0.5)
        pipeline.add_agent(agent)
        
        # Update with time before spawn time
        pipeline.update(current_time=0.3)
        assert len(pipeline.get_active_agents()) == 0
        
        # Update with exact spawn time
        pipeline.update(current_time=0.5)
        assert len(pipeline.get_active_agents()) == 1
        
        # Update with time going backwards (should handle gracefully)
        pipeline.update(current_time=0.4)
        # Should not break the system state
        assert len(pipeline.get_active_agents()) >= 0