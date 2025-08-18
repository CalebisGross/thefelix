"""
Test suite for agent lifecycle management.

Following test-first development: these tests define the expected behavior
of agent spawning, lifecycle states, and timing BEFORE implementation.

Tests validate agent behavior matching the OpenSCAD model from thefelix.md:
- Random spawn timing using seed 42069
- Agent progression along helix path
- State transitions (spawning -> active -> completed)
"""

import pytest
from unittest.mock import Mock
from src.agents.agent import Agent, AgentState
from src.core.helix_geometry import HelixGeometry


class TestAgentLifecycle:
    """Test agent lifecycle management and state transitions."""
    
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
    def mock_task(self):
        """Create a mock task for testing."""
        task = Mock()
        task.id = "test_task_001"
        task.data = {"text": "hello world"}
        return task
    
    def test_agent_initialization(self, standard_helix):
        """Test agent can be initialized with required parameters."""
        agent = Agent(
            agent_id="agent_001",
            spawn_time=0.5,
            helix=standard_helix
        )
        
        assert agent.agent_id == "agent_001"
        assert agent.spawn_time == 0.5
        assert agent.helix == standard_helix
        assert agent.state == AgentState.WAITING
        assert agent.current_task is None
        assert agent.current_position is None
    
    def test_agent_state_enum_values(self):
        """Test agent state enumeration has expected values."""
        assert AgentState.WAITING.value == "waiting"
        assert AgentState.SPAWNING.value == "spawning"
        assert AgentState.ACTIVE.value == "active"
        assert AgentState.COMPLETED.value == "completed"
        assert AgentState.FAILED.value == "failed"
    
    def test_agent_spawn_timing_validation(self, standard_helix):
        """Test agent spawn time validation."""
        # Valid spawn times should work
        Agent(agent_id="valid", spawn_time=0.0, helix=standard_helix)
        Agent(agent_id="valid", spawn_time=0.5, helix=standard_helix)
        Agent(agent_id="valid", spawn_time=1.0, helix=standard_helix)
        
        # Invalid spawn times should raise errors
        with pytest.raises(ValueError, match="spawn_time must be between 0 and 1"):
            Agent(agent_id="invalid", spawn_time=-0.1, helix=standard_helix)
        
        with pytest.raises(ValueError, match="spawn_time must be between 0 and 1"):
            Agent(agent_id="invalid", spawn_time=1.1, helix=standard_helix)
    
    def test_agent_can_spawn_at_current_time(self, standard_helix):
        """Test agent spawn readiness based on current time."""
        early_agent = Agent(agent_id="early", spawn_time=0.2, helix=standard_helix)
        late_agent = Agent(agent_id="late", spawn_time=0.8, helix=standard_helix)
        
        # At time 0.5, early agent should be ready, late agent should not
        assert early_agent.can_spawn(current_time=0.5) is True
        assert late_agent.can_spawn(current_time=0.5) is False
        
        # At time 0.9, both should be ready
        assert early_agent.can_spawn(current_time=0.9) is True
        assert late_agent.can_spawn(current_time=0.9) is True
        
        # At time 0.1, neither should be ready
        assert early_agent.can_spawn(current_time=0.1) is False
        assert late_agent.can_spawn(current_time=0.1) is False
    
    def test_agent_spawn_state_transition(self, standard_helix, mock_task):
        """Test agent spawning updates state and position."""
        agent = Agent(agent_id="spawner", spawn_time=0.3, helix=standard_helix)
        
        # Before spawning
        assert agent.state == AgentState.WAITING
        assert agent.current_position is None
        
        # Spawn the agent at time 0.5
        agent.spawn(current_time=0.5, task=mock_task)
        
        # After spawning
        assert agent.state == AgentState.ACTIVE
        assert agent.current_task == mock_task
        assert agent.current_position is not None
        
        # Position should be at the top of the helix (t=0) when spawning
        expected_position = standard_helix.get_position(0.0)
        assert agent.current_position == expected_position
    
    def test_agent_cannot_spawn_before_time(self, standard_helix, mock_task):
        """Test agent cannot spawn before its designated time."""
        agent = Agent(agent_id="future", spawn_time=0.7, helix=standard_helix)
        
        with pytest.raises(ValueError, match="Cannot spawn agent before spawn_time"):
            agent.spawn(current_time=0.5, task=mock_task)
    
    def test_agent_cannot_spawn_twice(self, standard_helix, mock_task):
        """Test agent cannot be spawned multiple times."""
        agent = Agent(agent_id="single", spawn_time=0.2, helix=standard_helix)
        
        # First spawn should work
        agent.spawn(current_time=0.5, task=mock_task)
        assert agent.state == AgentState.ACTIVE
        
        # Second spawn should fail
        with pytest.raises(ValueError, match="Agent already spawned"):
            agent.spawn(current_time=0.6, task=mock_task)
    
    def test_agent_position_updates_with_progress(self, standard_helix, mock_task):
        """Test agent position updates as it progresses along helix."""
        agent = Agent(agent_id="mover", spawn_time=0.1, helix=standard_helix)
        agent.spawn(current_time=0.3, task=mock_task)
        
        # Initial position at spawn
        initial_position = agent.current_position
        
        # Update position at later time
        agent.update_position(current_time=0.6)
        updated_position = agent.current_position
        
        # Position should have changed
        assert updated_position != initial_position
        
        # Should be further along the helix (higher t value)
        # Progress = time elapsed since spawn = 0.6 - 0.3 = 0.3
        expected_t = 0.6 - 0.3  # current_time - spawn_timestamp
        expected_position = standard_helix.get_position(expected_t)
        assert agent.current_position == expected_position
    
    def test_agent_completes_at_helix_end(self, standard_helix, mock_task):
        """Test agent completes when reaching end of helix."""
        agent = Agent(agent_id="completer", spawn_time=0.0, helix=standard_helix)
        agent.spawn(current_time=0.0, task=mock_task)
        
        # Move to end of helix (t=1.0)
        agent.update_position(current_time=1.0)
        
        # Agent should be completed
        assert agent.state == AgentState.COMPLETED
        assert agent.progress == 1.0
    
    def test_agent_progress_calculation(self, standard_helix, mock_task):
        """Test agent progress is calculated correctly."""
        agent = Agent(agent_id="progressor", spawn_time=0.2, helix=standard_helix)
        agent.spawn(current_time=0.4, task=mock_task)
        
        # At spawn: progress should be 0.0 (always starts at top)
        assert abs(agent.progress - 0.0) < 1e-10
        
        # Move forward: progress should be time elapsed since spawn = 0.8 - 0.4 = 0.4
        agent.update_position(current_time=0.8)
        assert abs(agent.progress - 0.4) < 1e-10
    
    def test_agent_task_assignment(self, standard_helix):
        """Test agent task assignment and tracking."""
        agent = Agent(agent_id="worker", spawn_time=0.1, helix=standard_helix)
        
        task1 = Mock()
        task1.id = "task_001"
        task1.data = {"text": "first task"}
        
        # Assign task during spawn
        agent.spawn(current_time=0.3, task=task1)
        assert agent.current_task == task1
        
        # Agent should track task ID
        assert agent.get_task_id() == "task_001"
    
    def test_agent_string_representation(self, standard_helix):
        """Test agent string representation for debugging."""
        agent = Agent(agent_id="debug_agent", spawn_time=0.5, helix=standard_helix)
        
        repr_str = str(agent)
        assert "debug_agent" in repr_str
        assert "spawn_time=0.5" in repr_str
        assert "waiting" in repr_str


class TestAgentRandomSpawning:
    """Test random agent spawning following OpenSCAD model."""
    
    def test_generate_spawn_times_with_seed(self):
        """Test generation of random spawn times matching OpenSCAD."""
        from src.agents.agent import generate_spawn_times
        
        # OpenSCAD parameters from thefelix.md
        number_of_nodes = 133
        random_seed = 42069
        
        spawn_times = generate_spawn_times(
            count=number_of_nodes,
            seed=random_seed
        )
        
        # Should generate exactly the requested count
        assert len(spawn_times) == number_of_nodes
        
        # All spawn times should be in valid range [0, 1]
        for spawn_time in spawn_times:
            assert 0.0 <= spawn_time <= 1.0
        
        # With same seed, should generate identical results
        spawn_times_2 = generate_spawn_times(
            count=number_of_nodes,
            seed=random_seed
        )
        assert spawn_times == spawn_times_2
    
    def test_spawn_times_distribution(self):
        """Test spawn times have reasonable distribution."""
        from src.agents.agent import generate_spawn_times
        
        spawn_times = generate_spawn_times(count=1000, seed=42069)
        
        # Should have values across the full range
        assert min(spawn_times) < 0.1
        assert max(spawn_times) > 0.9
        
        # Should be roughly uniform (simple distribution check)
        early_count = sum(1 for t in spawn_times if t < 0.5)
        late_count = sum(1 for t in spawn_times if t >= 0.5)
        
        # Should be roughly balanced (allow 20% variance)
        assert abs(early_count - late_count) < 200
    
    def test_different_seeds_produce_different_results(self):
        """Test different seeds produce different spawn time sequences."""
        from src.agents.agent import generate_spawn_times
        
        spawn_times_1 = generate_spawn_times(count=100, seed=12345)
        spawn_times_2 = generate_spawn_times(count=100, seed=54321)
        
        # Different seeds should produce different sequences
        assert spawn_times_1 != spawn_times_2


class TestAgentEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def standard_helix(self):
        """Create helix with OpenSCAD model parameters."""
        return HelixGeometry(
            top_radius=33.0,
            bottom_radius=0.001,
            height=33.0,
            turns=33
        )
    
    def test_agent_with_empty_id(self, standard_helix):
        """Test agent initialization with empty ID."""
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            Agent(agent_id="", spawn_time=0.5, helix=standard_helix)
    
    def test_agent_position_update_before_spawn(self, standard_helix):
        """Test position update before spawning should fail."""
        agent = Agent(agent_id="unspawned", spawn_time=0.5, helix=standard_helix)
        
        with pytest.raises(ValueError, match="Cannot update position of unspawned agent"):
            agent.update_position(current_time=0.7)
    
    def test_agent_with_spawn_time_at_boundaries(self, standard_helix):
        """Test agents with spawn times at exact boundaries."""
        # Spawn time exactly 0.0
        early_agent = Agent(agent_id="start", spawn_time=0.0, helix=standard_helix)
        assert early_agent.can_spawn(current_time=0.0) is True
        
        # Spawn time exactly 1.0
        late_agent = Agent(agent_id="end", spawn_time=1.0, helix=standard_helix)
        assert late_agent.can_spawn(current_time=1.0) is True
        assert late_agent.can_spawn(current_time=0.999) is False