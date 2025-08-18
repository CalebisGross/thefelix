"""
Agent lifecycle management for the Felix Framework.

This module implements autonomous agents that traverse the helical path,
matching the behavior specified in the OpenSCAD model from thefelix.md.

Mathematical Foundation:
- Agent spawn distribution: T_i ~ U(0,1) (uniform random timing)
- Position progression: r_i(τ) = r(T_i + (τ - T_i)) along helix path
- Workload distribution analysis supports Hypothesis H1 validation
- Agent density evolution: ρ(t,τ) for attention focusing (Hypothesis H3)

Key Features:
- Random spawn timing using configurable seeds (matches OpenSCAD rands() function)
- Helix path traversal with precise position tracking
- State machine for lifecycle management (WAITING → ACTIVE → COMPLETED)
- Task assignment and processing with performance measurement

Mathematical references:
- docs/mathematical_model.md, Section 4: Agent distribution functions and density evolution
- docs/hypothesis_mathematics.md, Section H1: Workload distribution statistical analysis
- docs/hypothesis_mathematics.md, Section H3: Attention focusing mechanism and agent density

Implementation supports testing of Hypotheses H1 (task distribution) and H3 (attention focusing).
"""

import random
from enum import Enum
from typing import Optional, List, Tuple, Any
from src.core.helix_geometry import HelixGeometry


class AgentState(Enum):
    """Agent lifecycle states."""
    WAITING = "waiting"      # Before spawn time
    SPAWNING = "spawning"    # Transitioning to active
    ACTIVE = "active"        # Processing along helix
    COMPLETED = "completed"  # Reached end of helix
    FAILED = "failed"        # Error condition


class Agent:
    """
    Autonomous agent that traverses the helical path.
    
    Agents spawn at random times and progress along the helix while
    processing assigned tasks. Position and state are tracked throughout
    the lifecycle.
    """
    
    def __init__(self, agent_id: str, spawn_time: float, helix: HelixGeometry):
        """
        Initialize agent with lifecycle parameters.
        
        Args:
            agent_id: Unique identifier for the agent
            spawn_time: Time when agent becomes active (0.0 to 1.0)
            helix: Helix geometry for path calculation
            
        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_initialization(agent_id, spawn_time)
        
        self.agent_id = agent_id
        self.spawn_time = spawn_time
        self.helix = helix
        self.state = AgentState.WAITING
        self.current_task: Optional[Any] = None
        self.current_position: Optional[Tuple[float, float, float]] = None
        self._progress: float = 0.0
        self._spawn_timestamp: Optional[float] = None
    
    def _validate_initialization(self, agent_id: str, spawn_time: float) -> None:
        """Validate agent initialization parameters."""
        if not agent_id or agent_id.strip() == "":
            raise ValueError("agent_id cannot be empty")
        
        if not (0.0 <= spawn_time <= 1.0):
            raise ValueError("spawn_time must be between 0 and 1")
    
    def can_spawn(self, current_time: float) -> bool:
        """
        Check if agent can spawn at the current time.
        
        Args:
            current_time: Current simulation time (0.0 to 1.0)
            
        Returns:
            True if agent can spawn, False otherwise
        """
        return current_time >= self.spawn_time
    
    def spawn(self, current_time: float, task: Any) -> None:
        """
        Spawn the agent and begin processing.
        
        Args:
            current_time: Current simulation time
            task: Task to assign to the agent
            
        Raises:
            ValueError: If spawn conditions are not met
        """
        if not self.can_spawn(current_time):
            raise ValueError("Cannot spawn agent before spawn_time")
        
        if self.state != AgentState.WAITING:
            raise ValueError("Agent already spawned")
        
        # Agents always start at the top of the helix (progress = 0)
        self._progress = 0.0
        self.state = AgentState.ACTIVE
        self.current_task = task
        self.current_position = self.helix.get_position(self._progress)
        self._spawn_timestamp = current_time  # Record when agent actually spawned
    
    def update_position(self, current_time: float) -> None:
        """
        Update agent position based on current time.
        
        Args:
            current_time: Current simulation time
            
        Raises:
            ValueError: If agent hasn't been spawned
        """
        if self.state == AgentState.WAITING:
            raise ValueError("Cannot update position of unspawned agent")
        
        if self.state in [AgentState.COMPLETED, AgentState.FAILED]:
            return  # No further updates needed
        
        # Calculate progression from when agent actually spawned
        if self._spawn_timestamp is None:
            raise ValueError("Cannot update position: agent has not spawned")
        
        progression_time = current_time - self._spawn_timestamp
        self._progress = min(progression_time, 1.0)  # Cap at 1.0
        
        # Update position
        self.current_position = self.helix.get_position(self._progress)
        
        # Check for completion
        if self._progress >= 1.0:
            self.state = AgentState.COMPLETED
    
    @property
    def progress(self) -> float:
        """Get current progress along helix (0.0 to 1.0)."""
        return self._progress
    
    def get_task_id(self) -> Optional[str]:
        """Get ID of current task, if any."""
        if self.current_task and hasattr(self.current_task, 'id'):
            return self.current_task.id
        return None
    
    def __str__(self) -> str:
        """String representation for debugging."""
        return (f"Agent(id={self.agent_id}, spawn_time={self.spawn_time}, "
                f"state={self.state.value}, progress={self._progress:.3f})")
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return str(self)


def generate_spawn_times(count: int, seed: Optional[int] = None) -> List[float]:
    """
    Generate random spawn times matching OpenSCAD model.
    
    Replicates the OpenSCAD function:
    node_start_times = rands(0, 1, number_of_nodes, random_seed);
    
    Args:
        count: Number of spawn times to generate
        seed: Random seed for reproducibility (matches OpenSCAD seed)
        
    Returns:
        List of spawn times in range [0.0, 1.0]
    """
    if seed is not None:
        random.seed(seed)
    
    return [random.random() for _ in range(count)]


def create_agents_from_spawn_times(spawn_times: List[float], 
                                   helix: HelixGeometry) -> List[Agent]:
    """
    Create agent instances from spawn time list.
    
    Args:
        spawn_times: List of spawn times for agents
        helix: Helix geometry for agent path calculation
        
    Returns:
        List of initialized Agent instances
    """
    agents = []
    for i, spawn_time in enumerate(spawn_times):
        agent_id = f"agent_{i:03d}"
        agent = Agent(agent_id=agent_id, spawn_time=spawn_time, helix=helix)
        agents.append(agent)
    
    return agents


def create_openscad_agents(helix: HelixGeometry, 
                          number_of_nodes: int = 133,
                          random_seed: int = 42069) -> List[Agent]:
    """
    Create agents matching OpenSCAD model parameters.
    
    Uses the exact parameters from thefelix.md:
    - number_of_nodes = 133
    - random_seed = 42069
    
    Args:
        helix: Helix geometry for agent paths
        number_of_nodes: Number of agents to create
        random_seed: Random seed for spawn time generation
        
    Returns:
        List of Agent instances with OpenSCAD-compatible spawn times
    """
    spawn_times = generate_spawn_times(count=number_of_nodes, seed=random_seed)
    return create_agents_from_spawn_times(spawn_times, helix)