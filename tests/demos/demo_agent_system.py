#!/usr/bin/env python3
"""
Demonstration of the Felix Framework agent system.

This script shows agents spawning and progressing along the helix
using the exact parameters from thefelix.md.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.helix_geometry import HelixGeometry
from src.agents.agent import create_openscad_agents


def demo_agent_system():
    """Demonstrate agent system with OpenSCAD parameters."""
    
    print("=== Felix Framework Agent System Demo ===\n")
    
    # Create helix with OpenSCAD parameters
    helix = HelixGeometry(
        top_radius=33.0,
        bottom_radius=0.001,
        height=33.0,
        turns=33
    )
    
    print(f"Helix configuration: {helix}")
    print(f"Arc length: {helix.approximate_arc_length():.1f} units\n")
    
    # Create agents with OpenSCAD random seed
    agents = create_openscad_agents(helix, number_of_nodes=10, random_seed=42069)
    
    print(f"Created {len(agents)} agents with spawn times:")
    for agent in agents[:5]:  # Show first 5
        print(f"  {agent.agent_id}: spawn_time={agent.spawn_time:.3f}")
    print(f"  ... and {len(agents)-5} more\n")
    
    # Simulate time progression
    time_steps = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for current_time in time_steps:
        print(f"=== Time {current_time:.2f} ===")
        
        # Count agent states
        waiting = spawned = active = completed = 0
        
        for agent in agents:
            if agent.can_spawn(current_time) and agent.state.value == "waiting":
                # Create mock task for demonstration
                class MockTask:
                    def __init__(self, task_id):
                        self.id = task_id
                
                task = MockTask(f"task_{agent.agent_id}")
                agent.spawn(current_time, task)
                spawned += 1
            
            if agent.state.value in ["active", "spawning"]:
                agent.update_position(current_time)
                active += 1
            elif agent.state.value == "waiting":
                waiting += 1
            elif agent.state.value == "completed":
                completed += 1
        
        print(f"Agents - Waiting: {waiting}, Active: {active}, Completed: {completed}")
        
        # Show detailed status for first few agents
        print("Sample agent positions:")
        for agent in agents[:3]:
            if agent.current_position:
                x, y, z = agent.current_position
                print(f"  {agent.agent_id}: pos=({x:.3f}, {y:.3f}, {z:.3f}) "
                      f"progress={agent.progress:.3f} state={agent.state.value}")
            else:
                print(f"  {agent.agent_id}: waiting (spawn_time={agent.spawn_time:.3f})")
        print()
    
    print("Demo complete! Agent system validated ✅")


if __name__ == "__main__":
    demo_agent_system()