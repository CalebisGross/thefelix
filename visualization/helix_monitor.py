#!/usr/bin/env python3
"""
Real-time Helix Visualization for Felix Framework.

This module provides visualization tools for monitoring agent movement
and communication in the Felix helix-based multi-agent system.

Features:
- Real-time 3D helix visualization
- Agent position tracking with color coding
- Spoke communication visualization
- Terminal-based ASCII rendering for compatibility
- Web-based 3D rendering (optional, if matplotlib available)

Usage:
    python visualization/helix_monitor.py --mode terminal
    python visualization/helix_monitor.py --mode web
"""

import sys
import time
import math
import argparse
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.helix_geometry import HelixGeometry


@dataclass
class AgentVisualization:
    """Visual representation of an agent."""
    agent_id: str
    agent_type: str
    position: Tuple[float, float, float]
    state: str
    spawn_time: float
    progress: float
    color_code: str = "white"


@dataclass 
class SpokeVisualization:
    """Visual representation of a spoke connection."""
    agent_id: str
    agent_position: Tuple[float, float, float]
    central_position: Tuple[float, float, float]
    message_count: int = 0
    last_activity: float = 0.0


class TerminalHelixVisualizer:
    """
    ASCII-based terminal visualization of the helix system.
    
    Provides a real-time view of agent positions and movement
    using terminal graphics for broad compatibility.
    """
    
    def __init__(self, helix: HelixGeometry, width: int = 80, height: int = 24):
        """
        Initialize terminal visualizer.
        
        Args:
            helix: Helix geometry to visualize
            width: Terminal width in characters
            height: Terminal height in characters
        """
        self.helix = helix
        self.width = width
        self.height = height
        
        # Color codes for different agent types
        self.agent_colors = {
            "research": "\033[94m",    # Blue
            "analysis": "\033[93m",    # Yellow  
            "synthesis": "\033[92m",   # Green
            "critic": "\033[95m",      # Magenta
            "general": "\033[97m"      # White
        }
        self.reset_color = "\033[0m"
        
        # Visualization state
        self.agents: List[AgentVisualization] = []
        self.spokes: List[SpokeVisualization] = []
        self.frame_count = 0
    
    def update_agents(self, agent_data: List[Dict[str, Any]]) -> None:
        """Update agent positions for visualization."""
        self.agents = []
        
        for data in agent_data:
            if "position" in data and data["position"]:
                agent_viz = AgentVisualization(
                    agent_id=data["agent_id"],
                    agent_type=data.get("agent_type", "general"),
                    position=data["position"],
                    state=data.get("state", "unknown"),
                    spawn_time=data.get("spawn_time", 0.0),
                    progress=data.get("progress", 0.0)
                )
                agent_viz.color_code = self.agent_colors.get(agent_viz.agent_type, self.agent_colors["general"])
                self.agents.append(agent_viz)
    
    def update_spokes(self, spoke_data: List[Dict[str, Any]]) -> None:
        """Update spoke communication data."""
        self.spokes = []
        
        for data in spoke_data:
            if "agent_position" in data:
                spoke_viz = SpokeVisualization(
                    agent_id=data["agent_id"],
                    agent_position=data["agent_position"],
                    central_position=(0, 0, data["agent_position"][2]),  # Central post at same Z
                    message_count=data.get("message_count", 0),
                    last_activity=data.get("last_activity", 0.0)
                )
                self.spokes.append(spoke_viz)
    
    def project_3d_to_2d(self, position: Tuple[float, float, float]) -> Tuple[int, int]:
        """
        Project 3D helix position to 2D terminal coordinates.
        
        Args:
            position: (x, y, z) position in 3D space
            
        Returns:
            (col, row) terminal coordinates
        """
        x, y, z = position
        
        # Use isometric-style projection
        # Map X and Y to terminal columns, Z to terminal rows
        
        # Scale to fit terminal
        scale_x = (self.width - 10) / (2 * self.helix.top_radius)
        scale_y = (self.height - 5) / self.helix.height
        
        # Project to terminal coordinates
        col = int((x + self.helix.top_radius) * scale_x + 5)
        row = int((self.helix.height - z) * scale_y + 2)
        
        # Apply Y offset for pseudo-3D effect
        y_offset = int(y * 0.1)  # Small Y-axis depth effect
        col += y_offset
        
        # Clamp to terminal bounds
        col = max(0, min(self.width - 1, col))
        row = max(0, min(self.height - 1, row))
        
        return col, row
    
    def render_frame(self, current_time: float) -> str:
        """
        Render current frame as ASCII art.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            ASCII frame as string
        """
        # Create empty frame buffer
        frame = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw helix structure (simplified)
        self._draw_helix_structure(frame)
        
        # Draw central post
        self._draw_central_post(frame)
        
        # Draw spokes
        self._draw_spokes(frame)
        
        # Draw agents
        self._draw_agents(frame)
        
        # Add UI elements
        self._add_ui_elements(frame, current_time)
        
        # Convert to string
        frame_str = '\n'.join(''.join(row) for row in frame)
        
        self.frame_count += 1
        return frame_str
    
    def _draw_helix_structure(self, frame: List[List[str]]) -> None:
        """Draw helix wireframe."""
        # Draw helix path at several Z levels
        num_levels = 12
        points_per_level = 20
        
        for level in range(num_levels):
            z = (level / (num_levels - 1)) * self.helix.height
            radius = self.helix.get_radius(z)
            
            for point in range(points_per_level):
                angle = (point / points_per_level) * 2 * math.pi + (z / self.helix.height) * self.helix.turns * 2 * math.pi
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                
                col, row = self.project_3d_to_2d((x, y, z))
                if 0 <= row < len(frame) and 0 <= col < len(frame[0]):
                    frame[row][col] = '·'
    
    def _draw_central_post(self, frame: List[List[str]]) -> None:
        """Draw central coordination post."""
        # Draw vertical line at center
        center_col = self.width // 2
        
        for row in range(2, self.height - 3):
            if 0 <= center_col < len(frame[0]):
                frame[row][center_col] = '│'
    
    def _draw_spokes(self, frame: List[List[str]]) -> None:
        """Draw spoke connections."""
        for spoke in self.spokes:
            agent_col, agent_row = self.project_3d_to_2d(spoke.agent_position)
            central_col, central_row = self.project_3d_to_2d(spoke.central_position)
            
            # Draw line between agent and center (simplified)
            if spoke.message_count > 0:
                # Active spoke
                char = '═' if abs(agent_col - central_col) > abs(agent_row - central_row) else '║'
            else:
                # Inactive spoke
                char = '-' if abs(agent_col - central_col) > abs(agent_row - central_row) else '|'
            
            # Simple line drawing (midpoint)
            mid_col = (agent_col + central_col) // 2
            mid_row = (agent_row + central_row) // 2
            
            if 0 <= mid_row < len(frame) and 0 <= mid_col < len(frame[0]):
                frame[mid_row][mid_col] = char
    
    def _draw_agents(self, frame: List[List[str]]) -> None:
        """Draw agent positions."""
        for agent in self.agents:
            col, row = self.project_3d_to_2d(agent.position)
            
            if 0 <= row < len(frame) and 0 <= col < len(frame[0]):
                # Agent character based on type
                char_map = {
                    "research": "R",
                    "analysis": "A", 
                    "synthesis": "S",
                    "critic": "C",
                    "general": "●"
                }
                char = char_map.get(agent.agent_type, "●")
                
                # Add color if terminal supports it
                colored_char = f"{agent.color_code}{char}{self.reset_color}"
                frame[row][col] = colored_char
    
    def _add_ui_elements(self, frame: List[List[str]], current_time: float) -> None:
        """Add UI text elements."""
        # Title
        title = f"Felix Framework - Helix Visualization (Frame {self.frame_count})"
        if len(title) < self.width:
            title_start = (self.width - len(title)) // 2
            for i, char in enumerate(title):
                if title_start + i < len(frame[0]):
                    frame[0][title_start + i] = char
        
        # Time display
        time_str = f"Time: {current_time:.2f}"
        for i, char in enumerate(time_str):
            if i < len(frame[0]):
                frame[1][i] = char
        
        # Agent count
        agent_count_str = f"Agents: {len(self.agents)}"
        start_col = self.width - len(agent_count_str)
        for i, char in enumerate(agent_count_str):
            if start_col + i < len(frame[0]):
                frame[1][start_col + i] = char
        
        # Legend
        legend_row = self.height - 2
        legend_items = ["R=Research", "A=Analysis", "S=Synthesis", "C=Critic"]
        legend_text = " | ".join(legend_items)
        
        if len(legend_text) < self.width:
            for i, char in enumerate(legend_text):
                if i < len(frame[0]):
                    frame[legend_row][i] = char
    
    def clear_screen(self) -> None:
        """Clear terminal screen."""
        print("\033[2J\033[H", end="")
    
    def display_frame(self, current_time: float) -> None:
        """Display current frame in terminal."""
        self.clear_screen()
        frame = self.render_frame(current_time)
        print(frame)


class WebHelixVisualizer:
    """
    Web-based 3D visualization using matplotlib (optional).
    
    Provides interactive 3D visualization if matplotlib is available.
    """
    
    def __init__(self, helix: HelixGeometry):
        """Initialize web visualizer."""
        self.helix = helix
        self.fig = None
        self.ax = None
        
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            self.plt = plt
            self.has_matplotlib = True
        except ImportError:
            self.has_matplotlib = False
            print("Warning: matplotlib not available. Use terminal mode instead.")
    
    def initialize_plot(self) -> bool:
        """Initialize 3D plot."""
        if not self.has_matplotlib:
            return False
        
        self.fig = self.plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Draw helix structure
        self._draw_helix_3d()
        
        # Set up plot
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y') 
        self.ax.set_zlabel('Z')
        self.ax.set_title('Felix Framework - Helix Multi-Agent System')
        
        return True
    
    def _draw_helix_3d(self) -> None:
        """Draw 3D helix structure."""
        # Generate helix points
        t_values = [i / 1000.0 for i in range(1001)]
        x_values = []
        y_values = []
        z_values = []
        
        for t in t_values:
            pos = self.helix.get_position(t)
            x_values.append(pos[0])
            y_values.append(pos[1])
            z_values.append(pos[2])
        
        # Plot helix
        self.ax.plot(x_values, y_values, z_values, 'b-', alpha=0.3, linewidth=1)
        
        # Draw central post
        self.ax.plot([0, 0], [0, 0], [0, self.helix.height], 'k-', linewidth=3)
    
    def update_agents_3d(self, agent_data: List[Dict[str, Any]]) -> None:
        """Update 3D agent visualization."""
        if not self.has_matplotlib:
            return
        
        # Clear previous agents
        # (In a real implementation, you'd update existing artists)
        
        # Plot agents
        agent_colors = {'research': 'red', 'analysis': 'yellow', 'synthesis': 'green', 'critic': 'purple'}
        
        for data in agent_data:
            if "position" in data and data["position"]:
                pos = data["position"]
                agent_type = data.get("agent_type", "general")
                color = agent_colors.get(agent_type, 'blue')
                
                self.ax.scatter([pos[0]], [pos[1]], [pos[2]], c=color, s=100, alpha=0.8)
                
                # Draw spoke
                self.ax.plot([0, pos[0]], [0, pos[1]], [pos[2], pos[2]], 
                           color=color, alpha=0.3, linewidth=1)
    
    def show(self) -> None:
        """Display the plot."""
        if self.has_matplotlib and self.fig:
            self.plt.show()


class HelixMonitor:
    """
    Main monitoring class that coordinates visualization.
    
    Provides unified interface for different visualization modes
    and handles data collection from Felix agents.
    """
    
    def __init__(self, helix: HelixGeometry, mode: str = "terminal"):
        """
        Initialize helix monitor.
        
        Args:
            helix: Helix geometry to monitor
            mode: Visualization mode ("terminal" or "web")
        """
        self.helix = helix
        self.mode = mode
        
        if mode == "terminal":
            self.visualizer = TerminalHelixVisualizer(helix)
        elif mode == "web":
            self.visualizer = WebHelixVisualizer(helix)
            if not self.visualizer.initialize_plot():
                print("Web mode not available, falling back to terminal mode")
                self.mode = "terminal"
                self.visualizer = TerminalHelixVisualizer(helix)
        else:
            raise ValueError(f"Unknown visualization mode: {mode}")
        
        print(f"Helix Monitor initialized in {self.mode} mode")
    
    def monitor_simulation(self, simulation_data: Dict[str, Any], 
                         refresh_rate: float = 0.5) -> None:
        """
        Monitor a running simulation.
        
        Args:
            simulation_data: Data structure with agent and communication info
            refresh_rate: How often to refresh display (seconds)
        """
        print(f"Starting monitoring (refresh rate: {refresh_rate}s)")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            frame_count = 0
            while True:
                current_time = simulation_data.get("current_time", frame_count * 0.1)
                
                # Update visualizer with current data
                if self.mode == "terminal":
                    agent_data = simulation_data.get("agents", [])
                    spoke_data = simulation_data.get("spokes", [])
                    
                    self.visualizer.update_agents(agent_data)
                    self.visualizer.update_spokes(spoke_data)
                    self.visualizer.display_frame(current_time)
                
                elif self.mode == "web":
                    agent_data = simulation_data.get("agents", [])
                    self.visualizer.update_agents_3d(agent_data)
                    self.visualizer.show()
                
                time.sleep(refresh_rate)
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"Error during monitoring: {e}")


def create_demo_simulation() -> Dict[str, Any]:
    """Create demo simulation data for testing visualization."""
    helix = HelixGeometry(33.0, 0.001, 33.0, 33)
    
    # Create some demo agents at different positions
    demo_agents = []
    current_time = 0.5
    
    # Research agents (early, top of helix)
    for i in range(3):
        t = 0.1 + i * 0.1
        pos = helix.get_position(t)
        demo_agents.append({
            "agent_id": f"research_{i:03d}",
            "agent_type": "research",
            "position": pos,
            "state": "active",
            "spawn_time": t,
            "progress": t
        })
    
    # Analysis agents (middle)
    for i in range(2):
        t = 0.4 + i * 0.1
        pos = helix.get_position(t)
        demo_agents.append({
            "agent_id": f"analysis_{i:03d}",
            "agent_type": "analysis", 
            "position": pos,
            "state": "active",
            "spawn_time": t,
            "progress": t
        })
    
    # Synthesis agent (bottom)
    t = 0.8
    pos = helix.get_position(t)
    demo_agents.append({
        "agent_id": "synthesis_001",
        "agent_type": "synthesis",
        "position": pos,
        "state": "active", 
        "spawn_time": t,
        "progress": t
    })
    
    # Create spoke data
    demo_spokes = []
    for agent in demo_agents:
        demo_spokes.append({
            "agent_id": agent["agent_id"],
            "agent_position": agent["position"],
            "message_count": 5,
            "last_activity": current_time
        })
    
    return {
        "current_time": current_time,
        "agents": demo_agents,
        "spokes": demo_spokes
    }


def main():
    """Main function for helix monitor."""
    parser = argparse.ArgumentParser(description="Felix Framework Helix Monitor")
    parser.add_argument("--mode", choices=["terminal", "web"], default="terminal",
                       help="Visualization mode")
    parser.add_argument("--demo", action="store_true",
                       help="Run demo simulation")
    parser.add_argument("--refresh-rate", type=float, default=0.5,
                       help="Refresh rate in seconds")
    
    args = parser.parse_args()
    
    # Create helix geometry
    helix = HelixGeometry(33.0, 0.001, 33.0, 33)
    
    # Create monitor
    monitor = HelixMonitor(helix, mode=args.mode)
    
    if args.demo:
        # Run demo
        simulation_data = create_demo_simulation()
        
        if args.mode == "terminal":
            # Animate demo in terminal
            for frame in range(100):
                # Update simulation time
                simulation_data["current_time"] = frame * 0.1
                
                # Update agent positions slightly
                for agent in simulation_data["agents"]:
                    current_progress = agent["progress"] + 0.01
                    if current_progress <= 1.0:
                        agent["progress"] = current_progress
                        agent["position"] = helix.get_position(current_progress)
                
                monitor.visualizer.update_agents(simulation_data["agents"])
                monitor.visualizer.update_spokes(simulation_data["spokes"])
                monitor.visualizer.display_frame(simulation_data["current_time"])
                
                time.sleep(args.refresh_rate)
        else:
            # Static web view
            monitor.monitor_simulation(simulation_data, args.refresh_rate)
    else:
        print("Helix monitor ready. Use --demo to see a demonstration.")
        print("In a real scenario, connect this to your Felix simulation.")


if __name__ == "__main__":
    main()