#!/usr/bin/env python3
"""
Blog Writer Demo using Felix Framework Geometric Orchestration.

This demo showcases how the Felix Framework's helix-based multi-agent system
can be used for collaborative content creation, demonstrating the geometric
orchestration approach as an alternative to traditional multi-agent systems.

The demo creates a team of specialized LLM agents that work together to write
a blog post, with agents spawning at different times and converging naturally
through the helix geometry toward a final synthesis.

Usage:
    python examples/blog_writer.py "Write a blog post about quantum computing"

Requirements:
    - LM Studio running with a model loaded (http://localhost:1234)
    - openai Python package installed
"""

import sys
import time
import asyncio
import argparse
from typing import List, Dict, Any
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.helix_geometry import HelixGeometry
from llm.lm_studio_client import LMStudioClient, LMStudioConnectionError
from llm.token_budget import TokenBudgetManager
from agents.llm_agent import LLMTask
from agents.specialized_agents import create_specialized_team
from communication.central_post import CentralPost
from communication.spoke import SpokeManager


class FelixBlogWriter:
    """
    Blog writing system using Felix geometric orchestration.
    
    Demonstrates how helix-based agent coordination can create
    content through natural convergence rather than explicit
    workflow management.
    """
    
    def __init__(self, lm_studio_url: str = "http://localhost:1234/v1"):
        """
        Initialize the Felix blog writing system.
        
        Args:
            lm_studio_url: LM Studio API endpoint
        """
        # Create helix geometry (OpenSCAD parameters)
        self.helix = HelixGeometry(
            top_radius=33.0,
            bottom_radius=0.001,
            height=33.0,
            turns=33
        )
        
        # Initialize LLM client
        self.llm_client = LMStudioClient(base_url=lm_studio_url)
        
        # Initialize communication system
        self.central_post = CentralPost(max_agents=20, enable_metrics=True)
        self.spoke_manager = SpokeManager(self.central_post)
        
        # Initialize token budget manager
        self.token_budget_manager = TokenBudgetManager(
            base_budget=1200,  # Increased budget for blog writing
            min_budget=150,
            max_budget=800
        )
        
        # Agent team
        self.agents = []
        
        print(f"Felix Blog Writer initialized")
        print(f"Helix: {self.helix.turns} turns, {self.helix.top_radius}→{self.helix.bottom_radius} radius")
    
    def test_lm_studio_connection(self) -> bool:
        """Test connection to LM Studio."""
        try:
            if self.llm_client.test_connection():
                print("✓ LM Studio connection successful")
                return True
            else:
                print("✗ LM Studio connection failed")
                return False
        except LMStudioConnectionError as e:
            print(f"✗ LM Studio connection error: {e}")
            return False
    
    def create_blog_writing_team(self, complexity: str = "medium") -> None:
        """
        Create specialized team for blog writing.
        
        Args:
            complexity: Task complexity level
        """
        print(f"\nCreating {complexity} complexity blog writing team...")
        
        # Create specialized agents with token budget manager
        self.agents = create_specialized_team(
            helix=self.helix,
            llm_client=self.llm_client,
            task_complexity=complexity,
            token_budget_manager=self.token_budget_manager
        )
        
        # Register agents with communication system
        for agent in self.agents:
            self.spoke_manager.register_agent(agent)
        
        print(f"Created team of {len(self.agents)} specialized agents:")
        for agent in self.agents:
            print(f"  - {agent.agent_id} ({agent.agent_type}) spawns at t={agent.spawn_time:.2f}")
    
    def run_blog_writing_session(self, topic: str, simulation_time: float = 1.0) -> Dict[str, Any]:
        """
        Run collaborative blog writing session.
        
        Args:
            topic: Blog post topic
            simulation_time: Duration of simulation (0.0 to 1.0)
            
        Returns:
            Results from the writing session
        """
        print(f"\n{'='*60}")
        print(f"FELIX BLOG WRITING SESSION")
        print(f"Topic: {topic}")
        print(f"{'='*60}")
        
        # Create the main task
        main_task = LLMTask(
            task_id="blog_post_001",
            description=f"Write a comprehensive blog post about: {topic}",
            context=f"This is a collaborative writing project. Multiple agents will contribute research, analysis, and synthesis to create a high-quality blog post about {topic}."
        )
        
        # Track session results
        results = {
            "topic": topic,
            "agents_participated": [],
            "processing_timeline": [],
            "final_output": None,
            "session_stats": {}
        }
        
        # Run simulation
        current_time = 0.0
        time_step = 0.05
        session_start = time.perf_counter()
        
        print(f"\nStarting geometric orchestration simulation...")
        print(f"Simulation will continue until high-confidence result is absorbed by central post")
        print(f"Maximum timeout: {simulation_time * 10:.1f} time units (step size: {time_step})")
        
        # Track completion
        simulation_complete = False
        max_timeout = simulation_time * 10  # Safety timeout (much longer than needed)
        
        while current_time <= max_timeout and not simulation_complete:
            step_start = time.perf_counter()
            
            # Check for agents ready to spawn
            for agent in self.agents:
                if (agent.can_spawn(current_time) and 
                    agent.state.value == "waiting"):
                    
                    print(f"\n[t={current_time:.2f}] 🌀 Spawning {agent.agent_id} ({agent.agent_type}) at helix top")
                    agent.spawn(current_time, main_task)
            
            # Process all active agents as they descend the helix
            for agent in self.agents:
                if agent.state.value == "active":
                    # Update agent position (descend the helix)
                    agent.update_position(current_time)
                    
                    # Get current position info
                    pos_info = agent.get_position_info(current_time)
                    depth = pos_info.get("depth_ratio", 0.0)
                    radius = pos_info.get("radius", 0.0)
                    
                    # Continuous processing: agents work as they descend
                    # Process every few steps to show progression
                    if current_time % 0.1 < time_step or agent.processing_stage == 0:
                        try:
                            result = agent.process_task_with_llm(main_task, current_time)
                            
                            print(f"  [{current_time:.2f}] 🔄 {agent.agent_id} processing (stage {result.processing_stage})")
                            print(f"      Depth: {depth:.2f}, Confidence: {result.confidence:.2f}")
                            
                            # Share result with central post
                            message = agent.share_result_to_central(result)
                            self.spoke_manager.send_message(agent.agent_id, message)
                            
                            # Check for high-confidence acceptance
                            if self.central_post.accept_high_confidence_result(message):
                                print(f"      ✅ HIGH CONFIDENCE: {agent.agent_id} result accepted by central post!")
                                print(f"      🎯 SIMULATION COMPLETE: Central post absorbed high-confidence result!")
                                
                                # Set as final output
                                results["final_output"] = {
                                    "content": result.content,
                                    "agent_id": agent.agent_id,
                                    "confidence": result.confidence,
                                    "stage": result.processing_stage,
                                    "timestamp": current_time
                                }
                                
                                simulation_complete = True
                            
                            # Track results only for final output (when confidence is high enough)
                            if result.confidence >= 0.6:  # Lower threshold for tracking
                                results["agents_participated"].append({
                                    "agent_id": agent.agent_id,
                                    "agent_type": agent.agent_type,
                                    "spawn_time": current_time,
                                    "position_info": result.position_info,
                                    "content_preview": result.content[:100] + "...",
                                    "tokens_used": result.llm_response.tokens_used,
                                    "processing_time": result.processing_time,
                                    "confidence": result.confidence,
                                    "stage": result.processing_stage
                                })
                                
                                results["processing_timeline"].append({
                                    "timestamp": current_time,
                                    "agent_id": agent.agent_id,
                                    "action": "task_processed",
                                    "tokens": result.llm_response.tokens_used,
                                    "confidence": result.confidence
                                })
                                
                                # Final output is now set when central post accepts high-confidence result
                        
                        except Exception as e:
                            print(f"  ✗ Error processing task for {agent.agent_id}: {e}")
            
            # Update active agents' positions
            for agent in self.agents:
                if agent.state.value == "active":
                    agent.update_position(current_time)
            
            # Process communication system and advance time
            self.spoke_manager.process_all_messages()
            current_time += time_step
        
        session_end = time.perf_counter()
        session_duration = session_end - session_start
        
        # Check if simulation completed due to timeout
        if not simulation_complete and current_time >= max_timeout:
            print(f"\n⚠️  WARNING: Simulation reached maximum timeout ({max_timeout:.1f} time units)")
            print(f"    No agent achieved sufficient confidence for central post absorption.")
            print(f"    Consider increasing agent processing time or adjusting confidence thresholds.")
        elif simulation_complete:
            print(f"\n✅ Simulation completed successfully at t={current_time:.2f}")
        
        # Collect final statistics
        results["session_stats"] = {
            "total_duration": session_duration,
            "simulation_time": current_time,  # Actual time simulation ran to
            "max_timeout": max_timeout,       # Maximum allowed time
            "simulation_complete": simulation_complete,
            "agents_created": len(self.agents),
            "agents_participated": len(results["agents_participated"]),
            "total_tokens_used": sum(a["tokens_used"] for a in results["agents_participated"]),
            "total_messages_processed": self.central_post.total_messages_processed,
            "llm_client_stats": self.llm_client.get_usage_stats()
        }
        
        return results
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display session results in a readable format."""
        print(f"\n{'='*60}")
        print(f"SESSION RESULTS")
        print(f"{'='*60}")
        
        stats = results["session_stats"]
        print(f"Topic: {results['topic']}")
        print(f"Duration: {stats['total_duration']:.2f} seconds")
        print(f"Simulation Time: {stats['simulation_time']:.2f} units (completed: {stats['simulation_complete']})")
        print(f"Agents: {stats['agents_participated']}/{stats['agents_created']} participated")
        print(f"Tokens: {stats['total_tokens_used']} total")
        print(f"Messages: {stats['total_messages_processed']} processed")
        
        # Show token budget summary if available
        if hasattr(self, 'token_budget_manager'):
            budget_status = self.token_budget_manager.get_system_status()
            print(f"Token Efficiency: {budget_status['system_efficiency']:.1%} "
                  f"({budget_status['total_used']}/{budget_status['total_allocated']} allocated)")
        
        print(f"\nAgent Participation Timeline:")
        for agent_info in results["agents_participated"]:
            print(f"  {agent_info['spawn_time']:.2f}s: {agent_info['agent_id']} ({agent_info['agent_type']})")
            print(f"        Depth: {agent_info['position_info'].get('depth_ratio', 0):.2f}, "
                  f"Tokens: {agent_info['tokens_used']}")
        
        if results["final_output"]:
            print(f"\n{'='*60}")
            print(f"FINAL BLOG POST")
            print(f"{'='*60}")
            print(results["final_output"]["content"])
            print(f"\n[Generated by {results['final_output']['agent_id']} "
                  f"with confidence {results['final_output']['confidence']:.2f} "
                  f"at stage {results['final_output']['stage']} "
                  f"at time t={results['final_output']['timestamp']:.2f}]")
        else:
            print(f"\nNo final synthesis output was generated.")
    
    def save_results(self, results: Dict[str, Any], output_file: str = None) -> None:
        """Save results to file."""
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"blog_writing_session_{timestamp}.json"
        
        import json
        with open(output_file, 'w') as f:
            # Make results JSON serializable
            serializable_results = {
                "topic": results["topic"],
                "agents_participated": results["agents_participated"],
                "session_stats": results["session_stats"],
                "final_output": results["final_output"]
            }
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


def main():
    """Main function for blog writer demo."""
    parser = argparse.ArgumentParser(description="Felix Framework Blog Writer Demo")
    parser.add_argument("topic", help="Blog post topic")
    parser.add_argument("--complexity", choices=["simple", "medium", "complex"], 
                       default="medium", help="Task complexity level")
    parser.add_argument("--simulation-time", type=float, default=1.0,
                       help="Simulation duration (0.0 to 1.0)")
    parser.add_argument("--lm-studio-url", default="http://localhost:1234/v1",
                       help="LM Studio API URL")
    parser.add_argument("--save-output", help="Save results to file")
    
    args = parser.parse_args()
    
    # Create blog writer
    writer = FelixBlogWriter(lm_studio_url=args.lm_studio_url)
    
    # Test LM Studio connection
    if not writer.test_lm_studio_connection():
        print("\nPlease ensure LM Studio is running with a model loaded.")
        sys.exit(1)
    
    # Create team and run session
    writer.create_blog_writing_team(complexity=args.complexity)
    results = writer.run_blog_writing_session(
        topic=args.topic,
        simulation_time=args.simulation_time
    )
    
    # Display and optionally save results
    writer.display_results(results)
    
    if args.save_output:
        writer.save_results(results, args.save_output)


if __name__ == "__main__":
    main()