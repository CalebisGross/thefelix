#!/usr/bin/env python3
"""
Colony Design Demo using Felix Framework Geometric Orchestration.

This demo showcases how the Felix Framework's helix-based multi-agent system
can be used to collaboratively design a self-sustaining human colony on a hypothetical
exoplanet, Kepler-999z, addressing ecological, scientific, energy, governance, and
philosophical challenges. It demonstrates the geometric orchestration approach as an
alternative to traditional multi-agent systems.

The demo creates a team of specialized LLM agents that work together to produce a
comprehensive colony design plan, with agents spawning at different times and converging
naturally through the helix geometry toward a final synthesis.

Usage:
    python examples/colony_design.py

Requirements:
    - LM Studio running with a model loaded (http://localhost:1234)
    - openai Python package installed
"""

import sys
import time
import asyncio
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.helix_geometry import HelixGeometry
from llm.lm_studio_client import LMStudioClient, LMStudioConnectionError, RequestPriority
from llm.token_budget import TokenBudgetManager
from agents.llm_agent import LLMTask
from agents.specialized_agents import create_specialized_team
from communication.central_post import CentralPost, AgentFactory
from communication.spoke import SpokeManager


class FelixColonyDesigner:
    """
    Colony design system using Felix geometric orchestration.
    
    Demonstrates how helix-based agent coordination can create a comprehensive
    plan for a self-sustaining human colony on an exoplanet through natural
    convergence rather than explicit workflow management.
    """
    
    def __init__(self, lm_studio_url: str = "http://localhost:1234/v1", random_seed: int = None,
                 strict_mode: bool = False, max_concurrent_agents: int = 4):
        """
        Initialize the Felix colony design system.
        
        Args:
            lm_studio_url: LM Studio API endpoint
            random_seed: Seed for randomization (None for truly random behavior)
            strict_mode: Enable strict token budgets for lightweight models
            max_concurrent_agents: Maximum concurrent agent processing
        """
        self.random_seed = random_seed
        self.strict_mode = strict_mode
        self.max_concurrent_agents = max_concurrent_agents
        
        # Create helix geometry (OpenSCAD parameters)
        self.helix = HelixGeometry(
            top_radius=33.0,
            bottom_radius=0.001,
            height=33.0,
            turns=33
        )
        
        # Initialize LLM client
        self.llm_client = LMStudioClient(base_url=lm_studio_url, max_concurrent_requests=max_concurrent_agents)
        
        # Initialize communication system
        self.central_post = CentralPost(max_agents=20, enable_metrics=True)
        self.spoke_manager = SpokeManager(self.central_post)
        
        # Initialize token budget manager
        self.token_budget_manager = TokenBudgetManager(
            base_budget=1200 if not strict_mode else 400,  # Reduced budget for strict mode
            min_budget=150 if not strict_mode else 50,
            max_budget=800 if not strict_mode else 150,
            strict_mode=strict_mode
        )
        
        # Initialize agent factory for dynamic agent creation
        self.agent_factory = AgentFactory(
            helix=self.helix,
            llm_client=self.llm_client,
            token_budget_manager=self.token_budget_manager,
            random_seed=random_seed
        )
        
        # Agent team
        self.agents = []
        
        print(f"Felix Colony Designer initialized")
        print(f"Helix: {self.helix.turns} turns, {self.helix.top_radius}→{self.helix.bottom_radius} radius")
        if random_seed is not None:
            print(f"Random seed: {random_seed} (deterministic behavior)")
        else:
            print("Random seed: None (truly random behavior)")
    
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
    
    def create_colony_design_team(self, complexity: str = "complex") -> None:
        """
        Create specialized team for colony design with randomized spawn times.
        
        Args:
            complexity: Task complexity level (default: complex)
        """
        print(f"\nCreating {complexity} complexity colony design team...")
        
        # Create specialized agents with default roles, relying on task description for guidance
        self.agents = create_specialized_team(
            helix=self.helix,
            llm_client=self.llm_client,
            task_complexity=complexity,
            token_budget_manager=self.token_budget_manager,
            random_seed=self.random_seed
        )
        
        # Register agents with communication system
        for agent in self.agents:
            self.spoke_manager.register_agent(agent)
        
        print(f"Created team of {len(self.agents)} specialized agents:")
        for agent in self.agents:
            print(f"  - {agent.agent_id} ({agent.agent_type}) spawns at t={agent.spawn_time:.3f}")
        print(f"Agent spawn times are {'deterministic' if self.random_seed is not None else 'randomized'}")
    
    async def run_colony_design_session_async(self, simulation_time: float = 1.0) -> Dict[str, Any]:
        """
        Run collaborative colony design session with true parallel processing.
        
        Args:
            simulation_time: Duration of simulation (0.0 to 1.0)
            
        Returns:
            Results from the design session
        """
        topic = "Design a self-sustaining human colony on Kepler-999z"
        print(f"\n{'='*60}")
        print(f"FELIX PARALLEL COLONY DESIGN SESSION")
        print(f"Topic: {topic}")
        print(f"Mode: {'STRICT (Lightweight)' if self.strict_mode else 'NORMAL'}")
        print(f"{'='*60}")
        
        # Start async message processing
        await self.central_post.start_async_processing(max_concurrent_processors=2)
        
        # Create the main task
        main_task = LLMTask(
            task_id="colony_design_001",
            description=(
                "Design a self-sustaining human colony for 10,000 people on Kepler-999z, an exoplanet with "
                "Earth-like conditions but a silicon-oxygen polymer-based ecosystem and non-linear time dilation "
                "effects near pulsar-induced geological sites. Address: (1) sustainable ecological integration "
                "with silicon-based lifeforms, (2) safe utilization of time dilation sites for research while "
                "mitigating cognitive risks, (3) an energy-independent system leveraging the planet’s technetium-99 "
                "and pulsar radiation, (4) a governance model balancing human needs and ethical considerations for "
                "potentially sentient lifeforms, and (5) a speculative technological breakthrough grounded in 2025 "
                "science to address one major challenge. Reflect on the philosophical implications of colonizing a "
                "planet with non-linear time and sentient silicon life."
            ),
            context=(
                "This is a collaborative project. Agents must contribute specialized expertise (e.g., astrobiology, "
                "astrophysics, engineering, ethics) to create a comprehensive colony design plan. The plan must integrate "
                "all components cohesively, ensuring human survival, scientific advancement, and ethical coexistence. "
                "Agents should adapt their roles to address specific aspects of the task as needed."
            )
        )
        
        # Track session results
        results = {
            "topic": topic,
            "agents_participated": [],
            "processing_timeline": [],
            "final_output": None,
            "session_stats": {},
            "parallel_processing": True
        }
        
        session_start = time.perf_counter()
        
        # Performance targets for strict mode
        if self.strict_mode:
            target_completion_time = 30.0  # 30 seconds max
            target_total_tokens = 2000     # 2000 tokens max
        else:
            target_completion_time = simulation_time * 60  # More relaxed
            target_total_tokens = 10000
        
        print(f"\nStarting PARALLEL geometric orchestration...")
        print(f"Target: {'<30s, <2000 tokens' if self.strict_mode else 'Normal performance'}")
        
        # Run parallel processing
        final_result = await self._run_parallel_processing(
            main_task, 0.0, 0.05, simulation_time, results,
            target_completion_time, target_total_tokens
        )
        
        session_end = time.perf_counter()
        session_duration = session_end - session_start
        
        # Cleanup async resources
        await self.central_post.shutdown_async()
        if hasattr(self.llm_client, 'close_async'):
            await self.llm_client.close_async()
        
        # Final statistics
        total_tokens = sum(a.get("tokens_used", 0) for a in results["agents_participated"])
        results["session_stats"] = {
            "total_duration": session_duration,
            "simulation_time": simulation_time,
            "simulation_complete": final_result is not None,
            "total_tokens_used": total_tokens,
            "agents_created": len(self.agents),
            "agents_participated": len(results["agents_participated"]),
            "total_messages_processed": self.central_post.total_messages_processed,
            "llm_client_stats": self.llm_client.get_usage_stats(),
            "strict_mode": self.strict_mode,
            "performance_targets_met": {
                "time_target": session_duration < target_completion_time,
                "token_target": total_tokens < target_total_tokens
            }
        }
        
        return results
    
    def run_colony_design_session(self, simulation_time: float = 1.0) -> Dict[str, Any]:
        """
        Run collaborative colony design session.
        
        Args:
            simulation_time: Duration of simulation (0.0 to 1.0)
            
        Returns:
            Results from the design session
        """
        # Run the async version
        return asyncio.run(self.run_colony_design_session_async(simulation_time))
    
    async def _run_parallel_processing(self, main_task: LLMTask, current_time: float, 
                                     time_step: float, simulation_time: float,
                                     results: Dict[str, Any], target_completion_time: float,
                                     target_total_tokens: int) -> Optional[Dict[str, Any]]:
        """
        Run the core parallel processing loop.
        """
        simulation_complete = False
        max_timeout = simulation_time * 10
        
        while current_time <= max_timeout and not simulation_complete:
            step_start = time.perf_counter()
            
            # Collect agents ready for processing
            ready_agents = []
            for agent in self.agents:
                if (agent.can_spawn(current_time) and 
                    agent.state.value == "waiting"):
                    print(f"\n[t={current_time:.2f}] 🌀 Spawning {agent.agent_id} ({agent.agent_type})")
                    agent.spawn(current_time, main_task)
                    ready_agents.append(agent)
                elif agent.state.value == "active":
                    agent.update_position(current_time)
                    # Process every few steps or first stage
                    if current_time % 0.1 < time_step or agent.processing_stage == 0:
                        ready_agents.append(agent)
            
            # Process agents in parallel batches
            if ready_agents:
                batch_size = min(len(ready_agents), self.max_concurrent_agents)
                for i in range(0, len(ready_agents), batch_size):
                    batch = ready_agents[i:i + batch_size]
                    
                    # Process batch in parallel
                    print(f"[t={current_time:.2f}] 🚀 Processing {len(batch)} agents in parallel")
                    
                    batch_results = await self._process_agent_batch_async(
                        batch, main_task, current_time
                    )
                    
                    # Check results and update
                    synthesis_agents_ready = any(
                        agent.agent_type == "synthesis" and agent.state.value == "active"
                        for agent in self.agents
                    )
                    
                    for result in batch_results:
                        if result:
                            # Check for high-confidence acceptance
                            message = result['agent'].share_result_to_central(result['llm_result'])
                            await self.central_post.queue_message_async(message)
                            
                            can_complete = (synthesis_agents_ready or 
                                          current_time > 0.9)
                            
                            if can_complete and self.central_post.accept_high_confidence_result(message):
                                print(f"      ✅ HIGH CONFIDENCE: {result['agent'].agent_id} ({result['agent'].agent_type}) result accepted!")
                                print(f"      🎯 PARALLEL SIMULATION COMPLETE!")
                                
                                results["final_output"] = {
                                    "content": result['llm_result'].content,
                                    "agent_id": result['agent'].agent_id,
                                    "agent_type": result['agent'].agent_type,
                                    "confidence": result['llm_result'].confidence,
                                    "stage": result['llm_result'].processing_stage,
                                    "timestamp": current_time
                                }
                                simulation_complete = True
                                break
                            
                            # Track participation
                            if result['llm_result'].confidence >= 0.6:
                                agent_info = {
                                    "agent_id": result['agent'].agent_id,
                                    "agent_type": result['agent'].agent_type,
                                    "spawn_time": current_time,
                                    "position_info": result['llm_result'].position_info,
                                    "tokens_used": result['llm_result'].llm_response.tokens_used,
                                    "confidence": result['llm_result'].confidence,
                                    "stage": result['llm_result'].processing_stage
                                }
                                results["agents_participated"].append(agent_info)
                    
                    if simulation_complete:
                        break
            
            # Check performance targets in strict mode
            if self.strict_mode:
                current_duration = time.perf_counter() - step_start
                current_tokens = sum(a.get("tokens_used", 0) for a in results["agents_participated"])
                
                if current_duration > target_completion_time or current_tokens > target_total_tokens:
                    print(f"\n⚠️  Performance target exceeded - completing simulation")
                    simulation_complete = True
                    break
            
            current_time += time_step
        
        return results["final_output"] if simulation_complete else None
    
    async def _process_agent_batch_async(self, agents: List, main_task: LLMTask, 
                                       current_time: float) -> List[Dict[str, Any]]:
        """
        Process a batch of agents in parallel.
        """
        async def process_single_agent(agent):
            try:
                priority = RequestPriority.HIGH if self.strict_mode else RequestPriority.NORMAL
                result = await agent.process_task_with_llm_async(
                    main_task, current_time, priority
                )
                
                pos_info = agent.get_position_info(current_time)
                depth = pos_info.get("depth_ratio", 0.0)
                
                print(f"    ✓ {agent.agent_id} completed (depth: {depth:.2f}, "
                      f"confidence: {result.confidence:.2f}, tokens: {result.llm_response.tokens_used})")
                
                return {
                    "agent": agent,
                    "llm_result": result,
                    "success": True
                }
                
            except Exception as e:
                print(f"    ✗ {agent.agent_id} failed: {e}")
                return {
                    "agent": agent,
                    "llm_result": None,
                    "success": False,
                    "error": str(e)
                }
        
        results = await asyncio.gather(
            *[process_single_agent(agent) for agent in agents],
            return_exceptions=True
        )
        
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success", False)]
        return successful_results
    
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
            print(f"FINAL COLONY DESIGN PLAN")
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
            output_file = f"colony_design_session_{timestamp}.json"
        
        import json
        with open(output_file, 'w') as f:
            serializable_results = {
                "topic": results["topic"],
                "agents_participated": results["agents_participated"],
                "session_stats": results["session_stats"],
                "final_output": results["final_output"]
            }
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


def main():
    """Main function for colony design demo."""
    parser = argparse.ArgumentParser(description="Felix Framework Colony Design Demo")
    parser.add_argument("--complexity", choices=["simple", "medium", "complex"], 
                       default="complex", help="Task complexity level")
    parser.add_argument("--simulation-time", type=float, default=1.0,
                       help="Simulation duration (0.0 to 1.0)")
    parser.add_argument("--lm-studio-url", default="http://localhost:1234/v1",
                       help="LM Studio API URL")
    parser.add_argument("--save-output", help="Save results to file")
    parser.add_argument("--random-seed", type=int, 
                       help="Random seed for reproducibility (omit for truly random)")
    parser.add_argument("--strict-mode", action="store_true",
                       help="Enable strict token budgets for lightweight models")
    parser.add_argument("--max-concurrent", type=int, default=4,
                       help="Maximum concurrent agents (default: 4)")
    
    args = parser.parse_args()
    
    # Create colony designer
    designer = FelixColonyDesigner(
        lm_studio_url=args.lm_studio_url, 
        random_seed=args.random_seed,
        strict_mode=args.strict_mode,
        max_concurrent_agents=args.max_concurrent
    )
    
    # Test LM Studio connection
    if not designer.test_lm_studio_connection():
        print("\nPlease ensure LM Studio is running with a model loaded.")
        sys.exit(1)
    
    # Create team and run session
    designer.create_colony_design_team(complexity=args.complexity)
    results = designer.run_colony_design_session(
        simulation_time=args.simulation_time
    )
    
    # Display and optionally save results
    designer.display_results(results)
    
    if args.save_output:
        designer.save_results(results, args.save_output)


if __name__ == "__main__":
    main()