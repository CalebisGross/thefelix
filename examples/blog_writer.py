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
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.helix_geometry import HelixGeometry
from src.llm.lm_studio_client import LMStudioClient, LMStudioConnectionError, RequestPriority
from src.llm.multi_server_client import LMStudioClientPool
from src.llm.token_budget import TokenBudgetManager
from src.agents.llm_agent import LLMTask
from src.agents.specialized_agents import create_specialized_team
from src.communication.central_post import CentralPost, AgentFactory
from src.communication.spoke import SpokeManager


class FelixBlogWriter:
    """
    Blog writing system using Felix geometric orchestration.
    
    Demonstrates how helix-based agent coordination can create
    content through natural convergence rather than explicit
    workflow management.
    """
    
    def __init__(self, lm_studio_url: str = "http://localhost:1234/v1", random_seed: int = None,
                 strict_mode: bool = False, max_concurrent_agents: int = 4, debug_mode: bool = False,
                 server_config_path: str = None):
        """
        Initialize the Felix blog writing system.
        
        Args:
            lm_studio_url: LM Studio API endpoint (used if no server_config_path)
            random_seed: Seed for randomization (None for truly random behavior)
            strict_mode: Enable strict token budgets for lightweight models
            max_concurrent_agents: Maximum concurrent agent processing
            debug_mode: Enable verbose debugging output
            server_config_path: Path to multi-server configuration file
        """
        self.random_seed = random_seed
        self.strict_mode = strict_mode
        self.max_concurrent_agents = max_concurrent_agents
        self.debug_mode = debug_mode
        self.server_config_path = server_config_path
        
        # Create helix geometry (OpenSCAD parameters)
        self.helix = HelixGeometry(
            top_radius=33.0,
            bottom_radius=0.001,
            height=33.0,
            turns=33
        )
        
        # Initialize LLM client (single server or multi-server pool)
        if server_config_path:
            # Use multi-server client pool
            self.llm_client = LMStudioClientPool(config_path=server_config_path, debug_mode=debug_mode)
            print(f"Felix initialized with multi-server configuration: {server_config_path}")
            if debug_mode:
                self.llm_client.display_pool_status()
        else:
            # Use single LM Studio client
            self.llm_client = LMStudioClient(base_url=lm_studio_url, max_concurrent_requests=max_concurrent_agents, debug_mode=debug_mode)
            print(f"Felix initialized with single server: {lm_studio_url}")
        
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
        
        print(f"Felix Blog Writer initialized")
        print(f"Helix: {self.helix.turns} turns, {self.helix.top_radius}→{self.helix.bottom_radius} radius")
        if random_seed is not None:
            print(f"Random seed: {random_seed} (deterministic behavior)")
        else:
            print("Random seed: None (truly random behavior)")
    
    def test_lm_studio_connection(self) -> bool:
        """Test connection to LM Studio server(s)."""
        try:
            if isinstance(self.llm_client, LMStudioClientPool):
                # Test all servers in the pool
                import asyncio
                health_results = asyncio.run(self.llm_client.health_check_all_servers())
                
                healthy_servers = sum(1 for healthy in health_results.values() if healthy)
                total_servers = len(health_results)
                
                if healthy_servers > 0:
                    print(f"✓ Multi-server pool: {healthy_servers}/{total_servers} servers healthy")
                    if self.debug_mode:
                        for server, healthy in health_results.items():
                            status = "✓" if healthy else "✗"
                            print(f"  {status} {server}")
                    return True
                else:
                    print(f"✗ Multi-server pool: 0/{total_servers} servers healthy")
                    return False
            else:
                # Test single server
                if self.llm_client.test_connection():
                    print("✓ LM Studio connection successful")
                    return True
                else:
                    print("✗ LM Studio connection failed")
                    return False
        except Exception as e:
            print(f"✗ LM Studio connection error: {e}")
            return False
    
    def create_blog_writing_team(self, complexity: str = "medium") -> None:
        """
        Create specialized team for blog writing with randomized spawn times.
        
        Args:
            complexity: Task complexity level
        """
        print(f"\nCreating {complexity} complexity blog writing team...")
        
        # Create specialized agents with randomized spawn times
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
    
    async def run_blog_writing_session_async(self, topic: str, simulation_time: float = 1.0) -> Dict[str, Any]:
        """
        Run collaborative blog writing session with true parallel processing.
        
        Args:
            topic: Blog post topic
            simulation_time: Duration of simulation (0.0 to 1.0)
            
        Returns:
            Results from the writing session
        """
        print(f"\n{'='*60}")
        print(f"FELIX PARALLEL BLOG WRITING SESSION")
        print(f"Topic: {topic}")
        print(f"Mode: {'STRICT (Lightweight)' if self.strict_mode else 'NORMAL'}")
        print(f"{'='*60}")
        
        # Start async message processing
        await self.central_post.start_async_processing(max_concurrent_processors=2)
        
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
            target_total_tokens = 20000
        
        print(f"\nStarting PARALLEL geometric orchestration...")
        print(f"Target: {'<30s, <2000 tokens' if self.strict_mode else 'Normal performance'}")
        if self.debug_mode:
            print(f"🔧 Debug mode enabled - showing detailed agent processing")
            print(f"📋 Team: {[f'{a.agent_id}({a.agent_type})@{a.spawn_time:.2f}' for a in self.agents]}")
            if isinstance(self.llm_client, LMStudioClientPool):
                self.llm_client.display_pool_status()
        
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
        elif hasattr(self.llm_client, 'close_all'):
            await self.llm_client.close_all()
        
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
    
    def run_blog_writing_session(self, topic: str, simulation_time: float = 1.0) -> Dict[str, Any]:
        """
        Run collaborative blog writing session.
        
        Args:
            topic: Blog post topic
            simulation_time: Duration of simulation (0.0 to 1.0)
            
        Returns:
            Results from the writing session
        """
        # Run the async version
        return asyncio.run(self.run_blog_writing_session_async(topic, simulation_time))
    
    
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
                    if self.debug_mode:
                        pos_info = agent.get_position_info(current_time)
                        print(f"    📍 Position: x={pos_info.get('x', 0):.2f}, y={pos_info.get('y', 0):.2f}, z={pos_info.get('z', 0):.2f}")
                        print(f"    📐 Radius: {pos_info.get('radius', 0):.2f}, Depth: {pos_info.get('depth_ratio', 0):.2f}")
                        temp = agent.get_adaptive_temperature(current_time)
                        print(f"    🌡️  Temperature: {temp:.2f}")
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
                    # Check if synthesis agents have spawned and had a chance to process
                    synthesis_agents_ready = any(
                        agent.agent_type == "synthesis" and agent.state.value == "active"
                        for agent in self.agents
                    )
                    
                    for result in batch_results:
                        if result:
                            # Check for high-confidence acceptance (only synthesis agents can trigger completion)
                            message = result['agent'].share_result_to_central(result['llm_result'])
                            await self.central_post.queue_message_async(message)
                            
                            # Only accept synthesis results if synthesis agents are ready, or allow early completion
                            # if we've processed for a while and no synthesis agents are spawning
                            can_complete = (synthesis_agents_ready or 
                                          current_time > 0.9)  # Fallback if synthesis agents don't spawn
                            
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
            
            # Show periodic stats in debug mode
            if self.debug_mode and current_time % 0.2 < time_step:  # Every 0.2 time units
                self.display_real_time_stats(results, current_time)
            
            current_time += time_step
        
        return results["final_output"] if simulation_complete else None
    
    async def _process_agent_batch_async(self, agents: List, main_task: LLMTask, 
                                       current_time: float) -> List[Dict[str, Any]]:
        """
        Process a batch of agents in parallel.
        """
        async def process_single_agent(agent):
            try:
                # Determine priority based on agent type and strict mode
                if self.strict_mode:
                    priority = RequestPriority.HIGH  # High priority in strict mode
                else:
                    priority = RequestPriority.NORMAL
                
                result = await agent.process_task_with_llm_async(
                    main_task, current_time, priority
                )
                
                pos_info = agent.get_position_info(current_time)
                depth = pos_info.get("depth_ratio", 0.0)
                
                print(f"    ✓ {agent.agent_id} completed (depth: {depth:.2f}, "
                      f"confidence: {result.confidence:.2f}, tokens: {result.llm_response.tokens_used})")
                
                if self.debug_mode:
                    print(f"    🧠 Content preview: {result.content[:100]}...")
                    if hasattr(agent, '_last_confidence_breakdown'):
                        breakdown = agent._last_confidence_breakdown
                        print(f"    📊 Confidence breakdown:")
                        print(f"        Base ({agent.agent_type}): {breakdown['base_confidence']:.3f}")
                        print(f"        Content quality: +{breakdown['content_bonus']:.3f}")
                        print(f"        Stage bonus: +{breakdown['stage_bonus']:.3f}")
                        print(f"        Consistency: +{breakdown['consistency_bonus']:.3f}")
                        print(f"        Total: {breakdown['total_before_cap']:.3f} → {breakdown['final_confidence']:.3f} (capped at {breakdown['max_confidence']:.1f})")
                    print(f"    ⏱️  Processing time: {result.processing_time:.2f}s, Stage: {result.processing_stage}")
                
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
        
        # Process all agents in the batch concurrently
        results = await asyncio.gather(
            *[process_single_agent(agent) for agent in agents],
            return_exceptions=True
        )
        
        # Filter out exceptions and failed results
        successful_results = []
        for result in results:
            if isinstance(result, dict) and result.get("success", False):
                successful_results.append(result)
        
        return successful_results
    
    def display_real_time_stats(self, results: Dict[str, Any], current_time: float) -> None:
        """Display real-time statistics during processing."""
        if not self.debug_mode:
            return
            
        print(f"\n╭─ REAL-TIME STATS (t={current_time:.2f}) ─╮")
        
        # Agent statistics
        active_agents = [a for a in self.agents if a.state.value == "active"]
        waiting_agents = [a for a in self.agents if a.state.value == "waiting"]
        completed_agents = [a for a in self.agents if a.state.value == "completed"]
        
        print(f"│ Agents: {len(active_agents)} active, {len(waiting_agents)} waiting, {len(completed_agents)} completed")
        
        # Token statistics
        total_tokens = sum(a.get("tokens_used", 0) for a in results["agents_participated"])
        if hasattr(self.llm_client, 'total_tokens'):
            llm_total = self.llm_client.total_tokens
            print(f"│ Tokens: {total_tokens} (session), {llm_total} (LLM total)")
        else:
            print(f"│ Tokens: {total_tokens} (session)")
            
        # Confidence distribution
        confidences = [a.get("confidence", 0) for a in results["agents_participated"]]
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            max_conf = max(confidences)
            print(f"│ Confidence: avg={avg_conf:.2f}, max={max_conf:.2f}")
        
        # Processing timing
        if hasattr(self.llm_client, 'total_response_time') and self.llm_client.total_requests > 0:
            avg_time = self.llm_client.total_response_time / self.llm_client.total_requests
            print(f"│ LLM Timing: {avg_time:.2f}s avg, {self.llm_client.total_requests} requests")
        
        print(f"╰{'─'*40}╯")
    
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
    
    def _process_agent_interactions(self, current_time: float) -> None:
        """Process emergent behavior and agent interactions."""
        active_agents = [agent for agent in self.agents if agent.state.value == "active"]
        
        if len(active_agents) < 2:
            return  # Need at least 2 agents for interactions
        
        interactions_processed = 0
        
        for agent in active_agents:
            # Assess collaboration opportunities
            opportunities = agent.assess_collaboration_opportunities(active_agents, current_time)
            
            # Process top collaboration opportunities
            for opportunity in opportunities[:2]:  # Limit to top 2 to avoid chaos
                other_agent_id = opportunity["agent_id"]
                other_agent = next((a for a in active_agents if a.agent_id == other_agent_id), None)
                
                if other_agent and opportunity["compatibility"] > 0.6:
                    influence_type = opportunity["recommended_influence"]
                    influence_strength = opportunity["compatibility"] * 0.5  # Moderate influence
                    
                    agent.influence_agent_behavior(other_agent, influence_type, influence_strength)
                    interactions_processed += 1
                    
                    if interactions_processed % 5 == 0:  # Report occasional interactions
                        print(f"  [t={current_time:.2f}] 🤝 {agent.agent_id} → {other_agent.agent_id}: "
                              f"{influence_type} (strength: {influence_strength:.2f})")
    
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
    parser.add_argument("--random-seed", type=int, 
                       help="Random seed for reproducibility (omit for truly random)")
    parser.add_argument("--strict-mode", action="store_true", default=False,
                       help="Enable strict token budgets for lightweight models (opt-in by default)")
    parser.add_argument("--no-strict-mode", action="store_true",
                       help="Disable strict token budgets (use flexible mode)")
    parser.add_argument("--max-concurrent", type=int, default=4,
                       help="Maximum concurrent agents (default: 4)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable verbose debug output showing LLM calls and agent details")
    parser.add_argument("--server-config", 
                       help="Path to multi-server configuration JSON file")
    
    args = parser.parse_args()
    
    # Create blog writer
    # Handle strict mode logic
    use_strict_mode = args.strict_mode and not args.no_strict_mode
    
    writer = FelixBlogWriter(
        lm_studio_url=args.lm_studio_url, 
        random_seed=args.random_seed,
        strict_mode=use_strict_mode,
        max_concurrent_agents=args.max_concurrent,
        debug_mode=args.debug,
        server_config_path=args.server_config
    )
    
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
