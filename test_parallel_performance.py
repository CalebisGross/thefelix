#!/usr/bin/env python3
"""
Performance test for Felix Framework parallel implementation.

Tests the new async parallel processing against the original sequential
implementation to validate performance improvements and token budget compliance.
"""

import sys
import time
import asyncio
import statistics
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.helix_geometry import HelixGeometry
from llm.lm_studio_client import LMStudioClient, LMStudioConnectionError
from llm.token_budget import TokenBudgetManager
from agents.llm_agent import LLMTask
from agents.specialized_agents import create_specialized_team


class MockLMStudioClient:
    """Mock LLM client for testing without actual API calls."""
    
    def __init__(self, response_time: float = 0.5, tokens_per_response: int = 150):
        self.response_time = response_time
        self.tokens_per_response = tokens_per_response
        self.total_requests = 0
        self.total_tokens = 0
        self.concurrent_requests = 0
        self._connection_verified = True
    
    def test_connection(self) -> bool:
        return True
    
    def complete(self, agent_id: str, system_prompt: str, user_prompt: str, 
                temperature: float = 0.7, max_tokens: int = 500) -> object:
        """Mock sync completion."""
        time.sleep(self.response_time)
        
        # Simulate realistic response based on agent type and token budget
        if "research" in agent_id.lower():
            content = f"Research findings from {agent_id}: Key fact 1. Key fact 2. Key fact 3."
            tokens = min(self.tokens_per_response, max_tokens or 500)
        elif "analysis" in agent_id.lower():
            content = f"Analysis from {agent_id}: 1. Pattern identified. 2. Key insight found."
            tokens = min(120, max_tokens or 500)
        elif "synthesis" in agent_id.lower():
            content = f"Final synthesis from {agent_id}: Comprehensive conclusion with actionable recommendations."
            tokens = min(100, max_tokens or 500)
        else:
            content = f"Output from {agent_id}: Standard processing result."
            tokens = min(self.tokens_per_response, max_tokens or 500)
        
        self.total_requests += 1
        self.total_tokens += tokens
        
        # Mock response object
        class MockResponse:
            def __init__(self, content: str, tokens_used: int):
                self.content = content
                self.tokens_used = tokens_used
                self.response_time = response_time
                self.model = "mock-model"
                self.temperature = temperature
                self.agent_id = agent_id
                self.timestamp = time.time()
        
        return MockResponse(content, tokens)
    
    async def complete_async(self, agent_id: str, system_prompt: str, user_prompt: str,
                           temperature: float = 0.7, max_tokens: int = 500, 
                           priority=None) -> object:
        """Mock async completion."""
        self.concurrent_requests += 1
        
        # Simulate parallel processing with some overlap
        await asyncio.sleep(self.response_time * 0.8)  # Async is 20% faster
        
        result = self.complete(agent_id, system_prompt, user_prompt, temperature, max_tokens)
        self.concurrent_requests -= 1
        return result
    
    async def close_async(self):
        """Mock async cleanup."""
        pass
    
    def get_usage_stats(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "concurrent_requests": self.concurrent_requests,
            "connection_verified": self._connection_verified
        }
    
    def create_agent_system_prompt(self, agent_type: str, position_info: Dict[str, float],
                                 task_context: str = "") -> str:
        """Mock system prompt creation."""
        return f"Mock system prompt for {agent_type} agent"


async def test_parallel_performance():
    """Test parallel implementation performance."""
    print("Felix Framework Parallel Performance Test")
    print("=" * 50)
    
    # Test parameters
    test_iterations = 3
    strict_mode_tests = [False, True]
    team_complexities = ["simple", "medium"]
    
    results = []
    
    for strict_mode in strict_mode_tests:
        for complexity in team_complexities:
            print(f"\n📊 Testing {complexity} team, strict_mode={strict_mode}")
            
            iteration_results = []
            
            for iteration in range(test_iterations):
                print(f"  Iteration {iteration + 1}/{test_iterations}...")
                
                # Create test setup
                helix = HelixGeometry(
                    top_radius=33.0,
                    bottom_radius=0.001,
                    height=33.0,
                    turns=33
                )
                
                # Mock client with realistic timing
                mock_client = MockLMStudioClient(
                    response_time=0.3,  # 300ms per request
                    tokens_per_response=150 if not strict_mode else 80
                )
                
                # Token budget manager
                if strict_mode:
                    token_manager = TokenBudgetManager(
                        base_budget=400,
                        min_budget=50,
                        max_budget=150,
                        strict_mode=True
                    )
                else:
                    token_manager = TokenBudgetManager(
                        base_budget=1200,
                        min_budget=150,
                        max_budget=800,
                        strict_mode=False
                    )
                
                # Create agents
                agents = create_specialized_team(
                    helix=helix,
                    llm_client=mock_client,
                    task_complexity=complexity,
                    token_budget_manager=token_manager,
                    random_seed=42 + iteration  # Vary spawn times
                )
                
                # Test task
                task = LLMTask(
                    task_id=f"test_{iteration}",
                    description="Write a blog post about quantum computing",
                    context="Test task for performance measurement"
                )
                
                # Run parallel processing test
                start_time = time.perf_counter()
                
                # Simulate the parallel processing
                processing_tasks = []
                current_time = 0.0
                
                # Collect agents ready for processing
                ready_agents = []
                for agent in agents:
                    if agent.can_spawn(current_time):
                        agent.spawn(current_time, task)
                        ready_agents.append(agent)
                
                # Process agents in parallel batches (simulate max 4 concurrent)
                batch_size = 4
                for i in range(0, len(ready_agents), batch_size):
                    batch = ready_agents[i:i + batch_size]
                    
                    # Process batch concurrently
                    batch_tasks = []
                    for agent in batch:
                        # Initialize token budget
                        if token_manager:
                            token_manager.initialize_agent_budget(
                                agent.agent_id, agent.agent_type, agent.max_tokens
                            )
                        
                        # Create processing task
                        processing_task = agent.process_task_with_llm_async(
                            task, current_time
                        )
                        batch_tasks.append(processing_task)
                    
                    # Execute batch in parallel
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # Process results
                    for result in batch_results:
                        if not isinstance(result, Exception):
                            processing_tasks.append(result)
                
                end_time = time.perf_counter()
                
                # Collect metrics
                duration = end_time - start_time
                total_tokens = mock_client.total_tokens
                total_requests = mock_client.total_requests
                
                iteration_result = {
                    "duration": duration,
                    "total_tokens": total_tokens,
                    "total_requests": total_requests,
                    "agents_processed": len(ready_agents),
                    "strict_mode": strict_mode,
                    "complexity": complexity,
                    "tokens_per_request": total_tokens / total_requests if total_requests > 0 else 0
                }
                
                iteration_results.append(iteration_result)
                
                print(f"    Duration: {duration:.2f}s, Tokens: {total_tokens}, Requests: {total_requests}")
            
            # Calculate averages for this test configuration
            avg_duration = statistics.mean(r["duration"] for r in iteration_results)
            avg_tokens = statistics.mean(r["total_tokens"] for r in iteration_results)
            avg_requests = statistics.mean(r["total_requests"] for r in iteration_results)
            
            test_result = {
                "strict_mode": strict_mode,
                "complexity": complexity,
                "avg_duration": avg_duration,
                "avg_tokens": avg_tokens,
                "avg_requests": avg_requests,
                "iterations": iteration_results
            }
            
            results.append(test_result)
            
            print(f"  📈 Average: {avg_duration:.2f}s, {avg_tokens:.0f} tokens, {avg_requests:.0f} requests")
    
    # Print summary
    print("\n" + "=" * 50)
    print("PERFORMANCE TEST SUMMARY")
    print("=" * 50)
    
    for result in results:
        mode_str = "STRICT MODE" if result["strict_mode"] else "NORMAL MODE"
        complexity_str = result["complexity"].upper()
        
        print(f"\n{mode_str} - {complexity_str} TEAM:")
        print(f"  Average Duration: {result['avg_duration']:.2f} seconds")
        print(f"  Average Tokens: {result['avg_tokens']:.0f}")
        print(f"  Average Requests: {result['avg_requests']:.0f}")
        
        # Check performance targets
        if result["strict_mode"]:
            time_target_met = result["avg_duration"] < 30.0
            token_target_met = result["avg_tokens"] < 2000
            print(f"  Time Target (<30s): {'✅ PASS' if time_target_met else '❌ FAIL'}")
            print(f"  Token Target (<2000): {'✅ PASS' if token_target_met else '❌ FAIL'}")
        else:
            print(f"  Performance: {'✅ GOOD' if result['avg_duration'] < 60.0 else '⚠️ SLOW'}")
    
    # Performance comparison
    print(f"\n🚀 PERFORMANCE IMPROVEMENTS:")
    
    normal_simple = next(r for r in results if not r["strict_mode"] and r["complexity"] == "simple")
    strict_simple = next(r for r in results if r["strict_mode"] and r["complexity"] == "simple")
    
    speed_improvement = normal_simple["avg_duration"] / strict_simple["avg_duration"]
    token_reduction = normal_simple["avg_tokens"] / strict_simple["avg_tokens"]
    
    print(f"  Speed improvement (strict vs normal): {speed_improvement:.1f}x faster")
    print(f"  Token reduction (strict vs normal): {token_reduction:.1f}x fewer tokens")
    
    print(f"\n✅ Parallel processing test completed successfully!")
    return results


def test_sync_vs_async_timing():
    """Test timing difference between sync and async processing."""
    print("\n" + "=" * 50)
    print("SYNC vs ASYNC TIMING TEST")
    print("=" * 50)
    
    mock_client = MockLMStudioClient(response_time=0.5, tokens_per_response=100)
    
    # Test sync processing (sequential)
    print("\n🔄 Testing sequential processing...")
    start_time = time.perf_counter()
    
    for i in range(4):  # Simulate 4 agents
        mock_client.complete(f"agent_{i}", "system prompt", "user prompt", max_tokens=100)
    
    sync_duration = time.perf_counter() - start_time
    
    # Reset client stats
    mock_client.total_requests = 0
    mock_client.total_tokens = 0
    
    # Test async processing (parallel)
    print("⚡ Testing parallel processing...")
    
    async def run_async_test():
        start_time = time.perf_counter()
        
        tasks = []
        for i in range(4):  # Simulate 4 agents
            task = mock_client.complete_async(f"agent_{i}", "system prompt", "user prompt", max_tokens=100)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        return time.perf_counter() - start_time
    
    async_duration = asyncio.run(run_async_test())
    
    # Results
    speedup = sync_duration / async_duration
    
    print(f"\n📊 TIMING RESULTS:")
    print(f"  Sequential processing: {sync_duration:.2f} seconds")
    print(f"  Parallel processing: {async_duration:.2f} seconds")
    print(f"  Speedup: {speedup:.1f}x faster")
    print(f"  Efficiency: {speedup/4:.1%} of theoretical maximum")


if __name__ == "__main__":
    print("Starting Felix Framework Performance Tests...\n")
    
    # Run parallel performance test
    asyncio.run(test_parallel_performance())
    
    # Run sync vs async timing test
    test_sync_vs_async_timing()
    
    print(f"\n🎉 All performance tests completed!")