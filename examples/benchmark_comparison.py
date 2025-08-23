#!/usr/bin/env python3
"""
Performance Benchmark: Felix vs Linear Multi-Agent Systems.

This benchmark compares the Felix Framework's geometric orchestration
against traditional linear multi-agent approaches for LLM-powered tasks.

The comparison helps validate whether helix-based coordination provides
measurable advantages over sequential processing for real-world tasks.

Usage:
    python examples/benchmark_comparison.py --task "research quantum computing"
    python examples/benchmark_comparison.py --task-file tasks.txt --runs 5

Requirements:
    - LM Studio running with a model loaded
    - openai Python package installed
"""

import sys
import time
import json
import asyncio
import argparse
import statistics
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.helix_geometry import HelixGeometry
from src.llm.lm_studio_client import LMStudioClient, LMStudioConnectionError
from src.agents.llm_agent import LLMTask
from src.agents.specialized_agents import create_specialized_team
from src.communication.central_post import CentralPost
from src.communication.spoke import SpokeManager


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    approach: str
    task_description: str
    run_id: int
    total_time: float
    total_tokens: int
    final_output: str
    output_quality_score: float
    agent_count: int
    communication_messages: int
    memory_usage_estimate: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class BenchmarkSummary:
    """Summary statistics across multiple runs."""
    approach: str
    task_description: str
    run_count: int
    avg_total_time: float
    avg_tokens: float
    avg_quality_score: float
    time_std_dev: float
    tokens_std_dev: float
    quality_std_dev: float
    success_rate: float
    best_output: str
    worst_output: str


class LinearMultiAgentSystem:
    """
    Traditional linear multi-agent system for comparison.
    
    Processes tasks sequentially through a pipeline of agents,
    representing the traditional approach to multi-agent coordination.
    """
    
    def __init__(self, llm_client: LMStudioClient):
        """Initialize linear system."""
        self.llm_client = llm_client
        self.central_post = CentralPost(max_agents=10, enable_metrics=True)
        
    def process_task_linear(self, task: LLMTask) -> Dict[str, Any]:
        """
        Process task through linear pipeline.
        
        Args:
            task: Task to process
            
        Returns:
            Processing results
        """
        start_time = time.perf_counter()
        results = []
        
        # Stage 1: Research (single agent)
        try:
            research_response = self.llm_client.complete(
                agent_id="linear_research",
                system_prompt="You are a research agent. Gather comprehensive information about the given topic.",
                user_prompt=task.description,
                temperature=0.7
            )
            results.append(("research", research_response))
        except Exception as e:
            return {"success": False, "error": f"Research stage failed: {e}"}
        
        # Stage 2: Analysis (single agent, uses research results)
        try:
            analysis_prompt = f"Analyze the following research findings and organize them:\n\n{research_response.content}"
            analysis_response = self.llm_client.complete(
                agent_id="linear_analysis",
                system_prompt="You are an analysis agent. Process and organize information from research.",
                user_prompt=analysis_prompt,
                temperature=0.5
            )
            results.append(("analysis", analysis_response))
        except Exception as e:
            return {"success": False, "error": f"Analysis stage failed: {e}"}
        
        # Stage 3: Synthesis (single agent, uses all previous results)
        try:
            synthesis_prompt = f"""Create a final comprehensive response based on:
            
Research: {research_response.content}

Analysis: {analysis_response.content}

Original task: {task.description}"""
            
            synthesis_response = self.llm_client.complete(
                agent_id="linear_synthesis",
                system_prompt="You are a synthesis agent. Create the final comprehensive output.",
                user_prompt=synthesis_prompt,
                temperature=0.3
            )
            results.append(("synthesis", synthesis_response))
        except Exception as e:
            return {"success": False, "error": f"Synthesis stage failed: {e}"}
        
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_tokens = sum(r[1].tokens_used for r in results)
        total_time = end_time - start_time
        final_output = synthesis_response.content
        
        return {
            "success": True,
            "total_time": total_time,
            "total_tokens": total_tokens,
            "final_output": final_output,
            "agent_count": 3,  # Linear: 1 per stage
            "communication_messages": 2,  # Research->Analysis, Analysis->Synthesis
            "memory_usage_estimate": 100.0,  # Simple sequential memory
            "stage_results": results
        }


class FelixMultiAgentSystem:
    """
    Felix geometric orchestration system.
    
    Processes tasks using helix-based agent coordination with
    natural convergence and spoke-based communication.
    """
    
    def __init__(self, llm_client: LMStudioClient):
        """Initialize Felix system."""
        self.llm_client = llm_client
        self.helix = HelixGeometry(
            top_radius=33.0,
            bottom_radius=0.001,
            height=33.0,
            turns=33
        )
        self.central_post = CentralPost(max_agents=20, enable_metrics=True)
        self.spoke_manager = SpokeManager(self.central_post)
    
    def process_task_felix(self, task: LLMTask) -> Dict[str, Any]:
        """
        Process task using Felix geometric orchestration.
        
        Args:
            task: Task to process
            
        Returns:
            Processing results
        """
        start_time = time.perf_counter()
        
        # Create specialized team
        agents = create_specialized_team(
            helix=self.helix,
            llm_client=self.llm_client,
            task_complexity="medium"
        )
        
        # Register agents
        for agent in agents:
            self.spoke_manager.register_agent(agent)
        
        # Run geometric orchestration simulation
        current_time = 0.0
        time_step = 0.05
        simulation_time = 1.0
        final_output = None
        agent_results = []
        
        while current_time <= simulation_time and not final_output:
            # Process agents based on spawn timing
            for agent in agents:
                if (agent.can_spawn(current_time) and 
                    agent.state.value == "waiting"):
                    
                    try:
                        # Spawn and process
                        agent.spawn(current_time, task)
                        result = agent.process_task_with_llm(task, current_time)
                        
                        # Share results with central post via spoke communication
                        message = agent.share_result_to_central(result)
                        self.spoke_manager.send_message(agent.agent_id, message)
                        # Central post will handle distribution through spoke system
                        
                        agent_results.append(result)
                        
                        # Check for final synthesis
                        if agent.agent_type == "synthesis":
                            if hasattr(agent, 'finalize_output'):
                                final_output_data = agent.finalize_output(result)
                                final_output = final_output_data["content"]
                            else:
                                final_output = result.content
                    
                    except Exception as e:
                        return {"success": False, "error": f"Agent {agent.agent_id} failed: {e}"}
            
            # Update positions and process communication
            for agent in agents:
                if agent.state.value == "active":
                    agent.update_position(current_time)
            
            self.spoke_manager.process_all_messages()
            current_time += time_step
        
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_tokens = sum(r.llm_response.tokens_used for r in agent_results)
        total_time = end_time - start_time
        
        if not final_output and agent_results:
            # Fallback: use last synthesis result
            synthesis_results = [r for r in agent_results if "synthesis" in r.agent_id]
            final_output = synthesis_results[-1].content if synthesis_results else agent_results[-1].content
        
        return {
            "success": True,
            "total_time": total_time,
            "total_tokens": total_tokens,
            "final_output": final_output or "No final output generated",
            "agent_count": len(agents),
            "communication_messages": self.central_post.total_messages_processed,
            "memory_usage_estimate": len(agents) * 20.0,  # Geometric memory overhead
            "agent_results": agent_results
        }


class BenchmarkRunner:
    """
    Main benchmark runner that coordinates comparisons.
    
    Runs both systems on the same tasks and collects performance metrics
    for statistical comparison.
    """
    
    def __init__(self, lm_studio_url: str = "http://localhost:1234/v1"):
        """Initialize benchmark runner."""
        self.llm_client = LMStudioClient(base_url=lm_studio_url)
        self.linear_system = LinearMultiAgentSystem(self.llm_client)
        self.felix_system = FelixMultiAgentSystem(self.llm_client)
        
        print("Benchmark Runner initialized")
    
    def test_connection(self) -> bool:
        """Test LM Studio connection."""
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
    
    def calculate_quality_score(self, output: str, task_description: str) -> float:
        """
        Calculate simple quality score for output.
        
        Args:
            output: Generated output
            task_description: Original task
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        # Simple heuristics for quality assessment
        length_score = min(len(output) / 1000, 1.0)  # Longer is better up to 1000 chars
        
        # Check for task-relevant content
        task_words = task_description.lower().split()
        output_lower = output.lower()
        relevance_score = sum(1 for word in task_words if word in output_lower) / len(task_words)
        
        # Structure score (check for organized content)
        structure_indicators = ["introduction", "conclusion", "summary", "analysis", "research"]
        structure_score = sum(0.1 for indicator in structure_indicators if indicator in output_lower)
        structure_score = min(structure_score, 0.5)
        
        return (length_score + relevance_score + structure_score) / 2.5
    
    def run_single_benchmark(self, task_description: str, approach: str, run_id: int) -> BenchmarkResult:
        """
        Run single benchmark for one approach.
        
        Args:
            task_description: Task to perform
            approach: "linear" or "felix"
            run_id: Run identifier
            
        Returns:
            Benchmark result
        """
        print(f"  Running {approach} approach (run {run_id})...")
        
        task = LLMTask(
            task_id=f"benchmark_{run_id}",
            description=task_description,
            context="This is a benchmark comparison task."
        )
        
        try:
            if approach == "linear":
                results = self.linear_system.process_task_linear(task)
            else:  # felix
                results = self.felix_system.process_task_felix(task)
            
            if not results["success"]:
                return BenchmarkResult(
                    approach=approach,
                    task_description=task_description,
                    run_id=run_id,
                    total_time=0.0,
                    total_tokens=0,
                    final_output="",
                    output_quality_score=0.0,
                    agent_count=0,
                    communication_messages=0,
                    memory_usage_estimate=0.0,
                    success=False,
                    error_message=results.get("error", "Unknown error")
                )
            
            quality_score = self.calculate_quality_score(results["final_output"], task_description)
            
            return BenchmarkResult(
                approach=approach,
                task_description=task_description,
                run_id=run_id,
                total_time=results["total_time"],
                total_tokens=results["total_tokens"],
                final_output=results["final_output"],
                output_quality_score=quality_score,
                agent_count=results["agent_count"],
                communication_messages=results["communication_messages"],
                memory_usage_estimate=results["memory_usage_estimate"],
                success=True
            )
        
        except Exception as e:
            return BenchmarkResult(
                approach=approach,
                task_description=task_description,
                run_id=run_id,
                total_time=0.0,
                total_tokens=0,
                final_output="",
                output_quality_score=0.0,
                agent_count=0,
                communication_messages=0,
                memory_usage_estimate=0.0,
                success=False,
                error_message=str(e)
            )
    
    def run_benchmark_comparison(self, task_description: str, runs: int = 3) -> Dict[str, Any]:
        """
        Run complete benchmark comparison.
        
        Args:
            task_description: Task to benchmark
            runs: Number of runs per approach
            
        Returns:
            Comparison results
        """
        print(f"\n{'='*60}")
        print(f"BENCHMARK COMPARISON")
        print(f"Task: {task_description}")
        print(f"Runs per approach: {runs}")
        print(f"{'='*60}")
        
        all_results = []
        
        # Run linear approach
        print(f"\nRunning Linear Pipeline Approach...")
        for run_id in range(runs):
            result = self.run_single_benchmark(task_description, "linear", run_id)
            all_results.append(result)
            if result.success:
                print(f"    Run {run_id}: {result.total_time:.2f}s, {result.total_tokens} tokens, quality={result.output_quality_score:.2f}")
            else:
                print(f"    Run {run_id}: FAILED - {result.error_message}")
        
        # Run Felix approach  
        print(f"\nRunning Felix Geometric Orchestration...")
        for run_id in range(runs):
            result = self.run_single_benchmark(task_description, "felix", run_id)
            all_results.append(result)
            if result.success:
                print(f"    Run {run_id}: {result.total_time:.2f}s, {result.total_tokens} tokens, quality={result.output_quality_score:.2f}")
            else:
                print(f"    Run {run_id}: FAILED - {result.error_message}")
        
        # Analyze results
        return self.analyze_benchmark_results(all_results, task_description)
    
    def analyze_benchmark_results(self, results: List[BenchmarkResult], task_description: str) -> Dict[str, Any]:
        """Analyze and summarize benchmark results."""
        # Separate by approach
        linear_results = [r for r in results if r.approach == "linear" and r.success]
        felix_results = [r for r in results if r.approach == "felix" and r.success]
        
        # Calculate summaries
        summaries = {}
        
        for approach, approach_results in [("linear", linear_results), ("felix", felix_results)]:
            if approach_results:
                times = [r.total_time for r in approach_results]
                tokens = [r.total_tokens for r in approach_results] 
                qualities = [r.output_quality_score for r in approach_results]
                
                # Find best and worst outputs
                best_result = max(approach_results, key=lambda r: r.output_quality_score)
                worst_result = min(approach_results, key=lambda r: r.output_quality_score)
                
                summary = BenchmarkSummary(
                    approach=approach,
                    task_description=task_description,
                    run_count=len(approach_results),
                    avg_total_time=statistics.mean(times),
                    avg_tokens=statistics.mean(tokens),
                    avg_quality_score=statistics.mean(qualities),
                    time_std_dev=statistics.stdev(times) if len(times) > 1 else 0.0,
                    tokens_std_dev=statistics.stdev(tokens) if len(tokens) > 1 else 0.0,
                    quality_std_dev=statistics.stdev(qualities) if len(qualities) > 1 else 0.0,
                    success_rate=len(approach_results) / sum(1 for r in results if r.approach == approach),
                    best_output=best_result.final_output,
                    worst_output=worst_result.final_output
                )
                summaries[approach] = summary
        
        return {
            "task_description": task_description,
            "raw_results": results,
            "summaries": summaries,
            "comparison": self._compare_approaches(summaries) if len(summaries) == 2 else None
        }
    
    def _compare_approaches(self, summaries: Dict[str, BenchmarkSummary]) -> Dict[str, Any]:
        """Compare the two approaches statistically."""
        linear = summaries["linear"]
        felix = summaries["felix"]
        
        comparison = {
            "time_improvement": ((linear.avg_total_time - felix.avg_total_time) / linear.avg_total_time) * 100,
            "token_efficiency": ((linear.avg_tokens - felix.avg_tokens) / linear.avg_tokens) * 100,
            "quality_improvement": ((felix.avg_quality_score - linear.avg_quality_score) / linear.avg_quality_score) * 100,
            "winner_by_time": "felix" if felix.avg_total_time < linear.avg_total_time else "linear",
            "winner_by_quality": "felix" if felix.avg_quality_score > linear.avg_quality_score else "linear",
            "winner_by_tokens": "felix" if felix.avg_tokens < linear.avg_tokens else "linear"
        }
        
        return comparison
    
    def display_results(self, analysis: Dict[str, Any]) -> None:
        """Display benchmark results."""
        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS")
        print(f"{'='*60}")
        
        summaries = analysis["summaries"]
        
        print(f"\nTask: {analysis['task_description']}")
        print(f"\nPerformance Summary:")
        
        for approach, summary in summaries.items():
            print(f"\n{approach.upper()} APPROACH:")
            print(f"  Success Rate: {summary.success_rate:.1%}")
            print(f"  Avg Time: {summary.avg_total_time:.2f}s (±{summary.time_std_dev:.2f})")
            print(f"  Avg Tokens: {summary.avg_tokens:.0f} (±{summary.tokens_std_dev:.0f})")
            print(f"  Avg Quality: {summary.avg_quality_score:.3f} (±{summary.quality_std_dev:.3f})")
        
        if analysis["comparison"]:
            comp = analysis["comparison"]
            print(f"\nCOMPARISON:")
            print(f"  Time: Felix is {comp['time_improvement']:+.1f}% vs Linear")
            print(f"  Tokens: Felix uses {comp['token_efficiency']:+.1f}% tokens vs Linear")
            print(f"  Quality: Felix is {comp['quality_improvement']:+.1f}% quality vs Linear")
            print(f"  Best Time: {comp['winner_by_time']}")
            print(f"  Best Quality: {comp['winner_by_quality']}")
            print(f"  Best Token Efficiency: {comp['winner_by_tokens']}")
        
        # Show best outputs
        for approach, summary in summaries.items():
            print(f"\n{'='*60}")
            print(f"BEST OUTPUT - {approach.upper()}")
            print(f"{'='*60}")
            print(summary.best_output[:500] + ("..." if len(summary.best_output) > 500 else ""))
    
    def save_results(self, analysis: Dict[str, Any], output_file: str) -> None:
        """Save benchmark results to JSON file."""
        # Convert dataclasses to dicts for JSON serialization
        serializable_analysis = {
            "task_description": analysis["task_description"],
            "raw_results": [asdict(r) for r in analysis["raw_results"]],
            "summaries": {k: asdict(v) for k, v in analysis["summaries"].items()},
            "comparison": analysis["comparison"]
        }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_analysis, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


def main():
    """Main function for benchmark comparison."""
    parser = argparse.ArgumentParser(description="Felix vs Linear Multi-Agent Benchmark")
    parser.add_argument("--task", help="Task description to benchmark")
    parser.add_argument("--task-file", help="File containing task descriptions (one per line)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per approach")
    parser.add_argument("--lm-studio-url", default="http://localhost:1234/v1",
                       help="LM Studio API URL")
    parser.add_argument("--output", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    if not args.task and not args.task_file:
        parser.error("Must provide either --task or --task-file")
    
    # Create benchmark runner
    runner = BenchmarkRunner(lm_studio_url=args.lm_studio_url)
    
    # Test connection
    if not runner.test_connection():
        print("\nPlease ensure LM Studio is running with a model loaded.")
        sys.exit(1)
    
    # Get tasks to benchmark
    tasks = []
    if args.task:
        tasks = [args.task]
    elif args.task_file:
        try:
            with open(args.task_file, 'r') as f:
                tasks = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading task file: {e}")
            sys.exit(1)
    
    # Run benchmarks
    all_analyses = []
    for i, task in enumerate(tasks):
        print(f"\n{'#'*60}")
        print(f"BENCHMARK {i+1}/{len(tasks)}")
        print(f"{'#'*60}")
        
        analysis = runner.run_benchmark_comparison(task, runs=args.runs)
        all_analyses.append(analysis)
        runner.display_results(analysis)
        
        if args.output:
            output_file = args.output if len(tasks) == 1 else f"{args.output}_{i+1}.json"
            runner.save_results(analysis, output_file)
    
    print(f"\nBenchmark comparison completed!")


if __name__ == "__main__":
    main()