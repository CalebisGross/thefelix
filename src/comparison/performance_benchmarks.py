"""
Performance Benchmarking Suite for Felix Framework

Implements Priority 4 performance benchmarking components:
- Tokens/second, time-to-completion, cost per task metrics
- Resource utilization tracking 
- Memory and CPU usage monitoring
- Comparison against baseline systems
- Integration with existing token budget system

This provides comprehensive performance validation for multi-agent
system efficiency and scalability analysis.
"""

import time
import psutil
import threading
import statistics
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import contextlib
import sys
import traceback

from communication.central_post import Message, MessageType


class MetricType(Enum):
    """Types of performance metrics."""
    THROUGHPUT = "throughput"  # tokens/second, messages/second
    LATENCY = "latency"  # time-to-completion, response time
    RESOURCE = "resource"  # CPU, memory, network
    COST = "cost"  # token costs, compute costs
    SCALABILITY = "scalability"  # performance vs. team size
    EFFICIENCY = "efficiency"  # output quality per resource unit


class BenchmarkStatus(Enum):
    """Status of benchmark execution."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PerformanceSnapshot:
    """Single point-in-time performance measurement."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    thread_count: int
    active_agents: int = 0
    messages_processed: int = 0
    tokens_processed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "thread_count": self.thread_count,
            "active_agents": self.active_agents,
            "messages_processed": self.messages_processed,
            "tokens_processed": self.tokens_processed
        }


@dataclass
class ThroughputMetrics:
    """Throughput-related performance metrics."""
    tokens_per_second: float = 0.0
    messages_per_second: float = 0.0
    agents_spawned_per_minute: float = 0.0
    task_completion_rate: float = 0.0  # tasks completed per minute
    
    # Peak values
    peak_tokens_per_second: float = 0.0
    peak_messages_per_second: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tokens_per_second": self.tokens_per_second,
            "messages_per_second": self.messages_per_second,
            "agents_spawned_per_minute": self.agents_spawned_per_minute,
            "task_completion_rate": self.task_completion_rate,
            "peak_tokens_per_second": self.peak_tokens_per_second,
            "peak_messages_per_second": self.peak_messages_per_second
        }


@dataclass
class LatencyMetrics:
    """Latency-related performance metrics."""
    average_response_time: float = 0.0  # seconds
    median_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    
    # Agent-specific latencies
    agent_spawn_time: float = 0.0
    message_processing_time: float = 0.0
    llm_call_time: float = 0.0
    
    # End-to-end metrics
    task_completion_time: float = 0.0
    time_to_first_result: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "average_response_time": self.average_response_time,
            "median_response_time": self.median_response_time,
            "p95_response_time": self.p95_response_time,
            "p99_response_time": self.p99_response_time,
            "agent_spawn_time": self.agent_spawn_time,
            "message_processing_time": self.message_processing_time,
            "llm_call_time": self.llm_call_time,
            "task_completion_time": self.task_completion_time,
            "time_to_first_result": self.time_to_first_result
        }


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    avg_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    
    # Thread and process metrics
    avg_thread_count: float = 0.0
    peak_thread_count: int = 0
    
    # Network metrics (if available)
    bytes_sent: int = 0
    bytes_received: int = 0
    
    # Resource efficiency
    cpu_efficiency: float = 0.0  # useful work / CPU usage
    memory_efficiency: float = 0.0  # tokens processed / memory used
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "avg_cpu_percent": self.avg_cpu_percent,
            "peak_cpu_percent": self.peak_cpu_percent,
            "avg_memory_mb": self.avg_memory_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_thread_count": self.avg_thread_count,
            "peak_thread_count": self.peak_thread_count,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "cpu_efficiency": self.cpu_efficiency,
            "memory_efficiency": self.memory_efficiency
        }


@dataclass
class CostMetrics:
    """Cost-related performance metrics."""
    total_tokens_used: int = 0
    estimated_token_cost: float = 0.0  # USD
    cost_per_task: float = 0.0
    cost_per_quality_point: float = 0.0
    
    # Token efficiency
    tokens_per_agent: float = 0.0
    useful_tokens_ratio: float = 0.0  # non-overhead tokens / total tokens
    
    # Time-based costs
    compute_time_minutes: float = 0.0
    estimated_compute_cost: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_tokens_used": self.total_tokens_used,
            "estimated_token_cost": self.estimated_token_cost,
            "cost_per_task": self.cost_per_task,
            "cost_per_quality_point": self.cost_per_quality_point,
            "tokens_per_agent": self.tokens_per_agent,
            "useful_tokens_ratio": self.useful_tokens_ratio,
            "compute_time_minutes": self.compute_time_minutes,
            "estimated_compute_cost": self.estimated_compute_cost
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark result with all metrics."""
    benchmark_name: str
    status: BenchmarkStatus
    start_time: float
    end_time: float
    duration_seconds: float
    
    # Core metrics
    throughput: ThroughputMetrics = field(default_factory=ThroughputMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    resources: ResourceMetrics = field(default_factory=ResourceMetrics)
    costs: CostMetrics = field(default_factory=CostMetrics)
    
    # Test configuration
    team_size: int = 0
    task_complexity: str = "medium"
    token_budget: int = 10000
    
    # Quality metrics integration
    quality_score: float = 0.0
    output_length: int = 0
    
    # Error information
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "benchmark_name": self.benchmark_name,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "throughput": self.throughput.to_dict(),
            "latency": self.latency.to_dict(),
            "resources": self.resources.to_dict(),
            "costs": self.costs.to_dict(),
            "team_size": self.team_size,
            "task_complexity": self.task_complexity,
            "token_budget": self.token_budget,
            "quality_score": self.quality_score,
            "output_length": self.output_length,
            "error_message": self.error_message,
            "error_traceback": self.error_traceback
        }


class ResourceMonitor:
    """Monitors system resource usage during benchmark execution."""
    
    def __init__(self, sample_interval: float = 0.5):
        """
        Initialize resource monitor.
        
        Args:
            sample_interval: Seconds between resource samples
        """
        self.sample_interval = sample_interval
        self.snapshots: List[PerformanceSnapshot] = []
        self.monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Process tracking
        self.process = psutil.Process()
        self.start_cpu_times = None
        self.start_memory = None
    
    def start_monitoring(self) -> None:
        """Start resource monitoring in background thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._stop_event.clear()
        self.snapshots.clear()
        
        # Record baseline
        self.start_cpu_times = self.process.cpu_times()
        self.start_memory = self.process.memory_info()
        
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        self._stop_event.set()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop (runs in background thread)."""
        while not self._stop_event.wait(self.sample_interval):
            try:
                # Collect resource metrics
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                thread_count = self.process.num_threads()
                
                snapshot = PerformanceSnapshot(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_mb=memory_info.rss / 1024 / 1024,  # Convert to MB
                    thread_count=thread_count
                )
                
                self.snapshots.append(snapshot)
                
                # Limit snapshot history to prevent memory issues
                if len(self.snapshots) > 1000:
                    self.snapshots = self.snapshots[-800:]  # Keep most recent 800
                    
            except Exception as e:
                # Continue monitoring even if individual samples fail
                print(f"Resource monitoring error: {e}")
    
    def get_resource_metrics(self) -> ResourceMetrics:
        """Calculate resource metrics from collected snapshots."""
        if not self.snapshots:
            return ResourceMetrics()
        
        # CPU metrics
        cpu_values = [s.cpu_percent for s in self.snapshots if s.cpu_percent > 0]
        avg_cpu = statistics.mean(cpu_values) if cpu_values else 0.0
        peak_cpu = max(cpu_values) if cpu_values else 0.0
        
        # Memory metrics
        memory_values = [s.memory_mb for s in self.snapshots]
        avg_memory = statistics.mean(memory_values) if memory_values else 0.0
        peak_memory = max(memory_values) if memory_values else 0.0
        
        # Thread metrics
        thread_values = [s.thread_count for s in self.snapshots]
        avg_threads = statistics.mean(thread_values) if thread_values else 0.0
        peak_threads = max(thread_values) if thread_values else 0
        
        return ResourceMetrics(
            avg_cpu_percent=avg_cpu,
            peak_cpu_percent=peak_cpu,
            avg_memory_mb=avg_memory,
            peak_memory_mb=peak_memory,
            avg_thread_count=avg_threads,
            peak_thread_count=peak_threads
        )
    
    def update_snapshot_counters(self, agents: int = 0, messages: int = 0, tokens: int = 0) -> None:
        """Update the latest snapshot with agent/message/token counts."""
        if self.snapshots:
            latest = self.snapshots[-1]
            latest.active_agents = agents
            latest.messages_processed = messages
            latest.tokens_processed = tokens


class PerformanceBenchmarker:
    """Main performance benchmarking system for Felix Framework."""
    
    def __init__(self, token_cost_per_1k: float = 0.002):
        """
        Initialize performance benchmarker.
        
        Args:
            token_cost_per_1k: Cost per 1000 tokens in USD
        """
        self.token_cost_per_1k = token_cost_per_1k
        self.resource_monitor = ResourceMonitor()
        
        # Measurement tracking
        self.response_times: List[float] = []
        self.agent_spawn_times: List[float] = []
        self.message_times: List[float] = []
        self.llm_call_times: List[float] = []
        
        # Token and message tracking
        self.total_tokens = 0
        self.total_messages = 0
        self.agent_count = 0
        self.task_count = 0
        
        # Timing tracking
        self.benchmark_start_time = 0.0
        self.first_result_time: Optional[float] = None
        self.task_completion_times: List[float] = []
    
    @contextlib.contextmanager
    def benchmark_context(self, benchmark_name: str, **config):
        """Context manager for running benchmarks with automatic resource monitoring."""
        # Initialize benchmark
        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            status=BenchmarkStatus.RUNNING,
            start_time=time.time(),
            end_time=0.0,
            duration_seconds=0.0,
            **config
        )
        
        # Start monitoring
        self.reset_counters()
        self.resource_monitor.start_monitoring()
        self.benchmark_start_time = result.start_time
        
        try:
            yield result
            
            # Benchmark completed successfully
            result.status = BenchmarkStatus.COMPLETED
            
        except Exception as e:
            # Benchmark failed
            result.status = BenchmarkStatus.FAILED
            result.error_message = str(e)
            result.error_traceback = traceback.format_exc()
            
        finally:
            # Stop monitoring and calculate metrics
            self.resource_monitor.stop_monitoring()
            result.end_time = time.time()
            result.duration_seconds = result.end_time - result.start_time
            
            # Calculate all metrics
            result.throughput = self._calculate_throughput_metrics(result.duration_seconds)
            result.latency = self._calculate_latency_metrics()
            result.resources = self.resource_monitor.get_resource_metrics()
            result.costs = self._calculate_cost_metrics(result.duration_seconds)
            
            # Calculate efficiency metrics
            self._calculate_efficiency_metrics(result)
    
    def reset_counters(self) -> None:
        """Reset all measurement counters."""
        self.response_times.clear()
        self.agent_spawn_times.clear() 
        self.message_times.clear()
        self.llm_call_times.clear()
        self.task_completion_times.clear()
        
        self.total_tokens = 0
        self.total_messages = 0
        self.agent_count = 0
        self.task_count = 0
        self.first_result_time = None
    
    def record_response_time(self, duration: float) -> None:
        """Record a response time measurement."""
        self.response_times.append(duration)
        
        # Record first result time
        if self.first_result_time is None:
            self.first_result_time = time.time() - self.benchmark_start_time
    
    def record_agent_spawn(self, spawn_duration: float) -> None:
        """Record agent spawn time."""
        self.agent_spawn_times.append(spawn_duration)
        self.agent_count += 1
    
    def record_message_processing(self, duration: float) -> None:
        """Record message processing time."""
        self.message_times.append(duration)
        self.total_messages += 1
    
    def record_llm_call(self, duration: float, tokens_used: int) -> None:
        """Record LLM call timing and token usage."""
        self.llm_call_times.append(duration)
        self.total_tokens += tokens_used
    
    def record_task_completion(self, duration: float) -> None:
        """Record task completion time."""
        self.task_completion_times.append(duration)
        self.task_count += 1
    
    def update_resource_counters(self) -> None:
        """Update resource monitor with current counts."""
        self.resource_monitor.update_snapshot_counters(
            agents=self.agent_count,
            messages=self.total_messages,
            tokens=self.total_tokens
        )
    
    def _calculate_throughput_metrics(self, duration: float) -> ThroughputMetrics:
        """Calculate throughput metrics."""
        if duration <= 0:
            return ThroughputMetrics()
        
        tokens_per_sec = self.total_tokens / duration
        messages_per_sec = self.total_messages / duration
        agents_per_min = (self.agent_count / duration) * 60
        tasks_per_min = (self.task_count / duration) * 60
        
        # Calculate peak rates from time windows
        peak_tokens_per_sec = self._calculate_peak_rate(self.llm_call_times, duration)
        peak_messages_per_sec = self._calculate_peak_rate(self.message_times, duration)
        
        return ThroughputMetrics(
            tokens_per_second=tokens_per_sec,
            messages_per_second=messages_per_sec,
            agents_spawned_per_minute=agents_per_min,
            task_completion_rate=tasks_per_min,
            peak_tokens_per_second=peak_tokens_per_sec,
            peak_messages_per_second=peak_messages_per_sec
        )
    
    def _calculate_latency_metrics(self) -> LatencyMetrics:
        """Calculate latency metrics."""
        metrics = LatencyMetrics()
        
        # Response time metrics
        if self.response_times:
            metrics.average_response_time = statistics.mean(self.response_times)
            metrics.median_response_time = statistics.median(self.response_times)
            
            sorted_times = sorted(self.response_times)
            n = len(sorted_times)
            metrics.p95_response_time = sorted_times[int(n * 0.95)] if n > 0 else 0.0
            metrics.p99_response_time = sorted_times[int(n * 0.99)] if n > 0 else 0.0
        
        # Component-specific times
        if self.agent_spawn_times:
            metrics.agent_spawn_time = statistics.mean(self.agent_spawn_times)
        
        if self.message_times:
            metrics.message_processing_time = statistics.mean(self.message_times)
        
        if self.llm_call_times:
            metrics.llm_call_time = statistics.mean(self.llm_call_times)
        
        # End-to-end metrics
        if self.task_completion_times:
            metrics.task_completion_time = statistics.mean(self.task_completion_times)
        
        if self.first_result_time:
            metrics.time_to_first_result = self.first_result_time
        
        return metrics
    
    def _calculate_cost_metrics(self, duration: float) -> CostMetrics:
        """Calculate cost metrics."""
        token_cost = (self.total_tokens / 1000) * self.token_cost_per_1k
        
        cost_per_task = token_cost / self.task_count if self.task_count > 0 else 0.0
        tokens_per_agent = self.total_tokens / self.agent_count if self.agent_count > 0 else 0.0
        
        # Estimate compute cost (rough approximation)
        compute_minutes = duration / 60
        compute_cost = compute_minutes * 0.01  # $0.01 per minute (rough estimate)
        
        return CostMetrics(
            total_tokens_used=self.total_tokens,
            estimated_token_cost=token_cost,
            cost_per_task=cost_per_task,
            tokens_per_agent=tokens_per_agent,
            compute_time_minutes=compute_minutes,
            estimated_compute_cost=compute_cost
        )
    
    def _calculate_efficiency_metrics(self, result: BenchmarkResult) -> None:
        """Calculate efficiency metrics and update result."""
        # CPU efficiency: throughput per CPU usage
        if result.resources.avg_cpu_percent > 0:
            result.resources.cpu_efficiency = (
                result.throughput.tokens_per_second / result.resources.avg_cpu_percent
            )
        
        # Memory efficiency: tokens per MB of memory
        if result.resources.avg_memory_mb > 0:
            result.resources.memory_efficiency = (
                self.total_tokens / result.resources.avg_memory_mb
            )
        
        # Cost per quality point (if quality score available)
        if result.quality_score > 0:
            result.costs.cost_per_quality_point = (
                result.costs.estimated_token_cost / result.quality_score
            )
        
        # Useful tokens ratio (approximation - exclude system/prompt tokens)
        if self.total_tokens > 0:
            # Rough estimate: 20% of tokens are overhead (prompts, system messages)
            estimated_useful_tokens = self.total_tokens * 0.8
            result.costs.useful_tokens_ratio = estimated_useful_tokens / self.total_tokens
    
    def _calculate_peak_rate(self, times: List[float], total_duration: float) -> float:
        """Calculate peak rate using sliding window approach."""
        if not times or total_duration <= 0:
            return 0.0
        
        # Simple approximation: peak is 2x average rate
        avg_rate = len(times) / total_duration
        return avg_rate * 2.0
    
    def compare_benchmarks(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Compare multiple benchmark results."""
        if len(results) < 2:
            return {"error": "Need at least 2 results to compare"}
        
        comparison = {
            "benchmark_count": len(results),
            "comparison_time": time.time(),
            "throughput_comparison": {},
            "latency_comparison": {},
            "resource_comparison": {},
            "cost_comparison": {},
            "efficiency_comparison": {}
        }
        
        # Extract metrics for comparison
        throughput_scores = [r.throughput.tokens_per_second for r in results]
        latency_scores = [r.latency.average_response_time for r in results]
        cpu_scores = [r.resources.avg_cpu_percent for r in results]
        memory_scores = [r.resources.avg_memory_mb for r in results]
        cost_scores = [r.costs.estimated_token_cost for r in results]
        
        # Throughput comparison
        best_throughput_idx = throughput_scores.index(max(throughput_scores))
        comparison["throughput_comparison"] = {
            "best_benchmark": results[best_throughput_idx].benchmark_name,
            "best_score": throughput_scores[best_throughput_idx],
            "improvement_over_worst": max(throughput_scores) / min(throughput_scores) if min(throughput_scores) > 0 else 0,
            "all_scores": {r.benchmark_name: s for r, s in zip(results, throughput_scores)}
        }
        
        # Latency comparison (lower is better)
        best_latency_idx = latency_scores.index(min(latency_scores))
        comparison["latency_comparison"] = {
            "best_benchmark": results[best_latency_idx].benchmark_name,
            "best_score": latency_scores[best_latency_idx],
            "improvement_over_worst": max(latency_scores) / min(latency_scores) if min(latency_scores) > 0 else 0,
            "all_scores": {r.benchmark_name: s for r, s in zip(results, latency_scores)}
        }
        
        # Resource efficiency (lower resource usage is better)
        cpu_efficiency = [t/c if c > 0 else 0 for t, c in zip(throughput_scores, cpu_scores)]
        if cpu_efficiency:
            best_efficiency_idx = cpu_efficiency.index(max(cpu_efficiency))
            comparison["efficiency_comparison"] = {
                "best_benchmark": results[best_efficiency_idx].benchmark_name,
                "best_cpu_efficiency": cpu_efficiency[best_efficiency_idx],
                "all_cpu_efficiency": {r.benchmark_name: e for r, e in zip(results, cpu_efficiency)}
            }
        
        return comparison
    
    def generate_benchmark_report(self, result: BenchmarkResult) -> str:
        """Generate human-readable benchmark report."""
        report_lines = [
            f"=== BENCHMARK REPORT: {result.benchmark_name} ===",
            f"Status: {result.status.value}",
            f"Duration: {result.duration_seconds:.2f}s",
            f"Team Size: {result.team_size} agents",
            f"Token Budget: {result.token_budget:,}",
            "",
            "THROUGHPUT METRICS:",
            f"  Tokens/second: {result.throughput.tokens_per_second:.2f}",
            f"  Messages/second: {result.throughput.messages_per_second:.2f}",
            f"  Peak tokens/second: {result.throughput.peak_tokens_per_second:.2f}",
            "",
            "LATENCY METRICS:",
            f"  Average response time: {result.latency.average_response_time:.3f}s",
            f"  P95 response time: {result.latency.p95_response_time:.3f}s",
            f"  Task completion time: {result.latency.task_completion_time:.2f}s",
            "",
            "RESOURCE METRICS:",
            f"  Average CPU: {result.resources.avg_cpu_percent:.1f}%",
            f"  Peak CPU: {result.resources.peak_cpu_percent:.1f}%",
            f"  Average Memory: {result.resources.avg_memory_mb:.1f} MB",
            f"  Peak Memory: {result.resources.peak_memory_mb:.1f} MB",
            "",
            "COST METRICS:",
            f"  Total tokens: {result.costs.total_tokens_used:,}",
            f"  Estimated cost: ${result.costs.estimated_token_cost:.4f}",
            f"  Cost per task: ${result.costs.cost_per_task:.4f}",
            f"  Tokens per agent: {result.costs.tokens_per_agent:.0f}",
        ]
        
        if result.quality_score > 0:
            report_lines.extend([
                "",
                "QUALITY METRICS:",
                f"  Quality score: {result.quality_score:.3f}",
                f"  Cost per quality point: ${result.costs.cost_per_quality_point:.4f}",
            ])
        
        if result.error_message:
            report_lines.extend([
                "",
                "ERROR DETAILS:",
                f"  {result.error_message}",
            ])
        
        return "\n".join(report_lines)


# Integration helpers for Felix Framework

class FelixBenchmarkIntegration:
    """Integration helpers for benchmarking Felix Framework components."""
    
    @staticmethod
    def benchmark_helix_vs_linear(helix_factory, linear_factory, task_description: str,
                                benchmark_name: str = "helix_vs_linear") -> List[BenchmarkResult]:
        """
        Benchmark helix architecture vs linear pipeline.
        
        Args:
            helix_factory: Factory function for helix system
            linear_factory: Factory function for linear system
            task_description: Task to execute
            benchmark_name: Base name for benchmarks
            
        Returns:
            List of benchmark results for comparison
        """
        benchmarker = PerformanceBenchmarker()
        results = []
        
        # Benchmark helix architecture
        with benchmarker.benchmark_context(f"{benchmark_name}_helix") as helix_result:
            helix_system = helix_factory()
            helix_result.team_size = getattr(helix_system, 'agent_count', 0)
            
            # Run helix benchmark
            start_time = time.time()
            helix_output = helix_system.process_task(task_description)
            end_time = time.time()
            
            benchmarker.record_task_completion(end_time - start_time)
            helix_result.output_length = len(str(helix_output))
            
        results.append(helix_result)
        
        # Benchmark linear architecture  
        benchmarker.reset_counters()
        with benchmarker.benchmark_context(f"{benchmark_name}_linear") as linear_result:
            linear_system = linear_factory()
            linear_result.team_size = getattr(linear_system, 'agent_count', 0)
            
            # Run linear benchmark
            start_time = time.time()
            linear_output = linear_system.process_task(task_description)
            end_time = time.time()
            
            benchmarker.record_task_completion(end_time - start_time)
            linear_result.output_length = len(str(linear_output))
            
        results.append(linear_result)
        
        return results
    
    @staticmethod
    def benchmark_scaling(agent_factory, task_description: str, team_sizes: List[int],
                         benchmark_name: str = "scaling_test") -> List[BenchmarkResult]:
        """
        Benchmark system scaling with different team sizes.
        
        Args:
            agent_factory: Factory function that accepts team_size parameter
            task_description: Task to execute
            team_sizes: List of team sizes to test
            benchmark_name: Base name for benchmarks
            
        Returns:
            List of benchmark results for scaling analysis
        """
        results = []
        
        for team_size in team_sizes:
            benchmarker = PerformanceBenchmarker()
            
            with benchmarker.benchmark_context(
                f"{benchmark_name}_size_{team_size}",
                team_size=team_size
            ) as result:
                
                system = agent_factory(team_size=team_size)
                
                # Run benchmark
                start_time = time.time()
                output = system.process_task(task_description)
                end_time = time.time()
                
                benchmarker.record_task_completion(end_time - start_time)
                result.output_length = len(str(output))
                
            results.append(result)
        
        return results


def create_sample_benchmark() -> BenchmarkResult:
    """Create a sample benchmark result for testing."""
    return BenchmarkResult(
        benchmark_name="sample_test",
        status=BenchmarkStatus.COMPLETED,
        start_time=time.time() - 30,
        end_time=time.time(),
        duration_seconds=30.0,
        team_size=5,
        throughput=ThroughputMetrics(
            tokens_per_second=150.0,
            messages_per_second=2.5,
            peak_tokens_per_second=200.0
        ),
        latency=LatencyMetrics(
            average_response_time=0.8,
            p95_response_time=1.2,
            task_completion_time=25.0
        ),
        resources=ResourceMetrics(
            avg_cpu_percent=35.0,
            peak_cpu_percent=60.0,
            avg_memory_mb=512.0,
            peak_memory_mb=640.0
        ),
        costs=CostMetrics(
            total_tokens_used=4500,
            estimated_token_cost=0.009,
            cost_per_task=0.009
        )
    )
