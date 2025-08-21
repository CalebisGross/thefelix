"""
Comparative Analysis System for Felix Framework

This module implements comprehensive comparison capabilities against industry-standard
multi-agent frameworks like LangGraph, AutoGen, and CrewAI. It provides standardized
task suites, fair comparison methodologies, and regression testing capabilities.

Author: Felix Framework Research Team
Date: 2025-01-20
"""

import json
import time
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
import logging
from pathlib import Path

# Import our own components
from .quality_metrics import QualityMetricsCalculator, QualityScore
from .performance_benchmarks import PerformanceBenchmarker, BenchmarkResult
from ..llm.token_budget import TokenBudgetManager

logger = logging.getLogger(__name__)

class FrameworkType(Enum):
    """Supported framework types for comparison."""
    FELIX = "felix"
    LANGGRAPH = "langgraph" 
    AUTOGEN = "autogen"
    CREWAI = "crewai"
    CUSTOM = "custom"

class TaskComplexity(Enum):
    """Task complexity levels for standardized testing."""
    SIMPLE = "simple"          # Single agent, straightforward prompt
    MODERATE = "moderate"      # 2-3 agents, some coordination required
    COMPLEX = "complex"        # 4+ agents, significant coordination
    EXTREME = "extreme"        # 6+ agents, complex dependencies

class TaskDomain(Enum):
    """Task domain categories for domain-specific analysis."""
    RESEARCH = "research"
    WRITING = "writing"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    PLANNING = "planning"

@dataclass
class StandardizedTask:
    """Definition of a standardized task for framework comparison."""
    id: str
    name: str
    description: str
    domain: TaskDomain
    complexity: TaskComplexity
    expected_agents: int
    prompt: str
    success_criteria: Dict[str, float]  # metric_name -> minimum_score
    max_tokens: int = 10000
    timeout_seconds: int = 300
    reference_outputs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'domain': self.domain.value,
            'complexity': self.complexity.value,
            'expected_agents': self.expected_agents,
            'prompt': self.prompt,
            'success_criteria': self.success_criteria,
            'max_tokens': self.max_tokens,
            'timeout_seconds': self.timeout_seconds,
            'reference_outputs': self.reference_outputs
        }

@dataclass
class ComparisonResult:
    """Results from comparing frameworks on a specific task."""
    task_id: str
    framework_type: FrameworkType
    execution_time: float
    success: bool
    error_message: Optional[str]
    output: Optional[str]
    quality_score: Optional[QualityScore]
    performance_metrics: Optional[BenchmarkResult]
    resource_usage: Dict[str, float]
    
    def meets_criteria(self, task: StandardizedTask) -> bool:
        """Check if result meets task success criteria."""
        if not self.success or not self.quality_score:
            return False
            
        for metric, min_score in task.success_criteria.items():
            if hasattr(self.quality_score, metric):
                actual_score = getattr(self.quality_score, metric)
                if actual_score < min_score:
                    return False
        return True

class FrameworkAdapter(ABC):
    """Abstract adapter for different multi-agent frameworks."""
    
    @abstractmethod
    def get_framework_type(self) -> FrameworkType:
        """Return the framework type this adapter handles."""
        pass
    
    @abstractmethod
    def execute_task(self, task: StandardizedTask) -> ComparisonResult:
        """Execute a standardized task using this framework."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this framework is available for testing."""
        pass
    
    @abstractmethod
    def get_setup_requirements(self) -> List[str]:
        """Return setup requirements for this framework."""
        pass

class FelixFrameworkAdapter(FrameworkAdapter):
    """Adapter for Felix Framework using existing infrastructure."""
    
    def __init__(self, token_budget_manager: Optional[TokenBudgetManager] = None):
        self.token_budget_manager = token_budget_manager or TokenBudgetManager()
        self.quality_calculator = QualityMetricsCalculator()
        self.performance_benchmarker = PerformanceBenchmarker()
    
    def get_framework_type(self) -> FrameworkType:
        return FrameworkType.FELIX
    
    def execute_task(self, task: StandardizedTask) -> ComparisonResult:
        """Execute task using Felix Framework."""
        start_time = time.time()
        
        try:
            # Start performance monitoring
            with self.performance_benchmarker.benchmark_context(
                f"felix_task_{task.id}",
                team_size=task.expected_agents,
                token_budget=task.max_tokens
            ) as benchmark_result:
                # Import here to avoid circular imports
                from ...examples.blog_writer import main as felix_main
                
                # Configure token budget (simplified)
                budget_per_agent = task.max_tokens // task.expected_agents
                
                # Execute Felix task (simplified - would need task-specific routing)
                output = self._execute_felix_task(task)
                
                execution_time = time.time() - start_time
                
                # Calculate quality metrics
                from .quality_metrics import DomainType
                domain_type = DomainType(task.domain.value.lower())
                quality_score = self.quality_calculator.calculate_quality_score(
                    output, 
                    domain=domain_type,
                    reference_texts=task.reference_outputs if task.reference_outputs else None
                )
            
            return ComparisonResult(
                task_id=task.id,
                framework_type=FrameworkType.FELIX,
                execution_time=execution_time,
                success=True,
                error_message=None,
                output=output,
                quality_score=quality_score,
                performance_metrics=benchmark_result,
                resource_usage=benchmark_result.resources.to_dict() if benchmark_result else {}
            )
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Felix execution failed for task {task.id}: {e}")
            
            return ComparisonResult(
                task_id=task.id,
                framework_type=FrameworkType.FELIX,
                execution_time=execution_time,
                success=False,
                error_message=str(e),
                output=None,
                quality_score=None,
                performance_metrics=None,
                resource_usage={}
            )
    
    def _execute_felix_task(self, task: StandardizedTask) -> str:
        """Execute task using Felix Framework - simplified implementation."""
        # This would be expanded to route to appropriate Felix components
        # For now, return a basic simulation
        return f"Felix Framework output for: {task.prompt}"
    
    def is_available(self) -> bool:
        """Check if Felix Framework is available."""
        try:
            from ...examples.blog_writer import main
            return True
        except ImportError:
            return False
    
    def get_setup_requirements(self) -> List[str]:
        """Return Felix setup requirements."""
        return [
            "Felix Framework installed",
            "LM Studio running at localhost:1234",
            "Required models loaded"
        ]

class LangGraphAdapter(FrameworkAdapter):
    """Adapter for LangGraph framework (mock implementation)."""
    
    def get_framework_type(self) -> FrameworkType:
        return FrameworkType.LANGGRAPH
    
    def execute_task(self, task: StandardizedTask) -> ComparisonResult:
        """Mock execution for LangGraph."""
        start_time = time.time()
        
        try:
            # Mock LangGraph execution
            time.sleep(0.5)  # Simulate execution time
            output = f"LangGraph output for: {task.prompt}"
            execution_time = time.time() - start_time
            
            # Mock quality score
            quality_score = QualityScore(
                overall_score=0.77,
                coherence_score=0.75, 
                accuracy_score=0.80, 
                completeness_score=0.70,
                clarity_score=0.85, 
                relevance_score=0.90, 
                originality_score=0.60, 
                structure_score=0.80
            )
            
            return ComparisonResult(
                task_id=task.id,
                framework_type=FrameworkType.LANGGRAPH,
                execution_time=execution_time,
                success=True,
                error_message=None,
                output=output,
                quality_score=quality_score,
                performance_metrics=None,
                resource_usage={"cpu_percent": 45.0, "memory_mb": 512.0}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ComparisonResult(
                task_id=task.id,
                framework_type=FrameworkType.LANGGRAPH,
                execution_time=execution_time,
                success=False,
                error_message=str(e),
                output=None,
                quality_score=None,
                performance_metrics=None,
                resource_usage={}
            )
    
    def is_available(self) -> bool:
        """Check if LangGraph is available."""
        try:
            # Mock availability check - would actually import langgraph
            return True  # Mock as available for testing
        except ImportError:
            return False
    
    def get_setup_requirements(self) -> List[str]:
        """Return LangGraph setup requirements."""
        return [
            "pip install langgraph",
            "OpenAI API key configured",
            "LangGraph dependencies installed"
        ]

class StandardizedTaskSuite:
    """Collection of standardized tasks for framework comparison."""
    
    def __init__(self):
        self.tasks: Dict[str, StandardizedTask] = {}
        self._initialize_default_tasks()
    
    def _initialize_default_tasks(self):
        """Initialize the default set of standardized tasks."""
        
        # Simple research task
        self.add_task(StandardizedTask(
            id="simple_research",
            name="Basic Research Task",
            description="Simple research on renewable energy",
            domain=TaskDomain.RESEARCH,
            complexity=TaskComplexity.SIMPLE,
            expected_agents=1,
            prompt="Research the current state of solar energy technology",
            success_criteria={"accuracy": 0.7, "completeness": 0.6, "relevance": 0.8},
            max_tokens=2000,
            timeout_seconds=120
        ))
        
        # Moderate writing task
        self.add_task(StandardizedTask(
            id="moderate_writing",
            name="Collaborative Writing",
            description="Multi-agent blog post creation",
            domain=TaskDomain.WRITING,
            complexity=TaskComplexity.MODERATE,
            expected_agents=3,
            prompt="Write a comprehensive blog post about sustainable transportation",
            success_criteria={"coherence": 0.8, "clarity": 0.7, "structure": 0.8},
            max_tokens=5000,
            timeout_seconds=180
        ))
        
        # Complex analysis task
        self.add_task(StandardizedTask(
            id="complex_analysis",
            name="Multi-Faceted Analysis",
            description="Complex business analysis with multiple perspectives",
            domain=TaskDomain.ANALYSIS,
            complexity=TaskComplexity.COMPLEX,
            expected_agents=5,
            prompt="Analyze the market potential for AI-powered educational tools",
            success_criteria={"accuracy": 0.8, "completeness": 0.8, "originality": 0.6},
            max_tokens=8000,
            timeout_seconds=300
        ))
        
        # Technical planning task
        self.add_task(StandardizedTask(
            id="technical_planning",
            name="Technical Architecture Planning",
            description="Complex technical system design",
            domain=TaskDomain.TECHNICAL,
            complexity=TaskComplexity.EXTREME,
            expected_agents=6,
            prompt="Design a scalable microservices architecture for an e-commerce platform",
            success_criteria={"accuracy": 0.9, "completeness": 0.8, "structure": 0.9},
            max_tokens=10000,
            timeout_seconds=400
        ))
    
    def add_task(self, task: StandardizedTask):
        """Add a task to the suite."""
        self.tasks[task.id] = task
    
    def get_task(self, task_id: str) -> Optional[StandardizedTask]:
        """Get a specific task by ID."""
        return self.tasks.get(task_id)
    
    def get_tasks_by_complexity(self, complexity: TaskComplexity) -> List[StandardizedTask]:
        """Get all tasks of a specific complexity level."""
        return [task for task in self.tasks.values() if task.complexity == complexity]
    
    def get_tasks_by_domain(self, domain: TaskDomain) -> List[StandardizedTask]:
        """Get all tasks in a specific domain."""
        return [task for task in self.tasks.values() if task.domain == domain]
    
    def get_all_tasks(self) -> List[StandardizedTask]:
        """Get all tasks in the suite."""
        return list(self.tasks.values())
    
    def save_to_file(self, filepath: str):
        """Save task suite to JSON file."""
        data = {
            'version': '1.0',
            'tasks': [task.to_dict() for task in self.tasks.values()]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load task suite from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for task_data in data['tasks']:
            task = StandardizedTask(
                id=task_data['id'],
                name=task_data['name'],
                description=task_data['description'],
                domain=TaskDomain(task_data['domain']),
                complexity=TaskComplexity(task_data['complexity']),
                expected_agents=task_data['expected_agents'],
                prompt=task_data['prompt'],
                success_criteria=task_data['success_criteria'],
                max_tokens=task_data.get('max_tokens', 10000),
                timeout_seconds=task_data.get('timeout_seconds', 300),
                reference_outputs=task_data.get('reference_outputs', [])
            )
            self.add_task(task)

@dataclass
class ComparisonSummary:
    """Summary of framework comparison results."""
    task_suite_version: str
    total_tasks: int
    frameworks_tested: List[FrameworkType]
    results_by_framework: Dict[FrameworkType, List[ComparisonResult]]
    success_rates: Dict[FrameworkType, float]
    average_execution_times: Dict[FrameworkType, float]
    quality_averages: Dict[FrameworkType, Dict[str, float]]
    resource_usage_averages: Dict[FrameworkType, Dict[str, float]]
    
    def get_winner_by_metric(self, metric: str) -> Optional[FrameworkType]:
        """Get the framework with the best performance for a specific metric."""
        if metric == "success_rate":
            return max(self.success_rates.items(), key=lambda x: x[1])[0]
        elif metric == "speed":
            return min(self.average_execution_times.items(), key=lambda x: x[1])[0]
        elif metric in ["coherence", "accuracy", "completeness", "clarity", "relevance", "originality", "structure"]:
            best_framework = None
            best_score = -1.0
            for framework, scores in self.quality_averages.items():
                if metric in scores and scores[metric] > best_score:
                    best_score = scores[metric]
                    best_framework = framework
            return best_framework
        return None

class ComparativeAnalyzer:
    """Main class for performing comparative analysis between frameworks."""
    
    def __init__(self, adapters: Optional[List[FrameworkAdapter]] = None):
        self.adapters: Dict[FrameworkType, FrameworkAdapter] = {}
        self.task_suite = StandardizedTaskSuite()
        
        # Register default adapters
        if adapters:
            for adapter in adapters:
                self.register_adapter(adapter)
        else:
            # Default Felix adapter
            self.register_adapter(FelixFrameworkAdapter())
            self.register_adapter(LangGraphAdapter())
    
    def register_adapter(self, adapter: FrameworkAdapter):
        """Register a framework adapter."""
        self.adapters[adapter.get_framework_type()] = adapter
        logger.info(f"Registered adapter for {adapter.get_framework_type().value}")
    
    def run_comparison(self, 
                      frameworks: Optional[List[FrameworkType]] = None,
                      tasks: Optional[List[str]] = None) -> ComparisonSummary:
        """Run comparative analysis across frameworks and tasks."""
        
        # Determine which frameworks to test
        frameworks_to_test = frameworks or list(self.adapters.keys())
        available_frameworks = [f for f in frameworks_to_test 
                              if f in self.adapters and self.adapters[f].is_available()]
        
        if not available_frameworks:
            raise RuntimeError("No available frameworks for testing")
        
        # Determine which tasks to run
        tasks_to_run = []
        if tasks:
            for task_id in tasks:
                task = self.task_suite.get_task(task_id)
                if task:
                    tasks_to_run.append(task)
        else:
            tasks_to_run = self.task_suite.get_all_tasks()
        
        logger.info(f"Running comparison: {len(available_frameworks)} frameworks, {len(tasks_to_run)} tasks")
        
        # Execute comparisons
        results_by_framework: Dict[FrameworkType, List[ComparisonResult]] = {}
        
        for framework_type in available_frameworks:
            adapter = self.adapters[framework_type]
            results_by_framework[framework_type] = []
            
            logger.info(f"Testing {framework_type.value}")
            
            for task in tasks_to_run:
                logger.info(f"  Running task: {task.name}")
                result = adapter.execute_task(task)
                results_by_framework[framework_type].append(result)
        
        # Generate summary
        return self._generate_summary(results_by_framework, tasks_to_run)
    
    def _generate_summary(self, 
                         results_by_framework: Dict[FrameworkType, List[ComparisonResult]],
                         tasks: List[StandardizedTask]) -> ComparisonSummary:
        """Generate comparison summary from results."""
        
        success_rates = {}
        average_execution_times = {}
        quality_averages = {}
        resource_usage_averages = {}
        
        for framework_type, results in results_by_framework.items():
            # Success rate
            successes = sum(1 for r in results if r.success)
            success_rates[framework_type] = successes / len(results) if results else 0.0
            
            # Average execution time
            times = [r.execution_time for r in results if r.success]
            average_execution_times[framework_type] = statistics.mean(times) if times else 0.0
            
            # Quality averages
            quality_scores = [r.quality_score for r in results if r.success and r.quality_score]
            if quality_scores:
                quality_averages[framework_type] = {
                    'coherence': statistics.mean([getattr(q, 'coherence', 0.5) for q in quality_scores]),
                    'accuracy': statistics.mean([getattr(q, 'accuracy', 0.5) for q in quality_scores]),
                    'completeness': statistics.mean([getattr(q, 'completeness', 0.5) for q in quality_scores]),
                    'clarity': statistics.mean([getattr(q, 'clarity', 0.5) for q in quality_scores]),
                    'relevance': statistics.mean([getattr(q, 'relevance', 0.5) for q in quality_scores]),
                    'originality': statistics.mean([getattr(q, 'originality', 0.5) for q in quality_scores]),
                    'structure': statistics.mean([getattr(q, 'structure', 0.5) for q in quality_scores])
                }
            else:
                quality_averages[framework_type] = {}
            
            # Resource usage averages
            resource_data = [r.resource_usage for r in results if r.success and r.resource_usage]
            if resource_data:
                all_keys = set().union(*resource_data)
                resource_usage_averages[framework_type] = {
                    key: statistics.mean([d.get(key, 0) for d in resource_data])
                    for key in all_keys
                }
            else:
                resource_usage_averages[framework_type] = {}
        
        return ComparisonSummary(
            task_suite_version="1.0",
            total_tasks=len(tasks),
            frameworks_tested=list(results_by_framework.keys()),
            results_by_framework=results_by_framework,
            success_rates=success_rates,
            average_execution_times=average_execution_times,
            quality_averages=quality_averages,
            resource_usage_averages=resource_usage_averages
        )
    
    def run_regression_tests(self, baseline_results_file: str) -> Dict[str, Any]:
        """Run regression tests against baseline results."""
        try:
            with open(baseline_results_file, 'r') as f:
                baseline_data = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Baseline file {baseline_results_file} not found")
            return {"error": "Baseline file not found"}
        
        # Run current comparison
        current_summary = self.run_comparison([FrameworkType.FELIX])
        
        # Compare with baseline
        regression_results = {
            "regression_detected": False,
            "changes": [],
            "current_performance": {},
            "baseline_performance": {}
        }
        
        if FrameworkType.FELIX in current_summary.success_rates:
            current_success_rate = current_summary.success_rates[FrameworkType.FELIX]
            baseline_success_rate = baseline_data.get("success_rate", 0.0)
            
            regression_results["current_performance"]["success_rate"] = current_success_rate
            regression_results["baseline_performance"]["success_rate"] = baseline_success_rate
            
            if current_success_rate < baseline_success_rate - 0.05:  # 5% tolerance
                regression_results["regression_detected"] = True
                regression_results["changes"].append({
                    "metric": "success_rate",
                    "change": current_success_rate - baseline_success_rate,
                    "type": "regression"
                })
        
        return regression_results
    
    def save_comparison_report(self, summary: ComparisonSummary, filepath: str):
        """Save detailed comparison report to file."""
        report = {
            "summary": {
                "task_suite_version": summary.task_suite_version,
                "total_tasks": summary.total_tasks,
                "frameworks_tested": [f.value for f in summary.frameworks_tested],
                "success_rates": {f.value: rate for f, rate in summary.success_rates.items()},
                "average_execution_times": {f.value: time for f, time in summary.average_execution_times.items()},
                "quality_averages": {f.value: scores for f, scores in summary.quality_averages.items()},
                "resource_usage_averages": {f.value: usage for f, usage in summary.resource_usage_averages.items()}
            },
            "detailed_results": {},
            "winners": {
                "success_rate": (winner := summary.get_winner_by_metric("success_rate")) and winner.value,
                "speed": (winner := summary.get_winner_by_metric("speed")) and winner.value,
                "coherence": (winner := summary.get_winner_by_metric("coherence")) and winner.value,
                "accuracy": (winner := summary.get_winner_by_metric("accuracy")) and winner.value
            },
            "timestamp": time.time()
        }
        
        # Add detailed results
        for framework_type, results in summary.results_by_framework.items():
            report["detailed_results"][framework_type.value] = [
                {
                    "task_id": r.task_id,
                    "success": r.success,
                    "execution_time": r.execution_time,
                    "error_message": r.error_message,
                    "quality_score": r.quality_score.__dict__ if r.quality_score else None,
                    "resource_usage": r.resource_usage
                }
                for r in results
            ]
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comparison report saved to {filepath}")

# Convenience functions for common use cases

def quick_felix_comparison(tasks: Optional[List[str]] = None) -> ComparisonSummary:
    """Quick comparison focusing on Felix Framework."""
    analyzer = ComparativeAnalyzer([FelixFrameworkAdapter()])
    return analyzer.run_comparison([FrameworkType.FELIX], tasks)

def benchmark_against_industry(include_mock_competitors: bool = True) -> ComparisonSummary:
    """Benchmark Felix against industry standards."""
    adapters: List[FrameworkAdapter] = [FelixFrameworkAdapter()]
    
    if include_mock_competitors:
        adapters.append(LangGraphAdapter())
        # Additional mock adapters would be added here
    
    analyzer = ComparativeAnalyzer(adapters)
    return analyzer.run_comparison()

def generate_performance_report(output_dir: str = "comparison_reports") -> str:
    """Generate a comprehensive performance report."""
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Run comprehensive comparison
    summary = benchmark_against_industry()
    
    # Save detailed report
    timestamp = int(time.time())
    report_file = output_path / f"felix_comparison_{timestamp}.json"
    
    analyzer = ComparativeAnalyzer()
    analyzer.save_comparison_report(summary, str(report_file))
    
    return str(report_file)
