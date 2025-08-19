"""
Experimental protocol design for Felix Framework research validation.

This module provides rigorous experimental design methods including
randomization, blocking, factorial experiments, and validation protocols
for scientific research methodology.

Mathematical Foundation:
- Randomized controlled trials for causal inference
- Factorial experimental designs for interaction effects
- Power analysis for sample size determination
- Confounding variable control through blocking

Key Features:
- Randomized experimental designs with proper controls
- Blocking procedures to control for confounding variables
- Factorial designs to test interaction effects
- Validation protocols with replication and randomization
- Statistical analysis integration for hypothesis testing

This enables scientifically rigorous experimental validation of
research hypotheses with proper controls and statistical methodology
suitable for peer review and publication.

Mathematical reference: docs/hypothesis_mathematics.md, Experimental Design
"""

import random
import itertools
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from core.helix_geometry import HelixGeometry
from .architecture_comparison import ArchitectureComparison, ExperimentalConfig


@dataclass
class ExperimentalDesign:
    """Experimental design specification."""
    factors: Dict[str, List[Any]]
    blocks: Optional[List[str]] = None
    replications: int = 1
    randomization_seed: Optional[int] = None
    control_variables: List[str] = field(default_factory=list)
    response_variables: List[str] = field(default_factory=list)


@dataclass
class ValidationProtocol:
    """Validation protocol specification."""
    architectures: List[str]
    experimental_conditions: List[Dict[str, Any]]
    statistical_tests: List[str]
    significance_level: float = 0.05
    power_requirement: float = 0.8
    effect_size_threshold: float = 0.5


class ExperimentalProtocol:
    """
    Experimental protocol designer for Felix Framework research validation.
    
    Provides comprehensive experimental design capabilities including
    randomization, blocking, factorial designs, and validation protocols
    for rigorous scientific research methodology.
    """
    
    def __init__(self, helix: HelixGeometry, 
                 control_variables: List[str],
                 response_variables: List[str]):
        """
        Initialize experimental protocol designer.
        
        Args:
            helix: Helix geometry for experiments
            control_variables: Variables to control in experiments
            response_variables: Variables to measure as outcomes
        """
        self.helix = helix
        self.control_variables = control_variables
        self.response_variables = response_variables
    
    def design_factorial_experiment(self, factors: Dict[str, List[Any]], 
                                   replications: int = 1,
                                   randomization_seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Design factorial experiment with all factor combinations.
        
        Args:
            factors: Dictionary mapping factor names to level lists
            replications: Number of replications per combination
            randomization_seed: Seed for randomization
            
        Returns:
            List of experimental conditions
        """
        if randomization_seed is not None:
            random.seed(randomization_seed)
        
        # Generate all factor combinations
        factor_names = list(factors.keys())
        factor_levels = list(factors.values())
        combinations = list(itertools.product(*factor_levels))
        
        # Create experimental conditions
        experiments = []
        for rep in range(replications):
            for combo in combinations:
                experiment = dict(zip(factor_names, combo))
                experiment["replication"] = rep + 1
                experiments.append(experiment)
        
        # Randomize order
        random.shuffle(experiments)
        
        return experiments
    
    def randomized_block_design(self, treatments: List[str], 
                               blocks: List[str],
                               replications: int = 1,
                               randomization_seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Design randomized block experiment to control confounding.
        
        Args:
            treatments: List of treatment conditions
            blocks: List of blocking factors
            replications: Number of replications per block
            randomization_seed: Seed for randomization
            
        Returns:
            List of experimental conditions with blocking
        """
        if randomization_seed is not None:
            random.seed(randomization_seed)
        
        experiments = []
        
        for block in blocks:
            for rep in range(replications):
                # Randomize treatment order within each block
                block_treatments = treatments.copy()
                random.shuffle(block_treatments)
                
                for treatment in block_treatments:
                    experiment = {
                        "treatment": treatment,
                        "block": block,
                        "replication": rep + 1
                    }
                    experiments.append(experiment)
        
        return experiments
    
    def latin_square_design(self, treatments: List[str],
                           size: Optional[int] = None,
                           randomization_seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Design Latin square experiment for two blocking factors.
        
        Args:
            treatments: List of treatment conditions
            size: Size of the Latin square (default: len(treatments))
            randomization_seed: Seed for randomization
            
        Returns:
            List of experimental conditions in Latin square arrangement
        """
        if size is None:
            size = len(treatments)
        
        if len(treatments) != size:
            raise ValueError("Number of treatments must equal square size")
        
        if randomization_seed is not None:
            random.seed(randomization_seed)
        
        # Generate Latin square
        square = []
        for i in range(size):
            row = []
            for j in range(size):
                treatment_idx = (i + j) % size
                row.append(treatments[treatment_idx])
            square.append(row)
        
        # Randomize rows and columns
        random.shuffle(square)
        for row in square:
            random.shuffle(row)
        
        # Convert to experimental conditions
        experiments = []
        for row_idx, row in enumerate(square):
            for col_idx, treatment in enumerate(row):
                experiment = {
                    "treatment": treatment,
                    "row_block": f"row_{row_idx + 1}",
                    "col_block": f"col_{col_idx + 1}",
                    "position": (row_idx + 1, col_idx + 1)
                }
                experiments.append(experiment)
        
        return experiments
    
    def power_analysis_sample_size(self, effect_size: float,
                                  power: float = 0.8,
                                  alpha: float = 0.05) -> int:
        """
        Calculate required sample size for desired statistical power.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            power: Desired statistical power
            alpha: Significance level
            
        Returns:
            Required sample size per group
        """
        # Simplified power analysis calculation
        # For more precise calculation, would use statsmodels or similar
        
        if effect_size <= 0:
            return 1000  # Very large sample for no effect
        
        # Approximate sample size calculation
        # Based on Cohen's formulas for t-test
        if effect_size >= 0.8:  # Large effect
            base_n = 20
        elif effect_size >= 0.5:  # Medium effect
            base_n = 50
        else:  # Small effect
            base_n = 100
        
        # Adjust for power and alpha
        power_adjustment = (0.8 / power) ** 2
        alpha_adjustment = (alpha / 0.05) ** 0.5
        
        required_n = int(base_n * power_adjustment * alpha_adjustment)
        
        return max(required_n, 5)  # Minimum of 5 per group
    
    def run_validation_protocol(self, architectures: List[str],
                               agent_counts: List[int],
                               replications: int = 3,
                               random_seed: int = 42069) -> Dict[str, Any]:
        """
        Run comprehensive validation protocol across architectures.
        
        Args:
            architectures: List of architecture names to test
            agent_counts: List of agent counts to test
            replications: Number of replications per condition
            random_seed: Random seed for reproducibility
            
        Returns:
            Comprehensive validation results
        """
        # Create architecture comparison framework
        comparison = ArchitectureComparison(
            helix=self.helix,
            max_agents=max(agent_counts),
            enable_detailed_metrics=True
        )
        
        # Design factorial experiment
        factors = {
            "architecture": architectures,
            "agent_count": agent_counts
        }
        
        experiments = self.design_factorial_experiment(
            factors=factors,
            replications=replications,
            randomization_seed=random_seed
        )
        
        # Run experiments
        experimental_data = []
        for experiment in experiments:
            config = ExperimentalConfig(
                agent_count=experiment["agent_count"],
                simulation_time=1.0,
                task_load=experiment["agent_count"] * 5,
                random_seed=random_seed + experiment["replication"]
            )
            
            # Run specific architecture experiment
            if experiment["architecture"] == "helix_spoke":
                results = comparison.run_helix_experiment(config)
            elif experiment["architecture"] == "linear_pipeline":
                results = comparison.run_linear_experiment(config)
            elif experiment["architecture"] == "mesh_communication":
                results = comparison.run_mesh_experiment(config)
            else:
                continue
            
            # Store experimental data
            experiment_data = {
                **experiment,
                "throughput": results.throughput,
                "completion_time": results.task_completion_time,
                "communication_overhead": results.communication_overhead,
                "memory_usage": results.memory_usage
            }
            experimental_data.append(experiment_data)
        
        # Perform statistical analysis
        statistical_analysis = self._analyze_experimental_data(experimental_data)
        
        # Run hypothesis tests
        hypothesis_tests = self._run_hypothesis_tests(experimental_data)
        
        return {
            "experimental_data": experimental_data,
            "statistical_analysis": statistical_analysis,
            "hypothesis_tests": hypothesis_tests,
            "validation_summary": self._generate_validation_summary(
                experimental_data, statistical_analysis, hypothesis_tests
            )
        }
    
    def create_balanced_design(self, treatments: List[str],
                              subjects: int,
                              periods: int,
                              randomization_seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Create balanced experimental design for repeated measures.
        
        Args:
            treatments: List of treatment conditions
            subjects: Number of subjects
            periods: Number of time periods
            randomization_seed: Seed for randomization
            
        Returns:
            Balanced experimental design
        """
        if randomization_seed is not None:
            random.seed(randomization_seed)
        
        experiments = []
        
        # Create balanced assignment of treatments to subjects and periods
        for subject in range(1, subjects + 1):
            # Randomize treatment sequence for each subject
            treatment_sequence = treatments * (periods // len(treatments) + 1)
            treatment_sequence = treatment_sequence[:periods]
            random.shuffle(treatment_sequence)
            
            for period, treatment in enumerate(treatment_sequence, 1):
                experiment = {
                    "subject": subject,
                    "period": period,
                    "treatment": treatment
                }
                experiments.append(experiment)
        
        return experiments
    
    def validate_experimental_design(self, design: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate experimental design for balance and randomization.
        
        Args:
            design: List of experimental conditions
            
        Returns:
            Design validation report
        """
        validation_report = {
            "total_experiments": len(design),
            "balance_check": {},
            "randomization_check": {},
            "design_quality": {}
        }
        
        # Check balance of factors
        if design:
            factors = [key for key in design[0].keys() if key != "replication"]
            
            for factor in factors:
                factor_counts = {}
                for experiment in design:
                    value = experiment.get(factor)
                    factor_counts[value] = factor_counts.get(value, 0) + 1
                
                # Check if balanced
                counts = list(factor_counts.values())
                is_balanced = len(set(counts)) <= 1
                
                validation_report["balance_check"][factor] = {
                    "is_balanced": is_balanced,
                    "counts": factor_counts,
                    "total_levels": len(factor_counts)
                }
        
        # Check for adequate replication
        replication_counts = {}
        for experiment in design:
            rep = experiment.get("replication", 1)
            replication_counts[rep] = replication_counts.get(rep, 0) + 1
        
        validation_report["randomization_check"] = {
            "replication_counts": replication_counts,
            "adequate_replication": len(replication_counts) >= 3
        }
        
        # Overall design quality assessment
        balance_score = sum(1 for check in validation_report["balance_check"].values() 
                          if check["is_balanced"])
        total_factors = len(validation_report["balance_check"])
        
        validation_report["design_quality"] = {
            "balance_score": balance_score / max(total_factors, 1),
            "adequate_size": len(design) >= 20,
            "overall_quality": "good" if balance_score / max(total_factors, 1) > 0.8 else "needs_improvement"
        }
        
        return validation_report
    
    def _analyze_experimental_data(self, experimental_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze experimental data with descriptive statistics."""
        from .statistical_analysis import StatisticalAnalyzer
        analyzer = StatisticalAnalyzer()
        
        # Group data by architecture
        arch_groups = {}
        for data in experimental_data:
            arch = data["architecture"]
            if arch not in arch_groups:
                arch_groups[arch] = {
                    "throughput": [],
                    "completion_time": [],
                    "communication_overhead": [],
                    "memory_usage": []
                }
            
            arch_groups[arch]["throughput"].append(data["throughput"])
            arch_groups[arch]["completion_time"].append(data["completion_time"])
            arch_groups[arch]["communication_overhead"].append(data["communication_overhead"])
            arch_groups[arch]["memory_usage"].append(data["memory_usage"])
        
        # Calculate statistics for each metric
        analysis = {}
        for metric in ["throughput", "completion_time", "communication_overhead", "memory_usage"]:
            metric_data = [arch_groups[arch][metric] for arch in arch_groups.keys()]
            
            if len(metric_data) >= 2 and all(len(data) > 1 for data in metric_data):
                f_stat, p_value = analyzer.one_way_anova(metric_data)
                eta_squared = analyzer.calculate_eta_squared(metric_data)
                
                analysis[metric] = {
                    "f_statistic": f_stat,
                    "p_value": p_value,
                    "effect_size": eta_squared,
                    "significant": p_value < 0.05
                }
        
        return analysis
    
    def _run_hypothesis_tests(self, experimental_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run specific hypothesis tests on experimental data."""
        from .statistical_analysis import StatisticalAnalyzer
        analyzer = StatisticalAnalyzer()
        
        # Separate data by architecture
        helix_data = [d for d in experimental_data if d["architecture"] == "helix_spoke"]
        linear_data = [d for d in experimental_data if d["architecture"] == "linear_pipeline"]
        mesh_data = [d for d in experimental_data if d["architecture"] == "mesh_communication"]
        
        hypothesis_tests = {}
        
        # H2: Communication overhead comparison
        if helix_data and mesh_data:
            helix_overhead = [d["communication_overhead"] for d in helix_data]
            mesh_overhead = [d["communication_overhead"] for d in mesh_data]
            
            t_stat, p_value = analyzer.two_sample_t_test(helix_overhead, mesh_overhead)
            effect_size = analyzer.calculate_cohens_d(helix_overhead, mesh_overhead)
            
            hypothesis_tests["H2_communication_overhead"] = {
                "t_statistic": t_stat,
                "p_value": p_value,
                "effect_size": effect_size,
                "conclusion": "Helix has lower overhead" if t_stat < 0 and p_value < 0.05 else "No significant difference"
            }
        
        # Additional hypothesis tests would be added here
        
        return hypothesis_tests
    
    def _generate_validation_summary(self, experimental_data: List[Dict[str, Any]],
                                   statistical_analysis: Dict[str, Any],
                                   hypothesis_tests: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation summary report."""
        # Count significant results
        significant_metrics = sum(1 for analysis in statistical_analysis.values()
                                if analysis.get("significant", False))
        total_metrics = len(statistical_analysis)
        
        # Count supported hypotheses
        supported_hypotheses = sum(1 for test in hypothesis_tests.values()
                                 if "lower" in test.get("conclusion", "").lower() or 
                                    "better" in test.get("conclusion", "").lower())
        total_hypotheses = len(hypothesis_tests)
        
        return {
            "experiment_count": len(experimental_data),
            "architectures_tested": len(set(d["architecture"] for d in experimental_data)),
            "significant_metrics": f"{significant_metrics}/{total_metrics}",
            "supported_hypotheses": f"{supported_hypotheses}/{total_hypotheses}",
            "validation_strength": "strong" if significant_metrics / max(total_metrics, 1) > 0.6 else "moderate",
            "research_recommendation": "Publication ready" if supported_hypotheses >= total_hypotheses * 0.6 else "Additional research needed"
        }