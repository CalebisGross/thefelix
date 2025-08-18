"""
Architecture comparison framework for the Felix Framework.

This module provides comprehensive comparison capabilities between the helix-based
Felix architecture and traditional alternatives (linear pipeline, mesh communication)
for rigorous hypothesis testing and research validation.

Key Components:
- ArchitectureComparison: Unified comparison framework
- StatisticalAnalyzer: Statistical validation methods
- ExperimentalProtocol: Controlled experiment design
- HypothesisValidator: Automated hypothesis testing

The system supports:
- Performance benchmarking across all architectures
- Statistical significance testing with proper experimental design
- Hypothesis validation for H1, H2, H3 research claims
- Publication-quality research methodology and documentation

Mathematical references:
- docs/hypothesis_mathematics.md: Statistical frameworks for all hypotheses
- docs/mathematical_model.md: Theoretical foundation for comparisons
"""

from .architecture_comparison import (
    ArchitectureComparison, 
    ComparisonResults, 
    ExperimentalConfig,
    PerformanceMetrics
)
from .statistical_analysis import (
    StatisticalAnalyzer,
    StatisticalResults,
    HypothesisValidator
)
from .experimental_protocol import (
    ExperimentalProtocol,
    ExperimentalDesign,
    ValidationProtocol
)

__all__ = [
    'ArchitectureComparison',
    'ComparisonResults',
    'ExperimentalConfig',
    'PerformanceMetrics',
    'StatisticalAnalyzer',
    'StatisticalResults', 
    'HypothesisValidator',
    'ExperimentalProtocol',
    'ExperimentalDesign',
    'ValidationProtocol'
]