# Felix Framework LLM Integration

The Felix Framework now supports LLM-powered agents using LM Studio for local inference, transforming it into a competitive alternative to LangGraph and similar multi-agent orchestration systems.

## Quick Start

### Prerequisites
1. **LM Studio** running with a model loaded at `http://localhost:1234`
2. **Python dependencies**: `pip install openai httpx` (if not already installed)

### Installation
```bash
# Navigate to Felix project
cd /home/hubcaps/Projects/thefelix

# Install additional dependencies (if needed)
pip install openai httpx

# Verify LM Studio connection
python -c "from src.llm.lm_studio_client import LMStudioClient; print('✓ Connection OK' if LMStudioClient().test_connection() else '✗ Connection Failed')"
```

## Usage Examples

### 1. Blog Writer Demo
Collaborative blog writing using geometric orchestration:

```bash
# Write a blog post about any topic
python examples/blog_writer.py "The future of artificial intelligence"

# Different complexity levels
python examples/blog_writer.py "Quantum computing basics" --complexity simple
python examples/blog_writer.py "Advanced machine learning techniques" --complexity complex

# Save output
python examples/blog_writer.py "Climate change solutions" --save-output results.json
```

**What happens**: Research agents spawn early (top of helix) for broad exploration, analysis agents spawn mid-way for focused processing, and synthesis agents spawn late (bottom of helix) for final integration. The geometric tapering naturally creates an editorial funnel.

### 2. Code Reviewer Demo
Multi-perspective code review with natural convergence:

```bash
# Review a Python file
python examples/code_reviewer.py path/to/your/code.py

# Review code directly
python examples/code_reviewer.py --code-string "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
```

**What happens**: Different agents examine code structure, performance, security, and maintainability, with critics providing quality assurance and final synthesis creating comprehensive review.

### 3. Performance Benchmark
Compare Felix vs traditional linear approaches:

```bash
# Benchmark a specific task
python examples/benchmark_comparison.py --task "Research renewable energy technologies"

# Multiple runs for statistical significance
python examples/benchmark_comparison.py --task "Analyze market trends" --runs 5

# Save detailed results
python examples/benchmark_comparison.py --task "Write technical documentation" --output benchmark_results.json
```

### 4. Visualization Tool
Watch agents move through the helix in real-time:

```bash
# Terminal-based visualization
python visualization/helix_monitor.py --mode terminal --demo

# Web-based 3D visualization (if matplotlib available)
python visualization/helix_monitor.py --mode web --demo
```

## Key Concepts

### Geometric Orchestration vs Graph-Based Systems

**Traditional (LangGraph-style)**:
```python
# Explicit graph definition
graph = Graph()
graph.add_node("research", research_function)
graph.add_node("analysis", analysis_function)
graph.add_edge("research", "analysis")
```

**Felix Framework**:
```python
# Geometric convergence
helix = HelixGeometry(33.0, 0.001, 33.0, 33)
agents = create_specialized_team(helix, llm_client, "medium")
# Agents naturally converge through geometry
```

### Natural Attention Focusing

- **Top of helix (wide)**: High creativity, broad exploration
- **Middle of helix**: Focused analysis, balanced processing  
- **Bottom of helix (narrow)**: Precise synthesis, low temperature

Temperature automatically adjusts based on position: `temperature = 0.1 + (0.9 - 0.1) * (1 - depth_ratio)`

### Agent Specialization

- **ResearchAgent**: Broad information gathering (spawn early)
- **AnalysisAgent**: Process and organize findings (spawn mid)
- **SynthesisAgent**: Final integration (spawn late)
- **CriticAgent**: Quality assurance (spawn as needed)

## Configuration

### LM Studio Settings
- Default URL: `http://localhost:1234/v1`
- No API key required (local inference)
- Any model supported by LM Studio works

### Agent Configuration
```python
# Custom team creation
agents = [
    ResearchAgent("research_001", 0.1, helix, llm_client, "technical"),
    AnalysisAgent("analysis_001", 0.5, helix, llm_client, "critical"),
    SynthesisAgent("synthesis_001", 0.8, helix, llm_client, "report")
]
```

### Temperature Ranges
```python
agent = LLMAgent(
    agent_id="example",
    spawn_time=0.5,
    helix=helix,
    llm_client=llm_client,
    temperature_range=(0.2, 0.8)  # Min/max based on helix position
)
```

## Architecture Comparison

| Feature | LangGraph | Felix Framework |
|---------|-----------|-----------------|
| Coordination | Explicit graphs | Geometric convergence |
| Agent spawning | Manual triggers | Time-based natural spawning |
| Communication | Defined edges | Spoke-based (O(N)) |
| Debugging | Log analysis | Visual 3D monitoring |
| Mental model | State machines | "Agents spiral to consensus" |
| Scalability | Graph complexity | Geometric constraints |

## Performance Characteristics

Based on initial testing:

**Strengths**:
- **Memory efficient**: O(N) communication vs O(N²) mesh
- **Visual debugging**: Watch agents converge in 3D space
- **Natural bottlenecking**: Geometric tapering for quality control
- **Intuitive**: Easier to understand than complex state machines

**Trade-offs**:
- **Computational overhead**: Geometric calculations
- **Fixed convergence pattern**: Less flexible than arbitrary graphs
- **New paradigm**: Learning curve for developers

## Troubleshooting

### Connection Issues
```bash
# Test LM Studio connection
curl http://localhost:1234/v1/models

# Check if model is loaded in LM Studio interface
```

### Import Errors
```bash
# Make sure you're in the project directory
cd /home/hubcaps/Projects/thefelix

# Check Python path
python -c "import sys; print(sys.path)"
```

### Performance Issues
- Reduce agent count for faster testing
- Use simpler tasks for initial validation
- Monitor token usage to manage costs

## Integration with Existing Code

```python
# Replace LangGraph workflow
from src.core.helix_geometry import HelixGeometry
from src.llm.lm_studio_client import LMStudioClient
from src.agents.specialized_agents import create_specialized_team

# Initialize
helix = HelixGeometry(33.0, 0.001, 33.0, 33)
llm_client = LMStudioClient()
agents = create_specialized_team(helix, llm_client, "medium")

# Process task (replaces graph execution)
task = LLMTask("task_001", "Your task description", "Context")
# ... run geometric orchestration simulation ...
```

This integration transforms Felix from a mathematical research project into a working LangGraph competitor with unique geometric advantages for multi-agent coordination.