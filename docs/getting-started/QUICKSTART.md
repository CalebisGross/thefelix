# Felix Framework - Quick Start Guide

Welcome to the Felix Framework! This guide will get you up and running with LLM-powered geometric orchestration in minutes.

## Prerequisites

1. **LM Studio** - Local LLM inference server
   - Download from [https://lmstudio.ai/](https://lmstudio.ai/)
   - Install and load a model (any chat model works)
   - Start the server (default: http://localhost:1234)

2. **Python 3.12+** and **Git** (Python 3.8+ supported but 3.12+ recommended)

## Step-by-Step Setup

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd thefelix

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify LM Studio Connection

```bash
# Test if LM Studio is running
curl http://localhost:1234/v1/models

# Test from Python
python -c "from src.llm.lm_studio_client import LMStudioClient; print('✓ Connected' if LMStudioClient().test_connection() else '✗ Failed')"
```

### 3. Run Your First Demo

```bash
# Simple blog writer demo
python examples/blog_writer.py "Write about renewable energy" --complexity simple

# Code reviewer demo
python examples/code_reviewer.py --code-string "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"

# Performance benchmark
python examples/benchmark_comparison.py --task "Research AI safety" --runs 3

# Real-time visualization
python visualization/helix_monitor.py --mode terminal --demo
```

## What You'll See

### Blog Writer
- **3-6 agents** spawn at different times
- **Research agents** (top of helix): Broad exploration, high creativity
- **Analysis agents** (middle): Focused processing 
- **Synthesis agents** (bottom): Final integration, low temperature
- **Natural convergence** through geometric constraints

### Code Reviewer
- **Multi-perspective analysis**: Structure, performance, security, style
- **Quality assurance**: Bug detection, best practices
- **Comprehensive report**: Final synthesis of all reviews

### Benchmark Comparison
- **Felix vs Linear**: Statistical comparison of approaches
- **Performance metrics**: Time, tokens, quality scores
- **Geometric advantages**: Natural bottlenecking, memory efficiency

## Key Concepts

### Geometric Orchestration
Instead of explicit graphs (like LangGraph), Felix uses **3D helix geometry**:

```python
# Traditional approach
graph.add_node("research", research_function)
graph.add_edge("research", "analysis")

# Felix approach  
helix = HelixGeometry(33.0, 0.001, 33.0, 33)
agents = create_specialized_team(helix, llm_client, "medium")
# Agents naturally converge through geometry
```

### Position-Aware Behavior
Agent behavior adapts based on helix position:
- **Top (wide)**: Temperature 0.9, broad exploration
- **Middle**: Temperature 0.5, focused analysis
- **Bottom (narrow)**: Temperature 0.1, precise synthesis

### Spoke Communication
- **O(N) complexity** vs O(N²) mesh systems
- **Central coordination** with distributed processing
- **Natural bottlenecking** for quality control

## Configuration

### Adjust Token Limits
```python
# In examples or your code
llm_client = LMStudioClient(timeout=120.0)  # 2 minute timeout
agent = LLMAgent(..., max_tokens=300)       # Shorter responses
```

### Team Complexity
```python
# Simple: 3 agents (1 research, 1 analysis, 1 synthesis)
agents = create_specialized_team(helix, llm_client, "simple")

# Medium: 6 agents (2 research, 2 analysis, 1 critic, 1 synthesis)  
agents = create_specialized_team(helix, llm_client, "medium")

# Complex: 9 agents (3 research, 3 analysis, 2 critics, 1 synthesis)
agents = create_specialized_team(helix, llm_client, "complex")
```

## Troubleshooting

### "Connection Failed"
```bash
# Check LM Studio is running
curl http://localhost:1234/v1/models

# Restart LM Studio and ensure model is loaded
# Check firewall isn't blocking port 1234
```

### "Import Errors"
```bash
# Ensure you're in the right directory
cd /path/to/thefelix

# Activate virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install --force-reinstall openai httpx numpy scipy
```

### "Request Timeout"
- **Reduce complexity**: Use `--complexity simple`
- **Smaller model**: Use a faster model in LM Studio
- **Reduce tokens**: Lower max_tokens in agent creation

### "No Output Generated"
- **Check LM Studio console** for activity
- **Try simpler prompts** first
- **Verify model responses** work in LM Studio UI

## Next Steps

1. **Experiment with prompts**: Try different topics and complexity levels
2. **Review your code**: Use the code reviewer on your own files  
3. **Run benchmarks**: Compare Felix vs traditional approaches
4. **Customize agents**: Create your own specialized agent types
5. **Monitor in real-time**: Use the visualization tools

## Getting Help

- **Documentation**: Check `/docs/` folder for detailed explanations
  - **Navigation Guide**: See `docs/getting-started/README.md` for documentation structure
  - **Architecture**: Review `docs/architecture/PROJECT_OVERVIEW.md` for high-level overview
- **Research Log**: See `RESEARCH_LOG.md` for development insights
- **Mathematical Model**: Review `docs/architecture/core/mathematical_model.md` for theory
- **LLM Integration**: Full details in `docs/guides/llm-integration/LLM_INTEGRATION.md`
- **Development**: See `docs/guides/development/DEVELOPMENT_RULES.md` for contribution guidelines

## Examples to Try

```bash
# Creative writing
python examples/blog_writer.py "The future of space exploration" --complexity medium

# Technical analysis  
python examples/code_reviewer.py examples/blog_writer.py

# Research comparison
python examples/benchmark_comparison.py --task "Analyze climate change solutions" --runs 5

# Watch agents work
python visualization/helix_monitor.py --mode terminal --demo
```

Welcome to geometric orchestration! 🌀

---

*Felix Framework: Where geometry meets intelligence*