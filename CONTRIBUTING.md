# Contributing to Felix Framework

Welcome to the Felix Framework! We're excited you're interested in contributing to our helix-based multi-agent cognitive architecture research project. 🌪️

## 🎯 Project Philosophy

Felix Framework is **research-grade software** built with scientific rigor. We prioritize:
- **Evidence-based development** with measurable outcomes
- **Test-driven development** (tests before code, always)
- **Hypothesis-driven research** with statistical validation
- **Complete documentation** of decisions and failures
- **Reproducible experiments** with version-controlled data

Before contributing, please read our [DEVELOPMENT_RULES.md](./DEVELOPMENT_RULES.md) for detailed standards.

## 🚀 Quick Start for Contributors

### Prerequisites
- **Python 3.12+** 
- **Git** with basic familiarity
- **Research mindset** - we document everything!

### Setup Development Environment

```bash
# 1. Fork and clone the repository
git clone https://github.com/CalebisGross/thefelix.git
cd thefelix

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify setup
python validate_felix_framework.py
python -m pytest tests/unit/ -v

# 5. Run benchmark to ensure everything works
python benchmark_enhanced_systems.py
```

If all tests pass and benchmarks show 100% success rate, you're ready to contribute!

## 🤝 How to Contribute

We welcome several types of contributions:

### 🔬 Research Contributions
- **New hypotheses** for testing helix-based coordination
- **Experimental implementations** of cognitive architectures
- **Performance analysis** and optimization studies
- **Comparative studies** with other multi-agent frameworks

### 💻 Code Contributions
- **Core framework improvements** (helix geometry, agent systems)
- **New agent specializations** (research, analysis, synthesis types)
- **LLM integrations** and multi-model orchestration
- **Performance optimizations** with measurable impact

### 📊 Testing & Validation
- **Test coverage improvements** (we aim for >95%)
- **Integration test scenarios** across system boundaries
- **Performance benchmarks** and regression tests
- **Property-based tests** using hypothesis library

### 📚 Documentation
- **Research documentation** in markdown format
- **Code examples** and usage patterns
- **Architecture Decision Records** (ADRs)
- **Tutorial content** for complex features

## 🔍 Finding Your First Contribution

### Good First Issues
Look for issues labeled with:
- `good-first-issue` - Well-scoped for newcomers
- `research` - Research-oriented contributions
- `testing` - Test improvement opportunities
- `documentation` - Documentation enhancements

### Areas Needing Help
1. **Test Coverage**: Expand test coverage for edge cases
2. **Performance Benchmarks**: Add benchmarks for scalability analysis
3. **Documentation**: Improve code examples and tutorials
4. **Research Experiments**: Implement new hypothesis tests
5. **LLM Integration**: Enhance multi-model orchestration

## 🧪 Development Process

### Our Test-First Approach

**CRITICAL**: All code contributions must follow test-driven development:

```bash
# 1. Write tests FIRST (this is mandatory)
# Create test file: tests/unit/test_your_feature.py

# 2. Run tests to ensure they fail
python -m pytest tests/unit/test_your_feature.py -v

# 3. Implement code to make tests pass
# Create/modify: src/your_module/your_feature.py

# 4. Verify tests pass
python -m pytest tests/unit/test_your_feature.py -v

# 5. Run full test suite
python -m pytest tests/unit/ -v
```

### Hypothesis-Driven Development

Every significant change should:
1. **State a clear hypothesis** in your PR description
2. **Predict expected outcomes** with measurable criteria
3. **Document alternatives considered** and why rejected
4. **Provide validation evidence** through tests/benchmarks

### Documentation Requirements

Before submitting any PR:
- [ ] Update relevant documentation
- [ ] Add docstrings to new functions/classes
- [ ] Create/update tests with good coverage
- [ ] Update RESEARCH_LOG.md if research-related
- [ ] Add ADR if architectural decision made

## 📝 Pull Request Guidelines

### Branch Naming
- `feature/description` - New functionality
- `experiment/hypothesis-name` - Research experiments
- `fix/issue-description` - Bug fixes
- `docs/section-name` - Documentation updates

### PR Checklist

Before submitting your pull request:

#### Code Quality
- [ ] All new code has corresponding tests
- [ ] Tests pass locally: `python -m pytest tests/unit/ -v`
- [ ] Code follows project style guidelines
- [ ] No unused imports or dead code
- [ ] Docstrings added for public functions/classes

#### Research Standards
- [ ] Hypothesis clearly stated in PR description
- [ ] Expected outcomes documented
- [ ] Validation methodology described
- [ ] Performance impact measured (if applicable)

#### Documentation
- [ ] README.md updated if user-facing changes
- [ ] RESEARCH_LOG.md updated if research contribution
- [ ] ADR created if architectural decision
- [ ] Code comments explain "why", not "what"

#### Testing
- [ ] Unit tests cover new functionality
- [ ] Integration tests updated if cross-system changes
- [ ] Performance benchmarks run: `python benchmark_enhanced_systems.py`
- [ ] No regression in existing test coverage

### Commit Message Format

Use our structured commit format from DEVELOPMENT_RULES.md:

```
[TYPE]: Brief description (max 50 chars)

WHY: Detailed explanation of the problem/need
WHAT: Specific changes made
EXPECTED: Predicted outcome/behavior
ALTERNATIVES: Other approaches considered and why rejected
TESTS: How this change will be validated

[Optional: BREAKING CHANGES, NOTES, etc.]
```

**Types**: `feat`, `fix`, `test`, `docs`, `refactor`, `experiment`, `perf`

### PR Template

When you create a PR, please include:

```markdown
## Hypothesis
[State your hypothesis clearly]

## Changes Made
- [List specific changes]
- [Include rationale for each]

## Expected Outcomes
- [Measurable predictions]
- [Performance expectations]

## Validation
- [ ] Tests added/updated
- [ ] Benchmarks run
- [ ] Documentation updated

## Alternatives Considered
[Other approaches and why rejected]

## Breaking Changes
[If any, describe impact]
```

## 🧪 Testing Requirements

### Test Organization
```
tests/
├── unit/           # Individual component tests
├── integration/    # Cross-component tests
└── performance/    # Benchmark and performance tests
```

### Coverage Standards
- **Unit tests**: >95% line coverage for new code
- **Integration tests**: All public APIs tested
- **Performance tests**: Benchmarks for performance-critical code

### Running Tests

```bash
# All unit tests with coverage
python -m pytest tests/unit/ -v --cov=src --cov-report=html

# Specific test module
python -m pytest tests/unit/test_helix_geometry.py -v

# Integration tests
python -m pytest tests/integration/ -v

# Performance tests (marked as slow)
python -m pytest tests/performance/ -m slow -v

# Run with specific markers
python -m pytest -m "unit and not slow" -v
```

### Test Markers
Use pytest markers to categorize tests:
- `@pytest.mark.unit` - Unit tests for individual components
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance and benchmark tests
- `@pytest.mark.slow` - Tests taking >1 second
- `@pytest.mark.hypothesis` - Property-based tests

## 📊 Code Style & Standards

### Python Style
- **PEP 8** compliance with 88-character line limit
- **Type hints** for all public functions
- **Docstrings** in Google format for all public APIs
- **Descriptive variable names** (no abbreviations)

### Example Code Style
```python
from typing import List, Optional
import numpy as np

def generate_helix_points(
    num_turns: int = 33,
    nodes: int = 133,
    top_radius: float = 33.0,
    bottom_radius: float = 0.001
) -> List[HelixPoint]:
    """Generate helix points with mathematical precision.
    
    Args:
        num_turns: Number of complete helix rotations
        nodes: Total number of agent positions
        top_radius: Starting radius at helix top
        bottom_radius: Ending radius at helix bottom
        
    Returns:
        List of HelixPoint objects with x, y, z coordinates
        
    Raises:
        ValueError: If parameters result in invalid geometry
    """
    # Implementation with clear variable names
    angle_increment = 2 * np.pi / (nodes / num_turns)
    # ... rest of implementation
```

### Architecture Decisions
Document significant decisions in `decisions/ADR-XXX-title.md`:

```markdown
# ADR-XXX: Title of Decision

## Status
Accepted | Superseded | Deprecated

## Context
[Situation and problem]

## Decision
[What we decided]

## Consequences
[Positive and negative impacts]

## Alternatives Considered
[Other options and why rejected]
```

## 🔬 Research Contributions

### Proposing New Hypotheses
1. **Research existing literature** and document findings
2. **State hypothesis clearly** with measurable predictions
3. **Design validation methodology** before implementation
4. **Consider statistical power** and sample sizes needed

### Experimental Process
1. **Document hypothesis** in `research/hypothesis-name.md`
2. **Implement tests first** to validate hypothesis
3. **Build minimal implementation** to test hypothesis
4. **Collect and analyze data** with statistical rigor
5. **Document results** regardless of success/failure

### Failed Experiments
We preserve ALL experiments, including failures:
- Code preserved in `experiments/failed/`
- Documentation of why it failed
- Lessons learned and insights gained
- Analysis prevents repeating mistakes

### Research Log Entries
Update `RESEARCH_LOG.md` for research contributions:

```markdown
## 2025-XX-XX: [Your Contribution]
**Hypothesis**: [What you're testing]
**Progress**: [What was accomplished]
**Obstacles**: [What challenges encountered]
**Insights**: [What was learned]
**Next Steps**: [What comes next]
```

## 🐛 Bug Reports

### Before Reporting a Bug
1. **Search existing issues** for duplicates
2. **Test on latest main branch** to confirm bug exists
3. **Gather reproduction steps** with minimal example
4. **Check if it's a configuration issue** vs actual bug

### Bug Report Template
```markdown
**Bug Description**
Clear description of the issue

**Reproduction Steps**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.12.0]
- Felix Version: [e.g., commit hash]
- Dependencies: [relevant package versions]

**Additional Context**
- Log outputs
- Screenshots if relevant
- Related issues
```

## 💡 Feature Requests

### Before Requesting Features
1. **Check if it aligns** with core research objectives
2. **Search existing issues** for similar requests
3. **Consider the complexity** and maintenance burden
4. **Think about testing strategy** for the feature

### Feature Request Template
```markdown
**Problem Statement**
What problem does this solve?

**Proposed Solution**
Detailed description of proposed feature

**Research Justification**
How does this advance our research goals?

**Validation Methodology**
How would we test/validate this feature?

**Alternatives Considered**
Other ways to solve this problem

**Implementation Notes**
Technical considerations or challenges
```

## 🌟 Recognition

We value all contributions! Contributors are recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **Research papers** when contributions advance research
- **Project documentation** for major improvements

## 📞 Getting Help

### Communication Channels
- **GitHub Issues**: Technical questions and bug reports
- **GitHub Discussions**: Research discussions and brainstorming
- **Pull Request Comments**: Code review discussions

### Documentation Resources
- **[README.md](./README.md)**: Project overview and quick start
- **[DEVELOPMENT_RULES.md](./DEVELOPMENT_RULES.md)**: Detailed development standards
- **[PROJECT_INDEX.md](./PROJECT_INDEX.md)**: Complete project structure
- **[RESEARCH_LOG.md](./RESEARCH_LOG.md)**: Research progress and findings

### Research Resources
- **[research/](./research/)**: Research hypotheses and methodologies
- **[docs/](./docs/)**: Technical documentation and specifications
- **[decisions/](./decisions/)**: Architecture decision records

## 📋 Code of Conduct

### Our Standards
- **Research integrity** above all else
- **Respectful communication** in all interactions
- **Constructive feedback** focused on improving the work
- **Collaborative problem-solving** approach
- **Evidence-based discussions** rather than opinions

### Unacceptable Behavior
- Making claims without evidence
- Ignoring test-first development requirements
- Submitting code without documentation
- Personal attacks or unprofessional conduct
- Plagiarism or research misconduct

### Enforcement
Issues will be addressed through:
1. **Direct communication** for minor issues
2. **Documented warnings** for repeated violations
3. **Temporary restrictions** for serious violations
4. **Permanent exclusion** for research misconduct

## 🏁 Getting Started Checklist

Ready to contribute? Here's your checklist:

- [ ] Read [DEVELOPMENT_RULES.md](./DEVELOPMENT_RULES.md) thoroughly
- [ ] Set up development environment successfully
- [ ] Run tests and benchmarks to verify setup
- [ ] Choose a contribution area that interests you
- [ ] Look for good first issues or create a proposal
- [ ] Fork the repository and create a feature branch
- [ ] Write tests first, then implement your contribution
- [ ] Document your changes thoroughly
- [ ] Submit a pull request following our guidelines

## 🎉 Thank You!

Your contributions help advance multi-agent cognitive architecture research. Every test, every line of code, every documentation improvement, and every research insight makes Felix Framework better.

**Welcome to the helix revolution!** 🌪️

---

*For detailed development standards and research methodology, see [DEVELOPMENT_RULES.md](./DEVELOPMENT_RULES.md)*

*For project structure and component details, see [PROJECT_INDEX.md](./PROJECT_INDEX.md)*