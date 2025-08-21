# Development Rules and Governance Protocol

## Core Principle
**Every action must be justified, documented, and validated. No exceptions.**

## Rule 1: Documentation Requirements

### 1.1 Change Documentation
- **MANDATORY**: Every code change requires documentation of WHY, WHAT, and EXPECTED OUTCOME
- **Format**: Use structured commit messages following template below
- **Evidence**: Include reasoning, alternatives considered, and rejection rationale

### 1.2 Commit Message Template
```
[TYPE]: Brief description (max 50 chars)

WHY: Detailed explanation of the problem/need
WHAT: Specific changes made
EXPECTED: Predicted outcome/behavior
ALTERNATIVES: Other approaches considered and why rejected
TESTS: How this change will be validated
REFERENCES: Related issues, documents, or research

[Optional: BREAKING CHANGES, NOTES, etc.]
```

### 1.3 Daily Research Log
- **MANDATORY**: End-of-day summary in `RESEARCH_LOG.md`
- **Include**: Progress, obstacles, insights, questions raised
- **Format**: Date, objectives, outcomes, next steps

## Rule 2: Testing Protocol

### 2.1 Test-First Development
- **NO CODE** without corresponding tests
- Tests must be written BEFORE implementation
- Tests must validate the specific hypothesis being tested

### 2.2 Evidence-Based Claims
- **NO ASSERTIONS** without measurable evidence
- Performance claims require benchmarks
- Behavior claims require reproducible demonstrations

### 2.3 Validation Requirements
```
For every component:
1. Unit tests for individual functions
2. Integration tests for component interactions
3. Performance benchmarks vs baseline
4. Documentation of expected vs actual behavior
```

## Rule 3: Scientific Method Application

### 3.1 Hypothesis-Driven Development
- **BEFORE** implementing: State clear hypothesis
- **DURING** implementation: Document observations
- **AFTER** implementation: Measure against hypothesis
- **ALWAYS**: Document whether hypothesis was confirmed/rejected

### 3.2 Controlled Experiments
- Isolate variables when testing
- Maintain control groups/baseline comparisons
- Repeat experiments for consistency
- Document environmental factors

### 3.3 Failure Documentation
- **MANDATORY**: Document all failed attempts
- Include analysis of why failure occurred
- Preserve failed code in `experiments/failed/` directory
- Extract lessons learned for future attempts

## Rule 4: Code Standards

### 4.1 No Speculation in Code
- Comments must state facts, not intentions or guesses
- Use research notes for speculation
- Code should be self-documenting with clear variable names

### 4.2 Architecture Decisions
- **MANDATORY**: Document architecture decision records (ADRs)
- Include problem statement, options considered, decision rationale
- Update ADRs when decisions are reversed or modified

### 4.3 Performance Baselines
- Establish baseline metrics BEFORE optimization
- Measure impact of every performance-related change
- No optimization without proven performance problem

## Rule 5: Version Control Standards

### 5.1 Semantic Versioning
- **MAJOR**: Breaking changes to core architecture
- **MINOR**: New features that maintain backward compatibility
- **PATCH**: Bug fixes and documentation updates

### 5.2 Branch Strategy
- `main`: Stable, tested code only
- `develop`: Integration branch for features
- `feature/*`: Individual feature development
- `experiment/*`: Research and experimental code

### 5.3 Review Process
- **NO DIRECT COMMITS** to main or develop
- Self-review checklist required before any merge
- All merges require documented approval rationale

## Rule 6: Research Integrity

### 6.1 Bias Prevention
- Actively seek evidence against our hypotheses
- Document when results don't match expectations
- Include negative results in research documentation

### 6.2 Reproducibility
- **ALL** experiments must be reproducible
- Include exact environment specifications
- Provide step-by-step reproduction instructions

### 6.3 External Validation
- Seek feedback from unbiased sources
- Document external input and how it influenced decisions
- Maintain changelog of external influence

## Rule 7: Scope Management

### 7.1 Feature Creep Prevention
- **EVERY** new feature must directly support core research objectives
- Maintain feature justification log
- Regular scope reviews with documented decisions

### 7.2 Reality Checks
- Weekly review: "Does this align with our foundational concept?"
- Monthly review: "Are we solving the right problem?"
- Quarterly review: "Should we continue this research direction?"

### 7.3 Kill Criteria
- Predetermined conditions for abandoning approaches
- Document sunk cost fallacy prevention measures
- Clear exit strategies for failed hypotheses

## Rule 8: Data and Measurement

### 8.1 Quantifiable Metrics
- Define success metrics before implementation
- Establish measurement procedures and tools
- Regular metric collection and analysis

### 8.2 Data Integrity
- Raw data preservation in version control
- Analysis scripts under version control
- Audit trail for all data processing

### 8.3 Statistical Rigor
- Appropriate sample sizes for conclusions
- Statistical significance testing where applicable
- Confidence intervals for performance claims

## Rule 9: Communication Standards

### 9.1 Internal Documentation
- Technical decisions documented in `decisions/` directory
- Research insights in `research/` directory
- Meeting notes and discussions preserved

### 9.2 External Communication
- No claims about research without documented evidence
- Clearly distinguish between proven results and ongoing work
- Maintain research integrity in all public statements

## Rule 10: Compliance and Enforcement

### 10.1 Self-Auditing
- Weekly compliance review against these rules
- Document any rule violations and corrective actions
- Update rules based on lessons learned

### 10.2 Tool Support
- Automated checks where possible (linting, testing, etc.)
- Template systems to enforce documentation standards
- Regular backup and preservation of research artifacts

### 10.3 Rule Evolution
- Rules may only be changed with documented justification
- Changes require analysis of impact on research validity
- Version control for rule changes with rationale

---

## Enforcement Statement

**These rules are not suggestions - they are mandatory protocols for maintaining research integrity. Violation of these rules compromises the validity of our research and is unacceptable.**

**When in doubt, document first, code second.**

---

**Document Version**: 1.0  
**Effective Date**: 2025-08-18  
**Next Review**: 2025-09-18  
**Compliance**: Mandatory for all project contributors