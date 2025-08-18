# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Felix Framework is a research project exploring helix-based cognitive architecture for multi-agent systems. The project translates a 3D geometric helix model (`thefelix.md`) into a computational framework where autonomous agents navigate spiral processing paths with spoke-based communication to a central coordination system.

**This is active research, not production software.** All development must follow strict scientific methodology and documentation requirements.

## Core Architecture Concepts

- **Helix Path**: Non-linear processing pipeline where agents traverse a spiral from broad (top) to focused (bottom)
- **Nodes**: Autonomous agents with independent spawn timing and specialized functions
- **Spokes**: Communication channels connecting agents to the central coordination system
- **Central Post**: Core memory/coordination system maintaining system state
- **Geometric Tapering**: Natural attention focusing mechanism through radius reduction

The mathematical foundation uses parametric helix generation with 33 turns, tapering from radius 33 to 0.001, with 133 nodes representing cognitive agents.

## Critical Development Rules

**Every action must be justified, documented, and validated. No exceptions.**

Key requirements from DEVELOPMENT_RULES.md:
- NO CODE without corresponding tests written FIRST
- Mandatory commit message template with WHY/WHAT/EXPECTED/ALTERNATIVES/TESTS
- Daily research log entries in `RESEARCH_LOG.md`
- All failed experiments preserved in `experiments/failed/`
- Architecture Decision Records (ADRs) for all design decisions
- Hypothesis-driven development with measurable outcomes

## Project Structure

Current structure:
- `thefelix.md` - Original OpenSCAD prototype demonstrating core concepts
- `PROJECT_OVERVIEW.md` - Research objectives and theoretical foundation
- `DEVELOPMENT_RULES.md` - Mandatory governance protocols

Planned structure (per DEVELOPMENT_RULES.md):
- `research/` - Research insights and documentation
- `decisions/` - Architecture Decision Records
- `experiments/failed/` - Preserved failed attempts with analysis
- `RESEARCH_LOG.md` - Daily progress documentation

## Development Workflow

### Before Any Implementation
1. State clear hypothesis in research documentation
2. Write tests that validate the hypothesis
3. Document alternatives considered and rejection rationale
4. Update todo list with specific, measurable tasks

### Documentation Requirements
- Use structured commit messages with WHY/WHAT/EXPECTED sections
- Update `RESEARCH_LOG.md` daily with progress/obstacles/insights
- Document failures with analysis of why they occurred
- Create ADRs for architecture decisions

### Research Integrity
- Actively seek evidence against hypotheses
- Document negative results
- Include exact environment specifications for reproducibility
- Regular scope reviews to prevent feature creep

## Key Commands

Currently no build/test infrastructure exists. As the project evolves, document actual commands here.

Future placeholder commands:
- Test runner: TBD based on chosen implementation language
- Lint/format: TBD based on language and style choices
- Documentation generation: TBD based on doc system choice

## Working with This Codebase

1. **Read DEVELOPMENT_RULES.md first** - These are mandatory protocols, not suggestions
2. **Understand the geometric model** - Review `thefelix.md` to grasp the 3D visualization
3. **Follow scientific method** - Hypothesis → Test → Implement → Measure → Document
4. **Question everything** - This is research; be skeptical and validate assumptions
5. **Document extensively** - Over-documentation is preferred over under-documentation

## Research Context

This project requires maintaining research integrity while building software. The goal is not just working code, but scientifically valid exploration of whether helix-based cognitive architecture offers advantages over traditional multi-agent systems.

Success metrics include both functional performance and novel behavioral characteristics that emerge from the helical structure.