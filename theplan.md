# Felix Framework Initial Prototyping Plan

## Phase 1: Foundation & Setup

### 1. Technology Stack Selection
- **Language**: Python 3.11+ (strong scientific libraries, rapid prototyping)
- **Testing**: pytest with hypothesis for property-based testing
- **Documentation**: Sphinx for technical docs, markdown for research logs
- **Visualization**: matplotlib for 2D projections, optional plotly for 3D
- **Performance**: cProfile for benchmarking, memory_profiler for memory analysis

### 2. Project Structure Creation
```
thefelix/
├── src/
│   ├── core/           # Helix engine and math
│   ├── agents/         # Agent system implementation
│   ├── communication/  # Spoke-based messaging
│   └── pipeline/       # Processing pipeline
├── tests/
│   ├── unit/          # Component tests
│   ├── integration/   # System tests
│   └── performance/   # Benchmarks
├── research/          # Research documentation
├── decisions/         # Architecture Decision Records
├── experiments/       # Experimental code
│   └── failed/       # Failed attempts with analysis
├── benchmarks/        # Performance baselines
└── docs/             # Technical documentation
```

### 3. Initial Hypothesis Documentation
Create formal hypothesis for the prototype:
- **H1**: Helical agent paths provide better task distribution than linear pipelines
- **H2**: Spoke-based communication reduces coordination overhead vs mesh networks
- **H3**: Geometric tapering naturally implements attention focusing
- **Metrics**: Define measurable success criteria for each hypothesis

## Phase 2: Core Mathematical Model

### 4. Helix Mathematics Implementation
Following test-first development:
1. Write tests for parametric helix equations
2. Write tests for position calculation at any point
3. Write tests for tapering function
4. Implement `HelixGeometry` class with:
   - Position calculation: `get_position(t, turn_number)`
   - Radius at height: `get_radius(z)`
   - Arc length computation
   - Validation against OpenSCAD model values

### 5. Agent Positioning System
1. Write tests for agent spawn timing
2. Write tests for agent movement along helix
3. Implement `AgentPosition` class with:
   - Current position tracking
   - Movement velocity calculation
   - Path progress monitoring

## Phase 3: Basic Agent System

### 6. Agent Lifecycle Management
1. Write tests for agent spawning
2. Write tests for agent state transitions
3. Write tests for agent termination
4. Implement `Agent` base class with:
   - Spawn timing based on random seed
   - State machine (spawning → active → processing → complete)
   - Basic task assignment

### 7. Simple Communication Framework
1. Write tests for spoke connections
2. Write tests for message passing
3. Implement `SpokeChannel` class with:
   - Agent-to-center messaging
   - Basic message queue
   - Latency measurement

## Phase 4: Minimal Viable Prototype

### 8. Integration & First Run
1. Create simple test scenario: distributed word counting
2. Spawn 10 agents with different start times
3. Each agent processes text chunks while descending helix
4. Agents communicate word counts via spokes
5. Central post aggregates results

### 9. Performance Baseline
1. Implement same task with traditional pipeline
2. Measure and compare:
   - Total processing time
   - Memory usage
   - Message passing overhead
   - Result accuracy

## Phase 5: Documentation & Analysis

### 10. Research Documentation
- Create RESEARCH_LOG.md with daily entries
- Document all design decisions in ADRs
- Preserve failed attempts with analysis
- Generate initial performance reports

## Implementation Order & Timeline

**Week 1**: Foundation (Steps 1-3)
- Technology setup and project structure
- Initial hypothesis documentation
- First ADR for tech choices

**Week 2**: Mathematics (Steps 4-5)
- Test-driven helix implementation
- Validate against OpenSCAD model
- Performance benchmarks for calculations

**Week 3**: Agents (Steps 6-7)
- Basic agent system with tests
- Simple communication framework
- Integration tests

**Week 4**: Integration (Steps 8-10)
- Working prototype with test scenario
- Performance comparison
- Documentation and analysis

## Success Criteria

1. **Technical**: Working prototype with 10+ agents navigating helix
2. **Performance**: Baseline metrics established and documented
3. **Scientific**: Hypotheses tested with measurable results
4. **Documentation**: Complete ADRs, research logs, and test coverage

## Risk Mitigation

- Start with 2D projection if 3D math proves complex
- Use simple message passing before optimizing
- Focus on correctness over performance initially
- Document all failures as learning opportunities

## First Action: Save Plan to theplan.md

Once approved, I will:
1. Save this complete plan to `theplan.md`
2. Begin Phase 1 implementation following the strict development rules