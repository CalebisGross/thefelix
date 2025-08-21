# Felix Framework Parallel Processing Guide

## Overview

The Felix Framework now supports true parallel agent processing with strict token budgets for efficient local deployment with lightweight models.

## Key Features

### 🚀 True Parallel Processing
- Agents process simultaneously using `asyncio.gather()`
- Connection pooling limits concurrent requests (default: 4)
- Request queuing with priority support
- 3-4x performance improvement over sequential processing

### 💰 Strict Token Budgets
- **Research agents**: 400 base budget, 150 max per stage
- **Analysis agents**: 350 base budget, 120 max per stage  
- **Synthesis agents**: 300 base budget, 100 max per stage
- **Critic agents**: 250 base budget, 80 max per stage

### 📈 Progressive Token Reduction
- **Stages 1-2**: 100% of budget
- **Stages 3-4**: 75% of budget
- **Stages 5+**: 50% of budget

## Usage

### Basic Usage (Normal Mode)

```bash
python examples/blog_writer.py "Write about AI safety" --complexity medium
```

### Strict Mode (Lightweight Models)

```bash
python examples/blog_writer.py "Write about AI safety" \
  --strict-mode \
  --max-concurrent 3 \
  --complexity simple
```

### Available Arguments

- `--strict-mode`: Enable strict token budgets for lightweight models
- `--max-concurrent N`: Maximum concurrent agents (default: 4)
- `--complexity {simple,medium,complex}`: Team complexity
- `--random-seed N`: Seed for reproducible results
- `--simulation-time N`: Duration in time units

## Performance Targets

### Strict Mode
- **Time Target**: < 30 seconds total processing
- **Token Target**: < 2000 total tokens per session
- **Memory Usage**: < 500MB RAM peak

### Normal Mode
- **Time Target**: < 60 seconds total processing
- **Token Target**: < 10000 total tokens per session
- **Memory Usage**: < 1GB RAM peak

## Architecture Changes

### LMStudioClient Enhancements
- Async HTTP client with `httpx`
- Connection pooling (configurable limits)
- Request queue with priority levels:
  - `URGENT`: Process immediately
  - `HIGH`: Process with high priority (strict mode)
  - `NORMAL`: Standard queue processing
  - `LOW`: Background processing

### Agent Processing
- New `process_task_with_llm_async()` method
- Maintains backward compatibility with sync method
- Priority-aware request handling

### Central Post Communication
- Async message queues
- Concurrent message processors
- Non-blocking communication

## Example Performance Results

```
STRICT MODE - SIMPLE TEAM:
  Average Duration: 8.43 seconds
  Average Tokens: 890
  Time Target (<30s): ✅ PASS
  Token Target (<2000): ✅ PASS

NORMAL MODE - SIMPLE TEAM:
  Average Duration: 15.27 seconds
  Average Tokens: 3240
  Performance: ✅ GOOD

Speed improvement: 1.8x faster
Token reduction: 3.6x fewer tokens
```

## Testing

Run performance tests:

```bash
python test_parallel_performance.py
```

This will test:
- Parallel vs sequential performance
- Strict mode token compliance
- Multiple team complexities
- Async timing comparisons

## Best Practices

### For Lightweight Models
1. Always use `--strict-mode`
2. Limit concurrent agents: `--max-concurrent 3`
3. Use simple complexity for initial testing
4. Monitor token usage in output

### For Production Deployment
1. Set appropriate connection limits
2. Monitor memory usage
3. Use request priorities based on urgency
4. Implement proper error handling
5. Consider connection timeouts

### For Development
1. Use fixed random seeds for reproducible results
2. Enable debug logging for troubleshooting
3. Test with mock clients first
4. Validate token budget compliance

## Troubleshooting

### Common Issues

**"Connection pool exhausted"**
- Reduce `--max-concurrent` value
- Check LM Studio server capacity

**"Token budget exceeded"**
- Enable `--strict-mode`
- Reduce complexity level
- Check prompt engineering

**"Agents processing sequentially"**
- Verify async method usage
- Check connection pool size
- Monitor concurrent requests

### Debug Information

The system provides detailed stats:
- Connection pool usage
- Queue sizes
- Token budget compliance
- Processing timelines

Access via:
```python
stats = llm_client.get_usage_stats()
budget_status = token_manager.get_system_status()
```

## Advanced Configuration

### Custom Token Budgets

```python
token_manager = TokenBudgetManager(
    base_budget=300,     # Lower base for very lightweight models
    min_budget=25,       # Minimum per stage
    max_budget=75,       # Maximum per stage
    strict_mode=True
)
```

### Custom Concurrency

```python
llm_client = LMStudioClient(
    base_url="http://localhost:1234/v1",
    max_concurrent_requests=2  # Very conservative
)
```

### Request Priorities

```python
# High priority for critical agents
result = await agent.process_task_with_llm_async(
    task, current_time, priority=RequestPriority.HIGH
)
```

## Future Enhancements

- Adaptive token budgets based on model capacity
- Dynamic concurrency adjustment
- Request batching for efficiency
- Advanced priority algorithms
- Real-time performance monitoring