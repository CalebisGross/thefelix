# Multi-Model Setup Guide

This guide shows how to configure Felix Framework for concurrent processing with multiple models on a single LM Studio server.

## Overview

The Felix Framework can use different models for different agent types on a single LM Studio server, enabling:
- **Model specialization**: Each agent type uses an optimal model
- **Concurrent processing**: Multiple agents work simultaneously 
- **Resource efficiency**: Single server manages all models
- **Easy setup**: No need for multiple server instances

## Required Models

Ensure these models are available in your LM Studio:

1. **`qwen/qwen3-4b-2507`** - Fast 4B model for research agents
2. **`qwen/qwen3-4b-thinking-2507`** - Reasoning model for analysis/critic agents  
3. **`google/gemma-3-12b`** - High-quality 12B model for synthesis agents

## Setup Instructions

### 1. Start LM Studio Server

```bash
# Start LM Studio server on default port
lm-studio server start --port 1234
```

Or use the LM Studio GUI:
- Go to "Local Server" tab
- Click "Start Server" 
- Ensure port is set to 1234 (default)

### 2. Load Models in LM Studio

Make sure all three models are available:
- Download the models in LM Studio if not already present
- The server will automatically switch between models as needed

### 3. Verify Configuration

The multi-model configuration is already created at `config/multi_model_config.json`:

```json
{
  "agent_mapping": {
    "research": "research_fast",      // Uses qwen/qwen3-4b-2507
    "analysis": "thinking_analysis",  // Uses qwen/qwen3-4b-thinking-2507
    "synthesis": "synthesis_quality", // Uses google/gemma-3-12b
    "critic": "thinking_analysis"     // Uses qwen/qwen3-4b-thinking-2507
  }
}
```

## Usage Examples

### Basic Multi-Model Blog Writing

```bash
# Run with multi-model configuration
python examples/blog_writer.py "Quantum computing applications" \
    --server-config config/multi_model_config.json \
    --debug

# The debug output will show which model each agent uses:
# 🌐 research_001 (research) → research_fast (qwen/qwen3-4b-2507)
# 🌐 analysis_001 (analysis) → thinking_analysis (qwen/qwen3-4b-thinking-2507)
# 🌐 synthesis_001 (synthesis) → synthesis_quality (google/gemma-3-12b)
```

### Test Multi-Model Setup

```bash
# Verify everything is working
python examples/test_multi_model.py

# This will:
# - Check all models are accessible
# - Verify agent-to-model mappings
# - Test concurrent processing
# - Show evidence of parallelism
```

### Compare Single vs Multi-Model Performance

```bash
# Single model (baseline)
python examples/blog_writer.py "AI ethics" \
    --server-config config/single_server_config.json

# Multi-model (specialized)
python examples/blog_writer.py "AI ethics" \
    --server-config config/multi_model_config.json
```

## Expected Behavior

When running with multi-model configuration:

### 1. **Agent Specialization**
- **Research agents** use fast Qwen 4B for quick exploration
- **Analysis agents** use Qwen Thinking for reasoning tasks
- **Synthesis agents** use Gemma 12B for high-quality final output
- **Critic agents** use Qwen Thinking for validation

### 2. **Concurrent Processing Evidence**
```
[t=0.05] 🌀 Spawning research_001 (research)
🌐 research_001 (research) → research_fast (qwen/qwen3-4b-2507)

[t=0.10] 🌀 Spawning analysis_001 (analysis)  
🌐 analysis_001 (analysis) → thinking_analysis (qwen/qwen3-4b-thinking-2507)

[t=0.15] 🚀 Processing 2 agents in parallel
    ✓ research_001 completed (depth: 0.15, confidence: 0.60, tokens: 245)
    ✓ analysis_001 completed (depth: 0.25, confidence: 0.65, tokens: 312)
```

### 3. **Model Switching**
LM Studio automatically switches between models as requests arrive:
- No manual model loading required
- Models cached in memory for faster switching
- Request queue handled internally by LM Studio

## Performance Benefits

### Compared to Sequential Processing:
- ⚡ **Faster completion**: Overlapped processing reduces total time
- 🎯 **Better quality**: Each agent type uses optimal model
- 🔧 **Model efficiency**: Specialized models for specific tasks

### Compared to Single Model:
- 🧠 **Task specialization**: Research vs reasoning vs synthesis models
- 📈 **Quality improvement**: Larger model for final synthesis
- ⚖️ **Resource balance**: Fast models for simple tasks, powerful for complex

## Troubleshooting

### Common Issues

#### 1. "No available server for agent type"
```bash
# Check LM Studio is running
curl http://127.0.0.1:1234/v1/models

# Verify models are loaded
python examples/test_multi_model.py
```

#### 2. "Model not found" errors
- Ensure all three models are downloaded in LM Studio
- Check model names match exactly in configuration
- Try loading each model manually in LM Studio first

#### 3. Slow performance
```bash
# Check LM Studio settings:
# - Increase GPU layers for faster inference
# - Enable model caching if available
# - Reduce context window if memory constrained
```

#### 4. Not seeing concurrent behavior
- This is normal - concurrency happens at HTTP request level
- LM Studio processes requests as fast as possible
- Use `--debug` flag to see detailed agent processing

### Optimization Tips

#### 1. **LM Studio Settings**
- Set "Parallel Requests" to 4+ in LM Studio settings
- Enable GPU acceleration for all models
- Increase model cache size if possible

#### 2. **Model Selection**
- Use quantized versions for faster switching
- Consider smaller models for research/analysis if quality sufficient
- Reserve largest model (Gemma 12B) for synthesis only

#### 3. **Configuration Tuning**
```json
{
  "servers": [
    {
      "max_concurrent": 2,  // Adjust based on your GPU memory
      "timeout": 120.0      // Increase for slower models
    }
  ]
}
```

## Advanced Usage

### Custom Model Mapping

Edit `config/multi_model_config.json` to use different models:

```json
{
  "agent_mapping": {
    "research": "your_fast_model",
    "analysis": "your_reasoning_model", 
    "synthesis": "your_quality_model",
    "critic": "your_validation_model"
  }
}
```

### Multiple Iterations

```bash
# Run multiple sessions to see variation
for i in {1..3}; do
  python examples/blog_writer.py "Topic $i" \
    --server-config config/multi_model_config.json \
    --random-seed $i
done
```

### Performance Monitoring

```bash
# Monitor LM Studio server logs for model switching
# Watch GPU utilization during processing
# Check network traffic to confirm concurrent requests
```

## Technical Details

### How It Works

1. **Agent Creation**: Each agent type gets mapped to specific model
2. **Concurrent Spawning**: Agents spawn at different simulation times
3. **Parallel Requests**: Multiple HTTP requests sent to LM Studio simultaneously
4. **Model Switching**: LM Studio handles model loading/switching internally
5. **Response Processing**: Results processed as they arrive

### Request Flow

```
Research Agent  → HTTP Request (model: qwen/qwen3-4b-2507)        ↘
Analysis Agent  → HTTP Request (model: qwen/qwen3-4b-thinking-2507) → LM Studio Queue
Synthesis Agent → HTTP Request (model: google/gemma-3-12b)        ↗
```

### Bottlenecks

- **GPU Memory**: Limited by largest model loaded
- **LM Studio Queue**: Processes requests sequentially but efficiently
- **Model Switching**: Small overhead when changing models
- **Network**: Minimal impact with local server

This setup provides the best balance of specialization and performance with a single LM Studio server!