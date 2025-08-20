# LM Studio Multi-Server Configuration

This directory contains configuration files for running multiple LM Studio servers simultaneously with the Felix Framework.

## Configuration Files

### `multi_model_config.json` ⭐ **RECOMMENDED**
Multi-model configuration for single LM Studio server:
- **Research Model**: `qwen/qwen3-4b-2507` - Fast exploration
- **Analysis Model**: `qwen/qwen3-4b-thinking-2507` - Reasoning focused
- **Synthesis Model**: `google/gemma-3-12b` - High-quality output
- All models use same server: `http://127.0.0.1:1234/v1`

### `server_config.json`
Multi-server configuration supporting up to 4 different LM Studio servers:
- **Creative Server** (port 1234): Fast model for research agents
- **Analytical Server** (port 1235): Balanced model for analysis/critic agents  
- **Synthesis Server** (port 1236): High-quality model for synthesis agents
- **Fallback Server** (port 1237): Fast backup model (disabled by default)

### `single_server_config.json`
Single-server configuration for comparison testing.

## Setting Up Multiple LM Studio Servers

1. **Start LM Studio instances on different ports:**
   ```bash
   # Terminal 1: Creative server
   lms server start --port 1234 --model mistral-7b-instruct
   
   # Terminal 2: Analytical server  
   lms server start --port 1235 --model llama-3.1-8b-instruct
   
   # Terminal 3: Synthesis server
   lms server start --port 1236 --model mixtral-8x7b-instruct
   ```

2. **Or use LM Studio GUI:**
   - Launch multiple LM Studio instances
   - Set different ports in Settings → Developer → Port
   - Load different models in each instance
   - Start servers

## Agent Type Mappings

- **Research agents** → Creative server (Mistral) for broad exploration
- **Analysis agents** → Analytical server (Llama) for focused analysis  
- **Synthesis agents** → Synthesis server (Mixtral) for high-quality output
- **Critic agents** → Analytical server (Llama) for reasoning/validation

## Usage

### With Multi-Model Config (Recommended):
```bash
python examples/blog_writer.py "AI safety" --server-config config/multi_model_config.json --debug
```

### Test Multi-Model Setup:
```bash
python examples/test_multi_model.py
```

### With Multi-Server Config:
```bash
python examples/blog_writer.py "AI safety" --server-config config/server_config.json --debug
```

### With Single-Server Config:
```bash
python examples/blog_writer.py "AI safety" --server-config config/single_server_config.json --debug
```

### Performance Comparison:
```bash
python examples/test_multi_server_performance.py
```

## Configuration Options

### Server Settings:
- `name`: Unique server identifier
- `url`: LM Studio server URL
- `model`: Model name to use
- `timeout`: Request timeout in seconds
- `max_concurrent`: Maximum concurrent requests
- `weight`: Load balancing weight
- `enabled`: Whether server is active

### Load Balancing Strategies:
- `agent_type_mapping`: Use agent type mappings (recommended)
- `round_robin`: Rotate between servers
- `least_busy`: Use server with lowest load
- `fastest_response`: Use server with best response time

## Health Monitoring

The system automatically:
- Checks server health every 30 seconds
- Fails over to available servers
- Monitors response times and load
- Displays server status in debug mode

## Performance Benefits

Multi-server setup provides:
- **True parallelism**: Agents process simultaneously
- **Model specialization**: Each agent type uses optimal model
- **Load distribution**: Spread across multiple GPUs/servers
- **Fault tolerance**: Continue if one server fails
- **3-4x performance**: With proper server setup