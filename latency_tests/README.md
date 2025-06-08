# MIRROR Latency Testing Guide

This guide provides comprehensive documentation for testing the latency performance of the MIRROR system against baseline language models using realistic conversation scenarios.

## Overview

The MIRROR latency testing suite supports three types of testing:
- **MIRROR-only testing**: Test the MIRROR system performance in isolation
- **Baseline testing**: Test baseline models (OpenAI, OpenRouter) for comparison
- **Comparison testing**: Run both MIRROR and baseline tests with identical scenarios for direct performance comparison

### What We Test

The tests simulate realistic human-AI conversations using:
- **Multi-turn conversations** (5 turns each): Introduction + 3 distractor questions + critical recommendation request
- **Human behavior simulation**: Realistic typing speeds (40 WPM), reading speeds (250 WPM), with natural variation
- **Benchmark scenarios**: Real pragmatic scenarios from the LLM pragmatics benchmark
- **Background activity monitoring**: For MIRROR, track parallel processing queue activity

## Test Scripts

### 1. Enhanced Main Script: `core/test_latency_benchmark.py`

The primary testing script supporting all modes:

```bash
# Test both MIRROR and baseline GPT-4o (comparison mode)
python core/test_latency_benchmark.py --test-mode both

# Test only MIRROR
python core/test_latency_benchmark.py --test-mode mirror

# Test only baseline OpenAI model
python core/test_latency_benchmark.py --test-mode baseline --baseline-model gpt-4o

# Test with OpenRouter (Claude 3 Sonnet)
python core/test_latency_benchmark.py --test-mode baseline --api-provider openrouter --baseline-model anthropic/claude-3-sonnet

# Comparison: MIRROR vs OpenRouter Llama 2
python core/test_latency_benchmark.py --test-mode both --api-provider openrouter --baseline-model meta-llama/llama-2-70b-chat

# Use different benchmark file and custom settings
python core/test_latency_benchmark.py --test-mode both --benchmark inputs_80.xlsx --scenarios 10 --typing-speed 50
```

### 2. Standalone Baseline Script: `core/test_baseline_latency.py`

Lightweight script for testing baseline models only:

```bash
# Test OpenAI GPT-4o
python core/test_baseline_latency.py --model gpt-4o

# Test OpenRouter Claude 3 Sonnet
python core/test_baseline_latency.py --api-provider openrouter --model anthropic/claude-3-sonnet

# Test OpenRouter GPT-4 with custom settings
python core/test_baseline_latency.py --api-provider openrouter --model openai/gpt-4 --scenarios 8 --typing-speed 35
```

## API Provider Setup

### OpenAI API
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### OpenRouter API
```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
# Optional: Add site name for OpenRouter analytics
export OPENROUTER_SITE_NAME="MIRROR-Latency-Test"
```

## Supported Models

### OpenAI API Provider
- `gpt-4o` (default)
- `gpt-4`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

### OpenRouter API Provider
Popular models include:
- `anthropic/claude-3-sonnet`
- `anthropic/claude-3-haiku`
- `openai/gpt-4o`
- `openai/gpt-4`
- `meta-llama/llama-2-70b-chat`
- `meta-llama/llama-2-13b-chat`
- `mistralai/mixtral-8x7b-instruct`
- `google/gemini-pro`

See [OpenRouter models page](https://openrouter.ai/models) for the complete list.

## Command Line Arguments

### Enhanced Main Script Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--test-mode` | Test mode: `mirror`, `baseline`, `both` | `both` |
| `--mirror-endpoint` | MIRROR API endpoint | `http://localhost:5555/v1/chat/completions` |
| `--baseline-model` | Baseline model name | `gpt-4o` |
| `--api-provider` | API provider: `openai`, `openrouter` | `openai` |
| `--scenarios` | Number of test scenarios | `5` |
| `--typing-speed` | Typing speed in WPM | `40` |
| `--reading-speed` | Reading speed in WPM | `250` |
| `--benchmark` | Benchmark file to use | `inputs_80.xlsx` |

### Standalone Script Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model to test | `gpt-4o` |
| `--api-provider` | API provider: `openai`, `openrouter` | `openai` |
| `--scenarios` | Number of test scenarios | `5` |
| `--typing-speed` | Typing speed in WPM | `40` |
| `--reading-speed` | Reading speed in WPM | `250` |
| `--benchmark` | Benchmark file to use | `inputs_80.xlsx` |

## Test Scenarios

### Scenario Structure

Each test scenario follows this 5-turn pattern:

1. **Turn 1**: Introduction with constraint + personal fact
   - Example: "Allow me to introduce myself. I have a severe peanut allergy. I enjoy science fiction movies. Tell me about you?"

2. **Turns 2-4**: Distractor questions (trivia)
   - Example: "What's the capital of France?"
   - Example: "Who wrote Hamlet?"
   - Example: "What's the speed of light?"

3. **Turn 5**: Critical recommendation request
   - Example: "Do you think I should try this new Thai restaurant? First answer, then explain your reasoning."

### Human Simulation

- **Typing simulation**: Calculates realistic typing time based on text length and typing speed (±20% variation)
- **Reading simulation**: Simulates reading previous responses before typing new messages (±15% variation)
- **Natural variation**: Adds randomness to prevent unrealistic precision

## Key Metrics

### Response Time Metrics
- **Average response time**: Mean across all conversation turns
- **Median response time**: Middle value for robustness against outliers
- **Last turn response time**: Critical metric for recommendation requests
- **Min/Max response time**: Range of performance
- **Standard deviation**: Consistency measure

### Human Simulation Metrics
- **Typing time**: Simulated time for user to type messages
- **Reading time**: Simulated time for user to read AI responses
- **Total human time**: Combined typing and reading time
- **Human time percentage**: Proportion of total conversation time

### Background Activity (MIRROR only)
- **Queue size**: Number of requests in MIRROR's processing queue
- **Active requests**: Currently processing requests
- **Background processing percentage**: Time with active background threads

## Sample Output

### Comparison Mode Output
```
MIRROR vs GPT-4O COMPARISON SUMMARY (Using inputs_80.xlsx)
Baseline API Provider: openai
================================================================================
Metric                    | MIRROR       | BASELINE     | Difference  
----------------------------------------------------------------------
Total scenarios           | X            | X            | N/A         
Total turns               | X            | X            | N/A         
Average response time     | Xs           | Xs           | X%      
Median response time      | Xs           | Xs           | X%      
Avg typing time           | Xs           | Xs           | X%       
Avg reading time          | Xs           | Xs           | X%       
Last turn avg             | Xs           | Xs           | X%      
----------------------------------------------------------------------

KEY INSIGHTS:
✅ MIRROR is X% faster on average
✅ MIRROR is X% faster for recommendation requests
📊 MIRROR background queue avg size: X
```

## Result Files

Test results are automatically saved to `core/test_results/` with timestamps:

### File Naming Convention
- **MIRROR only**: `mirror_inputs_80_20241201_143022.json`
- **Baseline only**: `baseline_openai_gpt-4o_inputs_80_20241201_143022.json`
- **OpenRouter baseline**: `baseline_openrouter_anthropic_claude-3-sonnet_inputs_80_20241201_143022.json`
- **Comparison**: `comparison_mirror_vs_openai_gpt-4o_inputs_80_20241201_143022.json`

### Result File Structure
```json
{
  "test_config": {
    "test_mode": "comparison",
    "baseline_model": "gpt-4o",
    "api_provider": "openai",
    "num_scenarios": 5,
    "benchmark_file": "inputs_80.xlsx",
    "timestamp": "20241201_143022"
  },
  "mirror_metrics": { /* ... */ },
  "baseline_metrics": { /* ... */ },
  "comparison_analysis": {
    "avg_response_time_improvement_percent": 23.7,
    "last_turn_improvement_percent": 21.3
  },
  "mirror_scenarios": [ /* detailed per-scenario data */ ],
  "baseline_scenarios": [ /* detailed per-scenario data */ ],
  "background_queue_metrics": [ /* MIRROR queue activity */ ]
}
```

## Advanced Testing

### Batch Testing Script Example

Create a script to test multiple models:

```bash
#!/bin/bash
# batch_test.sh

# Test different OpenAI models
python core/test_latency_benchmark.py --test-mode both --baseline-model gpt-4o --scenarios 10
python core/test_latency_benchmark.py --test-mode both --baseline-model gpt-4 --scenarios 10

# Test OpenRouter models
python core/test_latency_benchmark.py --test-mode both --api-provider openrouter --baseline-model anthropic/claude-3-sonnet --scenarios 10
python core/test_latency_benchmark.py --test-mode both --api-provider openrouter --baseline-model meta-llama/llama-2-70b-chat --scenarios 10

# Test different human speeds
python core/test_latency_benchmark.py --test-mode both --typing-speed 25 --reading-speed 200 --scenarios 5  # Slow user
python core/test_latency_benchmark.py --test-mode both --typing-speed 60 --reading-speed 300 --scenarios 5  # Fast user
```

### Custom Benchmark Files

To use custom benchmark scenarios:

1. Place Excel file in `llm_prag_benchmark/` directory
2. Ensure columns: `User's hard constraint`, `Question (user only)`, `Category`
3. Specify with `--benchmark your_file.xlsx`

### Performance Profiling

For detailed performance analysis:

```bash
# Long test with many scenarios
python core/test_latency_benchmark.py --test-mode both --scenarios 20 --baseline-model gpt-4o

# Test with different typing speeds to understand human simulation impact
python core/test_latency_benchmark.py --test-mode both --typing-speed 20 --scenarios 5   # Very slow
python core/test_latency_benchmark.py --test-mode both --typing-speed 80 --scenarios 5   # Very fast
```

## Troubleshooting

### Common Issues

**Missing API Keys**
```
ValueError: OPENAI_API_KEY environment variable not set
```
Solution: Set the appropriate environment variable for your chosen API provider.

**MIRROR Connection Failed**
```
Error calling MIRROR: Connection refused
```
Solution: Ensure MIRROR system is running and accessible at the specified endpoint.

**OpenRouter Model Not Found**
```
Error calling openrouter with model xyz: Model not found
```
Solution: Check [OpenRouter models page](https://openrouter.ai/models) for correct model names.

**Benchmark File Not Found**
```
Error loading benchmark scenarios: File not found
```
Solution: Ensure benchmark files are in `llm_prag_benchmark/` directory.

### Performance Issues

**Slow Tests**
- Reduce `--scenarios` for quicker testing
- Increase `--typing-speed` and `--reading-speed` to reduce simulation time
- Use `--test-mode baseline` or `--test-mode mirror` instead of `both`

**Memory Issues**
- Use smaller scenario counts
- Clear result files periodically from `core/test_results/`

## Best Practices

### For Development Testing
```bash
# Quick comparison test
python core/test_latency_benchmark.py --test-mode both --scenarios 3 --typing-speed 60

# Test specific edge cases
python core/test_latency_benchmark.py --test-mode mirror --scenarios 1 --benchmark custom_scenarios.xlsx
```

### For Production Evaluation
```bash
# Comprehensive comparison
python core/test_latency_benchmark.py --test-mode both --scenarios 20 --baseline-model gpt-4o

# Multiple model comparison
python core/test_latency_benchmark.py --test-mode both --api-provider openrouter --baseline-model anthropic/claude-3-sonnet --scenarios 20
```

### For Research Analysis
- Use consistent scenario counts across tests
- Test multiple models with same parameters
- Save and analyze result JSON files for statistical significance
- Consider human variation factors when interpreting results

## Dependencies

Ensure these packages are installed:
```bash
pip install openai pandas openpyxl requests
```

For MIRROR testing, ensure the MIRROR system is running and accessible.

This testing suite provides comprehensive latency evaluation capabilities for comparing MIRROR against various baseline models using realistic conversation scenarios and human behavior simulation.

## Extending Baseline Providers

Latency tests rely on provider classes declared in `llm_prag_benchmark/providers`.
Two interfaces are available:

- **PipelineProvider** – for systems that embed their own prompting logic (for
  example `MirrorProvider`, `BaselineMirrorProvider` or `SuperPromptProvider`).
- **ModelProvider** – for wrappers around a base model that expose only a minimal
  generate interface.

Implement a subclass of one of these interfaces with a `generate_response()`
method and register it in `providers/__init__.py`. Large prompt text should live
under `llm_prag_benchmark/prompts`. Providers are selected at runtime with the
`--providers` or `--baseline-provider` arguments, so the benchmark code remains
unchanged when adding new strategies.
