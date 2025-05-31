# MIRROR: Modular Internal Reasoning, Reflection, Orchestration, and Response

This repository is the official implementation of **MIRROR: Modular Internal Reasoning, Reflection, Orchestration, and Response**.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Human intelligence relies on inner monologue to process complex information through simultaneous reflection, memory retrieval, and response formulation. MIRROR is a cognitive architecture that systematically implements these parallel reasoning capabilities in large language models.

MIRROR operates as a unified system with two distinct functional layers:
- **The Thinker**: Encompasses the Inner Monologue Manager and Cognitive Controller for parallel reasoning
- **The Talker**: Leverages integrated narrative for context-aware responses

### Key Results

When evaluated on the CuRaTe benchmark (testing personalized dialogue with safety-critical constraints, conflicting preferences, and multi-turn consistency):
- **156% relative improvement** in critical safety scenarios involving conflicting preferences
- **Average accuracy of >80%** across all scenarios  
- **21% average relative improvement (15 percentage points absolute)** across GPT-4o, Gemini 1.5 Pro, Claude 3.7 Sonnet, Llama 4, and Mistral 3 variants
- Addresses three critical LLM failure modes: sycophancy, attentional deficits, and inconsistent constraint prioritization

## Requirements

To install requirements:

```bash
conda create -n mirror python=3.10 -y
conda activate mirror
pip install -r requirements.txt
```


### Environment Setup

Set up your API keys using one of these methods:

```bash
# Option 1: Create .env file
cp .env.example .env
# Edit .env to add your API keys

# Option 2: Set environment variables directly
export OPENROUTER_API_KEY="your_openrouter_api_key"
export OPENAI_API_KEY="your_openai_api_key"          # Optional for baselines
export ANTHROPIC_API_KEY="your_anthropic_api_key"    # Optional for baselines
export GOOGLE_API_KEY="your_google_api_key"          # Optional for baselines
```

### API Keys Required

- **OpenRouter API Key**: Primary interface for MIRROR architecture
- **Additional API Keys**: Optional, for baseline comparisons with specific providers

## Usage

### Interactive Mode

Run MIRROR in interactive conversation mode:

```bash
cd core
python main.py
```

### Python API

```python
from core.components.mirror import Mirror

# Initialize with default model (GPT-4o)
mirror = Mirror()

# Or initialize with specific model
mirror = Mirror(model="anthropic/claude-3.5-sonnet")

# Process user input and get response
response = mirror.process_user_input("Your message here")

# Reset conversation state
mirror.reset_conversation()
```

### Architecture Components

```python
# Access individual components
mirror.monologue_manager    # Inner Monologue Manager
mirror.cognitive_controller # Cognitive Controller  
mirror.talker              # Response generation
```

## Evaluation

### Running CuRaTe Benchmark

To evaluate MIRROR on the CuRaTe benchmark (our primary evaluation):

```bash
cd llm_prag_benchmark

# Basic evaluation with default settings
python eval.py

# Evaluate specific model with MIRROR architecture
python eval.py --mirror-model "anthropic/claude-3.5-sonnet" --model-prefix "mirror-"

# Run baseline comparison
python eval.py --mirror-model "anthropic/claude-3.5-sonnet" --model-prefix "baseline-"

# Run complete comparison with parallel processing
./run_parallel.sh "anthropic/claude-3.5-sonnet" "anthropic/claude-3.5-sonnet" 6
```

### Latency Testing

To evaluate response latency and compare with baseline models:

```bash
cd core

# Compare MIRROR vs baseline latency
python test_latency_benchmark.py --test-mode both --baseline-model "gpt-4o" --scenarios 5

# Test only MIRROR latency
python test_latency_benchmark.py --test-mode mirror --scenarios 10

# Test baseline model latency with OpenRouter
python test_baseline_latency.py --model "anthropic/claude-3.5-sonnet" --api-provider openrouter
```

### Evaluation Arguments

**CuRaTe Benchmark** (`llm_prag_benchmark/eval.py`):
- `--mirror-model`: Model identifier (default: "openai/gpt-4o")
- `--scenario N`: Run specific scenario (1-5)
- `--parallel`: Enable parallel processing
- `--workers N`: Number of parallel workers
- `--results-dir DIR`: Output directory for results

**Latency Testing** (`core/test_latency_benchmark.py`):
- `--test-mode`: "mirror", "baseline", or "both"
- `--baseline-model`: Baseline model for comparison
- `--scenarios N`: Number of test scenarios
- `--api-provider`: "openai" or "openrouter"

## Pre-trained Models

MIRROR is a cognitive architecture that enhances existing language models rather than providing pre-trained weights. The system works with any OpenRouter-compatible model:

### Supported Models
- **OpenAI**: gpt-4o, gpt-4-turbo, gpt-3.5-turbo
- **Anthropic**: claude-3.5-sonnet, claude-3-opus, claude-3-haiku  
- **Google**: gemini-1.5-pro, gemini-1.5-flash
- **Meta**: llama-3.1-405b-instruct, llama-3.1-70b-instruct
- **Mistral**: mistral-large, mixtral-8x7b-instruct
- **Many others**: Available through [OpenRouter](https://openrouter.ai/models)

The architecture automatically adapts to different base models while maintaining consistent cognitive processing capabilities.

## Results

Our evaluation on the CuRaTe benchmark demonstrates significant improvements across multiple dimensions:

### Overall Performance Across All Models

| Model | Baseline | MIRROR | Absolute Improvement | Relative Improvement |
|-------|----------|---------|---------------------|---------------------|
| Llama 4 Maverick | 75.0% | 85.0% | +10.0pp | +13.3% |
| Llama 4 Scout | 73.0% | 91.0% | +18.0pp | +24.7% |
| Gemini 1.5 Pro | 51.0% | 78.0% | +27.0pp | +52.9% |
| GPT-4o | 70.0% | 80.0% | +10.0pp | +14.3% |
| Claude 3.7 Sonnet | 75.0% | 82.0% | +7.0pp | +9.3% |
| Mistral Medium 3 | 72.0% | 90.0% | +18.0pp | +25.0% |
| Mistral Small 3.1 24B | 65.0% | 82.0% | +17.0pp | +26.2% |
| **Average** | **68.7%** | **84.0%** | **+15.3pp** | **+21%** |

### Scenario-Specific Performance Gains

| Scenario | Description | Average Improvement | Notable Results |
|----------|-------------|-------------------|-----------------|
| Scenario 1 | Basic constraint tracking | +20.4% | Llama 4 Scout: +47.1% |
| Scenario 2 | User + 1 conflicting person | +28.8% | Gemini 1.5 Pro: +78.3% |
| Scenario 3 | User + 2 conflicting people | +15.3% | Gemini 1.5 Pro: +62.7% |
| Scenario 4 | User + 3 conflicting people | +40.6% | Gemini 1.5 Pro: **+156.2%** |
| Scenario 5 | Non-conflicting preferences | +28.5% | Mistral Medium 3: +49.2% |

### Response Time Performance (MIRROR with GPT-4o)

| Metric | Value |
|--------|-------|
| Average response time | 2.52s |
| Median response time | 2.16s |
| 75th percentile | ~3.0s |
| Background thread conflicts | 0.8% of turns |
| Average queue length | 0.01 threads |


## Architecture Details

### System Components

1. **Inner Monologue Manager**: Coordinates reasoning threads across cognitive dimensions
   - Goals Thread: Tracks user intentions and objectives
   - Reasoning Thread: Processes logical implications
   - Memory Thread: Manages relevant context and constraints

2. **Cognitive Controller**: Synthesizes parallel insights into coherent narratives

3. **Talker**: Generates context-aware responses using integrated understanding

### Processing Flow

```
User Input → Immediate Response (Talker)
     ↓
Background Processing:
     ↓
Inner Monologue Manager → Cognitive Controller → Enhanced Context
     ↓
Next User Input → Enhanced Response (Talker + Context)
```

## Contributing

We welcome contributions to improve MIRROR's cognitive architecture:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-capability`)
3. Make changes and add tests
4. Commit changes (`git commit -am 'Add new cognitive capability'`)
5. Push to branch (`git push origin feature/new-capability`)
6. Create a Pull Request

### Development Guidelines

- Follow existing code style and documentation patterns
- Add appropriate tests for new functionality
- Update documentation for API changes
- Test with multiple model providers before submitting


## License

This project is licensed under the Creative Commons Attribution 4.0 International License - see the [LICENSE](LICENSE) file for details.