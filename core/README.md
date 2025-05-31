# MIRROR: Cognitive Architecture for LLMs

MIRROR (Modular Internal Reasoning, Reflection, Orchestration, and Response) enhances LLMs with internal reasoning and reflection capabilities between conversation turns.

## Installation

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Set up environment variables:
   ```
   # Option 1: Create .env file from example
   cp .env.example .env
   nano .env  # Edit to add your API keys
   
   # Option 2: Set environment variable directly
   export OPENROUTER_API_KEY="your_api_key_here"
   ```

## Usage

### Interactive Mode

Run the system in interactive mode:

```
python -m core.main
```

Type 'exit' to quit the session.

### Using in Custom Applications

```python
from core.components.mirror import Mirror

# Initialize with default model (gpt-4o)
mirror = Mirror()

# Or initialize with specific model
mirror = Mirror(model="anthropic/claude-3-opus")

# Process user input and get response
response = mirror.process_user_input("Your message here")

# Reset conversation state
mirror.reset_conversation()
```

## Architecture Overview

MIRROR combines:

1. **Talker**: Generates immediate responses
2. **Thinker**: Processes in background with:
   - Inner Monologue Manager: Generates reasoning, memory, and goal insights
   - Cognitive Controller: Synthesizes insights into a coherent narrative

The system responds immediately to user input while asynchronously processing deeper reasoning for future turns.

## Configuration Options

- `model`: Specify the OpenRouter model identifier (default: "openai/gpt-4o")
- `api_key`: OpenRouter API key (defaults to OPENROUTER_API_KEY environment variable)

## Example Session

```
You: What are the ethical implications of AI?
AI: [Immediate response based on the query]

You: How might these change in the future?
AI: [Response informed by previous background thinking]
```

First responses are immediate. Subsequent responses benefit from background cognitive processing that occurs between turns.
