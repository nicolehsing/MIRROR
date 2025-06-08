# MIRROR Architecture

MIRROR (Modular Internal Reasoning, Reflection, Orchestration, and Response) pairs an immediate conversational agent with background reasoning that runs between turns. The architecture is implemented in the `core` package and exposed through provider classes in `llm_prag_benchmark` for evaluation.

## Core Components

1. **Talker** (`core/components/talker.py`)
   - Generates the user facing reply.
   - Accepts the current conversation history plus an "Internal Narrative" that contains insights from previous turns.
   - Uses OpenRouter to call the underlying LLM.

2. **Inner Monologue Manager** (`core/components/inner_monologue_manager.py`)
   - Launches an `InnerMonologue` thread that asks the LLM to produce three streams: reasoning, memory, and goal.
   - Maintains a `monologue_history` list with recent monologue outputs.

3. **Cognitive Controller** (`core/components/cognitive_controller.py`)
   - Integrates the latest monologue streams with the prior internal narrative.
   - Stores the resulting narrative in `insight_memory_block` for the next turn.

4. **Mirror** (`core/components/mirror.py`)
   - Orchestrates the full flow. After each user message it:
     1. Appends the message to `conversation_history`.
     2. Generates an immediate response using the Talker and any current insights.
     3. Spawns a background thread to run the Inner Monologue Manager and Cognitive Controller on the conversation snapshot.
     4. Saves the updated narrative for future turns.
   - Tracks additional state such as salience scores and raw monologue output for benchmarking.

## Provider Interfaces

Benchmark scripts use provider classes defined in `llm_prag_benchmark/providers`.

- **Model Providers** wrap a single model API (`OpenAIProvider`, `OpenRouterProvider`).
- **Pipeline Providers** encapsulate multi step prompting logic (`MirrorProvider`, `BaselineMirrorProvider`, `SuperPromptProvider`).
- All providers expose `generate_response(conversation)` so the evaluation code can swap between implementations by key name.

The mapping of provider keys to classes is declared in `llm_prag_benchmark/providers/__init__.py`.

## Processing Loop

```text
User Input → Talker responds immediately
           ↘
            Background thread: Inner Monologue → Cognitive Controller → updated narrative
```

The next user turn receives a reply informed by the narrative produced in the background. Conversation history grows over time, while the consolidated narrative is overwritten each cycle.

## Extending the System

New providers can be added by subclassing `ModelProvider` or `PipelineProvider` and registering the class in `AVAILABLE_PROVIDERS`. Benchmark scripts then load them via command line arguments without further code changes.

