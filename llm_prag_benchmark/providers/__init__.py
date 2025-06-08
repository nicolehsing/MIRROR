"""Provider registry for the benchmark.

Pipeline providers encapsulate full prompting logic or multi-step pipelines
while model providers expose raw model APIs with minimal wrapper logic.
"""
from ..mirror_provider import MirrorProvider
from ..super_prompt_provider import SuperPromptProvider
from ..baseline_mirror_provider import BaselineMirrorProvider
from .openai_provider import OpenAIProvider
from .openrouter_provider import OpenRouterProvider
from .base import ModelProvider, PipelineProvider

# Providers that access base models directly
MODEL_PROVIDERS = {
    "openai": OpenAIProvider,
    "openrouter": OpenRouterProvider,
}

# Providers that implement higher-level prompting or reasoning pipelines
PIPELINE_PROVIDERS = {
    "mirror": MirrorProvider,
    "baseline_mirror": BaselineMirrorProvider,
    "superprompt": SuperPromptProvider,
}

AVAILABLE_PROVIDERS = {**MODEL_PROVIDERS, **PIPELINE_PROVIDERS}

__all__ = [
    "ModelProvider",
    "PipelineProvider",
    "MirrorProvider",
    "BaselineMirrorProvider",
    "SuperPromptProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "AVAILABLE_PROVIDERS",
    "MODEL_PROVIDERS",
    "PIPELINE_PROVIDERS",
]
