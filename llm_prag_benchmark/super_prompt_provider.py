import os
from typing import List, Dict
from openai import OpenAI
from .providers.base import PipelineProvider

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "super_prompt.txt")
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    SUPER_PROMPT = f.read()

class SuperPromptProvider(PipelineProvider):
    """Pipeline provider that sends a consolidated prompt in a single call."""

    def __init__(self, api_key=None, model=None, temp=0.7):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        if model and model.startswith("openrouter/"):
            model = model[len("openrouter/"):]
        self.model = model or "gpt-4o"
        self.temperature = float(temp) if temp is not None else 0.7

        self.client = OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1")

    def generate_immediate_response(self, conversation: List[Dict[str, str]]) -> str:
        messages = [{"role": "system", "content": SUPER_PROMPT}]
        for msg in conversation:
            messages.append({"role": msg["role"], "content": msg["content"]})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=1000,
        )
        return response.choices[0].message.content

    def run_background_processor(self, conversation: List[Dict[str, str]]) -> None:
        pass

    def generate_response(self, conversation: List[Dict[str, str]]) -> str:
        return self.generate_immediate_response(conversation)
