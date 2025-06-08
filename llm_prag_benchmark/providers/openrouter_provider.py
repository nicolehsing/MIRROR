import os
from typing import List, Dict
from openai import OpenAI
from .base import ModelProvider

class OpenRouterProvider(ModelProvider):
    """Model provider using OpenRouter API."""
    def __init__(self, api_key=None, model="gpt-4o", temp=0.7, site_name=None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        if model and model.startswith("openrouter/"):
            model = model[len("openrouter/") :]
        self.model = model
        self.temperature = float(temp) if temp is not None else 0.7
        self.site_name = site_name or os.getenv("OPENROUTER_SITE_NAME", "MIRROR-Benchmark")
        self.client = OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1")

    def generate_response(self, conversation: List[Dict[str, str]]) -> str:
        extra_headers = {
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": self.site_name,
        }
        response = self.client.chat.completions.create(
            model=self.model,
            messages=conversation,
            temperature=self.temperature,
            max_tokens=1000,
            extra_headers=extra_headers,
        )
        return response.choices[0].message.content
