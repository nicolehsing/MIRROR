import os
from typing import List, Dict
from openai import OpenAI
from .base import ModelProvider

class OpenAIProvider(ModelProvider):
    """Model provider using the standard OpenAI API."""
    def __init__(self, api_key=None, model="gpt-4o", temp=0.7):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.model = model
        self.temperature = float(temp) if temp is not None else 0.7
        self.client = OpenAI(api_key=self.api_key)

    def generate_response(self, conversation: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=conversation,
            temperature=self.temperature,
            max_tokens=1000,
        )
        return response.choices[0].message.content
