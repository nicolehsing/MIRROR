from typing import List, Dict

class BaseProvider:
    """Common interface for all providers."""

    def generate_response(self, conversation: List[Dict[str, str]]) -> str:
        """Generate a response for the given conversation."""
        raise NotImplementedError

class ModelProvider(BaseProvider):
    """Interface for providers that wrap a raw model API."""
    pass

class PipelineProvider(BaseProvider):
    """Interface for providers that embed prompting or pipeline logic."""

    def generate_immediate_response(self, conversation: List[Dict[str, str]]) -> str:
        """Produce the direct assistant reply for the latest user message."""
        raise NotImplementedError

    def run_background_processor(self, conversation: List[Dict[str, str]]) -> None:
        """Process background reasoning based on the conversation history."""
        raise NotImplementedError

    def generate_response(self, conversation: List[Dict[str, str]]) -> str:
        """Default implementation using immediate response then background step."""
        response = self.generate_immediate_response(conversation)
        self.run_background_processor(conversation)
        return response
