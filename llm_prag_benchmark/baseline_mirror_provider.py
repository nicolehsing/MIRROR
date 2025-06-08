import os
import threading
from typing import List, Dict

from .providers.base import PipelineProvider
from core.components.mirror import Mirror

class BaselineMirrorProvider(PipelineProvider):
    """Pipeline provider exposing separate immediate and background steps."""

    def __init__(self, api_key=None, model=None, temp=0.7):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        if model and model.startswith("openrouter/"):
            model = model[len("openrouter/") :]
        self.model = model or "gpt-4o"
        self.temperature = float(temp) if temp is not None else 0.7

        self.mirror = Mirror(api_key=self.api_key, model=self.model)

    def generate_immediate_response(self, conversation: List[Dict[str, str]]) -> str:
        """Generate the assistant reply using the Talker component."""
        if not conversation:
            return ""

        latest_user = conversation[-1]["content"]

        # Reset if this conversation doesn't match stored history
        if self.mirror.conversation_history != conversation[:-1]:
            self.mirror.reset_conversation()
            self.mirror.conversation_history = conversation[:-1]

        self.mirror.conversation_history.append({"role": "user", "content": latest_user})
        insights = self.mirror.current_insights
        response = self.mirror.talker.respond(self.mirror.conversation_history, insights)
        self.mirror.conversation_history.append({"role": "assistant", "content": response})
        return response

    def run_background_processor(self, conversation: List[Dict[str, str]]) -> None:
        """Run background reasoning to update internal insights."""
        history_snapshot = [msg.copy() for msg in conversation]
        self.mirror.process_background_thinking(history_snapshot)

    def generate_response(self, conversation: List[Dict[str, str]]) -> str:
        response = self.generate_immediate_response(conversation)
        thread_history = [msg.copy() for msg in self.mirror.conversation_history]
        t = threading.Thread(target=self.run_background_processor, args=(thread_history,), daemon=True)
        t.start()
        return response
