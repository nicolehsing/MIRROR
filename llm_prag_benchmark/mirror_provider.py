import os
import sys
from typing import List, Dict, Any, Tuple, Union, Optional

# Add parent directory to Python path so core module can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute import from the mirror package
from core.components.mirror import Mirror

class MirrorProvider:
    """
    Model provider implementation for the mirror architecture.
    This class adapts the mirror architecture to work with the multichallenge benchmark.
    """
    
    def __init__(self, api_key=None, model=None, temp=0.7, **kwargs):
        """
        Initialize the mirror provider.
        
        Args:
            api_key: API key for OpenRouter (optional, can use environment variable)
            model: Model name (not used directly, but required for API compatibility)
            temp: Temperature for generation
            **kwargs: Additional arguments
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.temperature = float(temp) if temp is not None else 0.7
        
        # Fix model ID format if it has the openrouter/ prefix
        if model and model.startswith("openrouter/"):
            model = model[len("openrouter/"):]
            print(f"Stripped 'openrouter/' prefix from model ID. Using: {model}")

        # Initialize the mirror instance, passing the model
        self.mirror = Mirror(api_key=self.api_key, model=model)
        
        # Track if background thinking has been processed
        self.is_first_turn = True
    
    def generate(self, conversation: List[Dict[str, str]]) -> str:
        """
        Generate a response for the given conversation history.
        This is an alias for generate_response to match the expected interface
        of the benchmark framework.
        
        Args:
            conversation: A list of message dictionaries with 'role' and 'content'
            
        Returns:
            The model's response
        """
        return self.generate_response(conversation)
    
    def generate_response(self, conversation: List[Dict[str, str]]) -> str:
        """
        Generate a response for the given conversation history.
        
        Args:
            conversation: A list of message dictionaries with 'role' and 'content'
            
        Returns:
            The model's response
        """
        # Get response and metadata but only return the response
        response_data = self.generate_response_with_metadata(conversation)
        if isinstance(response_data, tuple):
            return response_data[0]  # Return just the response string
        return response_data  # If not a tuple, return as is
    
    def generate_response_with_metadata(self, conversation: List[Dict[str, str]]) -> Union[str, Tuple[str, Optional[float], Optional[bool], Optional[Dict], Optional[str]]]:
        """
        Generate a response and return metadata about the thinking process.
        
        Args:
            conversation: A list of message dictionaries with 'role' and 'content'
            
        Returns:
            Tuple containing (response, salience_score, monologue_activated, 
                             raw_monologue, consolidated_narrative)
        """
        # Convert conversation to the format expected by mirror
        mirror_conversation = []
        for message in conversation:
            # Mapping benchmark format to mirror format
            mirror_message = {
                "role": message["role"],
                "content": message["content"]
            }
            mirror_conversation.append(mirror_message)
        
        # Determine task complexity for timeout adjustment
        timeout = 60.0  # Default timeout
        
        # Check if this is a new conversation
        is_new_conversation = (not mirror_conversation or 
                             self.mirror.conversation_history != mirror_conversation[:-1])
        
        # If this is a new conversation, reset mirror state
        if is_new_conversation:
            print("Starting a new conversation...")
            self.mirror.reset_conversation()
            self.is_first_turn = True
        
        # Get the latest user message
        latest_user_message = mirror_conversation[-1]["content"]
        
        # Process user input
        print(f"Generating response for turn {'1' if self.is_first_turn else 'N'}")
        print(f"DEBUG: Input to process_user_input: '{latest_user_message}'")
        
        # Store the current salience score and activation decision
        # We need to reset these first to capture the latest values
        self.mirror.last_salience_score = None
        self.mirror.last_monologue_activated = None
        
        # Generate the response
        response = self.mirror.process_user_input(latest_user_message)

        # Extract the salience score and activation decision
        salience_score = getattr(self.mirror, "last_salience_score", None)
        monologue_activated = getattr(self.mirror, "last_monologue_activated", None)
        
        # Extract the raw monologue and consolidated narrative from the previous turn's background processing
        raw_monologue = getattr(self.mirror, "last_raw_monologue", None)
        consolidated_narrative = getattr(self.mirror, "last_consolidated_narrative", None)
        
        # Check for empty or None response
        if response is None or response.strip() == "":
            print("WARNING: Received empty response from mirror system")
            fallback_response = "I apologize, but I encountered a limitation while generating a response. This may be due to reaching the maximum token limit or another technical constraint. Please try rephrasing your question or breaking it into smaller parts."
            # Return all metadata even with fallback response
            return (fallback_response, salience_score, monologue_activated, raw_monologue, consolidated_narrative)
        
        # No longer first turn after first response
        self.is_first_turn = False
        
        # Return response along with all metadata
        return (response, salience_score, monologue_activated, raw_monologue, consolidated_narrative) 