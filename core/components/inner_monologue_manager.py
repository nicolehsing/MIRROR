import json
import time
import math
from typing import List, Dict, Any
from collections import deque # Added for storing recent embeddings

# Removed sentence-transformer imports

from core.clients.openrouter_client import OpenRouterClient
from core.components.inner_monologue_thread import InnerMonologue

# Removed sentence-transformer logging suppression

class MonologueManager:
    def __init__(self,
                client: OpenRouterClient,
                model: str = "openai/gpt-4o",
                ):
        """
        Initialize the manager for the inner monologue implementation.
        Manages the salience gate for activating background processing based 
        primarily on logprob signals and conversational dynamics.

        Args:
            client: OpenRouter client for API calls
            model: Model identifier to use for monologue processing
        """
        self.client = client
        self.model = model
        
        # Initialize the monologue processor
        self.monologue_processor = InnerMonologue(client, model)
        print(f"INFO: Initialized inner monologue processor")
        
        # --- Salience Gate Initialization ---
        # Track conversation history for lexical novelty detection
        self.recent_user_tokens = set()  
        self.last_reflection_turn = 0  
        
    
    def process_user_input(self, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Process user input through the unified cognitive framework.
        (No changes needed here, it just calls the processor)
        
        Args:
            conversation_history: The current conversation history
            
        Returns:
            Dictionary with results containing belief, memory and goal aspects
        """
        start_time = time.time()
        print(f"INFO: Processing cognitive analysis...")
        
        # Process the input through the monologue processor
        result = self.monologue_processor.think(conversation_history)
        
        # Format results for consumption by higher-level components
        formatted_results = [
            {"name": "Reasoning Thread", "output": result.get("reasoning", "No reasoning output")},
            {"name": "Memory Thread", "output": result.get("memory", "No memory output")},
            {"name": "Goal Thread", "output": result.get("goal", "No goal output")}
        ]
        

        processing_time = time.time() - start_time
        print(f"INFO: Unified cognitive analysis completed in {processing_time:.2f}s")
        
        return {
            "results": formatted_results,
            "raw_response": result,
            "processing_time": processing_time
        } 

    def get_token_history_stats(self) -> Dict[str, Any]:
        """
        Return statistics about the current token history for debugging purposes.
        (No changes needed here)
        
        Returns:
            Dictionary containing token history statistics
        """
        total_tokens = len(self.recent_user_tokens)
        
        # Get most common meaningful tokens (length > 3)
        meaningful_tokens = [t for t in self.recent_user_tokens if len(t) > 3]
        
        # Get a sample of tokens for inspection
        sample_size = min(20, total_tokens)
        token_sample = list(self.recent_user_tokens)[:sample_size]
        
        return {
            "total_tokens": total_tokens,
            "meaningful_tokens": len(meaningful_tokens),
            "token_sample": token_sample,
            # Renamed for clarity
            "turn_index_of_last_reflection": self.last_reflection_turn 
        } 