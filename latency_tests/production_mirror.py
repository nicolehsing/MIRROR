import os
import math
import threading
import time
import queue
from typing import List, Dict, Any, Optional
import sys

from ..core.clients.openrouter_client import OpenRouterClient
from ..core.components.inner_monologue_manager import MonologueManager
from ..core.components.cognitive_controller import CognitiveController
from ..core.components.talker import Talker

class ProductionMirror:
    """
    Production version of Mirror that never waits for background cognitive processes.
    This implements a true asynchronous operation where background threads can queue
    up and run in parallel without blocking user interactions.
    """

    def __init__(self, 
                 api_key: str = None,
                 model: str = "openai/gpt-4o",
                 max_background_threads: int = 3,
                 ):
        """
        Initialize the ProductionMirror system optimized for production environments.

        Args:
            api_key: OpenRouter API key (defaults to environment variable)
            model: The OpenRouter model identifier string to use (e.g., "openai/gpt-4o")
            max_background_threads: Maximum number of background cognitive threads allowed simultaneously
        """
        # Get API key from environment if not provided
        if not api_key:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable must be set")
        
        # Store the model identifier
        self.model = model
        print(f"INFO: Initializing ProductionMirror with model: {self.model}")
        sys.stdout.flush()

        # Initialize conversation history
        self.conversation_history = []
        
        # Create OpenRouter client
        self.client = OpenRouterClient(api_key)
        
        # Create components (same as standard Mirror)
        self.monologue_manager = MonologueManager(self.client, model=self.model)
        self.cognitive_controller = CognitiveController(self.client, model=self.model)
        self.talker = Talker(
            client=self.client,
            model=self.model
        )
        
        # Initialize turn counter
        self.turn_count = 0
        
        # Initialize insights queue with thread safety
        self.insights_lock = threading.Lock()
        self.current_insights = None
        
        # Track background threads
        self.max_background_threads = max_background_threads
        self.background_threads = []
        self.active_thread_count = 0
        
        # Queue for tracking completed insights in order
        self.insights_queue = queue.Queue()
        
        # Status tracking
        self.queue_status = []  # List to track thread queue status for metrics

    def reset_conversation(self):
        """
        Resets all internal state to start a fresh conversation.
        Does not wait for background threads to complete in production mode.
        """
        print("INFO: Performing production reset of conversation state...")
        sys.stdout.flush()
        
        # Acquire lock to ensure thread safety during reset
        with self.insights_lock:
            # Reset conversation state
            self.conversation_history = []
            self.turn_count = 0
            self.current_insights = None
            
            # Mark background threads as abandoned but don't wait for them
            # They'll continue running but results will be ignored
            self.queue_status = []
            
            # Clear insights queue
            while not self.insights_queue.empty():
                try:
                    self.insights_queue.get_nowait()
                except queue.Empty:
                    break
        
        print("INFO: Conversation reset completed")
        sys.stdout.flush()

    def truncate_conversation_history(self, conversation: List[Dict[str, str]], max_tokens: int = 20000) -> List[Dict[str, str]]:
        """
        Truncate conversation history to fit within token limits.
        Same implementation as standard Mirror.
        """
        if not conversation:
            return conversation
        
        # Estimate token count (rough approximation: 4 chars ≈ 1 token)
        def estimate_tokens(messages: List[Dict[str, str]]) -> int:
            return sum(len(msg.get("content", "")) // 4 + 20 for msg in messages)
    
        # If conversation is already within limits, return as is
        estimated_tokens = estimate_tokens(conversation)
        if estimated_tokens <= max_tokens:
            return conversation
    
        # Extract system messages (always preserve these)
        system_messages = [msg for msg in conversation if msg.get("role") == "system"]
    
        # Find and preserve the first user message (contains document context)
        first_user_idx = next((i for i, msg in enumerate(conversation) 
                            if msg.get("role") == "user"), None)
        first_user_message = [conversation[first_user_idx]] if first_user_idx is not None else []
    
        # Start with essential messages (system + first user message)
        essential_messages = system_messages + first_user_message
    
        # Try keeping essential messages + last N messages
        for n_recent in [10, 6, 4, 2]:
            # Skip the first user message if it's within the recent messages
            if first_user_idx is not None and first_user_idx >= len(conversation) - n_recent:
                recent_messages = conversation[-n_recent:]
            else:
                # Otherwise, take recent messages but avoid duplicating the first user message
                recent_messages = [msg for idx, msg in enumerate(conversation[-n_recent:]) 
                              if not (msg.get("role") == "user" and 
                                        first_user_idx is not None and 
                                        idx + len(conversation) - n_recent == first_user_idx)]
        
            truncated = essential_messages + recent_messages
        
            if estimate_tokens(truncated) <= max_tokens:
                return truncated
    
        # Absolute last resort
        return system_messages + first_user_message

    def process_user_input(self, user_input: str) -> str:
        """
        Process user input through the ProductionMirror architecture.
        Never waits for background threads in production mode.

        Args:
            user_input: The user's message

        Returns:
            The AI's response
        """
        # Increment turn counter
        self.turn_count += 1
        print(f"Production mode: Processing turn {self.turn_count}")
        sys.stdout.flush()

        # Add user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})

        # Try to use any available insights from previous turn's thinking
        # But NEVER wait for background threads to complete
        current_turn_insights = None
        with self.insights_lock:
            # Check if we have new insights from completed background threads
            while not self.insights_queue.empty():
                try:
                    # Get the latest completed insight
                    turn_num, insights = self.insights_queue.get_nowait()
                    # Update current insights
                    self.current_insights = insights
                    print(f"Applied completed insights from turn {turn_num}")
                except queue.Empty:
                    break
            
            # Use whatever insights we have now (might be None)
            current_turn_insights = self.current_insights

        # Generate response with talker
        try:
            response = self.talker.respond(
                self.conversation_history, 
                current_turn_insights
            )
        except Exception as e:
            print(f"Error during response generation: {e}")
            sys.stdout.flush()
            response = f"I apologize, but I encountered an error: {str(e)}"

        # Add assistant response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Start background thinking thread but don't wait for it
        self._start_background_thinking(self.conversation_history, self.turn_count)
        
        return response
    
    def _start_background_thinking(self, history_snapshot: List[Dict[str, str]], turn_num: int):
        """
        Start a new background thinking thread.
        Manages thread pool to avoid creating too many concurrent threads.
        """
        # Clean up completed threads first
        self._cleanup_completed_threads()
        
        # Track queue status for metrics
        with self.insights_lock:
            active_count = sum(1 for t in self.background_threads if t.is_alive())
            self.active_thread_count = active_count
            self.queue_status.append(active_count)
        
        # Make a deep copy of history to avoid concurrent modification
        import copy
        history_copy = copy.deepcopy(history_snapshot)
        
        # Create and start a new background thread
        thread = threading.Thread(
            target=self._process_background_thinking,
            args=(history_copy, turn_num),
            daemon=True
        )
        
        self.background_threads.append(thread)
        thread.start()
        
        print(f"Started background thinking for turn {turn_num} (active threads: {self.active_thread_count})")
        sys.stdout.flush()
    
    def _cleanup_completed_threads(self):
        """Remove completed threads from the tracking list."""
        self.background_threads = [t for t in self.background_threads if t.is_alive()]
    
    def _process_background_thinking(self, conversation_history_snapshot: List[Dict[str, str]], turn_num: int):
        """
        Process background thinking and store results in the insights queue.
        """
        import time
        
        start_time = time.time()
        print(f"Background thinking for turn {turn_num} started")
        sys.stdout.flush()
        
        try:
            # Truncate conversation history
            truncated_conversation = self.truncate_conversation_history(conversation_history_snapshot)
            
            # Process through monologue manager
            monologue_result = self.monologue_manager.process_user_input(truncated_conversation)
            thread_results = monologue_result.get("results", [])
            
            # Consolidate results with cognitive controller
            try:
                consolidation_result = self.cognitive_controller.consolidate(thread_results)
                
                # Put result in the insights queue for future turns
                self.insights_queue.put((turn_num, consolidation_result))
                
                print(f"Background thinking for turn {turn_num} completed successfully")
            except Exception as e:
                # If consolidation fails, fall back to raw thread results
                fallback_insights = "\n".join([f"{t.get('name', 'Thread')}: {t.get('output', 'No output')}" 
                                          for t in thread_results])
                
                self.insights_queue.put((turn_num, f"Consolidation Error: {str(e)}\n{fallback_insights}"))
                print(f"Background thinking for turn {turn_num} completed with consolidation error: {e}")
        except Exception as e:
            print(f"Error in background thinking for turn {turn_num}: {e}")
            # Don't put failed insights in the queue
        
        total_time = time.time() - start_time
        print(f"Background thinking for turn {turn_num} took {total_time:.2f}s")
        sys.stdout.flush()
    
    def get_queue_metrics(self) -> Dict[str, Any]:
        """Return metrics about the background thread queue."""
        with self.insights_lock:
            queue_lengths = self.queue_status
            
            # Calculate metrics
            max_queue = max(queue_lengths) if queue_lengths else 0
            avg_queue = sum(queue_lengths) / len(queue_lengths) if queue_lengths else 0
            current_queue = self.active_thread_count
            
            return {
                "max_queue_length": max_queue,
                "avg_queue_length": avg_queue,
                "current_queue_length": current_queue,
                "queue_history": queue_lengths
            }

    def run_interactive(self):
        """Run the ProductionMirror system in interactive mode."""
        print("ProductionMirror system initialized! Type 'exit' to quit.")
        print("Note: This system runs in production mode with non-blocking background cognitive processes.")
        print("Responses are immediate, with background thinking happening asynchronously.")
        print("-" * 80)
        sys.stdout.flush()
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                sys.stdout.flush()
                break
            
            # Get AI response without waiting for any background processing
            response = self.process_user_input(user_input)
            
            # Show the response immediately
            print("\nAI:", response)
            
            # Also show the current queue state
            metrics = self.get_queue_metrics()
            print(f"[Background threads: {metrics['current_queue_length']} active]")
            
            print("-" * 80) 
            sys.stdout.flush() 