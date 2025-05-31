import os
import math
import threading
import time
from typing import List, Dict, Any
import sys

from core.clients.openrouter_client import OpenRouterClient
from core.components.inner_monologue_manager import MonologueManager
from core.components.cognitive_controller import CognitiveController
from core.components.talker import Talker

class Mirror:

    def __init__(self, 
                 api_key: str = None,
                 model: str = "openai/gpt-4o" # Add model parameter with default
                 ):
        """
        Initialize the Mirror system with all necessary components.

        Args:
            api_key: OpenRouter API key (defaults to environment variable)
            model: The OpenRouter model identifier string to use (e.g., "openai/gpt-4o")
        """
        # Get API key from environment if not provided
        if not api_key:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable must be set")
        
        # Store the model identifier and reflection flag
        self.model = model
        print(f"INFO: Initializing Mirror with model: {self.model}")
        sys.stdout.flush()

        # Initialize conversation history
        self.conversation_history = []
        
        # Create OpenRouter client
        self.client = OpenRouterClient(api_key)
        
        # Create monologue manager, passing the model and flag
        self.monologue_manager = MonologueManager(self.client, model=self.model)
        
        # Create cognitive controller, passing the model
        self.cognitive_controller = CognitiveController(self.client, model=self.model)
        
        # Create talker, passing the model
        self.talker = Talker(
            client=self.client,
            model=self.model
        )
        
        # Initialize turn counter
        self.turn_count = 0
        
        # Initialize insights
        self.current_insights = None
        
        
        # Add tracking for salience score and monologue activation
        self.last_salience_score = None
        self.last_monologue_activated = None
        self.last_raw_monologue = None  # Store the raw output from InnerMonologue
        self.last_consolidated_narrative = None # Store the output from CognitiveController
        
        # Add thread synchronization mechanisms
        self.insights_lock = threading.Lock()
        self.background_thread = None
        self.background_thread_active = False
        self.background_thread_completed = threading.Event()
        self.background_thread_completed.set()  # Initially set as completed

    def reset_conversation(self):
        """
        Resets all internal state to start a fresh conversation.
        Ensures complete cleanup of all components and waits for background processing to complete.
        """
        print("INFO: Performing thorough reset of Mirror conversation state...")
        sys.stdout.flush()
        
        # 1. Handle any active background threads
        if self.background_thread and self.background_thread.is_alive():
            print("  - Waiting for active background thread to complete")
            sys.stdout.flush()
            # Wait with timeout to avoid blocking indefinitely
            thread_completed = self.background_thread_completed.wait(timeout=100.0)
            if not thread_completed:
                print("  - WARNING: Background thread did not complete within timeout, forcing termination")
                sys.stdout.flush()
                # We can't directly terminate a thread in Python, but we can mark it as inactive
                self.background_thread_active = False
        
        # 2. Acquire lock to ensure thread safety during reset
        with self.insights_lock:
            # 3. Reset conversation state
            print("  - Resetting conversation history and turn counter")
            sys.stdout.flush()
            self.conversation_history = []
            self.turn_count = 0
            
            # 4. Reset all insight and narrative state
            print("  - Clearing all insights and narrative state")
            sys.stdout.flush()
            self.current_insights = None
            self.last_salience_score = None
            self.last_monologue_activated = None
            self.last_raw_monologue = None
            self.last_consolidated_narrative = None
            
            # 5. Reset Monologue Manager (deep reset)
            print("  - Performing deep reset of MonologueManager")
            sys.stdout.flush()
            if hasattr(self.monologue_manager, 'monologue_processor'):
                # Reset history array
                self.monologue_manager.monologue_processor.monologue_history = []
                
                # Reset any processor state
                if hasattr(self.monologue_manager.monologue_processor, 'last_result'):
                    self.monologue_manager.monologue_processor.last_result = None
                
                # Reset any salience or activation tracking
                if hasattr(self.monologue_manager, 'last_salience_score'):
                    self.monologue_manager.last_salience_score = None
                
                # Reset any reflection flags
                if hasattr(self.monologue_manager, 'last_reflection_active'):
                    self.monologue_manager.last_reflection_active = False
            
            # 6. Reset Cognitive Controller (deep reset)
            print("  - Performing deep reset of CognitiveController")
            sys.stdout.flush()
            if hasattr(self.cognitive_controller, 'insight_memory_block'):
                self.cognitive_controller.insight_memory_block = ""
            
            # Reset any narrative state or memory
            if hasattr(self.cognitive_controller, 'consolidated_understanding'):
                self.cognitive_controller.consolidated_understanding = None
                
            # Reset any tracking of past insights
            if hasattr(self.cognitive_controller, 'insight_history'):
                self.cognitive_controller.insight_history = []
                
            # 7. Reset Talker
            print("  - Resetting Talker state")
            sys.stdout.flush()
            if hasattr(self.talker, 'last_response'):
                self.talker.last_response = None
            
            # 8. Clean up threading resources
            print("  - Resetting thread state")
            sys.stdout.flush()
            self.background_thread = None
            self.background_thread_active = False
            self.background_thread_completed.set()  # Mark as completed
        
        # 9. Validate reset worked correctly
        if (len(self.conversation_history) > 0 or 
            self.turn_count != 0 or 
            self.current_insights is not None or
            self.last_consolidated_narrative is not None):
            print("WARNING: Reset validation failed - some state was not properly cleared!")
            sys.stdout.flush()
        else:
            print("INFO: Reset completed successfully - all state has been cleared for a new conversation")
            sys.stdout.flush()

    def truncate_conversation_history(self, conversation: List[Dict[str, str]], max_tokens: int = 20000) -> List[Dict[str, str]]:

        if not conversation:
            return conversation
        
        # Estimate token count (rough approximation: 4 chars ≈ 1 token)
        def estimate_tokens(messages: List[Dict[str, str]]) -> int:
            return sum(len(msg.get("content", "")) // 4 + 20 for msg in messages)
    
        # If conversation is already within limits, return as is
        estimated_tokens = estimate_tokens(conversation)
        if estimated_tokens <= max_tokens:
            print(f"Conversation history within limits ({estimated_tokens} tokens)")
            sys.stdout.flush()
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
                print(f"Truncated to essential messages + last {n_recent} messages ({estimate_tokens(truncated)} tokens)")
                sys.stdout.flush()
                return truncated
    
        # Absolute last resort
        return system_messages + first_user_message

    def process_user_input(self, user_input: str) -> str:
        """
        Process user input through the  Mirror architecture.

        Args:
            user_input: The user's message

        Returns:
            The AI's response
        """
        # Wait for any existing background thread to complete
        if self.background_thread and self.background_thread.is_alive():
            print(f"Waiting for previous background thinking thread to complete...")
            sys.stdout.flush()
            # Wait with a timeout to avoid blocking indefinitely
            self.background_thread_completed.wait(timeout=100.0)
            
        # Increment turn counter
        self.turn_count += 1
        print(f"Processing turn {self.turn_count}")
        sys.stdout.flush()

        # Add user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})

        if self.turn_count > 1:
            # Use insights from previous turn's thinking
            print(f"Turn {self.turn_count}: Applying insights from previous turn's thinking")
            sys.stdout.flush()
            with self.insights_lock:
                current_turn_insights = self.current_insights
        else:
            print("Turn 1: Generating immediate response (background thinking will start after)")
            sys.stdout.flush()
            current_turn_insights = None

        # Generate final response with talker (using full conversation history)
        try:
            # Try to get both response and probabilities
            result = self.talker.respond(
                self.conversation_history, 
                current_turn_insights
            )
            
            response = result
        except Exception as e:
            print(f"Error during response generation: {e}")
            sys.stdout.flush()
            response = f"I apologize, but I encountered an error: {str(e)}"

        # Add assistant response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        
        # Store salience score and activation decision for retrieval by benchmark
        if hasattr(self.monologue_manager, 'last_salience_score'):
            self.last_salience_score = self.monologue_manager.last_salience_score
        else:
            self.last_salience_score = None
            
        
        # Start background thinking in a background thread
        self.background_thread_completed.clear()
        self.background_thread_active = True
            
        # Create and start a new background thread, passing the relevant history snapshot
        self.background_thread = threading.Thread(
            target=self._background_thinking_wrapper,
            args=(self.conversation_history,), # Pass history as argument
            daemon=True
        )
        self.background_thread.start()
        print("Started background thinking thread")

        sys.stdout.flush()

        return response
    
    def _background_thinking_wrapper(self, history_snapshot: List[Dict[str, str]]):
        """Wrapper function for background thinking that ensures proper cleanup."""
        try:
            # Pass the history snapshot to the actual processing function
            self.process_background_thinking(history_snapshot)
        finally:
            self.background_thread_active = False
            self.background_thread_completed.set()
            print(" background thinking completed (from wrapper)")
            sys.stdout.flush()
    
    def process_background_thinking(self, conversation_history_snapshot: List[Dict[str, str]], timeout=120.0):
        """
        Process background thinking for the next turn using  architecture.
        
        Args:
            conversation_history_snapshot: The conversation history state.
            timeout: Maximum time to wait for processing (in seconds)
        """
        import time
        import threading
        import queue
        
        start_time = time.time()
        print(f"Starting  background thinking (timeout: {timeout}s)")
        sys.stdout.flush()
        
        try:
            # Truncate the specific conversation history snapshot passed to this thread
            truncated_conversation = self.truncate_conversation_history(conversation_history_snapshot)
            
            # Process through  monologue manager using the correct truncated history state
            # This is more efficient as it uses a single LLM call for all cognitive aspects
            monologue_result = self.monologue_manager.process_user_input(truncated_conversation)
            thread_results = monologue_result.get("results", [])
            
            # Store the raw monologue result
            with self.insights_lock: # Use lock for consistency, though likely safe
                self.last_raw_monologue = monologue_result 
            
            # Check timeout before cognitive controller consolidation
            elapsed = time.time() - start_time
            remaining_time = max(90.0, timeout - elapsed)  
            print(f" monologue processing took {elapsed:.2f}s, allocating {remaining_time:.2f}s for consolidation")
            sys.stdout.flush()
            
            if elapsed >= timeout:
                sys.stdout.flush()
                with self.insights_lock:
                    self.current_insights = "Background thinking timed out before consolidation."
                return
            
            # Consolidate insights with cognitive controller using a timeout mechanism
            result_queue = queue.Queue()
            
            def run_consolidation():
                try:
                    print("Starting cognitive controller consolidation...")
                    sys.stdout.flush()
                    consolidation_result = self.cognitive_controller.consolidate(thread_results)
                    print(f"Consolidation complete, result length: {len(consolidation_result) if consolidation_result else 0}")
                    sys.stdout.flush()
                    result_queue.put(("success", consolidation_result))
                except Exception as e:
                    print(f"Consolidation thread encountered error: {e}")
                    sys.stdout.flush()
                    result_queue.put(("error", str(e)))
            
            # Start consolidation in a separate thread
            consolidation_thread = threading.Thread(target=run_consolidation)
            consolidation_thread.daemon = True
            consolidation_thread.start()
            print(f"Consolidation thread started, waiting up to {remaining_time:.2f}s for completion")
            sys.stdout.flush()
            
            # Wait for the consolidation to complete or timeout
            try:
                status, result = result_queue.get(timeout=remaining_time)
                print(f"Got result from consolidation thread: status={status}")
                sys.stdout.flush()
                
                if status == "success":
                    with self.insights_lock:
                        self.current_insights = result
                        self.last_consolidated_narrative = result # Store successful consolidation
                else:  # Error case
                    print(f"ERROR in cognitive controller consolidation: {result}")
                    sys.stdout.flush()
                    fallback_insights = "\n".join([f"{t.get('name', 'Thread')}: {t.get('output', 'No output')}" 
                                              for t in thread_results])
                    with self.insights_lock:
                        self.current_insights = f"Consolidation Error: {result}\n{fallback_insights}"
                        self.last_consolidated_narrative = f"ERROR: {result}" # Store error state
            except queue.Empty:
                # Timeout waiting for consolidation
                print(f"WARNING: Cognitive controller consolidation timed out after {remaining_time:.2f}s")
                sys.stdout.flush()
                
                # Determine fallback insights and store timeout state
                fallback_reason = "Consolidation Timeout"
                if not result_queue.empty():
                    try:
                        status, result = result_queue.get_nowait()
                        if status == "success" and result:
                            print("Retrieved partial consolidation result")
                            sys.stdout.flush()
                            fallback_insights = result
                            fallback_reason = "Partial Consolidation on Timeout"
                        else:
                            raise Exception("No valid partial result")
                    except Exception:
                        fallback_insights = "\n".join([f"{t.get('name', 'Thread')}: {t.get('output', 'No output')}" 
                                                for t in thread_results])
                else:
                    fallback_insights = "\n".join([f"{t.get('name', 'Thread')}: {t.get('output', 'No output')}" 
                                              for t in thread_results])
                
                with self.insights_lock:
                    self.current_insights = f"{fallback_reason}: Using raw/partial outputs\n{fallback_insights}"
                    self.last_consolidated_narrative = f"TIMEOUT: {fallback_reason}" # Store timeout state
                
        except Exception as e:
            print(f"ERROR in background thinking: {e}")
            sys.stdout.flush()
            with self.insights_lock:
                self.current_insights = f"Background thinking error: {str(e)}"
                self.last_raw_monologue = {"error": f"Background thinking error: {str(e)}"} # Store error
                self.last_consolidated_narrative = f"ERROR: Background thinking error: {str(e)}" # Store error
            
        total_time = time.time() - start_time
        print(f" background thinking processing time: {total_time:.2f}s")
        sys.stdout.flush()

    def run_interactive(self):
        """Run the  Mirror system in interactive mode."""
        import sys
        print(" Mirror system initialized! Type 'exit' to quit.")
        print("Note: This system uses the  architecture with all cognitive aspects processed in a single call.")
        print("The first response will be immediate, while subsequent responses will benefit from background processing.")
        print("-" * 80)
        sys.stdout.flush()
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                sys.stdout.flush()
                break
            
            turn_number = self.turn_count + 1
            
            if turn_number == 1:
                print("Turn 1: Generating immediate response (background thinking will start after)")
                sys.stdout.flush()
            else:
                print(f"Turn {turn_number}: Applying insights from previous turn's thinking")
                sys.stdout.flush()
                
            # Get AI response first
            response = self.process_user_input(user_input)
            
            # Show the response immediately
            print("\nAI:", response)
            sys.stdout.flush()
            
            print("-" * 80) 
            sys.stdout.flush() 