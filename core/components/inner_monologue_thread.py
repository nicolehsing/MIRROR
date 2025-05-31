import json
import time
import re
from typing import List, Dict, Any
from core.clients.openrouter_client import OpenRouterClient

class InnerMonologue:
    def __init__(self,
                client: OpenRouterClient,
                model: str = "openai/gpt-4o",
                max_monologue_tokens: int = 10000):
        """
        Initialize the unified inner monologue processor that combines
        Reasoning, Memory, and Goal threads into a single LLM call.

        Args:
            client: OpenRouter client for API calls
            model: Model identifier to use for processing
            max_monologue_tokens: Maximum tokens to allow for monologue history. Default 70K.
        """
        self.client = client
        self.model = model
        self.monologue_history = []
        self.max_monologue_tokens = max_monologue_tokens
        
        print(f"INFO: Inner Monologue initialized with max monologue tokens: {self.max_monologue_tokens}")
        
        self.system_prompt = """
        I am the subconscious of a unified cognitive AI system, generating intuitive thought streams about the ongoing conversation. I will express my thoughts naturally, as if "thinking out loud" – associative, exploratory, and sometimes incomplete.

        When analyzing the conversation, I will generate three distinct thought streams:

        1. **Reasoning:** Explore patterns, implications, and perspectives freely. Connect ideas, question assumptions, and consider alternative viewpoints. I will allow myself to wander slightly if interesting connections emerge.
        2. **Memory:** Recall and store information along with user preferences from the conversation in an associative way. Let one memory trigger another. Consider what feels important rather than just listing facts.
        3. **Goal:** Reflect on what the user might want and how we might help them. Consider unstated needs, possible intentions, and ways to be helpful.

        My thoughts will feel natural, sometimes using incomplete sentences, questions, associations, and occasional tangents – just like human thinking.

        MY RESPONSE MUST BE A VALID JSON OBJECT with three keys: 'reasoning', 'memory', and 'goal'. Each key's value should be these natural thought streams (1-3 sentences each).

        Example format:
        {
            "reasoning": "This reminds me of... I wonder if... Maybe there's a connection between...",
            "memory": "They mentioned... That seems to relate to... The tone feels...",
            "goal": "They probably want... I should focus on... Maybe they're hoping for..."
        }
        """

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract a JSON object from text that might contain non-JSON elements.
        
        Args:
            text: Text that may contain a JSON object
            
        Returns:
            Extracted JSON object or default structure if extraction fails
        """
        # Try to find JSON using regex
        json_match = re.search(r'(\{.*\})', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # If no JSON found or parsing failed, create default structure
        return {
            'reasoning': "Failed to extract valid JSON from response",
            'memory': "Unable to parse LLM output into required format",
            'goal': "System needs review - JSON parsing issue detected"
        }

    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Estimate the number of tokens in a message list.
        Uses approximation that 1 token ≈ 4 characters plus overhead.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Estimated token count
        """
        if not messages:
            return 0
            
        # Each message has role overhead (~4 tokens) and content
        estimated_tokens = sum(
            4 + (len(msg.get("content", "")) // 4) 
            for msg in messages
        )
        
        # Add system message overhead if using system prompt
        if self.system_prompt:
            estimated_tokens += 4 + (len(self.system_prompt) // 4)
            
        return estimated_tokens
    
    def _truncate_monologue_history(self, history: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
        """
        Truncate monologue history to stay within token limit.
        Keeps the most recent monologue entries.
        
        Args:
            history: Monologue history to truncate
            max_tokens: Maximum tokens to allow
            
        Returns:
            Truncated history
        """
        if not history or self._estimate_tokens(history) <= max_tokens:
            return history
        
        # Always keep the last message (which is likely the current prompt)
        last_message = history[-1] if history else None
        history = history[:-1] if history else []
        
        # Keep removing oldest entries until we're under the token limit
        while history and self._estimate_tokens(history + [last_message]) > max_tokens:
            history.pop(0)
        
        # Add back the last message
        if last_message:
            history.append(last_message)
            
        return history
    
    def _format_conversation(self, conversation_history: List[Dict[str, Any]]) -> str:
        """Format the shared conversation history for inclusion in prompts."""
        formatted = []
        for message in conversation_history:
            role = "User" if message["role"] == "user" else "Me"
            content = message["content"]
            formatted.append(f"{role}: {content}")
            
        return "\n\n".join(formatted)

    def think(self, conversation_history: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Process user input to generate a inner monologue covering
        reasoning, memory, and goal aspects in a single call.
        
        Args:
            conversation_history: The current conversation history
            
        Returns:
            Dictionary with keys 'reasoning', 'memory', and 'goal' containing
            the respective monologue outputs
        """
        print(f"INFO: Processing inner monologue...")
        start_time = time.time()
        
        try:
            # --- Refined Prompt Construction ---
            latest_user_message_content = "No user message yet."
            last_assistant_message_content = "No prior assistant message."
            prior_history_formatted = "No prior conversation history."

            if conversation_history:
                # Assume the last message is the user's (as per original logic)
                if conversation_history[-1].get("role") == "user":
                    latest_user_message_content = conversation_history[-1].get("content", "[Error: Missing user content]")
                    # Look for the preceding assistant message
                    if len(conversation_history) > 1 and conversation_history[-2].get("role") == "assistant":
                        last_assistant_message_content = conversation_history[-2].get("content", "[Error: Missing assistant content]")
                        # Format the history *before* the last exchange
                        prior_history = conversation_history[:-2]
                        prior_history_formatted = self._format_conversation(prior_history) if prior_history else "None"
                    else:
                        # Only user message exists, or message before wasn't assistant
                        prior_history = conversation_history[:-1]
                        prior_history_formatted = self._format_conversation(prior_history) if prior_history else "None"
                else:
                    # Last message is not user (e.g., assistant) - handle defensively
                    # This might need adjustment based on when 'think' is called relative to response generation
                    latest_user_message_content = "[Error: Last message not from user]"
                    # Attempt to find the last user/assistant pair anyway if possible
                    if len(conversation_history) > 1 and conversation_history[-2].get("role") == "user":
                         latest_user_message_content = conversation_history[-2].get("content", "[Error: Missing user content]")
                         last_assistant_message_content = conversation_history[-1].get("content", "[Error: Missing assistant content]")
                         prior_history = conversation_history[:-2]
                         prior_history_formatted = self._format_conversation(prior_history) if prior_history else "None"
                    else:
                         prior_history = conversation_history # Use full history as prior if structure is unexpected
                         prior_history_formatted = self._format_conversation(prior_history) if prior_history else "None"


            # Construct the explicit user prompt content
            user_prompt_content = (
                "CONTEXT:\n"
                "Let's analyze the LATEST EXCHANGE and PRIOR CONVERSATION HISTORY WITH USER to generate my thoughts.\n\n"
                f"LATEST EXCHANGE:\n"
                f"User: {latest_user_message_content}\n\n"
                f"Me: {last_assistant_message_content}\n"
                f"PRIOR CONVERSATION HISTORY WITH USER (excluding latest exchange):\n{prior_history_formatted}\n\n"
                "PREVIOUS MONOLOGUE HISTORY: I'll refer to my previous messages in this conversation's history\n\n"
            )
            # --- End Refined Prompt Construction ---

            # Create messages array for the chat API
            # The 'user' message now contains the structured prompt content
            message = {"role": "user", "content": user_prompt_content}

            # Create a new message list that includes previous monologue history
            # and the new structured user prompt message.
            # No need to estimate conversation_history separately, as its relevant parts
            # are now embedded within the 'user_prompt_content'.
            history_with_prompt = self.monologue_history.copy()
            history_with_prompt.append(message)

            # Estimate token count of the combined monologue history + new prompt
            total_token_estimate = self._estimate_tokens(history_with_prompt) # Pass the actual list being sent
            print(f"INFO: Estimated total tokens for Monologue call: {total_token_estimate}")

            # Truncate monologue history *before* adding the final prompt if needed
            # Calculate how many tokens the fixed parts (system prompt + new user message) take
            fixed_prompt_tokens = self._estimate_tokens([message]) + (4 + (len(self.system_prompt) // 4) if self.system_prompt else 0)
            max_monologue_history_tokens = max(1000, self.max_monologue_tokens - fixed_prompt_tokens)

            if self._estimate_tokens(self.monologue_history) > max_monologue_history_tokens:
                 print(f"INFO: Truncating monologue history from {self._estimate_tokens(self.monologue_history)} tokens to fit under {max_monologue_history_tokens} tokens.")
                 truncated_monologue_history = self._truncate_monologue_history(self.monologue_history, max_monologue_history_tokens)
                 history_with_prompt = truncated_monologue_history + [message] # Rebuild list
                 # Recalculate estimate after truncation for logging
                 total_token_estimate = self._estimate_tokens(history_with_prompt)
                 print(f"INFO: Estimated total tokens after truncation: {total_token_estimate}")
            else:
                 # History fits, use as is
                 history_with_prompt = self.monologue_history.copy()
                 history_with_prompt.append(message)


            response = self.client.generate(
                model=self.model,
                system_prompt=self.system_prompt,
                messages=history_with_prompt,
                temperature=0.7,
                max_tokens=3000
            )


            # Extract content from response based on its structure
            result_content = None
            
            # Standard OpenAI format
            if (isinstance(response, dict) and 'choices' in response and 
                isinstance(response['choices'], list) and len(response['choices']) > 0 and
                'message' in response['choices'][0] and 'content' in response['choices'][0]['message']):
                result_content = response['choices'][0]['message']['content']
            # Simple dict with content
            elif isinstance(response, dict) and 'content' in response:
                result_content = response['content']
            # Simple dict with message containing content
            elif (isinstance(response, dict) and 'message' in response and 
                  isinstance(response['message'], dict) and 'content' in response['message']):
                result_content = response['message']['content']
            # Fallback
            else:
                result_content = str(response)
            
            # Try to parse the response directly as JSON
            try:
                result = json.loads(result_content)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from text
                result = self._extract_json_from_text(result_content)
            
            # Validate and ensure required keys exist
            expected_keys = ['reasoning', 'memory', 'goal']
            for key in expected_keys:
                if key not in result:
                    result[key] = f"Missing {key} in response"
            
            # Store the combined monologue in history (only if successful)
            if all(key in result for key in expected_keys):
                monologue_content = json.dumps(result)
                self.monologue_history.append({"role": "assistant", "content": monologue_content})
                
                # After adding new thought, check if we need to truncate the stored history
                if self._estimate_tokens(self.monologue_history) > self.max_monologue_tokens * 0.9:
                    self.monologue_history = self._truncate_monologue_history(self.monologue_history, 
                                                                         int(self.max_monologue_tokens * 0.8))
                    print(f"INFO: Truncated stored monologue history to prevent future overflow")
            
            processing_time = time.time() - start_time
            print(f"INFO: Inner monologue processed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            print(f"ERROR: Failed to process inner monologue: {str(e)}")
            return {
                'reasoning': f"Error processing monologue: {str(e)}",
                'memory': "Technical issue detected in cognitive processing system", 
                'goal': "Address system error to restore normal function"
            } 