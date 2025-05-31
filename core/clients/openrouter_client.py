import os
import requests
import json
import time
from typing import List, Dict, Any, Optional
import traceback

class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")

        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate(self,
                 model: str,
                 system_prompt: Optional[str] = None,
                 messages: List[Dict[str, str]] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 1024,
                 logprobs: bool = False,
                 top_logprobs: Optional[int] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        Generate a response using the specified model through OpenRouter.

        Args:
            model: Model identifier (e.g., "openai/gpt-4o")
            system_prompt: System prompt for the model (optional if included in messages)
            messages: List of message objects with role and content
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            logprobs: Whether to return log probabilities for generated tokens
            top_logprobs: Number of most likely tokens to return log probabilities for
            **kwargs: Additional parameters to pass to the API

        Returns:
            Response from the API standardized to match OpenAI format
        """
        url = f"{self.base_url}/chat/completions"

        # Prepare messages list
        _messages = []
        
        # Add system prompt as first message if provided
        if system_prompt:
            _messages.append({"role": "system", "content": system_prompt})
        
        # Add the rest of the messages
        if messages:
            # Filter out None values
            valid_messages = []
            for msg in messages:
                if not isinstance(msg, dict) or 'role' not in msg:
                    # print(f"DEBUG_ORC: Skipping malformed message (not dict or no role): {msg}")
                    continue # Skip malformed messages

                role = msg.get('role')
                
                # Common fields, content can be None for assistant tool_calls
                content = msg.get('content') 
                
                # Assistant specific
                tool_calls = msg.get('tool_calls')
                
                # Tool specific
                tool_call_id = msg.get('tool_call_id')
                # tool_name = msg.get('name') # 'name' is also part of a valid tool message

                if role == "system" or role == "user":
                    if content is not None: # System and User messages must have content
                        valid_messages.append(msg)
                    # else:
                        # print(f"DEBUG_ORC: Skipping {role} message with None content: {msg}")
                elif role == "assistant":
                    # Assistant messages can have:
                    # 1. Content only (text response)
                    # 2. Tool_calls only (content can be None or empty string)
                    # 3. Content and Tool_calls (though less common for pure tool use)
                    if tool_calls is not None: # If there are tool_calls, message is valid
                        valid_messages.append(msg)
                    elif content is not None: # If no tool_calls, content must be present
                        valid_messages.append(msg)
                    # else:
                        # print(f"DEBUG_ORC: Skipping assistant message with no tool_calls and None content: {msg}")
                elif role == "tool":
                    # Tool messages MUST have tool_call_id, name, and content
                    if tool_call_id and msg.get('name') and content is not None:
                        valid_messages.append(msg)
                    # else:
                        # print(f"DEBUG_ORC: Skipping incomplete tool message: {msg}")
                # else:
                    # print(f"DEBUG_ORC: Skipping message with unknown role: {msg}")
            
            _messages.extend(valid_messages)
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": _messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Add tools and tool_choice if present in kwargs
        if "tools" in kwargs:
            payload["tools"] = kwargs.pop("tools")
        if "tool_choice" in kwargs: # tool_choice is optional but good to support
            payload["tool_choice"] = kwargs.pop("tool_choice")

        # Add any additional parameters
        for key, value in kwargs.items():
            payload[key] = value

        try:
            print(f"Sending request to OpenRouter for model: {model}")
            response = requests.post(url, headers=self.headers, json=payload)
            # It's good practice to check HTTP status code here BEFORE trying to parse JSON
            # However, if OpenRouter sometimes returns 200 OK with an error in the JSON body,
            # just calling raise_for_status() might not be enough if the script proceeds.
            # For now, we will rely on the JSON parsing below but this is a place for future hardening.
            # response.raise_for_status() # This would throw an HTTPError for 4xx/5xx responses

            response_json = {} # Initialize to empty dict
            try:
                response_json = response.json()
            except json.JSONDecodeError as je:
                print(f"ERROR_OPENROUTER_CLIENT: Failed to decode JSON from OpenRouter. Status: {response.status_code}, Response text: {response.text}")
                # Return a structured error that your system expects
                return {
                    "choices": [
                        {
                            "message": {
                                "content": f"Error from OpenRouter: Failed to decode JSON response (Status: {response.status_code})"
                            }
                        }
                    ]
                }
            

            print(f"Request to OpenRouter for model is complete: {model}") # This log comes AFTER potential error interpretation

            # Check for error field in response - THIS IS THE CRITICAL LOGIC
            if "error" in response_json:
                error_msg = response_json.get("error", {}).get("message", str(response_json["error"]))
                print(f"OpenRouter API returned error for {model}: {error_msg}") # This is the error you see
                return {
                    "choices": [
                        {
                            "message": {
                                "content": f"Error from OpenRouter: {error_msg}"
                            }
                        }
                    ]
                }

            # Handle the Gemini models case through OpenRouter
            if "choices" not in response_json and "candidates" in response_json:
                print(f"Found 'candidates' in response from {model}, converting to standard format")
                candidates = response_json.get("candidates", [])
                if candidates and len(candidates) > 0:
                    candidate = candidates[0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        # This is the Gemini format through OpenRouter
                        content = candidate["content"]["parts"][0].get("text", "")
                        return {
                            "choices": [
                                {
                                    "message": {
                                        "content": content
                                    }
                                }
                            ]
                        }

            # Standard OpenRouter response check
            if "choices" not in response_json:
                print(f"WARNING_OPENROUTER_CLIENT: Unexpected response format (no 'choices') from {model}. Keys: {list(response_json.keys())}")
                return {
                    "choices": [
                        {
                            "message": {
                                "content": f"Error from OpenRouter: Unexpected response format from {model}"
                            }
                        }
                    ]
                }

            return response_json

        except requests.RequestException as e:
            print(f"ERROR_OPENROUTER_CLIENT: Request error with OpenRouter for {model}: {str(e)}")
            return {
                "choices": [
                    {
                        "message": {
                            "content": f"Error from OpenRouter: API request error: {str(e)}"
                        }
                    }
                ]
            }
        except Exception as e: # Catch-all for other unexpected errors during the process
            print(f"ERROR_OPENROUTER_CLIENT: Unexpected error in OpenRouterClient.generate for {model}: {str(e)}\n{traceback.format_exc()}")
            return {
                "choices": [
                    {
                        "message": {
                            "content": f"Error from OpenRouter: Unexpected client error: {str(e)}"
                        }
                    }
                ]
            } 
            
    def completion(self,
                  model: str,
                  prompt: str,
                  temperature: float = 0.7,
                  max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Generate a completion using the specified model through OpenRouter's completions endpoint.
        This is specifically designed for continuous inner monologue generation.

        Args:
            model: Model identifier (e.g., "deepseek/deepseek-r1-zero")
            prompt: The full text prompt to complete
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate

        Returns:
            Response from the API standardized to match OpenAI format
        """
        url = f"{self.base_url}/completions"

        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            print(f"Sending completion request to OpenRouter for model: {model}")
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()

            response_json = response.json()
            print(f"Completion request to OpenRouter for model is complete: {model}")

            # Check for error field in response
            if "error" in response_json:
                error_msg = response_json.get("error", {}).get("message", str(response_json["error"]))
                print(f"OpenRouter API returned error for {model}: {error_msg}")
                # Return a structured error response that won't break our code
                return {
                    "choices": [
                        {
                            "text": f"Error from OpenRouter: {error_msg}"
                        }
                    ]
                }

            # Convert completion format to a standardized format
            if "choices" in response_json:
                # Ensure the format includes text field for completions
                for choice in response_json["choices"]:
                    if "text" not in choice and "message" in choice and "content" in choice["message"]:
                        choice["text"] = choice["message"]["content"]
                    
                # Also add a standardized message format for compatibility
                for choice in response_json["choices"]:
                    if "text" in choice and "message" not in choice:
                        choice["message"] = {"content": choice["text"]}

            # Standard OpenRouter response check
            if "choices" not in response_json:
                print(f"WARNING: Unexpected response format from {model}")
                print(f"Response keys: {list(response_json.keys())}")

                # Return a standardized format that won't break our code
                return {
                    "choices": [
                        {
                            "text": f"Unexpected response format from {model}"
                        }
                    ]
                }

            return response_json

        except requests.RequestException as e:
            print(f"Request error with OpenRouter completion for {model}: {str(e)}")
            # Return a structured error response that won't break our code
            return {
                "choices": [
                    {
                        "text": f"API request error: {str(e)}"
                    }
                ]
            }
        except Exception as e:
            print(f"Unexpected error with OpenRouter completion request for {model}: {str(e)}")
            # Return a structured error response that won't break our code
            return {
                "choices": [
                    {
                        "text": f"Error: {str(e)}"
                    }
                ]
            } 