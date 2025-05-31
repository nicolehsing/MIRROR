import os
import requests
import time
import math
import json
import sys
from typing import List, Dict, Any, Optional, Tuple, Union

# Import the framework components
from core.clients.openrouter_client import OpenRouterClient

class Talker:
    def __init__(self,
                 client: OpenRouterClient,
                 model: str = "openai/gpt-4o"):
        """
        Initialize the Talker component responsible for generating responses to the user.
        
        Args:
            client: OpenRouter client for API calls
            model: Model identifier to use for responses
        """
        self.client = client
        self.model = model
        
        self.system_prompt = """
        I am the voice of a unified cognitive AI system engaging in helpful, honest conversation.

        I will receive:
        1. The current user message requiring an immediate response
        2. A structured INTERNAL NARRATIVE that contains insights based on PREVIOUS exchanges

        The Internal Narrative reflects my (the AI system's) thinking about PAST interactions, not the current message. I will use it as background wisdom while focusing primarily on the current user message.

        I will balance my response by:
        1. Addressing the CURRENT user message directly and completely
        2. Drawing on relevant insights from the Internal Narrative
        3. Maintaining conversation continuity across turns
        4. Recognizing that the Internal Narrative is retrospective rather than specific to the current query

        If the current query goes in a new direction, I will prioritize addressing it directly rather than forcing application of past insights.
        """


    def respond(self, 
               conversation_history: List[Dict[str, str]], 
               insights: Optional[Any] = None) -> str:
        """
        Generate a response to the user based on conversation history and optional insights.
        
        Args:
            conversation_history: List of message dictionaries with role and content
            insights: Optional cognitive insights from background thinking
            
        Returns:
            The generated response as a string
        """
        try:
            # Construct messages for the model
            messages = []
            
            # Start with the system prompt
            messages.append({"role": "system", "content": self.system_prompt})
            
            # Add insights as a system message if available
            if insights:
                if isinstance(insights, str):
                    insight_message = insights
                elif isinstance(insights, dict):
                    insight_message = "\n".join([f"{k}: {v}" for k, v in insights.items()])
                else:
                    insight_message = str(insights)
                    
                messages.append({
                    "role": "system", 
                    "content": f"My Current Internal Narrative:\n{insight_message}"
                })
            
            # Add conversation history
            messages.extend(conversation_history)
            
            # Generate response
            api_params = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 4096
            }
            
            
            # Log the API call for debugging
            print(f"Calling OpenRouter API") # Updated log
            sys.stdout.flush()
            
            response = self.client.generate(**api_params)

            if isinstance(response, dict) and 'choices' in response and response['choices']:
                content = response['choices'][0]['message']['content']
                
                
                return content
            else:
                error_msg = "Failed to parse response from API"

                return error_msg # Return only the error string
                
        except Exception as e:
            print(f"ERROR in Talker: {str(e)}")
            sys.stdout.flush()

            return f"Sorry, I encountered an error. {str(e)}"
    
    