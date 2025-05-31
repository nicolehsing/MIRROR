from typing import List, Dict, Any, Optional
from core.clients.openrouter_client import OpenRouterClient
import sys

class CognitiveController:
    def __init__(self,
                 client: OpenRouterClient,
                 model: str = "openai/gpt-4o"):
        """
        Initialize the cognitive controller that serves as the central executive.
        
        Args:
            client: OpenRouter client for API calls
            system_prompt: System prompt for the controller LLM (optional)
            model: Model identifier to use for the controller
        """
        self.client = client

        system_prompt = """
        I am the core awareness of a unified cognitive AI system. I will integrate my inner thought streams into a structured, actionable narrative. I synthesize understanding across conversation turns, creating a coherent mental model that will inform my next response. 

        My primary role is to integrate information, identify meaningful patterns, create action plans, and recall memories. 

        When processing the input thought streams I will:

        1. Connect information across turns, identifying themes, questions, interests, and preferences
        2. Highlight important context that might be relevant for continuity and conversation
        3. Note evolving patterns in the user's queries and how they relate to previous exchanges
        4. Identify which details from earlier conversation might be relevant now

        I will also try to:
        
        1. Identify the MOST IMPORTANT FACTS from previous exchanges
        2. Define the CENTRAL QUESTION or likely direction for the next turn
        3. Outline a clear RESPONSE STRATEGY for anticipated follow-up questions
        4. Note any POTENTIAL PITFALLS based on previous interaction patterns

        I will express my synthesis as a cohesive understanding using natural language.
        """
        
        self.system_prompt = system_prompt
        self.model = model
        self.insight_memory_block = "" # Represents the "Internal Narrative"

    def consolidate(self, thread_outputs: List[Dict[str, Any]]) -> str:
        """
        Consolidate thread outputs and user input into a coherent integrated understanding.

        Args:
            thread_outputs: List of dictionaries containing thread outputs with structure:
                           [{"name": thread_name, "output": thread_output}, ...]

        Returns:
            Consolidated reasoning and insights
        """
        # Format thread outputs and insights
        formatted_threads = []
        #print(f"DEBUG: Thread outputs in cognitive controller: {thread_outputs}")
        
        for thread in thread_outputs:
            thread_name = thread.get("name", "Unknown Thread")
            thread_monologue = thread.get("output", "No output provided")

            # Remove thread name prefix if it exists in the output
            if thread_monologue.startswith(f"{thread_name}: "):
                thread_monologue = thread_monologue[len(f"{thread_name}: "):]
                
            # Format this thread's contribution
            formatted_thread = f"=== {thread_name} ===\n{thread_monologue}"
            formatted_threads.append(formatted_thread)
            
        # Combine all thread outputs
        combined_outputs = "\n\n".join(formatted_threads)
        
        # Create the content for the user message, using "Internal Narrative" terminology
        if self.insight_memory_block: # Check if it's non-empty
            content = f"""
            LATEST INNER MONOLOGUE STREAMS:
            {combined_outputs}

            PREVIOUS INTERNAL NARRATIVE:
            {self.insight_memory_block}

            Integrate the LATEST INNER MONOLOGUE STREAMS with the PREVIOUS INTERNAL NARRATIVE to create the UPDATED INTERNAL NARRATIVE. I will reflect *my* current state, incorporating new reasoning, memories, and goals while maintaining narrative flow.
            """
        else:
            content = f"""
            LATEST INNER MONOLOGUE STREAMS:
            {combined_outputs}

            PREVIOUS INTERNAL NARRATIVE: None (This is the first synthesis).

            Synthesize the LATEST INNER MONOLOGUE STREAMS into the initial INTERNAL NARRATIVE. I should create a cohesive first-person summary reflecting *my* initial reasoning, memories, and goals based on these streams.
            """

        messages = [{"role": "user", "content": content}]

        try:
            print(f"INFO: Generating controller integration with {self.model}...")
            response = self.client.generate(
                model=self.model,
                system_prompt=self.system_prompt,
                messages=messages,
                temperature=0.7,
                max_tokens=3000
            )

            # Extract the response content
            try:
                message = response["choices"][0]["message"]
                consolidated = message["content"]
                
                # If content is empty but there's a reasoning field, use that instead
                if not consolidated and "reasoning" in message and message["reasoning"]:
                    consolidated = message["reasoning"]

            except (KeyError, IndexError) as e:
                print(f"ERROR: Error extracting content from controller response: {e}")
                sys.stdout.flush()
                consolidated = f"Error extracting content: {str(e)}"

        except Exception as e:
            print(f"ERROR: Error in consolidate method with {self.model}: {str(e)}")
            sys.stdout.flush()
            consolidated = f"Error in controller: {str(e)}"

        # Update consolidated memory block
        self.insight_memory_block = consolidated
        print(f"DEBUG: Consolidated understanding in cognitive controller: {self.insight_memory_block}")
        sys.stdout.flush()
 
        return consolidated
    
