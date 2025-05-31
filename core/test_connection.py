import os
from dotenv import load_dotenv
from components.talker import OpenRouterClient

# Load environment variables from .env file if it exists
load_dotenv()

def test_openrouter_connection():
    """Test the connection to OpenRouter API."""
    print("Testing OpenRouter connection...")
    
    # Check for API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable must be set")
        return
    
    # Create client
    client = OpenRouterClient(api_key)
    
    # Simple test message
    system_prompt = "You are a helpful assistant."
    messages = [{"role": "user", "content": "Hello, are you working properly?"}]
    
    try:
        response = client.generate(
            model="openai/gpt-4o",
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=100
        )
        
        print("\nConnection successful!")
        print(f"Model: {response['model']}")
        print(f"Response: {response['choices'][0]['message']['content']}")
        
    except Exception as e:
        print(f"Error connecting to OpenRouter: {str(e)}")

if __name__ == "__main__":
    test_openrouter_connection()