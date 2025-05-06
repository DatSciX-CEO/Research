
from google.adk.agents import Agent # type: ignore
from google.adk.models.lite_llm import LiteLlm # type: ignore

ollama_model = LiteLlm(model="ollama_chat/gemma3:4b")
gemini_model = "gemini-2.0-flash"

root_agent = Agent(
    name="greeting_agent",
  #  model=gemini_model,
    model= ollama_model,
    description="Greeting agent",
    instruction="""
    You are a helpful assistant that greets the user. 
    Ask for the user's name and greet them by name.
    """,
)



