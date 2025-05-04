from google.adk.agents import Agent

root_agent = Agent(
    name = "GreetingAgent",
    model = "gpt-3.5-turbo",
    description = "A simple agent that greets the user.",
    instruction = "You are a friendly assistant. Greet the user warmly and ask how you can help them today.",
)