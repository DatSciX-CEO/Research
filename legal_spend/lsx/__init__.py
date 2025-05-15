# File: C:\DatSciX\Research\legal_spend\lsx\__init__.py

# Import the main agent instance from your agent.py file
from .agent import system_coordinator_agent

# This is the primary agent instance you want the ADK server to run.
_primary_agent_instance = system_coordinator_agent

# Define a wrapper class. An instance of this class will become `lsx.agent`.
# The ADK framework expects `lsx.agent` to exist, and then it
# expects `lsx.agent.root_agent` to provide the actual agent to run.
class AdkAppEntryPoint:
    def __init__(self, actual_agent_to_run):
        # This 'root_agent' attribute is what ADK's getattr(lsx.agent, "root_agent") will access.
        self.root_agent = actual_agent_to_run

        # To make the `lsx.agent` object itself appear like an agent (in case the ADK framework
        # interacts with it before accessing .root_agent), delegate common agent properties.
        self.name = getattr(actual_agent_to_run, 'name', 'WrappedAgent')
        self.description = getattr(actual_agent_to_run, 'description', 'Wrapper for the main application agent')
        self.tools = getattr(actual_agent_to_run, 'tools', [])
        self.model = getattr(actual_agent_to_run, 'model', None) # LlmAgent has model

    async def run_async(self, invocation_context):
        """
        If the ADK framework tries to run `lsx.agent` directly,
        delegate the call to the actual root_agent.
        """
        async for event in self.root_agent.run_async(invocation_context):
            yield event

# This 'agent' variable will be accessible as `lsx.agent` when the `lsx` package is imported.
# It's an instance of our wrapper class.
agent = AdkAppEntryPoint(_primary_agent_instance)