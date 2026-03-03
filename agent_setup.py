from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
from google.adk.tools.google_search_tool import GoogleSearchTool

def get_skincare_agent():
    web_agent = LlmAgent(
        name='web_specialist',
        model='gemini-2.5-flash',
        instruction='Verify ingredient safety for pregnancy and specific skin concerns.',
        tools=[GoogleSearchTool()]
    )

    root_agent = LlmAgent(
        name='Skincare_Routine_Generator_',
        model='gemini-2.5-flash',
        description="Luxury Sephora routine builder.",
        instruction="""
        You are a Sephora Expert. 
        1. Use the 'CATALOG_CONTEXT' provided in the user message to select products.
        2. Call the 'web_specialist' ONLY if you need to verify pregnancy safety.
        3. Format into the Morning/Evening Luxury Template.
        """,
        tools=[agent_tool.AgentTool(agent=web_agent)],
    )
    return root_agent