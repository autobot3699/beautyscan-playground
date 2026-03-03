import os
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
from google.adk.tools.google_search_tool import GoogleSearchTool

def get_skincare_agent():
    
    model_name = 'gemini-2.0-flash'

    # 1. Web Specialist Agent
    web_agent = LlmAgent(
        name='web_specialist',
        model=model_name,
        instruction='Verify ingredient safety for pregnancy and specific skin concerns.',
        tools=[GoogleSearchTool()]
    )

    # 2. Root Sephora Agent
    root_agent = LlmAgent(
        name='Skincare_Routine_Generator',
        model=model_name,
        description="Luxury Sephora routine builder.",
        instruction="""
        You are a Senior Sephora Skincare Concierge. 
        1. STICK TO CATALOG: Use the 'CATALOG_CONTEXT' provided in the user message to select products. 
           Do NOT recommend products outside this context.
        2. SAFETY FIRST: Call the 'web_specialist' ONLY to verify pregnancy/breastfeeding safety for the specific products chosen.
        3. LUXURY FORMAT: Follow the Morning/Evening template provided in the user prompt exactly.
        """,
        tools=[agent_tool.AgentTool(agent=web_agent)]
    )
    
    return root_agent