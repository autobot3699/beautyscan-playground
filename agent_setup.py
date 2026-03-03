import os
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
from google.adk.tools.google_search_tool import GoogleSearchTool

def get_skincare_agent(credentials=None):
    # We define the model name and provide the explicit credentials 
    # and project ID to the agent's internal configuration.
    model_name = 'gemini-2.5-flash'
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "sephora-data-gke-apps")
    location = "us-central1"

    # Common config to force Vertex AI and provide auth
    model_kwargs = {
        "model": model_name,
        "project": project_id,
        "location": location,
        "credentials": credentials, # This is the critical fix
    }

    # 1. Web Specialist Agent
    web_agent = LlmAgent(
        name='web_specialist',
        instruction='Verify ingredient safety for pregnancy and specific skin concerns.',
        tools=[GoogleSearchTool()],
        **model_kwargs
    )

    # 2. Root Sephora Agent
    root_agent = LlmAgent(
        name='Skincare_Routine_Generator',
        description="Luxury Sephora routine builder.",
        instruction="""
        You are a Senior Sephora Skincare Concierge. 
        1. STICK TO CATALOG: Use the 'CATALOG_CONTEXT' provided in the user message.
        2. SAFETY FIRST: Call the 'web_specialist' ONLY for pregnancy safety checks.
        """,
        tools=[agent_tool.AgentTool(agent=web_agent)],
        **model_kwargs
    )
    
    return root_agent