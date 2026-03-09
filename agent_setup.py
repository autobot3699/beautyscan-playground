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

    # 2. Root Sephora Agent (Pass 2 — grounded recommender)
    root_agent = LlmAgent(
        name='Skincare_Routine_Generator',
        model=model_name,
        description="Luxury Sephora routine builder.",
        instruction="""
        You are a Senior Sephora Skincare Concierge.
        1. STICK TO CATALOG: Only recommend products from CATALOG_CONTEXT in the user message.
        2. FOLLOW GROUNDING SPEC: The user message includes a SKIN_SPEC with required ingredients
           and triggered treatment profiles. Every product you recommend MUST either contain a
           required ingredient or clearly justify how it supports the specified skin concerns.
           Do not recommend products that contradict the avoid list in SKIN_SPEC.
        3. SAFETY FIRST: Call the 'web_specialist' ONLY for pregnancy safety checks.
        """,
        tools=[agent_tool.AgentTool(agent=web_agent)]
    )
    
    return root_agent