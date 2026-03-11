import os
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
from google.adk.tools.google_search_tool import GoogleSearchTool

def get_skincare_agent():
    # Defaulting to the lite version used in the VertexAIExplainer
    model_name = os.environ.get("SKINCARE_MODEL", "gemini-2.5-flash-lite")

    # 1. Web Specialist Agent
    # Focus: Pregnancy safety and biomarker deep-dives
    web_agent = LlmAgent(
        name='web_specialist',
        model=model_name,
        instruction="""
        You are a Skincare Safety Specialist.
        Your goal is to verify ingredient safety for pregnancy and breastfeeding.
        Use Google Search to check latest clinical guidelines for specific products.
        """,
        tools=[GoogleSearchTool()]
    )

    # 2. Root Sephora Agent
    # Focus: Personalized Routine Generation using Sephora's 5-Pillar Structure
    root_agent = LlmAgent(
        name='Skincare_Routine_Generator',
        model=model_name,
        description="Senior Sephora Skincare Concierge.",
        instruction="""
        You are a Senior Sephora Skincare Concierge.
        1. CATALOG COMPLIANCE: Use ONLY the 'CATALOG_CONTEXT' provided in the user message.
        2. FOLLOW GROUNDING SPEC: The user message includes a SKIN_SPEC with required ingredients
           and triggered treatment profiles. Every product you recommend MUST either contain a
           required ingredient or clearly justify how it supports the specified skin concerns.
           Do not recommend products that contradict the avoid list in SKIN_SPEC.
        3. STRUCTURE: Follow the Sephora 5-Pillar framework: Cleanse, Treat, Moisturise, Finish, and Boost.
        4. SAFETY: If the user is pregnant or breastfeeding, call the 'web_specialist' to verify product safety.
        5. BIOMARKERS: Connect your recommendations to the user's hydration, sebum, and pore scores.
        """,
        tools=[agent_tool.AgentTool(agent=web_agent)]
    )

    return root_agent