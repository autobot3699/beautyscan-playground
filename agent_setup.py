import pandas as pd
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools import url_context

# --- CUSTOM CSV SEARCH TOOL ---
def sephora_catalog_search(query: str) -> str:
    """
    Searches the skincat.csv for products matching the query.
    Args:
        query: Keywords to search in the catalog (e.g., 'cleanser for oily skin').
    """
    try:
        df = pd.read_csv('skincat.csv')
        # Simple text search across all columns
        mask = df.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)
        results = df[mask].head(10) # Return top 10 matches
        if results.empty:
            return "No matching products found in the catalog."
        return results.to_string(index=False)
    except Exception as e:
        return f"Error accessing catalog: {str(e)}"

def get_skincare_agent():
    # Specialist Agents
    search_agent = LlmAgent(
        name='Skincare_Routine_Generator__google_search_agent',
        model='gemini-3-flash', # Updated to 2026 stable model
        instruction='Use the GoogleSearchTool to find information on the web.',
        tools=[GoogleSearchTool()],
    )

    url_agent = LlmAgent(
        name='Skincare_Routine_Generator__url_context_agent',
        model='gemini-3-flash',
        instruction='Use the UrlContextTool to fetch content from URLs.',
        tools=[url_context],
    )

    # Main Agent
    root_agent = LlmAgent(
        name='Skincare_Routine_Generator_',
        model='gemini-2.5-flash',
        description="Builds Sephora routines using the skincat.csv tool.",
        instruction="""
        The goal is to GIVE personalized skin recommendations to every client. 
        To do this, the agent must do two things:

        A. BUILD OR ROUND OUT A CLIENT’S ROUTINE
        Sephora Skin Routines include products to:
        1. Cleanse
        2. Treat
        3. Moisturise
        4. Finish
        5. Boost – optional for some Clients

        B. ADDRESS THE “MUST DOs” FOR THEIR CONCERN
        Example: Clogged pores & oiliness -> Focus on antibacterial, oil control, gentle exfoliation, and soothing inflammation.

        IMPORTANT DATA GUIDELINES:
        1. SOURCE OF TRUTH: You MUST use the 'sephora_catalog_search' tool to find products. The csv file attached (skincat.csv) contains the only valid products for SG, MY, TH, AU, NZ, HK, PH, ID.
        2. AVAILABILITY: Cross-reference the 'unavailableCountries' column in the tool to ensure the product is available in the user's specific region.
        3. ENRICHMENT: Use the Google Search sub-agent to cross-reference product ingredients found in the catalog with internet data to solve specific customer concerns.

        Product recommendations must collectively form a routine that addresses the client's skin goals.
        """,
        tools=[
            sephora_catalog_search, # Directly passing the function as a tool
            agent_tool.AgentTool(agent=search_agent),
            agent_tool.AgentTool(agent=url_agent)
        ],
    )
    return root_agent