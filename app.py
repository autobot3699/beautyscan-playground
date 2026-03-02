import streamlit as st
import pandas as pd
import asyncio
import uuid
import os
import vertexai
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from agent_setup import get_skincare_agent

# --- GCP CONFIGURATION ---
# Use 'us-central1' for the widest model availability if 'asia-southeast1' fails
PROJECT_ID = "sephora-data-gke-apps"
LOCATION = "us-central1" 

# Force Vertex AI backend
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"

# Initialize Vertex AI globally
vertexai.init(project=PROJECT_ID, location=LOCATION)

# --- INITIALIZATION ---
st.set_page_config(page_title="Sephora AI Skin Agent", layout="wide")

# Initialize Agent and Session Service (cached for performance)
@st.cache_resource
def init_agent_system():
    agent = get_skincare_agent()
    service = InMemorySessionService()
    return agent, service

root_agent, session_service = init_agent_system()

# Initialize Session States
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- ASYNC EXECUTION LOGIC ---
async def run_agent_turn(query):
    APP_NAME = "Sephora_Skincare_POC"
    USER_ID = "customer_user"
    SESSION_ID = st.session_state.session_id

    # 1. Initialize/Ensure the session exists in the service
    try:
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID
        )
    except Exception:
        pass # Session already exists

    # 2. Initialize the Runner
    runner = Runner(
        agent=root_agent, 
        session_service=session_service, 
        app_name=APP_NAME
    )
    
    # 3. Prepare the message
    new_msg = types.Content(role='user', parts=[types.Part(text=query)])
    final_text = ""
    
    # 4. Run the async loop
    async for event in runner.run_async(
        user_id=USER_ID, 
        session_id=SESSION_ID, 
        new_message=new_msg
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_text = event.content.parts[0].text
                
    return final_text

# --- UI STEPS ---

# STEP 1: MANDATORY FIELDS
if st.session_state.step == 1:
    st.header("Step 1: Your Skin Profile (Mandatory)")
    with st.form("mandatory_form"):
        skin_type = st.selectbox("Skin Type", ["Dry", "Oily", "Combination", "Normal"])
        
        col1, col2 = st.columns(2)
        with col1:
            hydration = st.slider("Hydration Score", 0, 100, 50)
            sebum = st.slider("Sebum Score", 0, 100, 50)
        with col2:
            pores = st.slider("Pores Score", 0, 100, 50)
            lines = st.slider("Lines Score", 0, 100, 50)
        
        if st.form_submit_button("Next: Personalize →"):
            st.session_state.form_data.update({
                "skin_type": skin_type, "hydration": hydration, 
                "sebum": sebum, "pores": pores, "lines": lines
            })
            st.session_state.step = 2
            st.rerun()

# STEP 2: OPTIONAL FIELDS
elif st.session_state.step == 2:
    st.header("Step 2: Personalize Your Results (Optional)")
    with st.form("optional_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 0, 120, 25)
            ethnicity = st.text_input("Ethnicity", placeholder="e.g. East Asian")
            pregnancy = st.selectbox("Are you pregnant or breastfeeding?", ["No", "Yes"])
            sun_exposure = st.selectbox("Daily Sun Exposure", ["Low", "Medium", "High"])
        with col2:
            customer_priority = st.text_input("Main Priority", placeholder="e.g. Brightening, Acne")
            current_routine = st.text_area("Current Routine")
            other_preferences = st.text_area("Other Preferences")

        btn1, btn2 = st.columns(2)
        with btn1:
            submit = st.form_submit_button("Generate Routine")
        with btn2:
            skip = st.form_submit_button("Skip & Generate Now")
        
        if submit or skip:
            if submit:
                st.session_state.form_data.update({
                    "age": age, "ethnicity": ethnicity, "pregnancy": pregnancy,
                    "sun_exposure": sun_exposure, "customer_priority": customer_priority,
                    "current_routine": current_routine, "other_preferences": other_preferences
                })
            else:
                st.session_state.form_data.update({
                    "age": "Not Specified", "ethnicity": "Not Specified", "pregnancy": "No",
                    "sun_exposure": "Medium", "customer_priority": "General Health"
                })
            st.session_state.step = 3
            st.rerun()

# STEP 3: AGENT EXECUTION
elif st.session_state.step == 3:
    st.header("✨ Your Personalized Sephora Routine")
    d = st.session_state.form_data
    
    # Structure the input for the Agent (Preserving your exact requested instructions)
    agent_prompt = f"""
    CONTEXT: Professional Sephora Skincare Expert.
    USER DATA:
    - Type: {d['skin_type']} | Hydration: {d['hydration']} | Sebum: {d['sebum']} | Pores: {d['pores']} | Lines: {d['lines']}
    - Profile: Age {d.get('age')}, Pregnancy Safe: {d.get('pregnancy')}, Priority: {d.get('customer_priority')}
    
    INSTRUCTIONS:
    1. MANDATORY: Use 'sephora_catalog_search' to find EXACT products from skincat.csv.
    2. LOGIC: If Hydration < 40, prioritize hydrating ingredients. If Sebum > 60, prioritize oil-control.
    3. SAFETY: If Pregnancy is 'Yes', EXCLUDE Retinoids/Salicylic Acid.
    4. OUTPUT: Provide a Morning and Evening routine. Use a Markdown table for each.
    5. RATIONALE: Explain why these products solve the user's specific biomarker scores.
    """
    
    with st.spinner("Agent is searching catalog and formulating your routine..."):
        result = asyncio.run(run_agent_turn(agent_prompt))
        st.markdown(result)
        
    if st.button("Start New Analysis"):
        st.session_state.step = 1
        st.session_state.form_data = {}
        st.rerun()