import streamlit as st
import pandas as pd
import asyncio
import uuid
import os
import vertexai
import json
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from agent_setup import get_skincare_agent

# --- GCP CONFIGURATION ---
PROJECT_ID = "sephora-data-gke-apps"
LOCATION = "us-central1" 

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
vertexai.init(project=PROJECT_ID, location=LOCATION)

# --- INITIALIZATION ---
st.set_page_config(page_title="Sephora AI Skin Agent", layout="wide")

@st.cache_resource
def init_agent_system():
    agent = get_skincare_agent()
    service = InMemorySessionService()
    return agent, service

root_agent, session_service = init_agent_system()

if 'step' not in st.session_state:
    st.session_state.step = 1
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

async def run_agent_turn(query):
    APP_NAME = "Sephora_Skincare_POC"
    USER_ID = "customer_user"
    SESSION_ID = st.session_state.session_id
    try:
        await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    except: pass 
    runner = Runner(agent=root_agent, session_service=session_service, app_name=APP_NAME)
    new_msg = types.Content(role='user', parts=[types.Part(text=query)])
    final_text = ""
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=new_msg):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_text = event.content.parts[0].text
    return final_text

# --- UI STEPS ---

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
            st.session_state.form_data.update({"skin_type": skin_type, "hydration": hydration, "sebum": sebum, "pores": pores, "lines": lines})
            st.session_state.step = 2
            st.rerun()

elif st.session_state.step == 2:
    st.header("Step 2: Personalize Your Results (Optional)")
    with st.form("optional_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 0, 120, 25)
            ethnicity = st.text_input("Ethnicity", placeholder="e.g. South Asian")
            pregnancy = st.selectbox("Are you pregnant or breastfeeding?", ["No", "Yes"])
            sun_exposure = st.selectbox("Daily Sun Exposure", ["Low", "Medium", "High"])
        with col2:
            # NEW PREFERENCE FIELD
            preferred_steps = st.selectbox("How many steps do you usually prefer in your skincare routine?", [3, 4, 5, 6], index=1)
            customer_priority = st.text_input("Main Priority", placeholder="e.g. Hyperpigmentation")
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
                    "current_routine": current_routine, "other_preferences": other_preferences,
                    "preferred_steps": preferred_steps
                })
            else:
                # Default values for skip
                st.session_state.form_data.update({
                    "age": 25, "ethnicity": "Not Specified", "pregnancy": "No", 
                    "sun_exposure": "Medium", "customer_priority": "General Health", 
                    "current_routine": "None", "other_preferences": "None",
                    "preferred_steps": 4 # Default to 4 steps if skipped
                })
            st.session_state.step = 3
            st.rerun()

elif st.session_state.step == 3:
    st.header("✨ Your Personalized Sephora Routine")
    d = st.session_state.form_data
    
    with st.spinner("Applying custom filtering logic..."):
        try:
            catalog_df = pd.read_csv('skincat.csv')
            def match_sephora_metadata(row_json, user_data):
                try:
                    meta = json.loads(row_json)
                    flat_meta = {item['filter_name']: item['filter_values'] for item in meta}
                    return user_data['skin_type'] in flat_meta.get('Skin Type', [])
                except: return False
            mask = catalog_df['filter_metadata'].apply(lambda x: match_sephora_metadata(x, d))
            # Increase head count to ensure agent has enough variety to pick 3-6 products
            catalog_context = catalog_df[mask].head(20).to_string(index=False)
        except: catalog_context = "Catalog items."

    # UPDATED PROMPT WITH PREFERRED STEPS
    agent_prompt = f"""
    ROLE: Senior Sephora Skincare Concierge.
    USER BIOMETRICS: {d['skin_type']} | Hydration: {d['hydration']}/100, Sebum: {d['sebum']}/100, Pores: {d['pores']}/100, Lines: {d['lines']}/100.
    CONTEXT: {d.get('age')}yo, {d.get('ethnicity')}, Pregnancy: {d.get('pregnancy')}, Sun: {d.get('sun_exposure')}.
    PREFERENCE: User prefers a {d.get('preferred_steps')}-step routine.
    
    CATALOG: {catalog_context}

    INSTRUCTIONS: Build a luxury routine. Use ONLY products in the catalog.
    
    OUTPUT FORMAT:
    You must wrap sections in these EXACT markers:
    [SUMMARY_START]
    (2-3 sentences connecting biomarkers to lifestyle/ethnicity)
    [SUMMARY_END]

    [MORNING_START]
    ☀️ Morning Routine: [Theme Name] | Goal: [Functional Goal]
    (Generate exactly {d.get('preferred_steps')} steps/products for this routine)
    - [Step]: [Brand] – [Product Name] | Why: [Specific Ingredients vs Biomarkers]
    Step by Step Method to Apply: [Tips]
    [MORNING_END]

    [EVENING_START]
    🌙 Evening Routine: [Theme Name] | Goal: [Repair Goal]
    (Generate exactly {d.get('preferred_steps')} steps/products for this routine)
    - [Step]: [Brand] – [Product Name] | Why: [Night Repair Logic]
    Step by Step Method to Apply: [Tips]
    [EVENING_END]

    STRICT: Use 'web_specialist' (Google Search) ONLY for pregnancy safety.
    """
    
    with st.spinner("Our AI Agent is formulating your routine..."):
        result = asyncio.run(run_agent_turn(agent_prompt))
        
        def get_section(text, start_marker, end_marker):
            try:
                return text.split(start_marker)[1].split(end_marker)[0].strip()
            except: return None

        summary = get_section(result, "[SUMMARY_START]", "[SUMMARY_END]")
        morning = get_section(result, "[MORNING_START]", "[MORNING_END]")
        evening = get_section(result, "[EVENING_START]", "[EVENING_END]")

        if summary:
            st.markdown("### 📝 Analysis Summary")
            st.info(summary)
            st.divider()

        if morning:
            st.markdown("### ☀️ Morning Routine")
            with st.container(border=True):
                st.markdown(morning)
            st.write("")

        if evening:
            st.markdown("### 🌙 Evening Routine")
            with st.container(border=True):
                st.markdown(evening)

        if not summary and not morning:
            st.warning("Routine generated in non-standard format.")
            st.markdown(result)
        
    if st.button("Start New Analysis"):
        st.session_state.step = 1
        st.session_state.form_data = {}
        st.rerun()