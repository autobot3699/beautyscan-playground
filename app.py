import streamlit as st
import pandas as pd
import asyncio
import uuid
import os
import json
import vertexai
from google.oauth2 import service_account
from google.genai import types

# ADK Specific Imports
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from agent_setup import get_skincare_agent

# --- GCP CONFIGURATION ---
PROJECT_ID = "sephora-data-gke-apps"
LOCATION = "us-central1"

def authenticate_gcp():
    try:
        # Check Streamlit Cloud Secrets (Production)
        if "gcp_service_account" in st.secrets:
            creds_info = dict(st.secrets["gcp_service_account"])
            return service_account.Credentials.from_service_account_info(creds_info)
    except Exception:
        pass
    
    # Check for local environment variable (Local Dev)
    env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if env_path and os.path.exists(env_path):
        return service_account.Credentials.from_service_account_file(env_path)
    return None

# 1. Initialize Credentials
credentials = authenticate_gcp()

# 2. Set Environment Variables for the SDK (Crucial for Production)
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
if PROJECT_ID:
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
    os.environ["GOOGLE_PROJECT_ID"] = PROJECT_ID  # Some SDK versions look for this

# 3. Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

# --- INITIALIZATION ---
st.set_page_config(page_title="Sephora AI Skin Agent", layout="wide")

@st.cache_resource
def init_agent_system():
    # Pass credentials or initialize within the authenticated context
    agent = get_skincare_agent()
    service = InMemorySessionService()
    return agent, service

root_agent, session_service = init_agent_system()

# --- CONTINUED: SESSION STATE & LOGIC ---
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
    except: 
        pass 
    
    runner = Runner(agent=root_agent, session_service=session_service, app_name=APP_NAME)
    new_msg = types.Content(role='user', parts=[types.Part(text=query)])
    final_text = ""
    
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=new_msg):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_text = event.content.parts[0].text
    return final_text

# --- UI STEPS (1 & 2) ---
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
                st.session_state.form_data.update({
                    "age": 25, "ethnicity": "Not Specified", "pregnancy": "No", 
                    "sun_exposure": "Medium", "customer_priority": "General Health", 
                    "current_routine": "None", "other_preferences": "None",
                    "preferred_steps": 4 
                })
            st.session_state.step = 3
            st.rerun()

# --- STEP 3: LOGIC & OUTPUT ---
elif st.session_state.step == 3:
    st.header("✨ Your Personalized Sephora Routine")
    d = st.session_state.form_data
    
    with st.spinner("Searching for the perfect products..."):
        try:
            catalog_df = pd.read_csv('skincat.csv')
            target_steps = d.get('preferred_steps', 4)
            
            def get_refined_catalog(df, user_data):
                def parse_meta(x):
                    try:
                        meta = json.loads(x)
                        return {item['filter_name']: [v.lower() for v in item['filter_values']] for item in meta}
                    except: return {}

                df['parsed_meta'] = df['filters_json'].apply(parse_meta)
                user_skin = user_data['skin_type'].lower()
                
                strict_mask = df['parsed_meta'].apply(lambda m: user_skin in m.get('Skin Type', []))
                results = df[strict_mask].copy()

                if len(results) < (target_steps + 4):
                    relaxed_mask = df['filters_json'].str.contains(user_skin, case=False, na=False)
                    results = df[relaxed_mask].copy()

                if results.empty:
                    results = df.head(30).copy()
                
                # DAILY RANDOMIZATION
                daily_seed = int(pd.Timestamp.now().strftime('%Y%m%d'))
                results = results.sample(frac=1, random_state=daily_seed)
                
                return results.head(25) 

            refined_df = get_refined_catalog(catalog_df, d)
            catalog_context = refined_df[['brand', 'Product']].to_string(index=False)
            st.caption(f"Found {len(refined_df)} products matching your profile.")

        except Exception as e:
            st.error(f"Catalog Error: {e}")
            catalog_context = "No products found."

    agent_prompt = f"""
    ROLE: Senior Sephora Skincare Concierge.
    STRICT SOURCE OF TRUTH: You MUST ONLY use the products listed below. 
    
    CATALOG LIST:
    {catalog_context}

    USER DATA: {d['skin_type']} skin, {d.get('age')}yo, Hydration: {d['hydration']}/100.
    PREFERENCE: Exactly {target_steps} steps.

    OUTPUT FORMAT (DO NOT MISS ANY SECTION):
    [SUMMARY_START]
    Write a detailed 3-sentence analysis connecting biomarkers to skin health.
    [SUMMARY_END]

    [MORNING_START]
    ☀️ Morning Routine: [Theme Name] | Goal: [Functional Goal]
    For each of the {target_steps} products, you MUST include:
    - **[Brand] – [Product Name]**
    - **Why**: [Detailed explanation of ingredients vs user's skin biomarkers]
    - **Step by Step Method to Apply**: [2-3 detailed luxury application tips]
    [MORNING_END]

    [EVENING_START]
    🌙 Evening Routine: [Theme Name] | Goal: [Repair Goal]
    For each of the {target_steps} products, you MUST include:
    - **[Brand] – [Product Name]**
    - **Why**: [Detailed night repair logic for dry/oily/combination skin]
    - **Step by Step Method to Apply**: [2-3 detailed luxury application tips]
    [EVENING_END]
    """

    with st.spinner("AI Agent is formulating your detailed skincare routine..."):
        result = asyncio.run(run_agent_turn(agent_prompt))
        
        def extract(text, start, end):
            try: 
                content = text.split(start)[1].split(end)[0].strip()
                return content
            except: return ""

        sum_out = extract(result, "[SUMMARY_START]", "[SUMMARY_END]")
        am_out = extract(result, "[MORNING_START]", "[MORNING_END]")
        pm_out = extract(result, "[EVENING_START]", "[EVENING_END]")

        if sum_out:
            st.info(f"**Skin Analysis:** {sum_out}")
            st.divider()

        if am_out:
            st.subheader("☀️ Morning Routine")
            with st.container(border=True):
                st.markdown(am_out)

        if pm_out:
            st.subheader("🌙 Evening Routine")
            with st.container(border=True):
                st.markdown(pm_out)

        if not am_out:
            st.error("The agent failed to generate the routine format. Raw output below:")
            st.write(result)
        
    if st.button("Start New Analysis"):
        st.session_state.step = 1
        st.session_state.form_data = {}
        st.rerun()