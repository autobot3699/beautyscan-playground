import tempfile
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
    if "gcp_service_account" in st.secrets:
        creds_info = dict(st.secrets["gcp_service_account"])
        # Create a temporary file to hold the JSON key for the SDK
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_key:
            json.dump(creds_info, temp_key)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_key.name
        return service_account.Credentials.from_service_account_info(creds_info)
    return None

credentials = authenticate_gcp()
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID

vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

@st.cache_resource
def init_agent_system():
    agent = get_skincare_agent() 
    service = InMemorySessionService()
    return agent, service

root_agent, session_service = init_agent_system()

if 'step' not in st.session_state: st.session_state.step = 1
if 'form_data' not in st.session_state: st.session_state.form_data = {}
if 'session_id' not in st.session_state: st.session_state.session_id = str(uuid.uuid4())

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
        with col2:
            preferred_steps = st.selectbox("Routine Steps Preference", [3, 4, 5, 6], index=1)
            customer_priority = st.text_input("Main Priority", placeholder="e.g. Hyperpigmentation")
        
        if st.form_submit_button("Generate Routine"):
            st.session_state.form_data.update({"age": age, "ethnicity": ethnicity, "pregnancy": pregnancy, "preferred_steps": preferred_steps, "customer_priority": customer_priority})
            st.session_state.step = 3
            st.rerun()

# --- STEP 3: LOGIC & OUTPUT ---
elif st.session_state.step == 3:
    st.header("✨ Your Structured Sephora Routine")
    d = st.session_state.form_data
    
    with st.spinner("Curating your personalized Sephora catalog..."):
        try:
            catalog_df = pd.read_csv('skincat.csv')
            
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

                if results.empty: results = df.head(50).copy()
                
                daily_seed = int(pd.Timestamp.now().strftime('%Y%m%d'))
                results = results.sample(frac=1, random_state=daily_seed)
                return results.head(40) 

            refined_df = get_refined_catalog(catalog_df, d)
            catalog_context = refined_df[['brand', 'Product']].to_string(index=False)
        except Exception as e:
            st.error(f"Catalog Error: {e}")
            catalog_context = "General Sephora List"

    # --- UPDATED 5-STEP STRUCTURE PROMPT ---
    agent_prompt = f"""
    ROLE: Senior Sephora Skincare Concierge.
    SOURCE: ONLY use products from this CATALOG:
    {catalog_context}

    USER PROFILE: {d['skin_type']} skin, {d.get('age')}yo. Priority: {d.get('customer_priority')}.

    INSTRUCTION: Organize the routine using the 5 Sephora Routine Steps below.
    For each chosen product, provide: **[Brand] - [Product Name]**, the 'Why' (linking to biomarkers), and the 'How' (application tips).

    [SUMMARY_START]
    Detailed analysis of user's skin profile.
    [SUMMARY_END]

    [ROUTINE_START]
    ### 🧼 1. Cleanse
    **Purpose**: Remove impurities such as excess sebum, dirt, sweat, makeup and sunscreen.
    **Product Categories**: Makeup Remover, Facial Cleansers, Toners & Essence.
    (Select 1-2 products)

    ### 🧪 2. Treat
    **Purpose**: Address specific concerns - Correct, target, and rebalance skin.
    **Product Categories**: Exfoliators, Masks, Serum and Oils.
    (Select 1-2 products)

    ### 💧 3. Moisturise
    **Purpose**: Replenish, hydrate, and repair moisture barrier to maintain hydration.
    **Product Categories**: Face/Eye/Lip Moisturisers & Hydrators.
    (Select 1 product)

    ### 🛡️ 4. Finish
    **Purpose**: Protect from sun damage, glow, blurring, and prepping.
    **Product Categories**: SPF, BB & CC Cream, Facial Mists.
    (Select 1 product)

    ### 🚀 5. Boost
    **Purpose**: Enhance daily routine or provide intensive care.
    **Product Categories**: Tools, Devices, Supplements.
    (Select 1 product if available)
    [ROUTINE_END]
    """

    with st.spinner("AI is formulating your 5-step routine..."):
        result = asyncio.run(run_agent_turn(agent_prompt))
        
        def extract(text, start, end):
            try: return text.split(start)[1].split(end)[0].strip()
            except: return ""

        sum_out = extract(result, "[SUMMARY_START]", "[SUMMARY_END]")
        routine_out = extract(result, "[ROUTINE_START]", "[ROUTINE_END]")

        if sum_out:
            st.info(f"**Expert Analysis:** {sum_out}")
            st.divider()

        if routine_out:
            st.markdown(routine_out)
        else:
            st.error("Structure parsing error. Showing raw output:")
            st.write(result)
        
    if st.button("Start New Analysis"):
        st.session_state.step = 1
        st.rerun()