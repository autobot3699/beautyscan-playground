import os
import tempfile

# Must be set before any google/requests imports to trust the corporate SSL cert.
# Combines certifi's standard CA bundle with the corporate cert so both are trusted.
_CORP_CERT = "/Users/spundir/Desktop/cert.pem"
if os.path.exists(_CORP_CERT):
    try:
        import certifi
        with open(certifi.where(), 'rb') as _f:
            _certifi_bundle = _f.read()
        with open(_CORP_CERT, 'rb') as _f:
            _corp_cert_data = _f.read()
        _combined = tempfile.NamedTemporaryFile(mode='wb', suffix='.pem', delete=False)
        _combined.write(_certifi_bundle + b'\n' + _corp_cert_data)
        _combined.close()
        os.environ['REQUESTS_CA_BUNDLE'] = _combined.name
        os.environ['SSL_CERT_FILE'] = _combined.name
        os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = _combined.name
    except ImportError:
        os.environ.setdefault("REQUESTS_CA_BUNDLE", _CORP_CERT)
        os.environ.setdefault("SSL_CERT_FILE", _CORP_CERT)
        os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", _CORP_CERT)

import streamlit as st
import pandas as pd
import asyncio
import uuid
import json
import vertexai
from google.oauth2 import service_account
from google.genai import types
from google import genai

# ADK Specific Imports
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from agent_setup import get_skincare_agent
from grounding_rules import derive_ingredient_spec

# --- GCP CONFIGURATION ---
PROJECT_ID = "sephora-data-gke-apps"
LOCATION = "us-central1"

def authenticate_gcp():
    try:
        if "gcp_service_account" in st.secrets:
            creds_info = dict(st.secrets["gcp_service_account"])
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_key:
                json.dump(creds_info, temp_key)
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_key.name
            return service_account.Credentials.from_service_account_info(creds_info)
    except Exception:
        pass  # No secrets.toml — fall through to Application Default Credentials
    return None

credentials = authenticate_gcp()
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

# ---------------------------------------------------------------------------
# Maps ingredient spec names → substrings present in filters_json
# Used for catalog scoring in filter_catalog_by_spec()
# ---------------------------------------------------------------------------
INGREDIENT_TAG_MAP = {
    "Hyaluronic Acid":  "Hyaluronic Acid",
    "Peptides":         "Peptides",
    "Squalane":         "Squalane",
    "Retinoid":         "Retinol",
    "Collagen":         "Collagen",
    "BHA/Salicylic Acid": "AHAs",
    "Niacinamide":      "Niacinamide",
    "AHA/Glycolic Acid": "AHAs/Glycolic Acid",
    "Vitamin C":        "Vitamins",
    "Ceramides":        "Ceramide",
}

# Keywords matched against the `slug` column to guarantee pillar coverage.
# At least MIN_PER_PILLAR products from each group are reserved in the catalog.
PILLAR_SLUG_KEYWORDS = {
    "Cleanse":    ["cleanser", "toner", "essence", "makeup-remover", "micellar", "cleansing"],
    "Treat":      ["serum", "-oil", "exfoliat", "mask", "peel", "treatment", "acid", "retinol"],
    "Moisturise": ["moisturis", "moisturiz", "hydrat", "eye-cream", "eye-gel", "lip-cream"],
    "Finish":     ["spf", "sunscreen", "sun-", "bb-cream", "cc-cream", "facial-mist", "primer", "setting"],
}
MIN_PER_PILLAR = 3   # reserved slots per pillar in the final 40-product catalog

# Regex patterns to exclude gift sets, kits, duos, and samples from catalog.
# Uses word boundaries (\b) so e.g. "Set" matches "Skincare Set" but not "(Re)setting".
EXCLUDED_PRODUCT_PATTERN = (
    r'\bSet\b|\bKit\b|\bDuo\b|\bTrio\b|\bCalendar\b|\bBundle\b|Travel Size|\bSample\b'
)


@st.cache_resource
def init_agent_system():
    agent = get_skincare_agent()
    service = InMemorySessionService()
    return agent, service

root_agent, session_service = init_agent_system()

if 'step' not in st.session_state: st.session_state.step = 1
if 'form_data' not in st.session_state: st.session_state.form_data = {}
if 'session_id' not in st.session_state: st.session_state.session_id = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Catalog filtering — re-ranks by ingredient spec alignment between passes
# ---------------------------------------------------------------------------

def filter_catalog_by_spec(df: pd.DataFrame, skin_type: str, spec: dict) -> pd.DataFrame:
    """
    1. Exclude gift sets and samples.
    2. Filter by skin_type.
    3. Reserve MIN_PER_PILLAR products per pillar category (slug-based).
    4. Score remaining pool by ingredient / concern alignment.
    5. Fill remaining slots from scored pool; return up to 40 products.
    """
    daily_seed = int(pd.Timestamp.now().strftime('%Y%m%d'))
    required = spec.get("required_ingredients", [])
    concerns  = spec.get("skin_concerns", [])

    # --- Step 1: exclude gift sets / samples ---
    df = df[~df['Product'].str.contains(EXCLUDED_PRODUCT_PATTERN, case=False, na=False, regex=True)].copy()

    # --- Step 2: filter by skin_type ---
    pool = df[df['filters_json'].str.contains(skin_type, case=False, na=False)].copy()
    if pool.empty:
        pool = df.copy()

    # Shuffle for daily variety before any selection
    pool = pool.sample(frac=1, random_state=daily_seed)

    # --- Step 3: reserve pillar seats ---
    reserved_indices: set = set()
    pillar_picks: list[pd.DataFrame] = []
    for pillar, keywords in PILLAR_SLUG_KEYWORDS.items():
        kw_pattern = "|".join(keywords)
        pillar_pool = pool[
            pool['slug'].str.contains(kw_pattern, case=False, na=False) &
            ~pool.index.isin(reserved_indices)
        ]
        picks = pillar_pool.head(MIN_PER_PILLAR)
        reserved_indices.update(picks.index)
        pillar_picks.append(picks)

    reserved_df = pd.concat(pillar_picks, ignore_index=True) if pillar_picks else pd.DataFrame()

    # --- Step 4: score the remaining pool ---
    def match_score(row: pd.Series) -> int:
        fj = str(row.get('filters_json', '')).lower()
        score = 0
        for ing in required:
            tag = INGREDIENT_TAG_MAP.get(ing, ing)
            if tag.lower() in fj:
                score += 2
        for concern in concerns:
            if concern.lower() in fj:
                score += 1
        return score

    remainder = pool[~pool.index.isin(reserved_indices)].copy()
    remainder['_match_score'] = remainder.apply(match_score, axis=1)
    remainder = remainder.sort_values('_match_score', ascending=False)
    remainder = remainder.drop(columns=['_match_score'])

    # --- Step 5: combine reserved + scored remainder, cap at 40 ---
    fill_slots = max(0, 40 - len(reserved_df))
    result = pd.concat([reserved_df, remainder.head(fill_slots)], ignore_index=True)
    return result.head(40)


# ---------------------------------------------------------------------------
# Pass 1 — Skin brief synthesiser (direct genai call, no tool use needed)
# ---------------------------------------------------------------------------

def _score_labels(scores: dict) -> dict:
    """
    Translate raw 0-100 biomarker scores into human-readable clinical labels.
    ALL scores use the same convention: higher = healthier / less concern.
      - Sebum:    high score = well-regulated (not oily); low = excess sebum / oily
      - Hydration: high = well-hydrated; low = dehydrated
      - Pores:    high = refined pores; low = enlarged / congested
      - Lines:    high = smooth; low = visible fine lines / wrinkles
    """
    def label(val: int, high: str, mid: str, low: str) -> str:
        if val >= 70: return high
        if val >= 40: return mid
        return low

    return {
        "hydration": label(scores["hydration"], "well-hydrated", "mildly dehydrated", "dehydrated"),
        "sebum":     label(scores["sebum"],     "well-regulated sebum (not oily)", "moderate sebum activity", "excess sebum / oily"),
        "pores":     label(scores["pores"],     "refined pores", "mildly enlarged pores", "enlarged / congested pores"),
        "lines":     label(scores["lines"],     "smooth (few lines)", "mild fine lines", "visible fine lines / wrinkles"),
    }


async def run_pass1(spec: dict, scores: dict) -> dict:
    """
    Lightweight LLM call that turns the rule-engine output (spec) into a
    structured clinical skin brief consumed by Pass 2.
    Returns a dict with keys: skin_brief, ingredient_rationale, priority_concerns.
    """
    profiles_str   = ', '.join(spec['profiles'])   if spec['profiles']   else 'None'
    ingredients_str = ', '.join(spec['required_ingredients']) if spec['required_ingredients'] else 'None'
    concerns_str   = ', '.join(spec['skin_concerns']) if spec['skin_concerns'] else 'None'
    notes_str      = '; '.join(spec['notes'])       if spec['notes']      else 'None'
    labels = _score_labels(scores)

    prompt = f"""You are a clinical skin scientist. Given these structured skin biomarker rules:

Matched Treatment Profiles : {profiles_str}
Required Ingredients        : {ingredients_str}
Key Skin Concerns           : {concerns_str}
Safety Notes                : {notes_str}

Biomarker readings (scores are 0-100; higher = healthier / less concern):
  Skin Type : {scores['skin_type']}
  Hydration : {scores['hydration']}/100 → {labels['hydration']}
  Sebum     : {scores['sebum']}/100 → {labels['sebum']}
  Pores     : {scores['pores']}/100 → {labels['pores']}
  Lines     : {scores['lines']}/100 → {labels['lines']}

Output ONLY valid JSON with this exact schema — no markdown, no commentary:
{{
  "skin_brief": "<2-sentence clinical summary of the skin condition and primary treatment direction>",
  "beauty_advisor_narrative": "<3 sentences as a warm Beauty Advisor speaking directly to the client as 'you'. Sentence 1: describe what their skin is telling us right now using the biomarker scores. Sentence 2: explain what it needs most and why, referencing the triggered treatment profiles naturally. Sentence 3: reassure them that today's routine has been curated specifically for their skin.>",
  "ingredient_rationale": {{
    "<ingredient_name>": "<why this score demands this ingredient>"
  }},
  "priority_concerns": ["<ranked list of concerns, most urgent first>"]
}}"""

    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    response = await asyncio.to_thread(
        client.models.generate_content,
        model='gemini-2.0-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type='application/json',
            temperature=0.1,
        ),
    )
    try:
        return json.loads(response.text)
    except (json.JSONDecodeError, AttributeError):
        return {
            "skin_brief": response.text if hasattr(response, 'text') else "",
            "beauty_advisor_narrative": "",
            "ingredient_rationale": {},
            "priority_concerns": spec['skin_concerns'],
        }


# ---------------------------------------------------------------------------
# Pass 2 — Grounded routine recommender (ADK agent with web_specialist tool)
# ---------------------------------------------------------------------------

async def run_pass2(query: str) -> str:
    APP_NAME = "Sephora_Skincare_POC"
    USER_ID  = "customer_user"
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


# ---------------------------------------------------------------------------
# UI — STEP 1: Mandatory skin profile
# ---------------------------------------------------------------------------
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
                "sebum": sebum, "pores": pores, "lines": lines,
            })
            st.session_state.step = 2
            st.rerun()

# ---------------------------------------------------------------------------
# UI — STEP 2: Optional personalisation
# ---------------------------------------------------------------------------
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
            preferred_steps = st.selectbox("How many steps do you prefer?", [3, 4, 5, 6], index=1)
            customer_priority = st.text_input("Main Priority", placeholder="e.g. Hyperpigmentation")
            current_routine = st.text_area("Current Routine", placeholder="List products you use now...")
            other_preferences = st.text_area("Other Preferences", placeholder="Allergies, specific brands, etc.")

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
                    "preferred_steps": preferred_steps,
                })
            else:
                st.session_state.form_data.update({
                    "age": 25, "sun_exposure": "Medium", "preferred_steps": 4,
                })
            st.session_state.step = 3
            st.rerun()

# ---------------------------------------------------------------------------
# UI — STEP 3: 2-pass generation
# ---------------------------------------------------------------------------
elif st.session_state.step == 3:
    st.header("✨ Your Personalized Sephora Routine")
    d = st.session_state.form_data

    # ------------------------------------------------------------------
    # Rule engine (pure Python — no LLM)
    # ------------------------------------------------------------------
    scores = {
        "skin_type": d["skin_type"],
        "hydration": d["hydration"],
        "sebum":     d["sebum"],
        "pores":     d["pores"],
        "lines":     d["lines"],
        "age":       d.get("age", 25),
        "pregnancy": d.get("pregnancy", "No"),
    }
    spec = derive_ingredient_spec(scores)
    labels = _score_labels(scores)

    # Show triggered profiles to the user
    if spec["profiles"]:
        profile_labels = ", ".join(f"**{p}**" for p in dict.fromkeys(spec["profiles"]))
        st.info(f"Triggered treatment profiles: {profile_labels}")
    with st.expander("See ingredient grounding spec"):
        st.write("**Required ingredients:**", ", ".join(spec["required_ingredients"]) or "None")
        st.write("**Skin concerns:**",         ", ".join(spec["skin_concerns"])        or "None")
        if spec["notes"]:
            st.warning(" | ".join(spec["notes"]))

    # ------------------------------------------------------------------
    # Catalog load + spec-aware re-filtering (between passes)
    # ------------------------------------------------------------------
    with st.spinner("Matching catalog to your skin profile..."):
        try:
            catalog_df  = pd.read_csv('skincat.csv')
            refined_df  = filter_catalog_by_spec(catalog_df, d["skin_type"], spec)
            catalog_context = refined_df[['brand', 'Product']].to_string(index=False)
        except Exception as e:
            st.error(f"Catalog Error: {e}")
            catalog_context = "Sephora Inventory"

    # ------------------------------------------------------------------
    # Pass 1 — Skin brief synthesis
    # ------------------------------------------------------------------
    with st.spinner("Pass 1: Analysing your skin biomarkers..."):
        skin_brief = asyncio.run(run_pass1(spec, scores))

    advisor_narrative = skin_brief.get("beauty_advisor_narrative", "")
    if advisor_narrative:
        st.info(f"**Expert Analysis:** {advisor_narrative}")
        st.divider()

    # ------------------------------------------------------------------
    # Pass 2 — Grounded routine recommendation
    # ------------------------------------------------------------------
    priority_concerns = skin_brief.get("priority_concerns", spec["skin_concerns"])
    ingredient_rationale = skin_brief.get("ingredient_rationale", {})

    rationale_block = "\n".join(
        f"  - {ing}: {reason}" for ing, reason in ingredient_rationale.items()
    ) or "  (see required ingredients above)"

    safety_block = f"\nSAFETY NOTES: {chr(10).join(spec['notes'])}" if spec["notes"] else ""

    pass2_prompt = f"""
    ROLE: Senior Sephora Skincare Concierge.
    SOURCE: ONLY use products from this CATALOG:
    {catalog_context}

    SKIN_SPEC (grounding — follow strictly for every product selection):
    - Triggered Profiles   : {', '.join(dict.fromkeys(spec['profiles'])) or 'None'}
    - Required Ingredients : {', '.join(spec['required_ingredients']) or 'None'}
    - Priority Concerns    : {', '.join(priority_concerns) or 'None'}
    - Ingredient Rationale :
{rationale_block}{safety_block}

    USER DATA:
    - Skin: {d['skin_type']}
    - Biomarker readings (0-100, higher = healthier / less concern):
        Hydration {d['hydration']}/100 → {labels['hydration']}
        Sebum {d['sebum']}/100 → {labels['sebum']}
        Pores {d['pores']}/100 → {labels['pores']}
        Lines {d['lines']}/100 → {labels['lines']}
    - Sun Exposure: {d.get('sun_exposure')}
    - Current Routine: {d.get('current_routine')}
    - Preferences: {d.get('other_preferences')}
    - PREFERRED TOTAL STEPS: {d.get('preferred_steps')}

    INSTRUCTION: Build a routine using the 5 Sephora Pillars.
    MANDATORY: Every pillar (Cleanse, Treat, Moisturise, Finish, Boost) MUST have at least one product.
    No pillar may be left empty — if the catalog doesn't perfectly match a pillar, choose the best available fit and explain why it serves that step.
    Distribute the remaining products across pillars to reach the user's PREFERRED TOTAL STEPS ({d.get('preferred_steps')}).
    For each product, "Why" MUST explain how its key ingredients address the client's specific skin scores and concerns. Use warm, client-facing Beauty Advisor language. Never mention "SKIN_SPEC", "grounding spec", "triggered profiles", or any internal labels.

    [ROUTINE_START]
    ### 🧼 1. Cleanse
    **Purpose**: Remove impurities such as excess sebum, dirt, sweat, makeup and sunscreen.
    **Categories**: Makeup Remover, Facial Cleansers, Toners & Essence.

    ### 🧪 2. Treat
    **Purpose**: Address specific concerns - Correct, target, and rebalance skin.
    **Categories**: Exfoliators, Masks, Serum and Oils.

    ### 💧 3. Moisturise
    **Purpose**: Replenish, hydrate, and repair moisture barrier.
    **Categories**: Face/Eye/Lip Moisturisers & Hydrators.

    ### 🛡️ 4. Finish
    **Purpose**: Protect from sun damage, glow, blurring, and prepping.
    **Categories**: SPF, BB & CC Cream, Facial Mists.

    ### 🚀 5. Boost
    **Purpose**: Enhance daily routine or provide intensive care.
    **Categories**: Tools, Devices, Supplements.

    For each selected product, include:
    - **[Brand] – [Product Name]**
    - **Why**: [Explain warmly and directly how this product's key ingredients target the client's skin scores and concerns — as a Beauty Advisor speaking to them in person]
    - **Step by Step Method to Apply**: [2-3 detailed application tips in cohesion with other products]
    [ROUTINE_END]
    """

    with st.spinner("Pass 2: Formulating your grounded routine..."):
        result = asyncio.run(run_pass2(pass2_prompt))

        def extract(text, start, end):
            try:
                return text.split(start)[1].split(end)[0].strip()
            except Exception:
                return ""

        routine_out = extract(result, "[ROUTINE_START]", "[ROUTINE_END]")

        if routine_out:
            st.markdown(routine_out)
        else:
            st.write(result)

    if st.button("Start New Analysis"):
        st.session_state.step = 1
        st.rerun()
