# beautyscan-playground

A Streamlit-based proof-of-concept for personalized skincare routine recommendations powered by Google Vertex AI and the ADK agent framework.

## How It Works

The app uses a **2-pass recommendation system**:

### Pass 1 — Skin Brief Synthesis
A direct Gemini call (no tool use) that transforms rule-engine output into a structured expert analysis:
- **Strengths**: Highlights what's working well in the skin profile
- **Areas to Address**: Empathetic breakdown of the top 1–2 concern areas
- **Routine Plan**: Skin pillars, key ingredients, and rationale

### Pass 2 — Grounded Routine Recommendation
An ADK agent (`Skincare_Routine_Generator`) that selects products exclusively from the filtered Sephora catalog and builds a routine across **5 Sephora Pillars**:
1. 🧼 **Cleanse** — Remove impurities (cleansers, toners, makeup removers)
2. 🧪 **Treat** — Target concerns (serums, exfoliators, masks)
3. 💧 **Moisturise** — Repair moisture barrier (moisturisers, eye creams)
4. 🛡️ **Finish** — Sun protection and priming (SPF, BB/CC creams, mists)
5. 🚀 **Boost** — Intensive care (tools, devices, supplements)

A sub-agent (`web_specialist`) is called for pregnancy/breastfeeding safety checks via Google Search.

## Rule Engine — Grounding Rules

The `grounding_rules.py` module maps biomarker scores to treatment profiles before any LLM call:

| Profile | Trigger Condition | Key Ingredients |
|---|---|---|
| **Barrier Boost** | Dry skin, hydration < 60, sebum > 75 | Hyaluronic Acid |
| **Firm Restore** | Lines score < 71 | Peptides, Squalane, Retinoid, Collagen |
| **Blemish Control** | Pores score < 75 | BHA/Salicylic Acid, Niacinamide |
| **Blemish Control** | Oily skin, sebum < 75 | BHA/Salicylic Acid, Niacinamide |
| **Radiance Reset** | Dry skin, hydration < 80, sebum > 14 | AHA/Glycolic Acid, Vitamin C, Ceramides |

Pregnancy safety: Retinoids are automatically removed from the ingredient spec and a safety note is passed to both passes.

## Catalog Filtering

Between Pass 1 and Pass 2, `skincat.csv` is filtered and scored:
1. Exclude gift sets, kits, duos, and sample-sized products
2. Filter by skin type
3. Reserve at least 3 products per pillar (slug-based keyword matching)
4. Score remaining products by ingredient/concern alignment (+2 per required ingredient match, +1 per concern match)
5. Return up to 40 products to the agent

## App Flow

**Step 1 — Skin Profile (Mandatory)**
- Skin type: Dry / Oily / Combination / Normal
- Biomarker sliders (0–100, higher = healthier): Hydration, Sebum, Pores, Lines

**Step 2 — Personalise (Optional)**
- Pregnancy / breastfeeding status
- Preferred routine steps (3–6)
- Client concerns multiselect (up to 3): Ageing, Blackheads, Dark Circles, Dullness, Firmness & Elasticity, Pigmentation & Dark Spots, Puffiness, Uneven Skin Texture, Uneven Skin Tone, Redness

**Step 3 — Results**
- Expert analysis panel (strengths + focus areas + routine plan)
- Full personalized routine with per-product Why and application steps

## Setup

### Requirements
```
streamlit
pandas
google-adk
google-genai
google-cloud-aiplatform
asyncio
google-auth
```

Install with:
```bash
pip install -r requirements.txt
```

### GCP Configuration

Set environment variables or use a `secrets.toml` file for service account credentials:

```toml
# .streamlit/secrets.toml
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "..."
private_key = "..."
client_email = "..."
...
```

Or set environment variables:
```bash
export GCP_PROJECT_ID="your-project-id"
export GCP_LOCATION="us-central1"
```

The model can be overridden via:
```bash
export SKINCARE_MODEL="gemini-2.0-flash"  # default: gemini-2.5-flash-lite
```

### Run

```bash
streamlit run app.py
```

Place `skincat.csv` in the same directory as `app.py`.
