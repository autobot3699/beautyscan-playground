import streamlit as st
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image

# --- CONFIGURATION ---
PROJECT_ID = "your-google-cloud-project-id"  # <--- CHANGE THIS
LOCATION = "us-central1" 
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-1.5-pro-002")

st.set_page_config(page_title="Skincare AI Scan", layout="wide")
st.title("📸 AI Skin Scan & Recommendation")

# --- 1. DATA LOADING ---
@st.cache_data
def load_data():
    return pd.read_csv('skincat.csv')

df = load_data()

# --- 2. SIDEBAR: CUSTOMER DATA ---
with st.sidebar:
    st.header("Profile")
    location = st.selectbox("Location", ["sg", "my", "th", "au", "ph", "id"])
    age = st.number_input("Age", 18, 100, 26)
    st.divider()
    st.info("Upload a clear photo of your skin to begin analysis.")

# --- 3. MAIN UI: IMAGE UPLOAD ---
uploaded_file = st.file_uploader("Upload Skin Scan (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    st.image(uploaded_file, caption="Skin Scan Uploaded", width=300)
    
    if st.button("Analyze & Generate Routine"):
        with st.spinner("Gemini is analyzing your skin texture..."):
            
            # Convert uploaded file to Vertex AI Image format
            image_bytes = uploaded_file.getvalue()
            skin_image = Part.from_data(data=image_bytes, mime_type="image/jpeg")
            
            # Filter catalog for geography
            catalog_snippet = df[~df['unavailableCountries'].str.contains(location, na=False)].to_string()

            # MULTIMODAL PROMPT (Guidelines + Image + Data)
            prompt = f"""
            SYSTEM GUIDELINES:
            1. Analyze the attached image for: Dryness/Flakiness, Pore Visibility, and Sensitivity.
            2. For Dry skin (Age {age}): Prioritize lipid-rich creams and non-foaming cleansers.
            3. For High Pores: Select products with PHAs or gentle enzymes from the catalog.
            4. Geography: User is in {location}.
            
            CATALOG DATA:
            {catalog_snippet}
            
            TASK:
            Output a professional analysis of the skin in the photo. 
            Then, provide a 3-step routine (Cleanse, Treat, Moisturize) using ONLY products from the catalog.
            Explain WHY each product was chosen based on the visual evidence in the photo.
            """

            # Call Gemini
            response = model.generate_content([prompt, skin_image])
            
            st.success("Analysis Complete!")
            st.markdown(response.text)

else:
    st.warning("Please upload a photo to get a personalized recommendation.")