#  Imports and load model
import streamlit as st
import numpy as np
from scipy.sparse import hstack, csr_matrix
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from utils import load_salary_model

st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Salary Predictor")
st.markdown(
    "Estimate a data science salary based on job title, "
    "seniority, and other factors. "
    "This model was trained on 2,108 real job postings "
    "with **R² = 0.289**."
)

model, tfidf, feature_info = load_salary_model()

# Input widgets
st.markdown("## Enter Job Details")

col1, col2 = st.columns(2)

with col1:
    job_title = st.text_input(
        "Job Title",
        value="Senior Data Scientist",
        help="Try different seniority levels: Junior, Mid, Senior, Principal"
    )
    
    job_family = st.selectbox(
        "Job Family",
        options=[
            "data_scientist", "ml_engineer", 
            "data_analyst", "data_engineer", 
            "nlp_engineer", "other"
        ],
        format_func=lambda x: x.replace("_", " ").title()
    )

with col2:
    source = st.selectbox(
        "Sector",
        options=["jsearch_api", "kaggle", "usajobs_api"],
        format_func=lambda x: {
            "jsearch_api": "Private sector (live postings)",
            "kaggle": "Private sector (industry benchmark)",
            "usajobs_api": "US Government"
        }[x]
    )
    
    is_remote = st.checkbox("Remote position")

# Build features and predict
from utils import extract_seniority

if st.button("Predict Salary", type="primary"):
    
    # Feature 1 — seniority from job title
    seniority = extract_seniority(job_title)
    
    # Feature 2 — job family encoding
    family_map = {
        "ml_engineer": 0,
        "data_scientist": 1,
        "data_analyst": 2,
        "data_engineer": 3,
        "nlp_engineer": 4,
        "other": 5
    }
    family_encoded = family_map[job_family]
    
    # Feature 3 — source encoding
    source_map = {
        "jsearch_api": 0,
        "kaggle": 1,
        "usajobs_api": 2
    }
    source_encoded = source_map[source]
    
    # Feature 4 — remote flag
    is_remote_encoded = 1 if is_remote else 0
    
    # Build structured features array
    structured = np.array([[
        seniority, family_encoded, 
        source_encoded, is_remote_encoded
    ]])
    
    # Build TF-IDF features from title
    title_features = tfidf.transform([job_title])
    
    # Combine structured + TF-IDF
    X_input = hstack([
        csr_matrix(structured),
        title_features
    ]).toarray()
    
    # Predict
    prediction = model.predict(X_input)[0]
    
    # Display result
    st.markdown("## Predicted Salary")
    st.metric(
        "Estimated Annual Salary",
        f"${prediction:,.0f}"
    )
    
    st.markdown(f"""
    **Based on:**
    - Seniority level: **{['', 'Entry', 'Mid', 'Senior', 'Principal'][seniority]}** 
      (detected from "{job_title}")
    - Job family: **{job_family.replace('_', ' ').title()}**
    - Sector: **{source}**
    - Remote: **{'Yes' if is_remote else 'No'}**
    
    *Note: RMSE = $47,290 — predictions can vary by roughly 
    this amount due to the wide salary range across 
    government and private sector roles.*
    """)


