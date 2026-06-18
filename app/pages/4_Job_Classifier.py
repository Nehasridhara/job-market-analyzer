# Imports and setup
import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from utils import load_classifier

st.set_page_config(
    page_title="Job Classifier",
    page_icon="🏷️",
    layout="wide"
)

st.title("🏷️ Job Role Classifier")
st.markdown(
    "Paste a job description below and the model will predict "
    "which of 5 data science roles it belongs to. "
    "This XGBoost model achieved **97% accuracy** on test data."
)

model, tfidf, le = load_classifier()

# Text input and prediction
st.markdown("## Paste a Job Description")

sample_descriptions = {
    "Select an example...": "",
    "Data Analyst example": 
        "Looking for a data analyst to create dashboards in Tableau, "
        "write SQL queries, and analyze business metrics. "
        "Experience with Excel and reporting required.",
    "ML Engineer example":
        "We need a machine learning engineer to build and deploy "
        "deep learning models using PyTorch and TensorFlow. "
        "Experience with Docker, Kubernetes, and MLOps pipelines required.",
    "NLP Engineer example":
        "Seeking an NLP engineer to work on transformer models, "
        "BERT, and large language models for text classification "
        "and named entity recognition tasks."
}

selected_example = st.selectbox(
    "Try an example, or select 'Select an example...' "
    "and type your own below:",
    options=list(sample_descriptions.keys())
)

# Only update text area when a NEW example is selected
if "last_example" not in st.session_state:
    st.session_state.last_example = selected_example
    st.session_state.description_text = sample_descriptions[selected_example]

if selected_example != st.session_state.last_example:
    st.session_state.description_text = sample_descriptions[selected_example]
    st.session_state.last_example = selected_example

description_input = st.text_area(
    "Job description text",
    value=st.session_state.description_text,
    height=200,
    key="description_text_area",
    placeholder="Paste a job description here..."
)

if st.button("Classify Job", type="primary"):
    
    if len(description_input.strip()) < 20:
        st.warning(
            "Please enter a longer job description "
            "for accurate classification."
        )
    else:
        # Transform text to TF-IDF features
        X_input = tfidf.transform([description_input])
        
        # Predict
        prediction_encoded = model.predict(X_input)[0]
        prediction_label = le.inverse_transform(
            [prediction_encoded]
        )[0]
        
        # Get probability for each class
        probabilities = model.predict_proba(X_input)[0]
        
        st.markdown("## Prediction Result")
        
        st.success(
            f"### Predicted Role: **{prediction_label.title()}**"
        )
        
        # Show probability breakdown
        prob_df = pd.DataFrame({
            "Role": [r.title() for r in le.classes_],
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)
        
        prob_df["Probability"] = prob_df["Probability"].apply(
            lambda x: f"{x*100:.1f}%"
        )
        
        st.markdown("### Confidence breakdown")
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
