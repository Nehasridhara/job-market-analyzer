import pandas as pd
import streamlit as st

@st.cache_data
def load_combined_data():
    return pd.read_csv("data/raw/jobs_combined.csv")

@st.cache_data
def load_skills_data():
    df = pd.read_csv("data/cleaned/jobs_with_skills.csv")
    df["skills"] = df["skills_str"].apply(
        lambda x: x.split("|") if isinstance(x, str) and x != "" else []
    )
    return df

@st.cache_data
def load_skill_counts():
    return pd.read_csv("data/cleaned/skill_counts.csv")

@st.cache_data
def load_cooccurrence():
    return pd.read_csv("data/cleaned/skill_cooccurrence.csv")

@st.cache_data
def load_trends():
    return pd.read_csv("data/cleaned/skill_trends.csv")

@st.cache_data
def load_growth():
    return pd.read_csv("data/cleaned/skill_growth.csv")

import pickle
import json

@st.cache_resource
def load_salary_model():
    with open("models/salary_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("models/feature_info.json", "r") as f:
        info = json.load(f)
    return model, tfidf, info

@st.cache_resource
def load_classifier():
    with open("models/job_classifier.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/tfidf_classifier.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, tfidf, le

import re

SKILLS = {
    "python", "sql", "r", "java", "scala",
    "matlab", "javascript", "sas", "stata",
    "tensorflow", "pytorch", "keras",
    "scikit-learn", "xgboost", "lightgbm",
    "huggingface", "transformers",
    "spark", "hadoop", "kafka", "airflow",
    "dbt", "pandas", "numpy", "dask",
    "postgresql", "mysql", "mongodb",
    "snowflake", "bigquery", "redshift",
    "databricks", "aws", "gcp", "azure",
    "docker", "kubernetes", "mlflow",
    "nlp", "bert", "gpt", "llm", "spacy",
    "langchain", "word2vec", "embeddings",
    "tableau", "power bi", "matplotlib",
    "plotly", "looker",
    "machine learning", "deep learning",
    "neural network", "regression",
    "classification", "clustering",
    "computer vision", "time series",
    "a/b testing", "statistics",
    "git", "github", "agile", "excel"
}

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[•·▪▸●◦]", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s\+\#\/\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_skills(text):
    cleaned = clean_text(text)
    found = []
    for skill in SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, cleaned):
            found.append(skill)
    return found

def extract_seniority(title):
    title_lower = str(title).lower()
    if any(w in title_lower for w in
           ["principal", "staff", "distinguished"]):
        return 4
    elif any(w in title_lower for w in
             ["senior", "sr", "lead", "manager"]):
        return 3
    elif any(w in title_lower for w in
             ["mid", "intermediate", "ii", "2"]):
        return 2
    elif any(w in title_lower for w in
             ["junior", "jr", "entry", "associate"]):
        return 1
    else:
        return 2
    
    