import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_combined_data

st.set_page_config(
    page_title="Data Science Job Market Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Science Job Market Analyzer")

st.markdown("""
Welcome! This dashboard analyzes the **2025-2026 US data science job market** 
using data from multiple sources:

- **JSearch API** — live job postings from LinkedIn, Indeed, Glassdoor
- **USAJobs API** — US government job postings
- **Kaggle DS Salaries dataset** — salary benchmarks

### What you'll find in this app

👈 Use the sidebar to navigate between pages:

- **Overview** — dataset statistics, role distribution, remote work trends
- **Skills** — most in-demand skills and how they connect
- **Salary Predictor** — get a salary estimate for a job profile
- **Job Classifier** — paste a job description, get the predicted role
- **Trends** — how skill demand has changed 2020-2026
""")

df = load_combined_data()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Jobs Analyzed", f"{len(df):,}")

with col2:
    salary_count = df["salary_min"].notna().sum()
    st.metric("Jobs with Salary Data", f"{salary_count:,}")

with col3:
    avg_salary = df["salary_min"].dropna().mean()
    st.metric("Average Salary", f"${avg_salary:,.0f}")

with col4:
    sources = df["source"].nunique()
    st.metric("Data Sources", sources)

st.markdown("---")
st.markdown("Built with Python, scikit-learn, XGBoost, and Streamlit")

