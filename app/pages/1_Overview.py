# Imports and setup
import streamlit as st
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from utils import load_combined_data

st.set_page_config(
    page_title="Overview",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Dataset Overview")

# Load data and show role distribution
df = load_combined_data()

st.markdown("## Jobs by Role")

role_counts = df["source"].value_counts()

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(role_counts.index, role_counts.values, 
       color=["#4C72B0", "#55A868", "#C44E52"])
ax.set_ylabel("Number of jobs")
ax.set_title("Jobs by data source")

for i, count in enumerate(role_counts.values):
    ax.text(i, count + 20, str(count), ha="center")

st.pyplot(fig)

# Remote work and salary charts side by side
st.markdown("## Remote Work & Salary Distribution")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Remote vs On-site")
    
    remote_counts = df["remote"].value_counts()
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        remote_counts.values,
        labels=["On-site" if not x else "Remote" 
                for x in remote_counts.index],
        autopct="%1.1f%%",
        colors=["#C44E52", "#55A868"],
        startangle=90
    )
    st.pyplot(fig)

with col2:
    st.markdown("### Salary Distribution")
    
    salary_data = df["salary_min"].dropna()
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(salary_data, bins=30, 
            color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Salary (USD)")
    ax.set_ylabel("Number of jobs")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"${x/1000:.0f}k")
    )
    st.pyplot(fig)

# Top locations and source summary table
st.markdown("## Geographic Distribution")

col1, col2 = st.columns([2, 1])

with col1:
    # Extract state from location for US-based jobs
    state_pattern = df["location"].str.extract(
        r",\s*([A-Z]{2})\b"
    )[0]
    
    top_states = state_pattern.value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(top_states.index, top_states.values, 
           color="#4C72B0")
    ax.set_title("Top 10 states by job count")
    ax.set_ylabel("Number of jobs")
    
    for i, count in enumerate(top_states.values):
        ax.text(i, count + 5, str(count), ha="center")
    
    st.pyplot(fig)

with col2:
    st.markdown("### Data Source Breakdown")
    
    source_summary = df.groupby("source").agg(
        total_jobs=("title", "count"),
        with_salary=("salary_min", lambda x: x.notna().sum()),
        avg_salary=("salary_min", "mean")
    ).round(0)
    
    source_summary.columns = [
        "Total Jobs", "With Salary", "Avg Salary"
    ]
    source_summary["Avg Salary"] = source_summary[
        "Avg Salary"
    ].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(source_summary, use_container_width=True)

st.markdown("---")
st.caption(
    f"Dataset last updated: based on {len(df):,} job postings "
    f"from JSearch API, USAJobs API, and Kaggle DS Salaries dataset"
)