# Imports, setup, and load data
import streamlit as st
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from utils import load_trends, load_growth

st.set_page_config(
    page_title="Skill Trends",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Skill Demand Trends (2020-2026)")

st.markdown("""
This analysis tracks how skill demand has changed over time, 
combining:
- **2020-2023**: Kaggle salary dataset (title-based skill inference)
- **2026**: Live job postings (description-based skill extraction)

⚠️ *The dashed line in charts marks where the data 
source/methodology changes — see README for details.*
""")

trend_matrix = load_trends()
growth_df = load_growth()

# Rename the unnamed index column to "year"
trend_matrix = trend_matrix.rename(
    columns={trend_matrix.columns[0]: "year"}
)
trend_matrix = trend_matrix.set_index("year")

# Trend line charts with skill selector
st.markdown("## Skill Demand Over Time")

all_skills = trend_matrix.columns.tolist()

selected_skills = st.multiselect(
    "Select skills to compare:",
    options=all_skills,
    default=["python", "machine learning", "llm", "pytorch"]
)

if len(selected_skills) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
              "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]
    
    for i, skill in enumerate(selected_skills):
        values = trend_matrix[skill].dropna()
        
        ax.plot(
            values.index.astype(int),
            values.values,
            marker="o",
            linewidth=2.5,
            markersize=6,
            color=colors[i % len(colors)],
            label=skill
        )
        
        ax.annotate(
            f"{values.values[-1]:.1f}%",
            xy=(values.index[-1], values.values[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=9,
            color=colors[i % len(colors)],
            fontweight="bold"
        )
    
    # Mark methodology change point
    ax.axvline(x=2024, color="gray", 
               linestyle="--", alpha=0.5, linewidth=1)
    ax.text(2024.1, ax.get_ylim()[1] * 0.9,
            "← Kaggle | JSearch/USAJobs →",
            fontsize=8, color="gray")
    
    ax.set_xlabel("Year")
    ax.set_ylabel("% of job postings")
    ax.set_title("Skill demand trends")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
else:
    st.info("Select at least one skill to see the trend chart.")

# Growth rates
st.markdown("## Skill Growth Rates (2020 → 2026)")

st.markdown(
    "Which skills are gaining or losing relative demand "
    "over the full period?"
)

# Sort by growth
growth_sorted = growth_df.sort_values(
    "growth", ascending=True
)

fig, ax = plt.subplots(figsize=(10, 7))

colors = [
    "#55A868" if g > 0 else "#C44E52" 
    for g in growth_sorted["growth"]
]

bars = ax.barh(
    growth_sorted["skill"],
    growth_sorted["growth"],
    color=colors
)

ax.axvline(x=0, color="black", 
           linewidth=0.8, linestyle="--")
ax.set_xlabel("Growth rate (%)")
ax.set_title(
    "Skill demand growth rate (2020 → 2026)\n"
    "Green = growing, Red = declining"
)

for bar, val in zip(bars, growth_sorted["growth"]):
    xpos = bar.get_width() + 1 if val >= 0 else bar.get_width() - 1
    align = "left" if val >= 0 else "right"
    ax.text(
        xpos, bar.get_y() + bar.get_height()/2,
        f"{val:+.0f}%", va="center", ha=align, fontsize=9
    )

st.pyplot(fig)

# Summary table
st.markdown("### Detailed growth data")
st.dataframe(
    growth_df.sort_values("growth", ascending=False),
    use_container_width=True,
    hide_index=True
)

# Key takeaways
st.markdown("## Key Takeaways")

top_growing = growth_df.sort_values(
    "growth", ascending=False
).iloc[0]
top_declining = growth_df.sort_values(
    "growth", ascending=True
).iloc[0]

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"**Fastest growing:** {top_growing['skill'].title()} "
        f"({top_growing['growth']:+.0f}%)"
    )

with col2:
    st.error(
        f"**Fastest declining:** {top_declining['skill'].title()} "
        f"({top_declining['growth']:+.0f}%)"
    )

