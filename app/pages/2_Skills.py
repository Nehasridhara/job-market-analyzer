#  Imports and setup
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from utils import load_skill_counts, load_cooccurrence

st.set_page_config(
    page_title="Skills Analysis",
    page_icon="🛠️",
    layout="wide"
)

st.title("🛠️ Skills Analysis")

# Top skills horizontal bar chart

skill_counts = load_skill_counts()

st.markdown("## Most In-Demand Skills")

top_n = st.slider(
    "Number of skills to show",
    min_value=5,
    max_value=25,
    value=15
)

top_skills = skill_counts.head(top_n)

fig, ax = plt.subplots(figsize=(10, top_n * 0.4))

colors = []
for skill in top_skills["skill"]:
    if skill in ["python", "r", "sql", "java", "scala", 
                  "sas", "stata", "matlab"]:
        colors.append("#4C72B0")
    elif skill in ["aws", "gcp", "azure", "docker", 
                    "kubernetes", "databricks", "snowflake"]:
        colors.append("#55A868")
    elif skill in ["machine learning", "deep learning", 
                    "statistics", "time series", "nlp", 
                    "computer vision", "regression",
                    "classification", "clustering", 
                    "neural network"]:
        colors.append("#C44E52")
    else:
        colors.append("#DD8452")

bars = ax.barh(
    top_skills["skill"][::-1],
    top_skills["percentage"][::-1],
    color=colors[::-1]
)

for bar, pct in zip(bars, top_skills["percentage"][::-1]):
    ax.text(
        bar.get_width() + 0.5,
        bar.get_y() + bar.get_height()/2,
        f"{pct}%", va="center", fontsize=9
    )

ax.set_xlabel("% of job postings mentioning this skill")
ax.set_title(f"Top {top_n} most in-demand skills")

st.pyplot(fig)

# Skill co-occurrence network
st.markdown("## Skill Co-occurrence Network")
st.markdown(
    "This network shows which skills frequently appear "
    "together in the same job posting. "
    "**Node size** = how common the skill is overall. "
    "**Edge thickness** = how often two skills appear together."
)

cooc_df = load_cooccurrence()

num_pairs = st.slider(
    "Number of skill connections to show",
    min_value=10,
    max_value=50,
    value=30
)

top_pairs = cooc_df.head(num_pairs)

G = nx.Graph()

for _, row in top_pairs.iterrows():
    G.add_edge(row["skill1"], row["skill2"], 
               weight=row["count"])

skill_freq = dict(zip(
    skill_counts["skill"], 
    skill_counts["count"]
))

node_sizes = [
    skill_freq.get(node, 10) * 15 
    for node in G.nodes()
]

edge_weights = [
    G[u][v]["weight"] * 0.4 
    for u, v in G.edges()
]

pos = nx.spring_layout(G, k=2, seed=42)

fig, ax = plt.subplots(figsize=(12, 9))

nx.draw_networkx_nodes(
    G, pos, node_size=node_sizes,
    node_color="#4C72B0", alpha=0.9, ax=ax
)

nx.draw_networkx_labels(
    G, pos, font_size=9,
    font_color="white", font_weight="bold", ax=ax
)

nx.draw_networkx_edges(
    G, pos, width=edge_weights,
    alpha=0.4, edge_color="#333333", ax=ax
)

ax.axis("off")
st.pyplot(fig)

# Searchable co-occurrence table
st.markdown("## Explore Skill Pairs")

col1, col2 = st.columns(2)

with col1:
    selected_skill = st.selectbox(
        "Select a skill to see what it pairs with:",
        options=sorted(skill_counts["skill"].unique())
    )

with col2:
    st.write("")  # spacing
    st.write("")

# Filter co-occurrence data for selected skill
filtered = cooc_df[
    (cooc_df["skill1"] == selected_skill) |
    (cooc_df["skill2"] == selected_skill)
].copy()

# Create a clean "paired with" column
filtered["paired_with"] = filtered.apply(
    lambda row: row["skill2"] 
    if row["skill1"] == selected_skill 
    else row["skill1"],
    axis=1
)

display_df = filtered[["paired_with", "count"]].sort_values(
    "count", ascending=False
).reset_index(drop=True)

display_df.columns = ["Pairs With", "Jobs Together"]

st.markdown(f"### Skills that pair with `{selected_skill}`")

if len(display_df) > 0:
    st.dataframe(display_df, use_container_width=True)
else:
    st.info(
        f"No co-occurrence data found for {selected_skill} "
        f"in the top pairs. Try increasing the slider above."
    )

