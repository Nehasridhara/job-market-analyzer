```markdown
# 📊 US Data Science Job Market Analyzer

An end-to-end data science pipeline that scrapes, integrates, and analyzes real job postings to understand the 2025-2026 US data science, ML, and AI job market — built as a portfolio project demonstrating the full stack from data engineering through deployed ML models.

**[Live Demo](#)** *(link added after deployment)*

---

## What this project does

This isn't a single notebook — it's a complete pipeline:

1. **Multi-source data collection** — live job postings via JSearch API, government jobs via USAJobs API, and a Kaggle salary benchmark dataset, integrated into one unified dataset
2. **NLP-based skill extraction** — dictionary-based NER identifies 60+ technical skills from raw job descriptions
3. **Three trained ML models** — salary prediction (regression), job role classification (97% accuracy), and skill demand trend analysis (2020-2026)
4. **Interactive Streamlit dashboard** — 7 pages covering EDA, skill networks, live salary prediction, live job classification, trend visualization, and a natural-language query interface powered by Gemini

## The meta-angle

This project uses data science to study the data science job market itself — analyzing what skills are actually in demand, how much they pay, and how that's changing over time, using real scraped and integrated data rather than a single static dataset.

---

## Key findings

- **Python appears in 68.9%** of all data science job postings — by far the most in-demand skill
- **PyTorch has overtaken TensorFlow** in job demand (21.1% vs 14.7%), and grew **+135%** from 2020 to 2026
- **LLM-related skills emerged from 0% to 3.3%** of postings in just a few years
- **Seniority level (not any specific skill) is the strongest salary predictor** — more important than knowing any particular framework
- **76% of jobs in this dataset cluster in the DC/Virginia/Maryland region**, reflecting the heavy government and defense contractor presence in data science hiring

---

## Architecture

```
JSearch API ─┐
USAJobs API ─┼─→ Data Integration → 2,249 unified job postings
Kaggle CSV ──┘            ↓
                  Skill Extraction (NLP)
                           ↓
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
  Salary Model      Job Classifier      Trend Analysis
  (Random Forest)      (XGBoost)         (2020-2026)
  R² = 0.289         97% accuracy      PyTorch +135%
        ↓                  ↓                  ↓
        └──────────────────┼──────────────────┘
                           ↓
              Streamlit Dashboard (7 pages)
                           ↓
              Gemini-powered NL query layer
```

---

## Tech stack

**Data collection:** Python, requests, JSearch API (RapidAPI), USAJobs API
**Data processing:** pandas, NumPy, regex
**NLP:** spaCy, NLTK, gensim (LDA topic modeling)
**Machine learning:** scikit-learn, XGBoost, TF-IDF vectorization
**Visualization:** Matplotlib, Seaborn, NetworkX (skill co-occurrence graphs)
**Dashboard:** Streamlit
**LLM integration:** Google Gemini API (natural language query layer)
**Deployment:** Streamlit Community Cloud

---

## Project structure

```
job-market-analyzer/
├── app/                      # Streamlit dashboard
│   ├── streamlit_app.py      # Home page
│   ├── utils.py               # Shared data/model loaders
│   └── pages/
│       ├── 1_Overview.py
│       ├── 2_Skills.py
│       ├── 3_Salary_Predictor.py
│       ├── 4_Job_Classifier.py
│       ├── 5_Trends.py
│       └── 6_Ask_AI.py
├── data/
│   ├── raw/                   # Combined dataset
│   └── cleaned/                # EDA outputs, charts, model artifacts
├── models/                     # Trained model files (.pkl)
├── notebooks/                  # Full analysis pipeline, phase by phase
│   ├── 01_scraping.ipynb
│   ├── 02_nlp.ipynb
│   ├── 03_ml_models.ipynb
│   ├── 04_data_integration.ipynb
│   └── 05_ml_models_v2.ipynb
└── requirements.txt
```

---

## How to run locally

```bash
git clone https://github.com/YOUR_USERNAME/job-market-analyzer.git
cd job-market-analyzer
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

To use the "Ask AI" natural language query page, create a `.env` file in the root directory:
```
GEMINI_API_KEY=your_key_here
```

---

## The data integration story

Initial models trained only on the 190 directly-scraped JSearch postings performed poorly (salary model R² = -3.67) due to only 49 salary-labeled examples — far too few for reliable regression. Rather than treating this as a dead end, I integrated two additional sources:

- **USAJobs API** — 484 US government postings, 100% salary disclosure (legally required)
- **Kaggle DS Salaries dataset** — 3,028 jobs filtered to US-only, used as a salary benchmark

This required building a canonical data schema, handling three different date formats, resolving a deduplication bug that initially dropped 95% of Kaggle rows, and reconciling two fundamentally different skill-measurement methodologies (title-based vs. description-based) for the trend analysis. The final integrated dataset grew salary coverage from 25.8% to 93.7%, and the salary model's R² improved from -3.67 to +0.289.

## The AI query layer — architecture note

The "Ask AI" page deliberately avoids letting an LLM compute statistics directly, since that risks hallucinated numbers. Instead:

1. Gemini reads the question + a description of the dataset schema (not the data itself) and generates a single pandas query
2. That query executes in a sandboxed Python environment against the real dataset — the result is mathematically exact, not AI-generated
3. Gemini only converts the already-computed number into a natural sentence

This means the system can fail to understand a question, but it cannot invent a statistic.

## Honest limitations

- **Salary model R² = 0.289** — modest, primarily because the integrated dataset spans both government and private-sector pay scales with fundamentally different distributions
- **NLP Engineer classification** is weakest (50% recall) due to severe class imbalance — only 29 samples vs 638 for Data Engineer
- **Skill trend analysis (2020-2023)** relies on title-based skill inference for Kaggle data (no real descriptions available), normalized against description-based extraction for 2026 data — methodology is documented transparently in the dashboard
- **Dictionary-based skill extraction** can miss skills not in the predefined list or described using non-standard terminology; a future improvement would add a transformer-based NER model (e.g. JobBERT) as a complementary layer

## Future improvements

- Hybrid skill extraction combining dictionary NER with a fine-tuned transformer model
- Larger labeled salary dataset to reduce cross-source distribution shift
- Scheduled re-scraping to keep trend data current beyond 2026

## Author

Neha Sridhara 
