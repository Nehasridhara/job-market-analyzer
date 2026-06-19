# 📊 US Data Science Job Market Analyzer

An end-to-end data science pipeline that scrapes, integrates, and analyzes real job postings to understand the 2025-2026 US data science, ML, and AI job market.

**Live demo:** *(https://job-market-analyzer-neha.streamlit.app)*

---

## What this project does

This isn't a single notebook — it's a complete pipeline covering every stage of a real data science project:

- **Multi-source data collection** — live job postings via JSearch API, government jobs via USAJobs API, and a Kaggle salary benchmark dataset, all integrated into one unified dataset of 2,249 jobs
- **NLP-based skill extraction** — dictionary-based named entity recognition identifies 60+ technical skills across job descriptions
- **Three trained ML models** — salary prediction (regression), job role classification (97% accuracy), and skill demand trend analysis covering 2020-2026
- **Interactive Streamlit dashboard** — 7 pages including EDA, skill co-occurrence networks, live salary prediction, live job classification, trend charts, and a natural-language query interface powered by Google Gemini

The meta-angle: this project uses data science to study the data science job market itself — analyzing what skills are actually in demand, how much they pay, and how that's changing over time.

---

## Key findings

- Python appears in **68.9%** of all data science job postings
- **PyTorch has overtaken TensorFlow** in job demand (21.1% vs 14.7%) and grew +135% from 2020 to 2026
- **LLM-related skills emerged from 0% to 3.3%** of postings over the analysis period
- **Seniority level — not any specific skill — is the strongest salary predictor**, more important than knowing any particular framework
- **76% of jobs cluster in DC/Virginia/Maryland**, reflecting the heavy government and defense contractor presence in data science hiring

---

## Pipeline overview

**Step 1 — Data collection**
Three sources scraped and integrated: JSearch API (live postings), USAJobs API (government jobs with guaranteed salary disclosure), Kaggle DS Salaries dataset (salary benchmark)

**Step 2 — NLP and skill extraction**
Text cleaning, dictionary-based NER across 60+ skills, co-occurrence mapping, LDA topic modeling

**Step 3 — Machine learning**

| Model | Method | Result |
|---|---|---|
| Salary prediction | Random Forest regression | R² = 0.289, RMSE = $47,290 |
| Job role classifier | XGBoost + TF-IDF | 97% accuracy, CV = 90.9% |
| Skill trend analysis | Time-series aggregation | PyTorch +135%, SQL -35% (2020-2026) |

**Step 4 — Dashboard and deployment**
Streamlit app with 7 interactive pages, Gemini-powered natural language query layer, deployed on Streamlit Community Cloud

---

## Tech stack

| Category | Tools |
|---|---|
| Data collection | Python, requests, JSearch API, USAJobs API |
| Data processing | pandas, NumPy, regex |
| NLP | spaCy, NLTK, gensim |
| Machine learning | scikit-learn, XGBoost |
| Visualization | Matplotlib, Seaborn, NetworkX |
| Dashboard | Streamlit |
| LLM integration | Google Gemini API |
| Deployment | Streamlit Community Cloud |

---

## Project structure

**app/** — Streamlit dashboard (streamlit_app.py, utils.py, pages/)

**data/** — raw combined dataset and cleaned outputs (charts, processed CSVs)

**models/** — trained model files (.pkl) for salary prediction and job classification

**notebooks/** — full analysis pipeline across 5 notebooks (scraping → NLP → ML → data integration → final models)

---

## How to run locally

```bash
git clone https://github.com/Nehasridhara/job-market-analyzer.git
cd job-market-analyzer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

To use the Ask AI page, create a `.env` file in the root folder:GEMINI_API_KEY=your_key_here

---

## The data integration story

Initial models trained only on 190 scraped JSearch postings performed poorly — salary model R² = -3.67 — because only 49 of those jobs disclosed salary, far too few for regression. Rather than treating this as a dead end, I integrated two additional sources, which required building a canonical data schema, handling three different date formats, fixing a deduplication bug that initially dropped 95% of Kaggle rows, and reconciling two different skill-measurement methodologies for the trend analysis. The final integrated dataset grew salary coverage from 25.8% to 93.7%, and the salary model's R² improved from -3.67 to +0.289.

---

## The AI query layer

The Ask AI page deliberately avoids letting the LLM compute statistics directly — that risks hallucinated numbers. Instead, Gemini reads a question and the dataset schema (not the data itself) and writes a single pandas query. That query runs in a sandboxed Python environment against the real dataset, producing a mathematically exact result. Gemini then only converts that already-computed number into a natural sentence. The system can fail to understand a question, but it cannot invent a statistic.

---

## Honest limitations

- Salary model R² = 0.289 — modest, primarily because the dataset spans both government and private-sector pay scales with fundamentally different distributions
- NLP Engineer classification has 50% recall due to severe class imbalance (29 samples vs 638 for Data Engineer)
- Skill trend analysis (2020-2023) uses title-based inference for Kaggle data since no descriptions were available — methodology boundary is marked transparently in the dashboard charts
- Dictionary-based skill extraction misses skills not in the predefined list; a future improvement would add a transformer-based NER layer (JobBERT)

---

## Future improvements

- Hybrid skill extraction combining dictionary NER with a fine-tuned transformer model
- Larger labeled salary dataset to reduce cross-source distribution shift
- Scheduled re-scraping to keep trend data current

---

## Author

Neha Sridhara 
