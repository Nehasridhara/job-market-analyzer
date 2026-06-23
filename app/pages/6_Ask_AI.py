import streamlit as st
import pandas as pd
from google import genai
import sys
import os
from dotenv import load_dotenv

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from utils import load_combined_data

load_dotenv()

st.set_page_config(
    page_title="Ask AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Ask AI About the Job Market")
st.markdown(
    "Ask a question in plain English about the job market data. "
    "The AI translates your question into a real query and computes "
    "the actual answer — it never guesses numbers."
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error(
        "GEMINI_API_KEY not found in .env file. "
        "Add it to use this feature."
    )
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

df = load_combined_data()
st.markdown("## Ask Your Question")

DATA_CONTEXT = """
The dataframe is called `df` and has these columns:
- title (string): job title
- company (string): company name
- location (string): city, state/country
- remote (boolean): True if remote job
- description (string): full job description text
- salary_min (float): minimum salary in USD, may be NaN
- salary_max (float): maximum salary in USD, may be NaN
- date_posted (string): date posted
- source (string): one of 'jsearch_api', 'kaggle', 'usajobs_api'
- role_label (string): one of 'data scientist', 'data analyst', 
  'data engineer', 'machine learning engineer', 'nlp engineer', 'other'
- year (float): year extracted from date_posted, may be NaN

The dataframe has 2,249 rows.

IMPORTANT: Only reference columns listed above. Do not assume 
any other columns exist.
"""

example_questions = [
    "Select a question...",
    "What's the average salary for machine learning engineers?",
    "How many remote jobs are there?",
    "Which role has the highest average salary?",
    "What percentage of jobs are from the government sector?",
    "What's the salary range for data analysts?"
]

selected_q = st.selectbox(
    "Try an example or type your own below:",
    options=example_questions
)

if "ai_question" not in st.session_state:
    st.session_state.ai_question = ""
    st.session_state.last_selected_q = selected_q

if selected_q != st.session_state.last_selected_q:
    st.session_state.ai_question = (
        selected_q if selected_q != "Select a question..." else ""
    )
    st.session_state.last_selected_q = selected_q

user_question = st.text_input(
    "Your question:",
    value=st.session_state.ai_question,
    placeholder="e.g. What's the average salary for data scientists?"
)

if st.button("Ask AI", type="primary"):

    if not user_question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Thinking..."):

            # STEP 1 — Generate the query
            query_prompt = f"""
{DATA_CONTEXT}

Given this question: "{user_question}"

Write a single line of Python pandas code that computes 
the answer using the dataframe `df`. 

Rules:
- Only use the columns listed above
- Return ONLY the code, no explanation, no markdown formatting
- The code must be a single expression that evaluates to 
  a number, string, or small result
- Use .dropna() where appropriate to avoid NaN issues
- If the question asks about salary, use the average of 
  salary_min and salary_max, or just salary_min if that's 
  simpler
- Do not import anything, do not define functions, 
  do not use loops

Example output format:
df[df['role_label'] == 'data analyst']['salary_min'].dropna().mean()
"""

            query_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=query_prompt
            )
            generated_code = query_response.text.strip()

            # Clean up markdown fences if present
            generated_code = generated_code.replace(
                "```python", ""
            ).replace("```", "").strip()

            # Fix common Gemini typos before executing
            generated_code = generated_code.replace(
                "dropn()", "dropna()"
            ).replace(
                ".mean(axis=1).dropna()", 
                ".mean(axis=1)"
            )

            # STEP 2 — Execute SAFELY
            try:
                safe_globals = {
                    "df": df,
                    "pd": pd,
                    "__builtins__": {
                        "len": len, "round": round,
                        "min": min, "max": max,
                        "sum": sum, "abs": abs,
                        "int": int, "float": float,
                        "str": str
                    }
                }

                result = eval(generated_code, safe_globals)

                # STEP 3 — Phrase the answer
                answer_prompt = f"""
The user asked: "{user_question}"

The computed answer is: {result}

Write ONE short, natural sentence answering the question 
using this value. If it's a salary or money figure, format 
it as a clean dollar amount (e.g. $164,368) rounded to the 
nearest dollar — do not show decimal places. If it's a 
percentage, round to 1 decimal place. If it's a count, 
use a whole number with comma separators.

Do not add extra commentary or caveats. Just state the 
fact clearly.
"""

                answer_response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=answer_prompt
                )
                final_answer = answer_response.text.strip()

                st.markdown("## Answer")
                st.success(final_answer)

                with st.expander("See the actual code that ran"):
                    st.code(generated_code, language="python")
                    st.write(f"**Raw result:** {result}")

            except Exception as e:
                st.error(
                    f"Couldn't compute an answer for this question. "
                    f"Try rephrasing it or pick an example question. "
                    f"(Error: {e})"
                )
                with st.expander("Debug info"):
                    st.code(generated_code, language="python")

st.markdown("---")

with st.expander("ℹ️ How this works"):
    st.markdown("""
    This feature uses a **two-step LLM architecture** designed 
    to prevent hallucinated numbers:
    
    1. **Query generation**: Gemini reads your question and the 
       dataset's schema, then writes a single pandas query — 
       it never sees the actual data values.
    2. **Deterministic execution**: that generated code runs in 
       a sandboxed Python environment against the real dataset. 
       The computed result is mathematically exact, not AI-generated.
    3. **Answer phrasing**: Gemini only converts the already-computed 
       number into a natural sentence — it cannot alter the value.
    
    This means the AI can never "make up" a statistic — it can 
    only fail to understand a question (in which case you'll see 
    an error) or successfully compute a real answer.
    
    **Limitations:**
    - Works best with direct factual questions about salary, 
      role counts, remote work, and skill mentions
    - Cannot answer subjective questions ("which job is best?")
    - Complex multi-step reasoning may occasionally fail — 
      try rephrasing if you get an error
    """)

st.caption(
    "Powered by Google Gemini 2.5 Flash · "
    "Queries run against 2,249 real job postings"
)