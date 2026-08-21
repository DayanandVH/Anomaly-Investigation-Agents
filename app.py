"""
Streamlit demo UI for the anomaly-investigation agent.

Run with:
    streamlit run app.py

This is a thin UI layer over Weeks 1-3 (detector, knowledge base, agent) —
no new logic lives here, it just wires the pipeline up for interactive use.
"""

import os
import sys

# Let this file import from src/ regardless of where streamlit is launched from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
from dataclasses import asdict

from data import load_data
from detector import AnomalyDetector
from knowledge_base import build_knowledge_base
from agent import build_agent


st.set_page_config(page_title="Anomaly Investigation Agent", page_icon="🔍", layout="wide")


# --- Cached setup: these are expensive (model fitting, embeddings, DB connections)
# so we only want to do them ONCE per session, not on every button click/interaction.
# st.cache_resource is for objects (models, connections) that should be reused as-is.

@st.cache_resource
def get_data_and_detector():
    df, source = load_data()
    feature_cols = [c for c in df.columns if c != "Class"]
    detector = AnomalyDetector(contamination=0.02).fit(df, feature_cols)
    flags = detector.score(df)
    return df, source, flags


@st.cache_resource
def get_knowledge_base():
    return build_knowledge_base()


@st.cache_resource
def get_agent(_kb, llm_choice: str, groq_api_key: str = ""):
    # Leading underscore on _kb tells Streamlit's cache not to try hashing that
    # argument (Chroma collections aren't easily hashable) — cache keys off the
    # other args (llm_choice, groq_api_key) instead.
    if llm_choice == "Groq (cloud, free tier)":
        from langchain_groq import ChatGroq
        os.environ["GROQ_API_KEY"] = groq_api_key
        llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    else:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="llama3.1:8b", temperature=0)
    return build_agent(_kb, llm)


# --- Sidebar: backend selection ---
st.sidebar.title("Settings")
llm_choice = st.sidebar.radio(
    "LLM backend",
    ["Groq (cloud, free tier)", "Ollama (local)"],
    help="Groq needs a free API key from console.groq.com. Ollama needs a local server running.",
)

groq_api_key = ""
if llm_choice == "Groq (cloud, free tier)":
    groq_api_key = st.sidebar.text_input(
        "GROQ_API_KEY",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Get a free key at console.groq.com/keys",
    )

# --- Main content ---
st.title("🔍 Anomaly Investigation Agent")
st.caption(
    "Flags anomalous transactions, retrieves relevant fraud patterns via RAG, "
    "and asks an LLM to explain the flag and recommend an action."
)

df, source, flags = get_data_and_detector()

if source == "synthetic":
    st.info(
        "Using synthetic demo data (no real dataset found at data/creditcard.csv). "
        "Drop the Kaggle Credit Card Fraud dataset there to use real data instead.",
        icon="ℹ️",
    )

st.subheader(f"Flagged transactions ({len(flags)} of {len(df)} rows)")

# Build a summary table of flagged rows so the user can pick one to investigate.
summary_rows = [
    {
        "Row": f.row_index,
        "Anomaly score": f.anomaly_score,
        "Amount": f.raw_row.get("Amount", "-"),
        "Top feature": f.top_features[0]["feature"],
        "Top z-score": f.top_features[0]["z_score"],
    }
    for f in flags
]
summary_df = pd.DataFrame(summary_rows).sort_values("Anomaly score", ascending=False)
st.dataframe(summary_df, use_container_width=True, height=250)

selected_row = st.selectbox(
    "Select a row to investigate",
    options=summary_df["Row"].tolist(),
    format_func=lambda r: f"Row {r} (score={next(f.anomaly_score for f in flags if f.row_index == r)})",
)

selected_flag = next(f for f in flags if f.row_index == selected_row)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**Top contributing features**")
    feat_df = pd.DataFrame(selected_flag.top_features)
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("**Full row data**")
    st.json(selected_flag.raw_row)

investigate = st.button("🕵️ Investigate this anomaly", type="primary")

if investigate:
    if llm_choice == "Groq (cloud, free tier)" and not groq_api_key:
        st.error("Please enter your GROQ_API_KEY in the sidebar first.")
    else:
        with st.spinner("Retrieving relevant patterns and reasoning..."):
            kb = get_knowledge_base()
            agent = get_agent(kb, llm_choice, groq_api_key)
            anomaly_dict = asdict(selected_flag)
            result = agent.invoke({
                "anomaly": anomaly_dict, "query": "", "retrieved": [], "explanation": "",
            })

        st.markdown("### Investigation result")

        with st.expander("Retrieval query used", expanded=False):
            st.write(result["query"])

        with st.expander("Retrieved fraud patterns / rules", expanded=True):
            for r in result["retrieved"]:
                st.markdown(f"- **[{r['category']}]** {r['text']}")

        st.markdown("**Agent explanation & recommendation:**")
        st.success(result["explanation"])
