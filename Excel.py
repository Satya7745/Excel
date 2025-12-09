import os
import asyncio
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import google.generativeai as genai
import json

# ---------------------- Gemini Setup ---------------------- #
@st.cache_resource
def configure_gemini():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    genai.configure(api_key=key)
    return genai.GenerativeModel("gemini-2.5-flash")

# ---------------------- Utility Functions ---------------------- #
@st.cache_data
def load_data(file) -> pd.DataFrame:
    try:
        if file.name.lower().endswith(".csv"):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"File error: {e}")
        return pd.DataFrame()

def detect_date_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.85:
                cols.append(c)
        except:
            pass
    return cols

def get_numeric_and_categorical(df):
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric, cat

# ---------------------- AGENTS ---------------------- #
async def visualization_agent(df, date_col, target_col, group_col):
    out = {"time_series": None, "distributions": [], "category_charts": []}

    if date_col and target_col:
        try:
            tmp = df[[date_col, target_col]].dropna()
            tmp[date_col] = pd.to_datetime(tmp[date_col])
            fig = px.line(tmp, x=date_col, y=target_col, title=f"{target_col} over time")
            out["time_series"] = fig
        except:
            pass

    numeric, cat = get_numeric_and_categorical(df)
    for col in numeric[:6]:
        try:
            fig = px.histogram(df, x=col, title=f"Distribution: {col}")
            out["distributions"].append((col, fig))
        except:
            pass

    if target_col and group_col:
        try:
            grp = df.groupby(group_col)[target_col].mean().reset_index()
            fig = px.bar(grp, x=group_col, y=target_col)
            out["category_charts"].append(("Category Chart", fig))
        except:
            pass

    return out

async def business_metrics_agent(df, target_col, group_col):
    out = {"kpis": {}, "group_summary": None}
    if target_col:
        s = pd.to_numeric(df[target_col], errors="coerce").dropna()
        out["kpis"] = {
            "Total": float(s.sum()),
            "Average": float(s.mean()),
            "Median": float(s.median()),
            "Min": float(s.min()),
            "Max": float(s.max()),
            "Count": int(s.count()),
        }

    if group_col:
        out["group_summary"] = df.groupby(group_col)[target_col].sum().reset_index()

    return out

async def correlation_agent(df):
    out = {"corr_matrix": None, "top_pairs": []}
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] > 1:
        corr = numeric.corr()
        out["corr_matrix"] = corr
    return out

async def summary_agent_gemini(df, metrics, corr, target_col, group_col):
    model = configure_gemini()
    if model is None:
        return {"llm_report": "Gemini API Key not configured."}

    prompt = f"""
Dataset Shape: {df.shape}
Target KPI: {target_col}
Group Column: {group_col}
KPIs: {metrics.get('kpis')}
Top Correlations: {corr.get('top_pairs')}

Provide:
1. Executive Summary
2. Key Insights
3. Risks
4. Actionable Recommendations
"""

    response = await asyncio.to_thread(model.generate_content, prompt)
    return {"llm_report": response.text}

# ---------------------- ORCHESTRATOR ---------------------- #
async def run_all_agents(df, date_col, target_col, group_col):
    v_task = asyncio.create_task(visualization_agent(df, date_col, target_col, group_col))
    m_task = asyncio.create_task(business_metrics_agent(df, target_col, group_col))
    c_task = asyncio.create_task(correlation_agent(df))

    v, m, c = await asyncio.gather(v_task, m_task, c_task)
    s = await summary_agent_gemini(df, m, c, target_col, group_col)

    return {
        "visualizations": v,
        "business_metrics": m,
        "correlations": c,
        "summary": s,
    }

# ---------------------- STREAMLIT UI ---------------------- #
def main():
    st.set_page_config("Business Insights — Gemini", layout="wide")
    st.title("Business & Work Insights — Multi-Agent AI (Gemini)")

    uploaded = st.sidebar.file_uploader("Upload CSV or Excel", ["csv","xlsx"])

    if not uploaded:
        st.info("Upload a file to begin.")
        return

    df = load_data(uploaded)
    numeric, cat = get_numeric_and_categorical(df)
    date_cols = detect_date_columns(df)

    date_col = st.sidebar.selectbox("Date Column", ["None"] + date_cols)
    target_col = st.sidebar.selectbox("Target KPI", numeric)
    group_col = st.sidebar.selectbox("Group Column", ["None"] + cat)

    date_col = None if date_col == "None" else date_col
    group_col = None if group_col == "None" else group_col

    if not st.sidebar.button("Run Analysis"):
        return

    with st.spinner("Running multi-agent analysis..."):
        results = asyncio.run(run_all_agents(df, date_col, target_col, group_col))

    viz, metrics, corr, summary = (
        results["visualizations"],
        results["business_metrics"],
        results["correlations"],
        results["summary"],
    )

    t1,t2,t3,t4,t5 = st.tabs(["Executive Summary","Visuals","KPIs","Correlations","Data"])

    with t1:
        st.markdown(summary.get("llm_report","No output."))

    with t2:
        if viz["time_series"]:
            st.plotly_chart(viz["time_series"])
        for _, fig in viz["distributions"]:
            st.plotly_chart(fig)

    with t3:
        st.json(metrics)

    with t4:
        if corr["corr_matrix"] is not None:
            st.dataframe(corr["corr_matrix"])

    with t5:
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False), "data.csv")

if __name__ == "__main__":
    main()

