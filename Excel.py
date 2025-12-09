import os
import asyncio
from typing import List
from io import BytesIO
import re

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
        fname = file.name.lower()
        file.seek(0)
        raw = file.read()

        # 🚨 SharePoint HTML detection
        if raw[:50].lower().startswith(b"<html") or b"<!doctype html" in raw[:200].lower():
            st.error("Invalid Excel: SharePoint returned HTML instead of real file.")
            st.code(raw[:300])
            return pd.DataFrame()

        bio = BytesIO(raw)

        if fname.endswith((".xlsx", ".xlsm", ".xlsb")):
            sheets = pd.read_excel(bio, sheet_name=None, engine="openpyxl")
        elif fname.endswith(".xls"):
            sheets = pd.read_excel(bio, sheet_name=None, engine="xlrd")
        elif fname.endswith(".csv"):
            return pd.read_csv(bio)
        else:
            st.error("Unsupported file format.")
            return pd.DataFrame()

        df_list = []
        for sheet, df in sheets.items():
            if not df.empty:
                df["__sheet_name"] = sheet
                df_list.append(df)

        return pd.concat(df_list, ignore_index=True)

    except Exception as e:
        st.error(f"Excel parsing failed: {e}")
        return pd.DataFrame()

def detect_date_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() > 0.85:
            cols.append(c)
    return cols

def get_numeric_and_categorical(df):
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric, cat

# ---------------------- AGENTS ---------------------- #
async def visualization_agent(df, date_col, target_col, group_col):
    out = {"time_series": None, "distributions": [], "category_charts": []}

    if date_col and target_col:
        tmp = df[[date_col, target_col]].dropna()
        tmp[date_col] = pd.to_datetime(tmp[date_col])
        out["time_series"] = px.line(tmp, x=date_col, y=target_col)

    numeric, _ = get_numeric_and_categorical(df)
    for col in numeric[:6]:
        out["distributions"].append((col, px.histogram(df, x=col)))

    if target_col and group_col:
        grp = df.groupby(group_col)[target_col].mean().reset_index()
        out["category_charts"].append(px.bar(grp, x=group_col, y=target_col))

    return out

async def business_metrics_agent(df, target_col, group_col):
    out = {"kpis": {}, "group_summary": None}

    if target_col:
        s = pd.to_numeric(df[target_col], errors="coerce").dropna()
        out["kpis"] = {
            "total": round(float(s.sum()), 3),
            "mean": round(float(s.mean()), 3),
            "median": round(float(s.median()), 3),
            "min": round(float(s.min()), 3),
            "max": round(float(s.max()), 3),
            "count": int(s.count())
        }

    if target_col and group_col:
        out["group_summary"] = (
            df.groupby(group_col)[target_col]
            .sum()
            .reset_index()
            .to_dict(orient="records")
        )

    return out

async def correlation_agent(df):
    out = {"corr_matrix": None, "top_pairs": []}
    numeric = df.select_dtypes(include=["number"])

    if numeric.shape[1] > 1:
        corr = numeric.corr()
        out["corr_matrix"] = corr

        pairs = corr.abs().unstack().reset_index()
        pairs.columns = ["x", "y", "corr"]
        pairs = pairs[pairs["x"] != pairs["y"]]
        pairs = pairs.sort_values("corr", ascending=False)
        out["top_pairs"] = pairs.head(8).to_dict(orient="records")

    return out

# ---------------------- HALLUCINATION-SAFE LLM ---------------------- #
async def summary_agent_gemini(df, metrics, corr, target_col, group_col):
    model = configure_gemini()
    if model is None:
        return {"llm_report": "Gemini API Key not configured."}

    # 🚨 STRICT ANTI-HALLUCINATION PROMPT
    prompt = f"""
You are a strict financial and data analyst.

You are ONLY allowed to use the facts provided below.
You are NOT allowed to invent data, assume trends, or generalize.
If evidence is missing, explicitly state "INSUFFICIENT DATA".

FACTUAL INPUT:
Dataset Shape: {df.shape}
Target KPI: {target_col}
Group Column: {group_col}
KPI Metrics: {json.dumps(metrics.get("kpis"), indent=2)}
Group Summary: {json.dumps(metrics.get("group_summary"), indent=2)}
Top Correlations: {json.dumps(corr.get("top_pairs"), indent=2)}

OUTPUT FORMAT (STRICT JSON):
{{
  "executive_summary": "...",
  "key_insights": [
      {{"statement": "...", "evidence": "...", "confidence": 0.0-1.0}}
  ],
  "risks": [
      {{"statement": "...", "evidence": "...", "confidence": 0.0-1.0}}
  ],
  "recommendations": [
      {{"action": "...", "justification": "...", "confidence": 0.0-1.0}}
  ]
}}
"""

    response = await asyncio.to_thread(model.generate_content, prompt)
    raw_text = getattr(response, "text", "")

    # ✅ HARD JSON EXTRACTION
    try:
        clean_json = re.search(r"\{.*\}", raw_text, re.S).group(0)
        parsed = json.loads(clean_json)
    except Exception:
        parsed = {"error": "Model returned non-JSON. Insights blocked to avoid hallucination."}

    return parsed

# ---------------------- ORCHESTRATOR ---------------------- #
async def run_all_agents(df, date_col, target_col, group_col):
    v_task = asyncio.create_task(visualization_agent(df, date_col, target_col, group_col))
    m_task = asyncio.create_task(business_metrics_agent(df, target_col, group_col))
    c_task = asyncio.create_task(correlation_agent(df))

    v, m, c = await asyncio.gather(v_task, m_task, c_task)
    s = await summary_agent_gemini(df, m, c, target_col, group_col)

    return {"visualizations": v, "business_metrics": m, "correlations": c, "summary": s}

# ---------------------- STREAMLIT UI ---------------------- #
def main():
    st.set_page_config("Business Insights — Gemini", layout="wide")
    st.title("Business & Work Insights — No-Hallucination AI")

    uploaded = st.sidebar.file_uploader("Upload CSV or Excel", ["csv", "xls", "xlsx"])

    if not uploaded:
        st.info("Upload a file to begin.")
        return

    df = load_data(uploaded)
    if df.empty:
        return

    numeric, cat = get_numeric_and_categorical(df)
    date_cols = detect_date_columns(df)

    date_col = st.sidebar.selectbox("Date Column", ["None"] + date_cols)
    target_col = st.sidebar.selectbox("Target KPI", numeric)
    group_col = st.sidebar.selectbox("Group Column", ["None"] + cat)

    date_col = None if date_col == "None" else date_col
    group_col = None if group_col == "None" else group_col

    if not st.sidebar.button("Run Analysis"):
        return

    with st.spinner("Running verified analysis..."):
        results = asyncio.run(run_all_agents(df, date_col, target_col, group_col))

    viz, metrics, corr, summary = (
        results["visualizations"],
        results["business_metrics"],
        results["correlations"],
        results["summary"],
    )

    t1,t2,t3,t4,t5 = st.tabs(["Verified Insights","Visuals","KPIs","Correlations","Data"])

    with t1:
        st.json(summary)

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

if __name__ == "__main__":
    main()
