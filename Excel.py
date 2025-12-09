import os
import asyncio
import json
import re
from io import BytesIO

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import google.generativeai as genai


# =========================
# ✅ GEMINI SETUP
# =========================
@st.cache_resource
def configure_gemini():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    genai.configure(api_key=key)
    return genai.GenerativeModel("gemini-2.5-flash")


# =========================
# ✅ EXCEL CLEANER (UNSTRUCTURED SAFE)
# =========================
@st.cache_data
def load_and_clean_excel(file) -> pd.DataFrame:
    try:
        file.seek(0)
        raw = file.read()

        # --- Block HTML masquerading as Excel ---
        if raw[:50].lower().startswith(b"<html") or b"<!doctype html" in raw[:200].lower():
            st.error("Invalid Excel file (SharePoint returned HTML)")
            return pd.DataFrame()

        bio = BytesIO(raw)

        # --- Load all sheets without assuming headers ---
        sheets = pd.read_excel(bio, sheet_name=None, header=None, engine="openpyxl")

        def extract_table(df_raw: pd.DataFrame):
            df_raw = df_raw.dropna(how="all")

            header_idx = None
            for i in range(len(df_raw)):
                if df_raw.iloc[i].notna().sum() >= 3:
                    header_idx = i
                    break

            if header_idx is None:
                return None

            df = df_raw.iloc[header_idx + 1:].copy()
            df.columns = df_raw.iloc[header_idx]
            df = df.dropna(axis=1, how="all")
            df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
            return df.reset_index(drop=True)

        df_list = []
        for name, sheet in sheets.items():
            clean_df = extract_table(sheet)
            if clean_df is not None and not clean_df.empty:
                clean_df["__sheet_name"] = name
                df_list.append(clean_df)

        if not df_list:
            return pd.DataFrame()

        return pd.concat(df_list, ignore_index=True)

    except Exception as e:
        st.error(str(e))
        return pd.DataFrame()


# =========================
# ✅ AGENT 1 — METRICS
# =========================
def metrics_agent(df: pd.DataFrame):
    numeric = df.select_dtypes(include=["number"])

    metrics = {}
    for col in numeric.columns:
        s = numeric[col].dropna()
        if not s.empty:
            metrics[col] = {
                "mean": float(s.mean()),
                "sum": float(s.sum()),
                "min": float(s.min()),
                "max": float(s.max()),
                "count": int(s.count()),
            }
    return metrics


# =========================
# ✅ AGENT 2 — CORRELATION
# =========================
def correlation_agent(df: pd.DataFrame):
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] < 2:
        return None, []

    corr = numeric.corr()

    pairs = corr.abs().unstack().reset_index()
    pairs.columns = ["x", "y", "corr"]
    pairs = pairs[pairs["x"] != pairs["y"]]
    pairs = pairs.sort_values("corr", ascending=False)

    return corr, pairs.head(10).to_dict(orient="records")


# =========================
# ✅ AGENT 3 — DYNAMIC VISUALIZATION
# =========================
def visualization_agent(df: pd.DataFrame):
    visuals = []

    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Auto histogram for all numeric columns
    for col in numeric:
        visuals.append(px.histogram(df, x=col, title=f"Distribution of {col}"))

    # Auto bar charts for categorical vs numeric
    if categorical and numeric:
        for cat in categorical[:2]:
            for num in numeric[:2]:
                grp = df.groupby(cat)[num].mean().reset_index()
                visuals.append(
                    px.bar(grp, x=cat, y=num, title=f"{num} by {cat}")
                )

    return visuals


# =========================
# ✅ AGENT 4 — STRICT INSIGHT LLM (NO HALLUCINATION)
# =========================
async def insight_agent(df, metrics, corr_pairs):
    model = configure_gemini()
    if model is None:
        return {"error": "Gemini API key missing"}

    prompt = f"""
You are a strict data analyst.

You are ONLY allowed to use the following facts.
DO NOT invent numbers or trends.
If data is insufficient, say "INSUFFICIENT DATA".

METRICS:
{json.dumps(metrics, indent=2)}

TOP CORRELATIONS:
{json.dumps(corr_pairs, indent=2)}

Respond ONLY in this strict JSON:

{{
  "summary": "...",
  "insights": [
      {{"statement": "...", "evidence": "..."}}
  ],
  "risks": [
      {{"statement": "...", "evidence": "..."}}
  ],
  "recommendations": [
      {{"action": "...", "justification": "..."}}
  ]
}}
"""

    response = await asyncio.to_thread(model.generate_content, prompt)
    txt = getattr(response, "text", "")

    try:
        clean_json = re.search(r"\{.*\}", txt, re.S).group(0)
        return json.loads(clean_json)
    except Exception:
        return {"error": "Model output rejected to prevent hallucination."}


# =========================
# ✅ STREAMLIT UI
# =========================
def main():
    st.set_page_config("Business Insights — Multi-Agent AI", layout="wide")
    st.title("Business & Work Insights — Cleaned & Enriched")

    uploaded = st.sidebar.file_uploader("Upload Excel", ["xlsx", "xlsm", "xls"])

    if not uploaded:
        st.info("Upload an Excel file to start")
        return

    df = load_and_clean_excel(uploaded)

    if df.empty:
        st.error("No usable table detected in this Excel")
        return

    st.success("Excel cleaned successfully")

    with st.expander("🔍 Preview Cleaned Data"):
        st.dataframe(df)

    if not st.sidebar.button("Run Multi-Agent Analysis"):
        return

    with st.spinner("Running agents..."):

        metrics = metrics_agent(df)
        corr_matrix, corr_pairs = correlation_agent(df)
        visuals = visualization_agent(df)
        insights = asyncio.run(insight_agent(df, metrics, corr_pairs))

    tabs = st.tabs(["Insights", "Visuals", "Metrics", "Correlations", "Data"])

    with tabs[0]:
        st.json(insights)

    with tabs[1]:
        for fig in visuals:
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.json(metrics)

    with tabs[3]:
        if corr_matrix is not None:
            st.dataframe(corr_matrix)

    with tabs[4]:
        st.dataframe(df)
        st.download_button("Download Clean CSV",
                           df.to_csv(index=False),
                           "cleaned_data.csv",
                           "text/csv")


if __name__ == "__main__":
    main()
