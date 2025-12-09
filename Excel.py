import os
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from io import BytesIO

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
    """
    Robust loader for CSV / Excel files coming from Streamlit uploader.
    - Accepts multi-sheet Excel files and concatenates all sheets, adding a `__sheet_name` column.
    - Detects HTML payloads (common when SharePoint returns a login page or error page).
    - Tries pandas default engine first, then falls back to openpyxl / xlrd.
    """
    try:
        fname = (file.name or "").lower()
        # CSV: let pandas handle the uploaded file-like object directly
        if fname.endswith(".csv"):
            file.seek(0)
            return pd.read_csv(file)

        # Read raw bytes (safe for in-memory and file objects)
        file.seek(0)
        content = file.read()
        if not content:
            st.error("Uploaded file is empty.")
            return pd.DataFrame()

        head = content[:512].lower()

        # Detect HTML / login pages from SharePoint (common root cause)
        if b"<html" in head or b"<!doctype html" in head:
            st.error(
                "Uploaded file appears to be HTML (SharePoint may have returned a login/error page). "
                "Confirm the download/authentication and re-upload the actual Excel file."
            )
            # Optionally log the first bytes for debugging
            st.write("Preview of file head (first 200 bytes):")
            st.code(head[:200])
            return pd.DataFrame()

        bio = BytesIO(content)

        # Try reading all sheets (sheet_name=None returns dict of DataFrames)
        try:
            sheets = pd.read_excel(bio, sheet_name=None)
        except Exception as ex:
            # Try common explicit engines as fallbacks
            bio.seek(0)
            try:
                sheets = pd.read_excel(bio, sheet_name=None, engine="openpyxl")
            except Exception:
                bio.seek(0)
                try:
                    sheets = pd.read_excel(bio, sheet_name=None, engine="xlrd")
                except Exception as final_ex:
                    st.error(f"File error while parsing Excel: {final_ex}")
                    return pd.DataFrame()

        # If pandas returned a dict (multiple sheets), concatenate
        if isinstance(sheets, dict):
            df_list = []
            for sheet_name, df in sheets.items():
                if df is None or df.empty:
                    continue
                df_copy = df.copy()
                # Preserve sheet source
                df_copy["__sheet_name"] = sheet_name
                df_list.append(df_copy)

            if not df_list:
                st.warning("No sheets with data were found in the uploaded Excel file.")
                return pd.DataFrame()

            combined = pd.concat(df_list, ignore_index=True, sort=False)
            return combined

        # Otherwise, single DataFrame returned
        return sheets

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
        except Exception:
            pass

    numeric, cat = get_numeric_and_categorical(df)
    for col in numeric[:6]:
        try:
            fig = px.histogram(df, x=col, title=f"Distribution: {col}")
            out["distributions"].append((col, fig))
        except Exception:
            pass

    if target_col and group_col:
        try:
            grp = df.groupby(group_col)[target_col].mean().reset_index()
            fig = px.bar(grp, x=group_col, y=target_col)
            out["category_charts"].append(("Category Chart", fig))
        except Exception:
            pass

    return out

async def business_metrics_agent(df, target_col, group_col):
    out = {"kpis": {}, "group_summary": None}
    if target_col and target_col in df.columns:
        s = pd.to_numeric(df[target_col], errors="coerce").dropna()
        out["kpis"] = {
            "Total": float(s.sum()) if not s.empty else 0.0,
            "Average": float(s.mean()) if not s.empty else 0.0,
            "Median": float(s.median()) if not s.empty else 0.0,
            "Min": float(s.min()) if not s.empty else 0.0,
            "Max": float(s.max()) if not s.empty else 0.0,
            "Count": int(s.count()),
        }

    if group_col and group_col in df.columns:
        try:
            out["group_summary"] = df.groupby(group_col)[target_col].sum().reset_index()
        except Exception:
            out["group_summary"] = None

    return out

async def correlation_agent(df):
    out = {"corr_matrix": None, "top_pairs": []}
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] > 1:
        corr = numeric.corr()
        out["corr_matrix"] = corr
        # find top absolute correlations (excluding self)
        corr_abs = corr.abs().unstack().reset_index()
        corr_abs.columns = ["x", "y", "corr"]
        corr_abs = corr_abs[corr_abs["x"] != corr_abs["y"]]
        corr_abs = corr_abs.sort_values("corr", ascending=False).drop_duplicates(subset=["corr"])
        top = corr_abs.head(10).to_dict(orient="records")
        out["top_pairs"] = top
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
    # `response` structure may vary depending on SDK version; adapt if needed
    text = getattr(response, "text", None) or json.dumps(response, default=str)
    return {"llm_report": text}

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

    uploaded = st.sidebar.file_uploader(
        "Upload CSV or Excel (single file — multiple sheets allowed)", 
        type=["csv", "xls", "xlsx", "xlsm", "xlsb"]
    )

    if not uploaded:
        st.info("Upload a file to begin.")
        return

    df = load_data(uploaded)
    if df.empty:
        return

    # Inform the user when data came from multiple sheets
    if "__sheet_name" in df.columns:
        st.sidebar.info("This Excel contained multiple sheets; all sheets were concatenated. Column `__sheet_name` indicates source sheet.")

    numeric, cat = get_numeric_and_categorical(df)
    date_cols = detect_date_columns(df)

    date_col = st.sidebar.selectbox("Date Column", ["None"] + date_cols)
    target_col = st.sidebar.selectbox("Target KPI", ["None"] + numeric)
    group_col = st.sidebar.selectbox("Group Column", ["None"] + cat)

    date_col = None if date_col == "None" else date_col
    target_col = None if target_col == "None" else target_col
    group_col = None if group_col == "None" else group_col

    if not st.sidebar.button("Run Analysis"):
        return

    with st.spinner("Running multi-agent analysis..."):
        # Use asyncio.run; keep existing behaviour. If Streamlit environment already has loop, adjust accordingly.
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
        for _, fig in viz["category_charts"]:
            st.plotly_chart(fig)

    with t3:
        st.json(metrics)

    with t4:
        if corr["corr_matrix"] is not None:
            st.dataframe(corr["corr_matrix"])

    with t5:
        st.dataframe(df)
        # Provide CSV download of the concatenated dataset
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv_bytes, "data.csv", mime="text/csv")

if __name__ == "__main__":
    main()
