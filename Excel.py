import os
import asyncio
import json
import re
from io import BytesIO
from typing import Tuple, List, Optional

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# Try to import Gemini SDK; handle absent package gracefully
try:
    import google.generativeai as genai
except Exception:
    genai = None


# -------------------------
# Gemini setup (optional)
# -------------------------
@st.cache_resource
def configure_gemini():
    if genai is None:
        return None
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    genai.configure(api_key=key)
    return genai.GenerativeModel("gemini-2.5-flash")


# -------------------------
# Excel loading & cleaning
# -------------------------
@st.cache_data
def load_and_clean_excel(file) -> pd.DataFrame:
    """
    Load an uploaded Excel/CSV, detect the header row in each sheet (heuristic),
    extract the main table per sheet, add __sheet_name, and concatenate.
    Returns a DataFrame or empty DataFrame on failure.
    """
    try:
        file.seek(0)
        raw = file.read()
        if not raw:
            return pd.DataFrame()

        # Quick check for SharePoint/html masquerade
        head = raw[:512].lower()
        if head.startswith(b"<html") or b"<!doctype html" in head:
            st.error("Upload appears to be an HTML page (SharePoint preview or auth page). Please download the real file and upload.")
            return pd.DataFrame()

        bio = BytesIO(raw)

        # Use pandas to read all sheets without assuming header (header=None)
        try:
            sheets = pd.read_excel(bio, sheet_name=None, header=None)
        except Exception as e:
            # Try explicit engines for robustness
            bio.seek(0)
            try:
                sheets = pd.read_excel(bio, sheet_name=None, header=None, engine="openpyxl")
            except Exception:
                bio.seek(0)
                try:
                    sheets = pd.read_excel(bio, sheet_name=None, header=None, engine="xlrd")
                except Exception as ex:
                    st.error(f"Failed to parse Excel: {ex}")
                    return pd.DataFrame()

        def extract_table(df_raw: pd.DataFrame) -> Optional[pd.DataFrame]:
            # drop completely empty rows
            df_raw = df_raw.dropna(how="all")
            if df_raw.shape[0] == 0:
                return None

            # Heuristic: first row with >= 3 non-null cells is header
            header_idx = None
            for i in range(min(10, len(df_raw))):  # only look in the first 10 rows
                if df_raw.iloc[i].notna().sum() >= 3:
                    header_idx = i
                    break

            if header_idx is None:
                # fallback: if the first row has any non-null values, use it
                if df_raw.iloc[0].notna().sum() > 0:
                    header_idx = 0
                else:
                    return None

            # Compose table: header row -> column names, following rows -> data
            df = df_raw.iloc[header_idx + 1 :].copy()
            df.columns = df_raw.iloc[header_idx].astype(str).tolist()

            # drop fully empty columns and Unnamed columns
            df = df.dropna(axis=1, how="all")
            df = df.loc[:, ~df.columns.str.match(r"^Unnamed", na=False)]

            # reset index and return
            return df.reset_index(drop=True)

        df_list = []
        for sheet_name, sheet_df in sheets.items():
            if sheet_df is None:
                continue
            table = extract_table(sheet_df)
            if table is not None and not table.empty:
                table["__sheet_name"] = sheet_name
                df_list.append(table)

        if not df_list:
            return pd.DataFrame()

        combined = pd.concat(df_list, ignore_index=True, sort=False)
        return combined

    except Exception as exc:
        st.error(f"Unexpected error reading file: {exc}")
        return pd.DataFrame()


# -------------------------
# Type inference (numeric & date)
# -------------------------
def infer_and_cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to cast columns to numeric or datetime where appropriate.
    Return a copy with inferred types.
    """
    df2 = df.copy()
    for col in df2.columns:
        series = df2[col]

        # Skip the sheet name column
        if col == "__sheet_name":
            continue

        # If column is already numeric or datetime, skip
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue

        # Try datetime first
        try:
            parsed = pd.to_datetime(series, errors="coerce", dayfirst=False, infer_datetime_format=True)
            if parsed.notna().mean() >= 0.8:
                df2[col] = parsed
                continue
        except Exception:
            pass

        # Try numeric
        try:
            num = pd.to_numeric(series, errors="coerce")
            if num.notna().mean() >= 0.8:
                df2[col] = num
                continue
        except Exception:
            pass

        # Leave as-is (object)
    return df2


# -------------------------
# Arrow-safe sanitizer (for st.dataframe & downloads)
# -------------------------
def make_df_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harden DataFrame so pyarrow/streamlit can serialize it reliably.
    Converts column names to strings, resets index, and turns values into simple scalars/strings.
    """
    safe = df.copy(deep=True)

    # Make sure column names are strings
    safe.columns = [str(c) for c in safe.columns]

    # Reset index to avoid exotic index objects in metadata
    safe = safe.reset_index(drop=True)

    # Replace problematic cell types with JSON-serialized strings or str()
    def safe_val(x):
        if pd.isna(x):
            return ""
        if isinstance(x, (dict, list, tuple, set)):
            try:
                return json.dumps(x, default=str)
            except Exception:
                return str(x)
        # For numpy types, convert to Python scalars
        if isinstance(x, (np.integer, np.floating, np.bool_)):
            return x.item()
        # datetimes -> ISO strings
        if pd.api.types.is_datetime64_any_dtype(type(x)) or hasattr(x, "isoformat"):
            try:
                return str(x)
            except Exception:
                return ""
        return str(x)

    for c in safe.columns:
        safe[c] = safe[c].apply(safe_val)

    # Force final dtype to string for maximum compatibility
    safe = safe.astype(str)

    return safe


# -------------------------
# Agent implementations
# -------------------------
def metrics_agent(df: pd.DataFrame) -> dict:
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


def correlation_agent(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], List[dict]]:
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] < 2:
        return None, []
    corr = numeric.corr()
    pairs = corr.abs().unstack().reset_index()
    pairs.columns = ["x", "y", "corr"]
    pairs = pairs[pairs["x"] != pairs["y"]]
    pairs = pairs.sort_values("corr", ascending=False)
    return corr, pairs.head(10).to_dict(orient="records")


def visualization_agent(df: pd.DataFrame) -> List:
    visuals = []
    # prefer typed df for plotting
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Numeric histograms
    for col in numeric_cols:
        try:
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            visuals.append(fig)
        except Exception:
            pass

    # Time series if we have a date + numeric
    if date_cols and numeric_cols:
        for date_col in date_cols[:2]:
            for num_col in numeric_cols[:2]:
                try:
                    tmp = df[[date_col, num_col]].dropna()
                    if not tmp.empty:
                        fig = px.line(tmp.sort_values(date_col), x=date_col, y=num_col, title=f"{num_col} over {date_col}")
                        visuals.append(fig)
                except Exception:
                    pass

    # Categorical vs numeric
    if categorical_cols and numeric_cols:
        for cat in categorical_cols[:3]:
            for num in numeric_cols[:3]:
                try:
                    grp = df.groupby(cat)[num].mean().reset_index()
                    if not grp.empty:
                        fig = px.bar(grp, x=cat, y=num, title=f"{num} by {cat}")
                        visuals.append(fig)
                except Exception:
                    pass

    return visuals


# -------------------------
# Gemini insight agent (strict, evidence-locked)
# -------------------------
async def insight_agent(df: pd.DataFrame, metrics: dict, corr_pairs: List[dict]) -> dict:
    model = configure_gemini()
    if model is None:
        return {"error": "Gemini API not configured. Insights disabled."}

    # Build a short evidence payload (trim large lists)
    evidence = {
        "shape": df.shape,
        "metrics_sample": {k: metrics[k] for k in list(metrics)[:8]},
        "top_correlations": corr_pairs[:8],
    }

    prompt = f"""
You are a rigorous data analyst. You MUST use only the JSON facts below.
DO NOT INVENT numbers, percentages, trends, or causal claims not present in the facts.
If data is insufficient for a claim, respond with "INSUFFICIENT DATA".

FACTS:
{json.dumps(evidence, indent=2)}

Output STRICT JSON:
{{"summary":"...","insights":[{{"statement":"...","evidence":"...","confidence":0.0}}],"risks":[{{"statement":"...","evidence":"...","confidence":0.0}}],"recommendations":[{{"action":"...","justification":"...", "confidence":0.0}}]}}
"""
    try:
        # run model.generate_content in thread to avoid blocking
        response = await asyncio.to_thread(model.generate_content, prompt)
        raw = getattr(response, "text", "")
        # extract JSON payload
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return {"error": "Model returned non-JSON output — insights blocked to avoid hallucination."}
        payload = json.loads(m.group(0))
        return payload
    except Exception as exc:
        return {"error": f"Insight agent failed: {exc}"}


# -------------------------
# Helper to run coroutines safely when streamlit may already have a running loop
# -------------------------
def run_coro(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


# -------------------------
# Streamlit app
# -------------------------
def main():
    st.set_page_config("Business Insights — Safe Multi-Agent", layout="wide")
    st.title("Business & Work Insights — Cleaned & Enriched (No Hallucination)")

    uploaded = st.sidebar.file_uploader("Upload Excel (XLS/XLSX) or CSV", type=["xlsx", "xls", "csv", "xlsm"])

    if not uploaded:
        st.info("Upload a file to begin (supports multi-sheet Excel).")
        return

    with st.spinner("Cleaning and inferring types..."):
        raw_df = load_and_clean_excel(uploaded)

    if raw_df.empty:
        st.error("No usable table detected in the uploaded file.")
        return

    # Keep a typed copy for analysis; keep raw_df for display/download
    df_typed = infer_and_cast_types(raw_df)

    # Prepare safe-for-UI DataFrame
    safe_df = make_df_arrow_safe(raw_df)

    st.success("File cleaned successfully.")
    with st.expander("🔍 Preview cleaned table"):
        st.dataframe(safe_df, use_container_width=True)

    # Allow optional quick auto-detection for suggested KPI / grouping
    numeric_cols = df_typed.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df_typed.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = [c for c in df_typed.columns if pd.api.types.is_datetime64_any_dtype(df_typed[c])]

    st.sidebar.markdown("### Optional selections (auto-detected):")
    target_kpi = st.sidebar.selectbox("Target KPI (numeric)", options=["None"] + numeric_cols, index=0)
    group_col = st.sidebar.selectbox("Group column (categorical)", options=["None"] + categorical_cols, index=0)

    target_kpi = None if target_kpi == "None" else target_kpi
    group_col = None if group_col == "None" else group_col

    if not st.sidebar.button("Run Multi-Agent Analysis"):
        return

    with st.spinner("Running analysis agents..."):

        # Metrics & correlation & visuals run synchronously (deterministic)
        metrics = metrics_agent(df_typed)
        corr_matrix, corr_pairs = correlation_agent(df_typed)
        visuals = visualization_agent(df_typed)

        # LLM insight (evidence-locked) - run via helper
        insights = run_coro(insight_agent(df_typed, metrics, corr_pairs)) if genai is not None else {"error": "Gemini not installed or API key missing."}

    # Present outputs in tabs
    tabs = st.tabs(["Insights", "Visuals", "Metrics", "Correlations", "Data"])

    with tabs[0]:
        st.subheader("Verified Insights (evidence-locked)")
        st.json(insights)

    with tabs[1]:
        st.subheader("Auto-generated Visualizations")
        if visuals:
            for fig in visuals:
                try:
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.write(f"Could not render a chart: {e}")
        else:
            st.info("No auto-visualizations available for this dataset.")

    with tabs[2]:
        st.subheader("Computed Metrics")
        st.json(metrics)

    with tabs[3]:
        st.subheader("Correlation matrix / top pairs")
        if corr_matrix is not None:
            try:
                st.dataframe(make_df_arrow_safe(corr_matrix.reset_index()))
            except Exception:
                st.write(corr_pairs)
        else:
            st.info("Not enough numeric columns to compute correlations.")

    with tabs[4]:
        st.subheader("Cleaned Data (downloadable)")
        st.dataframe(safe_df, use_container_width=True)
        csv_bytes = safe_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Clean CSV", csv_bytes, "cleaned_data.csv", "text/csv")

    st.sidebar.success("Analysis complete.")


if __name__ == "__main__":
    main()
