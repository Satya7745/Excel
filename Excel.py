"""
Streamlit app: Multi-agent (async) Business Insights
- Upload CSV / Excel
- Visualization agent (plotly)
- Business metrics agent (KPIs, group breakdown)
- Correlation agent
- Summary agent powered by Azure OpenAI

REQUIREMENTS (add to requirements.txt):
streamlit
pandas
numpy
plotly
openpyxl
openai

ENVIRONMENT VARIABLES (set in Streamlit Secrets or OS):
AZURE_OPENAI_API_KEY
AZURE_OPENAI_ENDPOINT (e.g. https://<resource-name>.openai.azure.com/)
AZURE_OPENAI_DEPLOYMENT (your model/deployment name)
AZURE_OPENAI_API_VERSION (e.g. 2024-02-15-preview)

Place this file as app.py and run:
streamlit run app.py

"""

import os
import asyncio
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import openai
import json

# ---------------------- Azure OpenAI setup ---------------------- #
@st.cache_resource
def configure_openai_client():
    key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not (key and endpoint and deployment and api_version):
        return None

    # configure openai Python library to target Azure
    openai.api_type = "azure"
    openai.api_key = key
    # make sure endpoint does not end with a slash
    openai.api_base = endpoint.rstrip("/")
    openai.api_version = api_version

    # return deployment name for convenience
    return {
        "deployment": deployment,
    }

# ---------------------- Utility functions ---------------------- #
@st.cache_data
def load_data(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    try:
        name = getattr(file, "name", "uploaded")
        if name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()
    return df


def detect_date_columns(df: pd.DataFrame) -> List[str]:
    date_cols: List[str] = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
            continue
        # try to parse
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.85:
                date_cols.append(col)
        except Exception:
            continue
    return date_cols


def get_numeric_and_categorical(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # treat object and category as categorical
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Also include low-cardinality numeric as categorical candidates
    for c in df.select_dtypes(include=["number"]).columns:
        if df[c].nunique(dropna=True) <= 10:
            if c not in cat_cols:
                cat_cols.append(c)
    return numeric_cols, cat_cols

# ---------------------- Agents ---------------------- #
async def visualization_agent(df: pd.DataFrame, date_col: Optional[str], target_col: Optional[str], group_col: Optional[str]) -> Dict[str, Any]:
    out = {"time_series": None, "distributions": [], "category_charts": []}

    # Time series
    if date_col and date_col in df.columns and target_col and target_col in df.columns:
        try:
            tmp = df[[date_col, target_col]].copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp = tmp.dropna(subset=[date_col, target_col])
            tmp = tmp.sort_values(date_col)
            if not tmp.empty:
                fig = px.line(tmp, x=date_col, y=target_col, title=f"{target_col} over time")
                out["time_series"] = fig
        except Exception:
            pass

    # Distributions
    numeric_cols, cat_cols = get_numeric_and_categorical(df)
    for col in numeric_cols[:6]:
        try:
            fig = px.histogram(df, x=col, nbins=30, title=f"Distribution: {col}")
            out["distributions"].append((col, fig))
        except Exception:
            continue

    # Category vs target
    try:
        if target_col and group_col and group_col in df.columns:
            grp = (
                df[[group_col, target_col]]
                .dropna()
                .groupby(group_col)[target_col]
                .agg("mean")
                .reset_index()
                .sort_values(target_col, ascending=False)
            )
            if not grp.empty:
                fig = px.bar(grp, x=group_col, y=target_col, title=f"Avg {target_col} by {group_col}")
                out["category_charts"].append((f"{target_col} by {group_col}", fig))
        elif target_col and cat_cols:
            auto = cat_cols[0]
            grp = (
                df[[auto, target_col]]
                .dropna()
                .groupby(auto)[target_col]
                .agg("mean")
                .reset_index()
                .sort_values(target_col, ascending=False)
                .head(20)
            )
            if not grp.empty:
                fig = px.bar(grp, x=auto, y=target_col, title=f"Avg {target_col} by {auto}")
                out["category_charts"].append((f"{target_col} by {auto}", fig))
    except Exception:
        pass

    await asyncio.sleep(0)
    return out


async def business_metrics_agent(df: pd.DataFrame, target_col: Optional[str], group_col: Optional[str]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"kpis": {}, "group_summary": None}
    if target_col and target_col in df.columns:
        series = pd.to_numeric(df[target_col], errors="coerce").dropna()
        if not series.empty:
            metrics["kpis"] = {
                "Total": float(series.sum()),
                "Average": float(series.mean()),
                "Median": float(series.median()),
                "Min": float(series.min()),
                "Max": float(series.max()),
                "Std": float(series.std(ddof=0)),
                "Count": int(series.count()),
            }
    if group_col and group_col in df.columns and target_col and target_col in df.columns:
        try:
            g = df[[group_col, target_col]].dropna()
            g[target_col] = pd.to_numeric(g[target_col], errors="coerce")
            g = g.dropna(subset=[target_col])
            if not g.empty:
                group_summary = (
                    g.groupby(group_col)[target_col]
                    .agg(["count", "sum", "mean"]) 
                    .rename(columns={"mean":"avg"})
                    .reset_index()
                    .sort_values("sum", ascending=False)
                )
                metrics["group_summary"] = group_summary
        except Exception:
            pass
    await asyncio.sleep(0)
    return metrics


async def correlation_agent(df: pd.DataFrame) -> Dict[str, Any]:
    out = {"corr_matrix": None, "top_pairs": []}
    numeric = df.select_dtypes(include=["number"]).copy()
    if numeric.shape[1] >= 2:
        corr = numeric.corr()
        out["corr_matrix"] = corr
        pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                pairs.append((cols[i], cols[j], corr.iloc[i,j]))
        pairs_sorted = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
        out["top_pairs"] = pairs_sorted[:10]
    await asyncio.sleep(0)
    return out


async def summary_agent_azure(df: pd.DataFrame, business_metrics: Dict[str, Any], correlation_info: Dict[str, Any], target_col: Optional[str], group_col: Optional[str]) -> Dict[str, str]:
    cfg = configure_openai_client()
    # If Azure not configured, return a deterministic summary
    if cfg is None:
        # Fallback small summary
        rows, cols = df.shape
        numeric_cols, cat_cols = get_numeric_and_categorical(df)
        missing = int(df.isna().sum().sum())
        overview = f"Dataset has {rows:,} rows and {cols} columns. Numeric cols: {len(numeric_cols)}; Categorical cols: {len(cat_cols)}. Missing values total: {missing}."
        insights = "Basic KPIs available when a numeric target is selected. Configure Azure OpenAI (AZURE_OPENAI_* env vars) for richer summaries."
        return {"overview": overview, "insights": insights}

    deployment = cfg["deployment"]

    # build compact context for the LLM
    kpis = business_metrics.get("kpis") or {}
    group_preview = ""
    if business_metrics.get("group_summary") is not None:
        try:
            group_preview = business_metrics["group_summary"].head(5).to_csv(index=False)
        except Exception:
            group_preview = ""
    top_corr = correlation_info.get("top_pairs") or []
    corr_preview = "\n".join([f"{a} vs {b}: {v:.2f}" for a,b,v in top_corr[:6]])

    system_prompt = (
        "You are a senior business analyst. Produce a concise, structured executive report for a manager.\n"
        "Include: 1) Executive summary (2-4 lines) 2) Key data-driven insights (bulleted) 3) Top risks or data quality issues 4) 2-3 actionable recommendations."
    )

    user_prompt = (
        f"Dataset shape: {df.shape}\n"
        f"Target KPI: {target_col}\n"
        f"Group column: {group_col}\n"
        f"KPIs: {json.dumps(kpis, default=str)}\n"
        f"Top group breakdown (csv):\n{group_preview}\n"
        f"Top correlations:\n{corr_preview}\n"
        "Be concise, use bullets and short sentences."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        # openai.ChatCompletion.create is a blocking call; run in thread
        def call_llm():
            return openai.ChatCompletion.create(
                engine=deployment,  # for azure, engine param is deployment name
                model=deployment,
                messages=messages,
                temperature=0.2,
                max_tokens=700,
            )

        resp = await asyncio.to_thread(call_llm)
        text = ""
        if resp and getattr(resp, "choices", None):
            text = resp.choices[0].message.content
        else:
            text = str(resp)

        return {"llm_report": text}
    except Exception as e:
        # On failure, return a short fallback summary
        rows, cols = df.shape
        fallback = f"LLM failed: {e}. Fallback summary: dataset {rows}x{cols}. KPIs: {kpis}."
        return {"llm_report": fallback}

# ---------------------- Orchestrator ---------------------- #
async def run_all_agents(df: pd.DataFrame, date_col: Optional[str], target_col: Optional[str], group_col: Optional[str]) -> Dict[str, Any]:
    viz_task = asyncio.create_task(visualization_agent(df, date_col, target_col, group_col))
    metrics_task = asyncio.create_task(business_metrics_agent(df, target_col, group_col))
    corr_task = asyncio.create_task(correlation_agent(df))

    metrics_result, corr_result, viz_result = await asyncio.gather(metrics_task, corr_task, viz_task)
    summary_result = await summary_agent_azure(df, metrics_result, corr_result, target_col, group_col)

    return {
        "visualizations": viz_result,
        "business_metrics": metrics_result,
        "correlations": corr_result,
        "summary": summary_result,
    }

# ---------------------- Streamlit App ---------------------- #

def main():
    st.set_page_config(page_title="Business Insights Agent (Azure OpenAI)", layout="wide")
    st.title("Business & Work Insights — Multi-Agent Analyzer")
    st.write("Upload a CSV/XLSX with your business or work data. The app runs multiple agents in parallel: visualizations, KPIs, correlations, and an Azure OpenAI powered executive summary.")

    with st.sidebar:
        st.header("Upload & Settings")
        uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xls", "xlsx"], accept_multiple_files=False)
        st.markdown("---")
        st.info("Set Azure variables in environment or Streamlit Secrets: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION")

    if uploaded is None:
        st.info("Upload a file in the sidebar to begin.")
        return

    df = load_data(uploaded)
    if df.empty:
        st.error("Uploaded file could not be read or is empty.")
        return

    # column suggestions
    date_candidates = detect_date_columns(df)
    numeric_cols, cat_cols = get_numeric_and_categorical(df)

    with st.sidebar:
        date_col = st.selectbox("Date column (optional)", options=["(none)"] + date_candidates if date_candidates else ["(none)"], index=0)
        if date_col == "(none)":
            date_col = None
        target_col = None
        if numeric_cols:
            target_col = st.selectbox("Main numeric business metric (target)", options=numeric_cols)
        group_col = None
        if cat_cols:
            group_col = st.selectbox("Group by (optional)", options=["(none)"] + cat_cols)
            if group_col == "(none)":
                group_col = None

        run = st.button("Run analysis")

    if not run:
        st.info("Configure analysis in the sidebar and click Run analysis.")
        return

    # Run agents
    with st.spinner("Running agents — visualizations, metrics, correlations, LLM summary..."):
        try:
            results = asyncio.run(run_all_agents(df, date_col, target_col, group_col))
        except RuntimeError:
            # if event loop already running, get current loop
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(run_all_agents(df, date_col, target_col, group_col))

    viz = results.get("visualizations", {})
    metrics = results.get("business_metrics", {})
    corr = results.get("correlations", {})
    summary = results.get("summary", {})

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Executive Summary", "Visual Insights", "Business Metrics", "Correlations", "Data Explorer"])

    with tab1:
        st.header("Executive / AI Report")
        llm_text = summary.get("llm_report") or summary.get("overview") or "No summary available."
        st.markdown(llm_text)

        st.markdown("---")
        st.subheader("Raw LLM JSON (debug)")
        st.json(summary)

    with tab2:
        st.header("Visual Insights")
        if viz.get("time_series") is not None:
            st.subheader("Time Series")
            st.plotly_chart(viz["time_series"], use_container_width=True)

        if viz.get("distributions"):
            st.subheader("Distributions")
            for col, fig in viz["distributions"]:
                st.plotly_chart(fig, use_container_width=True)

        if viz.get("category_charts"):
            st.subheader("Category / Segment Charts")
            for title, fig in viz["category_charts"]:
                st.plotly_chart(fig, use_container_width=True)

        if not (viz.get("time_series") or viz.get("distributions") or viz.get("category_charts")):
            st.info("No visualizations available. Check numeric/date columns.")

    with tab3:
        st.header("Business Metrics & KPIs")
        kpis = metrics.get("kpis") or {}
        if kpis:
            c1, c2, c3, c4 = st.columns([2,2,2,2])
            c1.metric("Total", f"{kpis.get('Total', 0):,.2f}")
            c2.metric("Average", f"{kpis.get('Average', 0):,.2f}")
            c3.metric("Median", f"{kpis.get('Median', 0):,.2f}")
            c4.metric("Count", f"{kpis.get('Count', 0):,}")
            st.markdown("---")
            st.subheader("Other KPIs")
            st.write({k:v for k,v in kpis.items() if k not in ['Total','Average','Median','Count']})
        else:
            st.info("No KPIs computed. Select a numeric target column in the sidebar.")

        st.subheader("Segment-level summary")
        gs = metrics.get("group_summary")
        if gs is not None:
            st.dataframe(gs, use_container_width=True)
        else:
            st.info("No group summary. Choose a grouping column.")

    with tab4:
        st.header("Correlations")
        corr_matrix = corr.get("corr_matrix")
        if corr_matrix is not None:
            st.dataframe(corr_matrix.style.background_gradient(axis=None), use_container_width=True)
            # plot heatmap
            try:
                fig = px.imshow(corr_matrix, title="Correlation matrix", zmin=-1, zmax=1)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass
            st.markdown("### Top correlated pairs")
            for a,b,v in corr.get("top_pairs", [])[:10]:
                st.write(f"- **{a}** vs **{b}**: {v:.2f}")
        else:
            st.info("Correlation matrix requires at least two numeric columns.")

    with tab5:
        st.header("Data Explorer")
        st.dataframe(df.head(1000), use_container_width=True)
        st.markdown("---")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download processed CSV", data=csv, file_name="processed_data.csv", mime="text/csv")


if __name__ == '__main__':
    main()
