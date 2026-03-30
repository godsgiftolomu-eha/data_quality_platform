import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

from utils.data_loader import load_csv, load_excel, get_sheet_names, load_google_sheet
from utils.qa_checks import (
    missing_values_report,
    duplicate_report,
    datatype_report,
    outlier_report,
    distribution_report,
    uniqueness_report,
    completeness_score,
)
from utils.standardization import (
    zscore_normalize,
    minmax_scale,
    trim_whitespace,
    change_case,
    normalize_dates,
    drop_duplicates,
    drop_missing_rows,
    fill_missing,
)
from utils.mlos_qa_checks import (
    run_all_mlos_checks,
    issues_to_dataframe,
    fix_generate_uuids,
)

st.set_page_config(page_title="Data Quality Platform", page_icon="📊", layout="wide")

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-card h2 { margin: 0; font-size: 2rem; }
    .metric-card p { margin: 0; color: #555; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Data Quality Platform")
st.markdown("Upload your dataset, run quality checks, standardize, and download the cleaned result.")

# ── Session State ───────────────────────────────────────────────────────────
for key, default in [
    ("df", None), ("df_cleaned", None), ("loaded_file_id", None),
    ("loaded_sheet", None), ("profiling_html", None),
    ("tp_df", None), ("tp_file_id", None), ("tp_loaded_sheet", None),
    ("mlos_issues", None), ("std_message", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar: Data Upload ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Data Source")
    source = st.radio("Choose source", ["Upload File (CSV / Excel)", "Google Sheet URL"])

    if source == "Upload File (CSV / Excel)":
        uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls"])
        if uploaded_file:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            is_new_file = file_id != st.session_state.loaded_file_id

            if uploaded_file.name.endswith(".csv"):
                if is_new_file:
                    st.session_state.df = load_csv(uploaded_file)
                    st.session_state.loaded_file_id = file_id
                    st.session_state.loaded_sheet = None
                    st.session_state.df_cleaned = None
                    st.session_state.profiling_html = None
                    st.session_state.mlos_issues = None
                st.success(f"Loaded **{uploaded_file.name}**")
            else:
                sheets = get_sheet_names(uploaded_file)
                if len(sheets) == 1:
                    if is_new_file:
                        st.session_state.df = load_excel(uploaded_file, sheet_name=sheets[0])
                        st.session_state.loaded_file_id = file_id
                        st.session_state.loaded_sheet = sheets[0]
                        st.session_state.df_cleaned = None
                        st.session_state.profiling_html = None
                        st.session_state.mlos_issues = None
                    st.success(f"Loaded **{uploaded_file.name}** (sheet: {sheets[0]})")
                else:
                    selected_sheet = st.selectbox("Select a sheet", sheets, key="main_sheet_select")
                    if is_new_file or selected_sheet != st.session_state.loaded_sheet:
                        st.session_state.df = load_excel(uploaded_file, sheet_name=selected_sheet)
                        st.session_state.loaded_file_id = file_id
                        st.session_state.loaded_sheet = selected_sheet
                        st.session_state.df_cleaned = None
                        st.session_state.profiling_html = None
                        st.session_state.mlos_issues = None
                    st.success(f"Loaded **{uploaded_file.name}** (sheet: {selected_sheet})")
        else:
            # File was removed
            if st.session_state.loaded_file_id is not None:
                st.session_state.df = None
                st.session_state.loaded_file_id = None
                st.session_state.loaded_sheet = None
                st.session_state.df_cleaned = None
                st.session_state.profiling_html = None
                st.session_state.mlos_issues = None
    else:
        sheet_url = st.text_input("Paste Google Sheet URL (must be publicly accessible)")
        if st.button("Load Sheet", key="btn_load_gsheet") and sheet_url:
            try:
                st.session_state.df = load_google_sheet(sheet_url)
                st.session_state.loaded_file_id = f"gsheet_{sheet_url}"
                st.session_state.df_cleaned = None
                st.session_state.profiling_html = None
                st.session_state.mlos_issues = None
                st.success("Google Sheet loaded!")
            except Exception as e:
                st.error(f"Failed to load sheet: {e}")

    if st.session_state.df is not None:
        st.markdown("---")
        st.markdown(f"**Rows:** {st.session_state.df.shape[0]}  \n**Columns:** {st.session_state.df.shape[1]}")


# ── Guard ──────────────────────────────────────────────────────────────────
if st.session_state.df is None:
    st.info("👈 Upload a dataset from the sidebar to get started.")
    st.stop()

df = st.session_state.df

# ── Tabs ───────────────────────────────────────────────────────────────────
tab_overview, tab_profiling, tab_qa, tab_mlos, tab_std, tab_download = st.tabs([
    "Overview", "Profiling Report", "Quality Checks", "MLoS QA", "Standardization", "Download"
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ═══════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(100), width="stretch")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Completeness", f"{completeness_score(df)}%")
    with col4:
        st.metric("Duplicates", int(df.duplicated().sum()))

    st.subheader("Column Types")
    st.dataframe(datatype_report(df), width="stretch")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — ydata-profiling Report
# ═══════════════════════════════════════════════════════════════════════════
with tab_profiling:
    st.subheader("Profiling Report (ydata-profiling)")
    st.caption("This may take a moment for large datasets.")

    profile_mode = st.selectbox("Report mode", ["Minimal (faster)", "Full (detailed)"],
                                key="profile_mode")

    if st.button("Generate Profiling Report", key="btn_profiling"):
        with st.spinner("Generating report..."):
            try:
                from ydata_profiling import ProfileReport

                minimal = profile_mode.startswith("Minimal")
                profile = ProfileReport(df, minimal=minimal, title="Data Quality Report")
                st.session_state.profiling_html = profile.to_html()
            except ImportError:
                st.warning(
                    "ydata-profiling is not installed. "
                    "Run: `pip install ydata-profiling`"
                )

    if st.session_state.profiling_html:
        st.components.v1.html(st.session_state.profiling_html, height=800, scrolling=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Quality Checks
# ═══════════════════════════════════════════════════════════════════════════
with tab_qa:
    st.subheader("Quality & Assurance Checks")

    check = st.selectbox("Select a check", [
        "Missing Values",
        "Duplicates",
        "Data Types",
        "Outlier Detection",
        "Distribution Stats",
        "Uniqueness / Cardinality",
    ])

    if check == "Missing Values":
        report = missing_values_report(df)
        st.dataframe(report, width="stretch")

        cols_with_missing = report[report["Missing Count"] > 0]
        if not cols_with_missing.empty:
            fig = px.bar(
                cols_with_missing.sort_values("Missing %", ascending=True),
                x="Missing %", y="Column",
                orientation="h",
                text="Missing %",
                color="Missing %",
                color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(
                title="Missing Values by Column",
                xaxis_title="Missing %",
                yaxis_title="",
                coloraxis_showscale=False,
                height=max(350, len(cols_with_missing) * 30 + 100),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.success("No missing values found!")

    elif check == "Duplicates":
        dup = duplicate_report(df)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Duplicate Rows", dup["duplicate_rows"])
        with col2:
            st.metric("Duplicate %", f"{dup['duplicate_percent']}%")

        # Donut chart for duplicate vs unique
        fig = go.Figure(go.Pie(
            labels=["Unique", "Duplicate"],
            values=[dup["total_rows"] - dup["duplicate_rows"], dup["duplicate_rows"]],
            hole=0.5,
            marker_colors=["#2ecc71", "#e74c3c"],
            textinfo="label+percent",
        ))
        fig.update_layout(title="Duplicate vs Unique Rows", height=350)
        st.plotly_chart(fig, width="stretch")

        if dup["duplicate_rows"] > 0:
            with st.expander("View duplicate rows"):
                st.dataframe(dup["duplicate_df"], width="stretch")

    elif check == "Data Types":
        dt_report = datatype_report(df)
        st.dataframe(dt_report, width="stretch")

        # Type distribution pie chart
        type_counts = dt_report["Inferred Type"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        fig = px.pie(type_counts, names="Type", values="Count",
                     title="Column Type Distribution",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")

    elif check == "Outlier Detection":
        report = outlier_report(df)
        if report.empty:
            st.info("No numeric columns with enough data for outlier detection.")
        else:
            st.dataframe(report, width="stretch")

            # Outlier % comparison chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="IQR Outlier %", x=report["Column"], y=report["IQR Outlier %"],
                marker_color="#3498db",
            ))
            fig.add_trace(go.Bar(
                name="Z-Score Outlier %", x=report["Column"], y=report["Z-Score Outlier %"],
                marker_color="#e74c3c",
            ))
            fig.update_layout(
                title="Outlier Percentage by Column",
                xaxis_title="Column", yaxis_title="Outlier %",
                barmode="group", height=400,
            )
            st.plotly_chart(fig, width="stretch")

            # Box plot for selected column
            chart_col = st.selectbox("Box plot for column", report["Column"].tolist())
            if chart_col:
                fig = px.box(df, y=chart_col, title=f"Box Plot: {chart_col}",
                             points="outliers")
                fig.update_layout(height=400)
                st.plotly_chart(fig, width="stretch")

    elif check == "Distribution Stats":
        report = distribution_report(df)
        if report.empty:
            st.info("No numeric columns found.")
        else:
            st.dataframe(report, width="stretch")

            chart_col = st.selectbox("Visualize distribution", report["Column"].tolist())
            if chart_col:
                col_data = df[chart_col].dropna()

                # Histogram with KDE
                fig = px.histogram(
                    col_data, x=chart_col, nbins=30,
                    title=f"Distribution: {chart_col}",
                    marginal="box",
                    color_discrete_sequence=["#3498db"],
                )
                fig.update_layout(
                    xaxis_title=chart_col, yaxis_title="Count",
                    height=450,
                )
                st.plotly_chart(fig, width="stretch")

    elif check == "Uniqueness / Cardinality":
        u_report = uniqueness_report(df)
        st.dataframe(u_report, width="stretch")

        fig = px.bar(
            u_report.sort_values("Cardinality %", ascending=True),
            x="Cardinality %", y="Column",
            orientation="h",
            text="Cardinality %",
            color="Cardinality %",
            color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            title="Cardinality by Column",
            xaxis_title="Cardinality %",
            yaxis_title="",
            coloraxis_showscale=False,
            height=max(350, len(u_report) * 30 + 100),
        )
        st.plotly_chart(fig, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — MLoS Domain QA
# ═══════════════════════════════════════════════════════════════════════════
with tab_mlos:
    st.subheader("MLoS Domain Validation")
    st.markdown(
        "Run domain-specific QA checks on your MLoS dataset. "
        "Upload a **takeoffpoint reference table** to enable cross-table validation (rules 1-4)."
    )

    # Takeoffpoint reference upload
    tp_file = st.file_uploader(
        "Upload Takeoffpoint table (CSV/Excel) — optional for cross-table checks",
        type=["csv", "xlsx", "xls"],
        key="tp_upload",
    )
    if tp_file:
        tp_fid = f"{tp_file.name}_{tp_file.size}"
        tp_is_new = tp_fid != st.session_state.tp_file_id

        if tp_file.name.endswith(".csv"):
            if tp_is_new:
                st.session_state.tp_df = load_csv(tp_file)
                st.session_state.tp_file_id = tp_fid
                st.session_state.tp_loaded_sheet = None
            st.success(f"Takeoffpoint table loaded: **{tp_file.name}** ({st.session_state.tp_df.shape[0]} rows)")
        else:
            tp_sheets = get_sheet_names(tp_file)
            if len(tp_sheets) == 1:
                if tp_is_new:
                    st.session_state.tp_df = load_excel(tp_file, sheet_name=tp_sheets[0])
                    st.session_state.tp_file_id = tp_fid
                    st.session_state.tp_loaded_sheet = tp_sheets[0]
            else:
                tp_selected_sheet = st.selectbox("Select takeoffpoint sheet", tp_sheets, key="tp_sheet_select")
                if tp_is_new or tp_selected_sheet != st.session_state.tp_loaded_sheet:
                    st.session_state.tp_df = load_excel(tp_file, sheet_name=tp_selected_sheet)
                    st.session_state.tp_file_id = tp_fid
                    st.session_state.tp_loaded_sheet = tp_selected_sheet
            st.success(f"Takeoffpoint table loaded: **{tp_file.name}** ({st.session_state.tp_df.shape[0]} rows)")
    else:
        if st.session_state.tp_file_id is not None:
            st.session_state.tp_df = None
            st.session_state.tp_file_id = None
            st.session_state.tp_loaded_sheet = None

    if st.button("Run MLoS QA Checks", key="btn_mlos_run"):
        with st.spinner("Running all MLoS checks..."):
            issues = run_all_mlos_checks(df, st.session_state.tp_df)
            st.session_state.mlos_issues = issues

    if st.session_state.mlos_issues is not None:
        issues = st.session_state.mlos_issues
        summary_df = issues_to_dataframe(issues)

        # Counters
        errors = sum(1 for i in issues if i.get("severity") == "ERROR")
        warnings = sum(1 for i in issues if i.get("severity") == "WARNING")
        passed = sum(1 for i in issues if i.get("severity") == "SKIPPED")
        infos = sum(1 for i in issues if i.get("severity") == "INFO")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Errors", errors)
        with c2:
            st.metric("Warnings", warnings)
        with c3:
            st.metric("Info", infos)
        with c4:
            st.metric("Skipped", passed)

        if errors == 0 and warnings == 0:
            st.success("All MLoS QA checks passed!")
        else:
            st.dataframe(summary_df, width="stretch")

            # Expandable detail for each issue
            for issue in issues:
                severity = issue.get("severity", "")
                label = f"[{severity}] Rule {issue.get('rule', '?')}: {issue.get('message', '')}"

                with st.expander(label):
                    if "rows" in issue and issue["rows"]:
                        st.write(f"**Affected rows:** {len(issue['rows'])} (showing first 50)")
                        row_indices = issue["rows"][:50]
                        st.dataframe(df.iloc[row_indices], width="stretch")
                    if "invalid_values" in issue:
                        st.write(f"**Sample invalid values:** {issue['invalid_values']}")

        # Auto-fix section
        st.markdown("---")
        st.subheader("Auto-Fix Options")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Generate missing fc_globalid UUIDs", key="btn_fix_fc"):
                st.session_state.df = fix_generate_uuids(df, ["fc_globalid", "fc_global_id"])
                st.session_state.mlos_issues = None
                st.session_state.std_message = "Generated new UUIDs for fc_globalid. Re-run checks to verify."
                st.rerun()
        with col_b:
            if st.button("Generate missing settlementarea_globlid UUIDs", key="btn_fix_sa"):
                st.session_state.df = fix_generate_uuids(df, [
                    "settlementarea_globlid", "settlementarea_globalid", "settlement_area_globalid"])
                st.session_state.mlos_issues = None
                st.session_state.std_message = "Generated new UUIDs for settlementarea_globlid. Re-run checks to verify."
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — Standardization
# ═══════════════════════════════════════════════════════════════════════════
with tab_std:
    st.subheader("Standardization & Cleaning")

    # Show pending message from previous action (survives rerun)
    if st.session_state.std_message:
        st.success(st.session_state.std_message)
        st.session_state.std_message = None

    working_df = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else df.copy()

    action = st.selectbox("Select action", [
        "Z-Score Normalization",
        "Min-Max Scaling",
        "Trim Whitespace",
        "Change Case",
        "Normalize Dates",
        "Drop Duplicates",
        "Drop Rows with Missing Values",
        "Fill Missing Values",
    ], key="std_action")

    numeric_cols = working_df.select_dtypes(include=np.number).columns.tolist()
    string_cols = working_df.select_dtypes(include="object").columns.tolist()
    all_cols = working_df.columns.tolist()

    if action == "Z-Score Normalization":
        cols = st.multiselect("Select numeric columns", numeric_cols, default=numeric_cols, key="std_zscore_cols")
        if st.button("Apply", key="btn_std_zscore") and cols:
            working_df = zscore_normalize(working_df, cols)
            st.session_state.df_cleaned = working_df
            st.session_state.std_message = "Z-Score normalization applied!"
            st.rerun()

    elif action == "Min-Max Scaling":
        cols = st.multiselect("Select numeric columns", numeric_cols, default=numeric_cols, key="std_minmax_cols")
        lo = st.number_input("Min", value=0.0, key="std_minmax_lo")
        hi = st.number_input("Max", value=1.0, key="std_minmax_hi")
        if st.button("Apply", key="btn_std_minmax") and cols:
            working_df = minmax_scale(working_df, cols, feature_range=(lo, hi))
            st.session_state.df_cleaned = working_df
            st.session_state.std_message = "Min-Max scaling applied!"
            st.rerun()

    elif action == "Trim Whitespace":
        cols = st.multiselect("Select text columns", string_cols, default=string_cols, key="std_trim_cols")
        if st.button("Apply", key="btn_std_trim") and cols:
            working_df = trim_whitespace(working_df, cols)
            st.session_state.df_cleaned = working_df
            st.session_state.std_message = "Whitespace trimmed!"
            st.rerun()

    elif action == "Change Case":
        cols = st.multiselect("Select text columns", string_cols, default=string_cols, key="std_case_cols")
        case_opt = st.selectbox("Case", ["lower", "upper", "title"], key="std_case_opt")
        if st.button("Apply", key="btn_std_case") and cols:
            working_df = change_case(working_df, cols, case=case_opt)
            st.session_state.df_cleaned = working_df
            st.session_state.std_message = f"Changed to {case_opt} case!"
            st.rerun()

    elif action == "Normalize Dates":
        cols = st.multiselect("Select date columns", all_cols, key="std_date_cols")
        fmt = st.text_input("Target format", value="%Y-%m-%d", key="std_date_fmt")
        if st.button("Apply", key="btn_std_dates") and cols:
            working_df = normalize_dates(working_df, cols, target_format=fmt)
            st.session_state.df_cleaned = working_df
            st.session_state.std_message = "Dates normalized!"
            st.rerun()

    elif action == "Drop Duplicates":
        if st.button("Apply", key="btn_std_dedup"):
            before = len(working_df)
            working_df = drop_duplicates(working_df)
            st.session_state.df_cleaned = working_df
            st.session_state.std_message = f"Dropped {before - len(working_df)} duplicate rows."
            st.rerun()

    elif action == "Drop Rows with Missing Values":
        thresh = st.slider("Drop rows missing more than X% of columns", 0, 100, 50, key="std_missing_thresh")
        if st.button("Apply", key="btn_std_dropmissing"):
            before = len(working_df)
            working_df = drop_missing_rows(working_df, threshold_pct=thresh)
            st.session_state.df_cleaned = working_df
            st.session_state.std_message = f"Dropped {before - len(working_df)} rows."
            st.rerun()

    elif action == "Fill Missing Values":
        cols = st.multiselect("Select columns", all_cols, key="std_fill_cols")
        strategy = st.selectbox("Strategy", ["mean", "median", "mode", "ffill", "bfill", "zero", "empty_string"],
                                key="std_fill_strategy")
        if st.button("Apply", key="btn_std_fill") and cols:
            working_df = fill_missing(working_df, cols, strategy=strategy)
            st.session_state.df_cleaned = working_df
            st.session_state.std_message = f"Filled missing values with '{strategy}'."
            st.rerun()

    if st.session_state.df_cleaned is not None:
        st.markdown("---")
        st.subheader("Preview (Cleaned)")
        st.dataframe(st.session_state.df_cleaned.head(100), width="stretch")

        if st.button("Reset to Original", key="btn_std_reset"):
            st.session_state.df_cleaned = None
            st.session_state.std_message = "Reset to original dataset."
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — Download
# ═══════════════════════════════════════════════════════════════════════════
with tab_download:
    st.subheader("Download Dataset")

    download_df = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else df

    st.dataframe(download_df.head(50), width="stretch")

    file_format = st.selectbox("File format", ["CSV", "Excel"], key="download_format")

    if file_format == "CSV":
        csv_data = download_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
    else:
        buffer = io.BytesIO()
        download_df.to_excel(buffer, index=False, engine="openpyxl")
        st.download_button(
            label="Download Excel",
            data=buffer.getvalue(),
            file_name="cleaned_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
