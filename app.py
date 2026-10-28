"""
EV CHARGING DEMAND PREDICTION DASHBOARD
======================================
Run:
    python -m pip install streamlit pandas matplotlib
    python -m streamlit run app.py

Use EV_Charging_Dashboard_Ready.csv, or upload an EV charging CSV.
The Run analysis button loads the selected dataset. The Apply filters button
applies the location and charger-type filters.
"""

from pathlib import Path
import hashlib

import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# 1. Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="EV Charging Demand Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ EV Charging Demand Prediction Dashboard")
st.caption("Explore charging demand by time, location, and charger type.")

APP_FOLDER = Path(__file__).resolve().parent
DEFAULT_FILE = APP_FOLDER / "EV_Charging_Dashboard_Ready.csv"

# Keep the last successfully loaded dataset while users change filters.
if "ev_source_df" not in st.session_state:
    st.session_state["ev_source_df"] = None
if "ev_source_name" not in st.session_state:
    st.session_state["ev_source_name"] = None
if "ev_source_signature" not in st.session_state:
    st.session_state["ev_source_signature"] = None
if "ev_filters" not in st.session_state:
    st.session_state["ev_filters"] = None

# ------------------------------------------------------------
# 2. Dataset selection and explicit Run Analysis button
# ------------------------------------------------------------
st.sidebar.header("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload EV charging CSV",
    type=["csv"],
    help="Upload EV_Charging_Dashboard_Ready.csv or another CSV with Demand, Location, and ChargerType columns.",
)

if uploaded_file is not None:
    selected_name = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    signature = hashlib.md5(file_bytes).hexdigest()
    source_kind = "upload"
elif DEFAULT_FILE.exists():
    selected_name = DEFAULT_FILE.name
    file_bytes = DEFAULT_FILE.read_bytes()
    signature = hashlib.md5(file_bytes).hexdigest()
    source_kind = "default"
    st.sidebar.success(f"Default dataset found: {selected_name}")
else:
    selected_name = None
    file_bytes = None
    signature = None
    source_kind = None
    st.sidebar.info("Upload a CSV file to begin.")

run_analysis = st.sidebar.button(
    "▶ Run analysis",
    type="primary",
    use_container_width=True,
    disabled=file_bytes is None,
)

if run_analysis and file_bytes is not None:
    try:
        from io import BytesIO

        loaded = pd.read_csv(BytesIO(file_bytes), low_memory=False)
        loaded.columns = [str(c).strip() for c in loaded.columns]
        loaded = loaded.dropna(how="all").drop_duplicates().copy()

        required = ["Demand", "Location", "ChargerType"]
        missing = [c for c in required if c not in loaded.columns]

        if missing:
            st.session_state["ev_source_df"] = None
            st.session_state["ev_source_name"] = None
            st.session_state["ev_source_signature"] = None
            st.session_state["ev_filters"] = None
            st.error(
                "This does not appear to be the EV charging dataset. "
                "Required columns missing: " + ", ".join(missing)
            )
            st.write("Columns found in your CSV:", list(loaded.columns))
            st.info(
                "Use the EV dataset containing Demand, Location, and ChargerType. "
                "A credit portfolio dataset will not work in this dashboard."
            )
        else:
            loaded["Demand"] = pd.to_numeric(loaded["Demand"], errors="coerce")
            if "Duration" in loaded.columns:
                loaded["Duration"] = pd.to_numeric(loaded["Duration"], errors="coerce")

            if "StartDatetime" in loaded.columns:
                loaded["StartDatetime"] = pd.to_datetime(
                    loaded["StartDatetime"], errors="coerce"
                )
            elif "StartDate" in loaded.columns:
                loaded["StartDatetime"] = pd.to_datetime(
                    loaded["StartDate"], errors="coerce"
                )

            if "StartDatetime" in loaded.columns:
                dt = loaded["StartDatetime"]
                if "Hour" not in loaded.columns:
                    loaded["Hour"] = dt.dt.hour
                if "DayOfWeek" not in loaded.columns:
                    loaded["DayOfWeek"] = dt.dt.dayofweek
                if "DayName" not in loaded.columns:
                    loaded["DayName"] = dt.dt.day_name()
                if "Month" not in loaded.columns:
                    loaded["Month"] = dt.dt.month
                if "Year" not in loaded.columns:
                    loaded["Year"] = dt.dt.year
                if "IsWeekend" not in loaded.columns:
                    loaded["IsWeekend"] = dt.dt.dayofweek.isin([5, 6]).astype("Int64")

            invalid_demand = int(loaded["Demand"].isna().sum())
            loaded = loaded.dropna(subset=["Demand"]).copy()

            if loaded.empty:
                st.session_state["ev_source_df"] = None
                st.error("No valid numeric Demand values were found in the CSV.")
            else:
                st.session_state["ev_source_df"] = loaded
                st.session_state["ev_source_name"] = selected_name
                st.session_state["ev_source_signature"] = signature
                st.session_state["ev_filters"] = None
                st.session_state["ev_invalid_demand"] = invalid_demand
                st.success(
                    f"Loaded {selected_name}: {len(loaded):,} valid rows. "
                    "Use the filters below and click Apply filters."
                )
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")

source_df = st.session_state.get("ev_source_df")

if source_df is None:
    st.info(
        "Choose or upload your EV charging CSV in the sidebar, then click **Run analysis**."
    )
    st.stop()

st.sidebar.caption(f"Loaded dataset: {st.session_state['ev_source_name']}")
st.sidebar.caption(f"Records loaded: {len(source_df):,}")

# ------------------------------------------------------------
# 3. Filters with an Apply button
# ------------------------------------------------------------
st.sidebar.header("Filters")
with st.sidebar.form("ev_filter_form"):
    all_locations = sorted(source_df["Location"].dropna().astype(str).unique().tolist())
    all_types = sorted(source_df["ChargerType"].dropna().astype(str).unique().tolist())

    current_filters = st.session_state.get("ev_filters") or {}
    default_locations = current_filters.get("locations", all_locations)
    default_types = current_filters.get("types", all_types)

    # If a new CSV is loaded, make sure stored selections still exist.
    default_locations = [v for v in default_locations if v in all_locations]
    default_types = [v for v in default_types if v in all_types]

    selected_locations = st.multiselect(
        "Location",
        options=all_locations,
        default=default_locations,
        help="Choose one or more locations. Select all to include every location.",
    )
    selected_types = st.multiselect(
        "Charger type",
        options=all_types,
        default=default_types,
        help="Choose one or more charger types.",
    )
    apply_filters = st.form_submit_button("Apply filters", use_container_width=True)

if apply_filters or st.session_state.get("ev_filters") is None:
    st.session_state["ev_filters"] = {
        "locations": selected_locations,
        "types": selected_types,
    }

filters = st.session_state["ev_filters"]
filtered = source_df[
    source_df["Location"].astype(str).isin(filters["locations"])
    & source_df["ChargerType"].astype(str).isin(filters["types"])
].copy()

if st.sidebar.button("Reset filters", use_container_width=True):
    st.session_state["ev_filters"] = {
        "locations": all_locations,
        "types": all_types,
    }
    st.rerun()

if filtered.empty:
    st.warning("No rows match your filters. Select at least one location and charger type, then click Apply filters.")
    st.stop()

# ------------------------------------------------------------
# 4. Summary cards
# ------------------------------------------------------------
st.subheader("Dataset overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Charging records", f"{len(filtered):,}")
m2.metric("Average demand", f"{filtered['Demand'].mean():,.2f}")
m3.metric("Highest demand", f"{filtered['Demand'].max():,.2f}")
if "Duration" in filtered.columns and filtered["Duration"].notna().any():
    m4.metric("Average duration", f"{filtered['Duration'].mean():,.2f}")
else:
    m4.metric("Locations selected", f"{filtered['Location'].nunique():,}")

st.caption(
    f"Dataset: {st.session_state['ev_source_name']} · "
    f"{len(filtered):,} records after filtering"
)

# ------------------------------------------------------------
# 5. Charts
# ------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Average demand by hour")
    if "Hour" in filtered.columns and filtered["Hour"].notna().any():
        hourly = filtered.groupby("Hour")["Demand"].mean().sort_index()
        st.bar_chart(hourly)
    else:
        st.info("Hour information is unavailable in this dataset.")

with right:
    st.subheader("Average demand by charger type")
    by_type = filtered.groupby("ChargerType")["Demand"].mean().sort_values(ascending=False)
    st.bar_chart(by_type)

left, right = st.columns(2)

with left:
    st.subheader("Average demand by location")
    by_location = filtered.groupby("Location")["Demand"].mean().sort_values(ascending=False)
    st.bar_chart(by_location)

with right:
    st.subheader("Average demand by day of week")
    if "DayName" in filtered.columns:
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        by_day = filtered.groupby("DayName")["Demand"].mean().reindex(order).dropna()
        st.bar_chart(by_day)
    elif "DayOfWeek" in filtered.columns:
        st.bar_chart(filtered.groupby("DayOfWeek")["Demand"].mean().sort_index())
    else:
        st.info("Day-of-week information is unavailable.")

st.subheader("Monthly average demand")
if "StartDatetime" in filtered.columns and filtered["StartDatetime"].notna().any():
    monthly_df = filtered.dropna(subset=["StartDatetime"]).copy()
    monthly_df["MonthDate"] = monthly_df["StartDatetime"].dt.to_period("M").astype(str)
    st.line_chart(monthly_df.groupby("MonthDate")["Demand"].mean().sort_index())
elif "Month" in filtered.columns:
    st.line_chart(filtered.groupby("Month")["Demand"].mean().sort_index())
else:
    st.info("Date information is unavailable for monthly trends.")

# ------------------------------------------------------------
# 6. Data quality and downloadable filtered rows
# ------------------------------------------------------------
st.subheader("Data quality checks")
q1, q2, q3 = st.columns(3)
q1.metric("Duplicate rows remaining", f"{int(filtered.duplicated().sum()):,}")
q2.metric(
    "Negative duration records",
    f"{int((filtered['Duration'] < 0).sum()):,}" if "Duration" in filtered.columns else "N/A",
)
q3.metric("Negative demand records", f"{int((filtered['Demand'] < 0).sum()):,}")
st.caption("Negative duration values are flagged for review; they are not automatically removed.")

st.subheader("Filtered charging records")
st.dataframe(filtered.head(500), use_container_width=True)

st.download_button(
    "Download filtered EV data",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="EV_Charging_Filtered_Data.csv",
    mime="text/csv",
)
