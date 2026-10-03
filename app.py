import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------
st.set_page_config(
    page_title="EV Charging Demand Prediction",
    page_icon="🔋",
    layout="wide",
)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def find_column(dataframe, possible_names):
    """Return the first matching column from a list of possible names."""
    lower_columns = {column.lower(): column for column in dataframe.columns}

    for name in possible_names:
        if name.lower() in lower_columns:
            return lower_columns[name.lower()]

    return None


def prepare_data(dataframe):
    """Clean the uploaded EV charging dataset and create useful features."""
    df = dataframe.copy()

    # Remove completely empty rows and duplicate rows.
    df = df.dropna(how="all")
    df = df.drop_duplicates()

    # Detect important columns.
    demand_col = find_column(df, ["Demand", "demand", "Energy", "energy"])
    start_datetime_col = find_column(
        df,
        ["StartDatetime", "StartDateTime", "Start DateTime", "start_datetime"],
    )
    start_day_col = find_column(df, ["StartDay", "Start Date", "Date", "date"])
    start_time_col = find_column(df, ["StartTime", "Start Time", "time"])
    duration_col = find_column(df, ["Duration", "duration"])

    # Convert demand to numeric when available.
    if demand_col:
        df[demand_col] = pd.to_numeric(df[demand_col], errors="coerce")

    # Create a datetime column.
    datetime_created = False

    if start_datetime_col:
        df["AnalysisDatetime"] = pd.to_datetime(
            df[start_datetime_col], errors="coerce"
        )
        datetime_created = df["AnalysisDatetime"].notna().any()

    if not datetime_created and start_day_col:
        if start_time_col:
            combined = (
                df[start_day_col].astype(str)
                + " "
                + df[start_time_col].astype(str)
            )
            df["AnalysisDatetime"] = pd.to_datetime(
                combined, errors="coerce"
            )
        else:
            df["AnalysisDatetime"] = pd.to_datetime(
                df[start_day_col], errors="coerce"
            )

    # Create duration when it is not already available.
    if duration_col is None:
        end_datetime_col = find_column(
            df,
            ["EndDatetime", "EndDateTime", "End DateTime", "end_datetime"],
        )

        if start_datetime_col and end_datetime_col:
            start = pd.to_datetime(df[start_datetime_col], errors="coerce")
            end = pd.to_datetime(df[end_datetime_col], errors="coerce")
            df["Duration"] = (end - start).dt.total_seconds() / 60
            duration_col = "Duration"

    if duration_col:
        df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")

    # Create time-based features.
    if "AnalysisDatetime" in df.columns:
        df["Hour"] = df["AnalysisDatetime"].dt.hour
        df["DayOfWeek"] = df["AnalysisDatetime"].dt.dayofweek
        df["Month"] = df["AnalysisDatetime"].dt.month
        df["Day"] = df["AnalysisDatetime"].dt.day
        df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    return df, demand_col, duration_col


def numeric_features(df, target_column):
    """Return numeric predictor columns suitable for machine learning."""
    numeric_df = df.select_dtypes(include=np.number).copy()

    if target_column in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_column])

    # Remove columns that are unlikely to be useful predictors.
    ignored = [
        "ChargingsessionID",
        "UserID",
        "ChargerID",
    ]

    numeric_df = numeric_df.drop(
        columns=[column for column in ignored if column in numeric_df.columns],
        errors="ignore",
    )

    return numeric_df


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("🔋 EV Charging Demand Prediction and Usage Pattern Analysis")
st.write(
    "Analyze EV charging usage patterns and predict charging demand "
    "using an uploaded CSV dataset."
)

st.divider()

# ---------------------------------------------------------
# CSV UPLOAD
# ---------------------------------------------------------
st.sidebar.header("📁 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload EV charging CSV",
    type=["csv"],
)

if uploaded_file is None:
    st.info(
        "Please upload an EV charging CSV file from the sidebar to start "
        "the analysis."
    )
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Unable to read the CSV file: {error}")
    st.stop()

df, demand_col, duration_col = prepare_data(raw_df)

if df.empty:
    st.error("The uploaded CSV does not contain usable rows.")
    st.stop()

if demand_col is None:
    st.error(
        "A Demand column is required for this project. "
        "Please upload a CSV containing a Demand column."
    )
    st.stop()

df = df.dropna(subset=[demand_col])

if df.empty:
    st.error("No valid numeric Demand values were found in the uploaded CSV.")
    st.stop()

# ---------------------------------------------------------
# SIDEBAR INFORMATION
# ---------------------------------------------------------
st.sidebar.success("CSV uploaded successfully.")
st.sidebar.write(f"Rows: {len(df):,}")
st.sidebar.write(f"Columns: {len(df.columns):,}")

location_col = find_column(df, ["Location"])
charger_type_col = find_column(df, ["ChargerType", "Charger Type"])
company_col = find_column(df, ["ChargerCompany", "Charger Company"])

# ---------------------------------------------------------
# 1. DATASET OVERVIEW
# ---------------------------------------------------------
st.header("1. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Charging Sessions", f"{len(df):,}")
col2.metric("Average Demand", f"{df[demand_col].mean():.2f}")
col3.metric(
    "Average Duration",
    f"{df[duration_col].mean():.2f}" if duration_col else "N/A",
)
col4.metric(
    "Locations",
    f"{df[location_col].nunique():,}" if location_col else "N/A",
)

st.dataframe(df.head(10), width="stretch", height=300)

# ---------------------------------------------------------
# 2. DATA QUALITY
# ---------------------------------------------------------
st.header("2. Data Quality")

quality_col1, quality_col2, quality_col3 = st.columns(3)

quality_col1.metric(
    "Missing Values",
    f"{int(df.isna().sum().sum()):,}",
)

quality_col2.metric(
    "Duplicate Rows",
    f"{int(df.duplicated().sum()):,}",
)

quality_col3.metric(
    "Number of Columns",
    f"{len(df.columns):,}",
)

missing_df = (
    df.isna()
    .sum()
    .reset_index()
    .rename(columns={"index": "Column", 0: "Missing Values"})
)

missing_df = missing_df[missing_df["Missing Values"] > 0]

if missing_df.empty:
    st.success("No missing values remain in the prepared dataset.")
else:
    st.dataframe(missing_df, width="stretch", height=220)

# ---------------------------------------------------------
# 3. EDA
# ---------------------------------------------------------
st.header("3. Exploratory Data Analysis")

eda_left, eda_right = st.columns(2)

with eda_left:
    if location_col:
        location_data = (
            df.groupby(location_col)[demand_col]
            .mean()
            .reset_index()
            .sort_values(demand_col, ascending=False)
        )

        fig = px.bar(
            location_data,
            x=location_col,
            y=demand_col,
            title="Average Demand by Location",
            labels={demand_col: "Average Demand"},
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Location column not available.")

with eda_right:
    if charger_type_col:
        charger_data = (
            df.groupby(charger_type_col)[demand_col]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            charger_data,
            x=charger_type_col,
            y=demand_col,
            title="Average Demand by Charger Type",
            labels={demand_col: "Average Demand"},
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("ChargerType column not available.")

if "Hour" in df.columns:
    hourly_data = (
        df.groupby("Hour")[demand_col]
        .mean()
        .reset_index()
    )

    fig = px.line(
        hourly_data,
        x="Hour",
        y=demand_col,
        markers=True,
        title="Average Demand by Hour",
        labels={demand_col: "Average Demand"},
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------
# 4. REGRESSION
# ---------------------------------------------------------
st.header("4. Regression - Demand Prediction")

reg_features = numeric_features(df, demand_col)

if len(reg_features.columns) >= 1 and len(df) >= 10:
    reg_data = pd.concat([reg_features, df[demand_col]], axis=1).dropna()

    if len(reg_data) >= 10:
        X = reg_data.drop(columns=[demand_col])
        y = reg_data[demand_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        reg_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        )
        reg_model.fit(X_train, y_train)
        reg_predictions = reg_model.predict(X_test)

        mae = mean_absolute_error(y_test, reg_predictions)
        rmse = np.sqrt(mean_squared_error(y_test, reg_predictions))
        r2 = r2_score(y_test, reg_predictions)

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("MAE", f"{mae:.2f}")
        metric2.metric("RMSE", f"{rmse:.2f}")
        metric3.metric("R² Score", f"{r2:.2f}")

        comparison = pd.DataFrame(
            {
                "Actual Demand": y_test.values,
                "Predicted Demand": reg_predictions,
            }
        )

        fig = px.scatter(
            comparison,
            x="Actual Demand",
            y="Predicted Demand",
            title="Actual vs Predicted Demand",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Not enough valid numeric data for regression.")
else:
    st.warning("At least one numeric feature and 10 rows are required for regression.")

# ---------------------------------------------------------
# 5. CLASSIFICATION
# ---------------------------------------------------------
st.header("5. Classification - High vs Normal Demand")

clf_features = numeric_features(df, demand_col)

if len(clf_features.columns) >= 1 and len(df) >= 10:
    clf_data = pd.concat([clf_features, df[demand_col]], axis=1).dropna()

    if len(clf_data) >= 10:
        threshold = clf_data[demand_col].median()

        clf_data["DemandClass"] = np.where(
            clf_data[demand_col] >= threshold,
            1,
            0,
        )

        X = clf_data.drop(columns=[demand_col, "DemandClass"])
        y = clf_data["DemandClass"]

        if y.nunique() >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )

            clf_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            )
            clf_model.fit(X_train, y_train)
            clf_predictions = clf_model.predict(X_test)

            accuracy = accuracy_score(y_test, clf_predictions)
            precision = precision_score(
                y_test,
                clf_predictions,
                zero_division=0,
            )
            recall = recall_score(
                y_test,
                clf_predictions,
                zero_division=0,
            )
            f1 = f1_score(
                y_test,
                clf_predictions,
                zero_division=0,
            )

            metric1, metric2, metric3, metric4 = st.columns(4)
            metric1.metric("Accuracy", f"{accuracy:.2f}")
            metric2.metric("Precision", f"{precision:.2f}")
            metric3.metric("Recall", f"{recall:.2f}")
            metric4.metric("F1 Score", f"{f1:.2f}")

            class_counts = (
                clf_data["DemandClass"]
                .map({0: "Normal Demand", 1: "High Demand"})
                .value_counts()
                .reset_index()
            )
            class_counts.columns = ["Demand Class", "Count"]

            fig = px.bar(
                class_counts,
                x="Demand Class",
                y="Count",
                title="Demand Class Distribution",
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("The Demand values create only one class.")
    else:
        st.warning("Not enough valid data for classification.")
else:
    st.warning("At least one numeric feature and 10 rows are required for classification.")

# ---------------------------------------------------------
# 6. TIME SERIES
# ---------------------------------------------------------
st.header("6. Time Series Analysis")

if "AnalysisDatetime" in df.columns:
    time_data = (
        df.dropna(subset=["AnalysisDatetime"])
        .set_index("AnalysisDatetime")[demand_col]
        .resample("D")
        .mean()
        .reset_index()
    )

    if len(time_data) >= 2:
        time_data["7-Day Moving Average"] = (
            time_data[demand_col]
            .rolling(window=7, min_periods=1)
            .mean()
        )

        fig = px.line(
            time_data,
            x="AnalysisDatetime",
            y=[demand_col, "7-Day Moving Average"],
            title="Daily Demand Trend and 7-Day Moving Average",
            labels={"value": "Demand", "AnalysisDatetime": "Date"},
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch")

        # Simple future forecast using a linear trend.
        if len(time_data) >= 3:
            forecast_days = 7
            x = np.arange(len(time_data))
            y = time_data[demand_col].values

            slope, intercept = np.polyfit(x, y, 1)

            future_x = np.arange(
                len(time_data),
                len(time_data) + forecast_days,
            )

            future_dates = pd.date_range(
                time_data["AnalysisDatetime"].max() + pd.Timedelta(days=1),
                periods=forecast_days,
                freq="D",
            )

            forecast_values = slope * future_x + intercept

            forecast_df = pd.DataFrame(
                {
                    "Date": future_dates,
                    "Forecast Demand": forecast_values,
                }
            )

            fig = px.line(
                forecast_df,
                x="Date",
                y="Forecast Demand",
                markers=True,
                title="7-Day Demand Forecast",
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, width="stretch")

            st.dataframe(forecast_df, width="stretch")
    else:
        st.warning("Not enough datetime information for time-series analysis.")
else:
    st.warning(
        "Datetime information was not found. Time-series analysis "
        "requires a StartDatetime or date/time column."
    )

# ---------------------------------------------------------
# 7. FEATURE IMPORTANCE
# ---------------------------------------------------------
st.header("7. Feature Importance")

importance_features = numeric_features(df, demand_col)

if len(importance_features.columns) >= 1 and len(df) >= 10:
    importance_data = pd.concat(
        [importance_features, df[demand_col]],
        axis=1,
    ).dropna()

    if len(importance_data) >= 10:
        X = importance_data.drop(columns=[demand_col])
        y = importance_data[demand_col]

        importance_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        )
        importance_model.fit(X, y)

        importance_df = pd.DataFrame(
            {
                "Feature": X.columns,
                "Importance": importance_model.feature_importances_,
            }
        ).sort_values("Importance", ascending=False)

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest Feature Importance",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Not enough data for feature importance.")
else:
    st.warning("Numeric features are required for feature importance.")

# ---------------------------------------------------------
# 8. PROJECT SUMMARY
# ---------------------------------------------------------
st.header("8. Project Summary")

st.markdown(
    """
    **Project objective:** Analyze EV charging usage patterns and predict
    charging demand.

    **Main steps completed:**
    - Uploaded and validated the CSV dataset
    - Removed duplicate rows and handled data types
    - Created date/time features
    - Performed exploratory data analysis
    - Built a Random Forest regression model for demand prediction
    - Built a Random Forest classification model for High/Normal Demand
    - Analyzed daily demand trends and a 7-day moving average
    - Generated a simple 7-day demand forecast
    - Displayed Random Forest feature importance
    - Presented the complete analysis through Streamlit
    """
)

st.success("EV Charging Demand Analysis completed successfully.")
