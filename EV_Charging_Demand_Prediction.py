"""
EV CHARGING DEMAND PREDICTION — COMPLETE SIMPLE PROJECT FILE

Easy workflow for faculty:
1. Load CSV
2. Check and clean data
3. Create time features
4. Explore demand patterns
5. Check outliers
6. Train/test split
7. Train Random Forest model
8. Evaluate predictions
9. Save results

Keep this Python file and "ChargingRecords (1).csv" in the same folder.
The original CSV is never overwritten.
"""

import pandas as pd
from pathlib import Path

# Machine-learning tools
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# STEP 1: Set file names
# ---------------------------------------------------------
FOLDER = Path(__file__).resolve().parent
INPUT_FILE = FOLDER / "ChargingRecords (1).csv"
OUTPUT_FOLDER = FOLDER / "EV_Project_Results"
OUTPUT_FOLDER.mkdir(exist_ok=True)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "CSV not found. Put 'ChargingRecords (1).csv' beside this Python file."
        )

    report = []

    def note(text=""):
        print(text)
        report.append(str(text))

    # ---------------------------------------------------------
    # STEP 2: Load the data
    # ---------------------------------------------------------
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    note("STEP 2 — LOAD DATA")
    note(f"Rows before cleaning: {len(df):,}")
    note(f"Columns before cleaning: {len(df.columns)}")
    note(f"Column names: {list(df.columns)}")
    note()

    # ---------------------------------------------------------
    # STEP 3: Check missing values
    # ---------------------------------------------------------
    note("STEP 3 — CHECK MISSING VALUES")
    missing = df.isna().sum()
    note(missing.to_string())
    note(f"Total missing cells: {int(missing.sum()):,}")
    missing.to_csv(OUTPUT_FOLDER / "missing_values_by_column.csv", header=["MissingCount"])
    note()

    # ---------------------------------------------------------
    # STEP 4: Remove empty rows and duplicates
    # ---------------------------------------------------------
    before = len(df)
    df = df.dropna(how="all").copy()
    note("STEP 4 — REMOVE EMPTY ROWS")
    note(f"Completely empty rows removed: {before - len(df):,}")

    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    note("STEP 5 — REMOVE DUPLICATES")
    note(f"Duplicate rows removed: {duplicate_count:,}")
    note()

    # ---------------------------------------------------------
    # STEP 6: Convert important columns to correct types
    # ---------------------------------------------------------
    if "Demand" not in df.columns:
        raise ValueError("The CSV needs a 'Demand' column for prediction.")

    df["Demand"] = pd.to_numeric(df["Demand"], errors="coerce")
    if "Duration" in df.columns:
        df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")
    if "StartDatetime" not in df.columns:
        raise ValueError("The CSV needs a 'StartDatetime' column for time features.")

    df["StartDatetime"] = pd.to_datetime(df["StartDatetime"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["Demand", "StartDatetime"]).copy()
    note("STEP 6 — CONVERT TYPES AND REMOVE INVALID TARGET/DATE")
    note(f"Rows removed for invalid/missing Demand or StartDatetime: {before - len(df):,}")
    note()

    # ---------------------------------------------------------
    # STEP 7: Create time-based features
    # ---------------------------------------------------------
    df["Hour"] = df["StartDatetime"].dt.hour
    df["DayOfWeek"] = df["StartDatetime"].dt.dayofweek
    df["Month"] = df["StartDatetime"].dt.month
    df["Day"] = df["StartDatetime"].dt.day
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    note("STEP 7 — CREATE FEATURES")
    note("Hour, DayOfWeek, Month, Day, and IsWeekend were added.")
    note("DayOfWeek: Monday=0, Sunday=6; IsWeekend: 1=weekend, 0=weekday.")
    note()

    # Save one cleaned dataset for later project use
    cleaned_file = OUTPUT_FOLDER / "EV_Charging_Cleaned.csv"
    df.to_csv(cleaned_file, index=False)
    note(f"Cleaned dataset saved: {cleaned_file.name}")
    note(f"Rows after cleaning: {len(df):,}")
    note(f"Columns after feature creation: {len(df.columns)}")
    note()

    # ---------------------------------------------------------
    # STEP 8: Explore average demand
    # ---------------------------------------------------------
    note("STEP 8 — AVERAGE DEMAND ANALYSIS")

    if "Location" in df.columns:
        by_location = df.groupby("Location", dropna=False)["Demand"].mean().sort_values(ascending=False)
        by_location.rename("AverageDemand").to_csv(OUTPUT_FOLDER / "average_demand_by_location.csv")
        note("Saved average_demand_by_location.csv")

    if "ChargerType" in df.columns:
        by_type = df.groupby("ChargerType", dropna=False)["Demand"].mean().sort_values(ascending=False)
        by_type.rename("AverageDemand").to_csv(OUTPUT_FOLDER / "average_demand_by_charger_type.csv")
        note("Saved average_demand_by_charger_type.csv")

    by_hour = df.groupby("Hour")["Demand"].mean().sort_index()
    by_hour.rename("AverageDemand").to_csv(OUTPUT_FOLDER / "average_demand_by_hour.csv")
    note("Saved average_demand_by_hour.csv")

    # Daily demand and 7-day moving average (trend summary, not future model forecast)
    daily = df.groupby(df["StartDatetime"].dt.date)["Demand"].mean().rename("AverageDemand").to_frame()
    daily.index = pd.to_datetime(daily.index)
    daily["7DayMovingAverage"] = daily["AverageDemand"].rolling(7, min_periods=1).mean()
    daily.to_csv(OUTPUT_FOLDER / "daily_demand_and_7day_average.csv")
    note("Saved daily_demand_and_7day_average.csv")
    note()

    # ---------------------------------------------------------
    # STEP 9: Detect possible outliers using IQR
    # ---------------------------------------------------------
    note("STEP 9 — OUTLIER CHECK (IQR METHOD)")
    outlier_rows = []
    for column in ["Demand", "Duration"]:
        if column in df.columns:
            values = df[column].dropna()
            if len(values) > 0:
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                count = int(((values < lower) | (values > upper)).sum())
                outlier_rows.append({
                    "Column": column,
                    "Q1": q1,
                    "Q3": q3,
                    "IQR": iqr,
                    "LowerBound": lower,
                    "UpperBound": upper,
                    "PossibleOutlierCount": count
                })
                note(f"{column}: {count:,} possible outliers by IQR.")
    pd.DataFrame(outlier_rows).to_csv(OUTPUT_FOLDER / "outlier_check.csv", index=False)
    note("Outliers are flagged for review, not automatically removed.")
    if "Duration" in df.columns:
        note(f"Negative Duration values needing review: {int((df['Duration'] < 0).sum()):,}")
    note()

    # ---------------------------------------------------------
    # STEP 10: Prepare features for Random Forest regression
    # ---------------------------------------------------------
    # Predict Demand using time and charger/location details.
    # Duration is excluded because it may not be known before a charging session.
    candidate_numeric = ["Hour", "DayOfWeek", "Month", "Day", "IsWeekend"]
    candidate_categorical = ["Location", "ChargerType", "ChargerCompany"]
    numeric_features = [c for c in candidate_numeric if c in df.columns]
    categorical_features = [c for c in candidate_categorical if c in df.columns]
    feature_columns = numeric_features + categorical_features

    if not feature_columns:
        raise ValueError("No usable prediction features were found in the CSV.")

    model_data = df[feature_columns + ["Demand"]].copy()
    model_data = model_data.dropna(subset=["Demand"])
    X = model_data[feature_columns]
    y = model_data["Demand"]

    note("STEP 10 — PREPARE MODEL DATA")
    note(f"Features used: {feature_columns}")
    note("Target to predict: Demand")
    note(f"Rows available for modelling: {len(model_data):,}")
    note()

    # ---------------------------------------------------------
    # STEP 11: Split data into training and testing sets
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    note("STEP 11 — TRAIN-TEST SPLIT")
    note("80% of data is used for training and 20% for testing.")
    note(f"Training rows: {len(X_train):,}; testing rows: {len(X_test):,}")
    note()

    # ---------------------------------------------------------
    # STEP 12: Train Random Forest regression model
    # ---------------------------------------------------------
    transformers = []
    if numeric_features:
        transformers.append(("numbers", "passthrough", numeric_features))
    if categorical_features:
        transformers.append((
            "categories",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ))

    preprocessing = ColumnTransformer(transformers=transformers)
    model = Pipeline(steps=[
        ("preprocessing", preprocessing),
        ("random_forest", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    note("STEP 12 — TRAIN RANDOM FOREST")
    note("Random Forest learns patterns from training data and predicts numeric Demand.")
    note()

    # STEP 12B: Show which features influenced the model most
    feature_names = model.named_steps["preprocessing"].get_feature_names_out()
    importances = model.named_steps["random_forest"].feature_importances_
    importance_table = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    importance_table.to_csv(OUTPUT_FOLDER / "random_forest_feature_importance.csv", index=False)
    note("Feature importance saved to random_forest_feature_importance.csv")
    note("Higher importance means the model relied more on that feature; it does not prove causation.")
    note()

    # ---------------------------------------------------------
    # STEP 13: Evaluate model performance
    # ---------------------------------------------------------
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    metrics = pd.DataFrame([{
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "TrainingRows": len(X_train),
        "TestingRows": len(X_test)
    }])
    metrics.to_csv(OUTPUT_FOLDER / "model_evaluation_metrics.csv", index=False)

    note("STEP 13 — EVALUATE MODEL")
    note(f"MAE:  {mae:.4f}  (average absolute prediction error)")
    note(f"RMSE: {rmse:.4f}  (penalizes larger errors more)")
    note(f"R²:   {r2:.4f}  (closer to 1 is generally better)")
    note("Metrics saved to model_evaluation_metrics.csv")
    note()

    # Save test predictions so actual and predicted Demand can be compared
    prediction_table = pd.DataFrame({
        "ActualDemand": y_test.to_numpy(),
        "PredictedDemand": predictions
    })
    prediction_table.to_csv(OUTPUT_FOLDER / "actual_vs_predicted_demand.csv", index=False)

    # ---------------------------------------------------------
    # STEP 14: Finish and save report
    # ---------------------------------------------------------
    note("STEP 14 — FINISH")
    note(f"Results folder: {OUTPUT_FOLDER}")
    note("Original CSV was not overwritten.")
    note("This script includes CSV preparation, basic analysis, outlier checking, and a Random Forest model.")
    note("A Streamlit dashboard is a separate app/interface and is not created by this script.")

    (OUTPUT_FOLDER / "PROJECT_RUN_SUMMARY.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print("\nProject run complete. Open the EV_Project_Results folder to view the outputs.")


if __name__ == "__main__":
    main()
