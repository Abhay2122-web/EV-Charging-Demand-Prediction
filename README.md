# EV Charging Demand Prediction and Usage Pattern Analysis

## Project Overview

This project analyzes EV charging station usage patterns and predicts charging demand using Python and Machine Learning.

The project follows an end-to-end workflow:

**Data Upload → Data Cleaning → EDA → Feature Engineering → Regression → Classification → Time Series → Model Explainability → Streamlit Dashboard**

The project is based on the requirements provided for the EV Charging Demand Prediction and Usage Pattern Analysis project.

---

## Problem Statement

The objective is to analyze EV charging station usage patterns and predict charging demand based on factors such as:

- Charger type
- Location
- Charging company
- Charging time
- Charging duration
- Historical usage patterns

The project includes regression, classification, time-series analysis, model evaluation, and model explainability.

---

## Project Objectives

1. Clean and preprocess EV charging data.
2. Analyze charging usage patterns using EDA.
3. Create useful time-based features.
4. Predict charging demand using regression.
5. Classify sessions into High Demand and Normal Demand.
6. Analyze charging demand trends over time.
7. Forecast future charging demand.
8. Identify important factors influencing demand.
9. Present the results through an interactive Streamlit dashboard.

---

## Dataset

The project uses an EV charging session CSV dataset.

The dataset contains fields related to charging sessions, including:

- Charging Session ID
- User ID
- Charger ID
- Charger Company
- Location
- Charger Type
- Charger Capacity
- Charger AC/DC
- Start Day
- Start Time
- End Day
- End Time
- Start Datetime
- End Datetime
- Duration
- Demand

### Dataset Upload

The dataset is uploaded through the Streamlit application.

The application contains a CSV file uploader in the sidebar:

**Dataset → Upload another CSV**

This makes the dataset replaceable. A compatible CSV can be uploaded without changing the Python source code.

The actual CSV dataset is not required to be stored in the GitHub repository.

---

## Main Project Sections

### 1. Dataset Overview

The dashboard displays:

- Total charging sessions
- Average demand
- Average charging duration
- Number of locations
- Processed dataset preview

### 2. Data Cleaning

The application performs basic preprocessing:

- Removes duplicate records
- Removes completely empty rows/columns
- Handles missing values
- Converts date/time fields
- Converts numerical fields to numeric format
- Prepares the data for analysis

### 3. Exploratory Data Analysis

The dashboard analyzes charging demand by:

- Location
- Charger type
- Charging hour

Charts are generated interactively using Plotly.

### 4. Regression

Regression is used to predict charging demand.

The project evaluates the regression model using:

- MAE
- RMSE
- R² Score

The main regression model used in the dashboard is:

**Random Forest Regressor**

### 5. Classification

Charging sessions are divided into two categories:

- **High Demand:** Demand >= median Demand
- **Normal Demand:** Demand < median Demand

The classification model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

The main classification model used is:

**Random Forest Classifier**

### 6. Time-Series Analysis

Charging demand is aggregated by day to identify demand trends.

The dashboard provides:

- Daily charging demand trend
- Future demand forecast
- Seven-day forecast

### 7. Model Explainability

Random Forest feature importance is used to identify the features that contribute most to demand prediction.

The dashboard displays:

- Feature importance chart
- Feature importance table

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit

---

## Project Structure

```text
EV_Charging_Project/
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

The CSV dataset is uploaded through Streamlit and therefore does not have to be stored in the repository.

---

## Requirements

Install the required libraries using:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Step 1: Open the project folder

```bash
cd EV_Charging_Project
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Start Streamlit

```bash
streamlit run app.py
```

### Step 4: Upload the dataset

When the Streamlit dashboard opens:

1. Go to the sidebar.
2. Find **Upload another CSV**.
3. Select the EV charging CSV file.
4. Wait for the dashboard to process the data.

---

## Dashboard Workflow

```text
CSV Upload
    ↓
Data Cleaning
    ↓
Feature Engineering
    ↓
EDA
    ↓
Regression
    ↓
Classification
    ↓
Time-Series Forecasting
    ↓
Feature Importance
    ↓
Final Dashboard
```

---

## Model Evaluation

### Regression

The regression model is evaluated using:

| Metric | Purpose |
|---|---|
| MAE | Average absolute prediction error |
| RMSE | Measures prediction error with greater penalty for large errors |
| R² | Measures how well the model explains variation in demand |

### Classification

The classification model is evaluated using:

| Metric | Purpose |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | Correct high-demand predictions among predicted high-demand sessions |
| Recall | High-demand sessions correctly identified |
| F1 Score | Balance between precision and recall |

---

## Business Use Cases

The analysis can support:

- Charging station demand forecasting
- EV charging station planning
- Resource and capacity planning
- Peak demand management
- Charger utilization analysis
- Operational optimization
- Customer experience improvement
- Future demand forecasting

---

## Key Project Outcome

The final application provides an interactive way to:

- Understand EV charging usage patterns
- Analyze charging demand
- Predict demand
- Identify High/Normal Demand sessions
- Study demand trends
- Forecast future demand
- Understand important demand-related features

---

## Important Note

The dashboard is designed to work with a replaceable CSV dataset.

The uploaded CSV should contain suitable EV charging information, especially a demand/energy target and relevant charging, location, duration, and date/time information.

The application validates the uploaded data and displays an error when required information is unavailable.

---

## Author

**Abhay Attri**

EV Charging Demand Prediction and Usage Pattern Analysis
