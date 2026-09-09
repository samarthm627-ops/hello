# hello
my first repository on github
import pandas as pd
import numpy as np

np.random.seed(42)

medicines = [
    ("Paracetamol", "Fever"),
    ("Cetirizine", "Allergy"),
    ("Azithromycin", "Antibiotic"),
    ("ORS", "Hydration"),
    ("Ibuprofen", "Painkiller"),
    ("Vitamin C", "Vitamin"),
    ("Cough Syrup", "Cough"),
    ("Antacid", "Digestive")
]

dates = pd.date_range(start="2023-01-01", end="2025-12-31")

data = []

for date in dates:
    month = date.month

    if month in [6, 7, 8, 9]:
        season = "Monsoon"
    elif month in [12, 1, 2]:
        season = "Winter"
    else:
        season = "Summer"

    temperature = np.random.uniform(20, 35)
    rainfall = np.random.uniform(0, 30)

    holiday = np.random.choice([0, 1], p=[0.9, 0.1])
    disease_level = np.random.choice(
        [1, 2, 3],
        p=[0.45, 0.40, 0.15]
    )

    for medicine, category in medicines:

        base_demand = {
            "Paracetamol": 100,
            "Cetirizine": 70,
            "Azithromycin": 45,
            "ORS": 80,
            "Ibuprofen": 60,
            "Vitamin C": 55,
            "Cough Syrup": 65,
            "Antacid": 50
        }[medicine]

        seasonal_effect = 0

        if season == "Monsoon" and medicine in [
            "Paracetamol",
            "Azithromycin",
            "Cough Syrup"
        ]:
            seasonal_effect = 30

        if season == "Summer" and medicine == "ORS":
            seasonal_effect = 45

        if season == "Winter" and medicine in [
            "Cetirizine",
            "Cough Syrup"
        ]:
            seasonal_effect = 25

        demand = (
            base_demand
            + seasonal_effect
            + disease_level * 10
            + holiday * 5
            + np.random.normal(0, 10)
        )

        sales = max(0, int(demand))

        price = np.random.uniform(20, 500)

        stock = sales + np.random.randint(20, 100)

        data.append([
            date,
            medicine,
            category,
            sales,
            price,
            season,
            temperature,
            rainfall,
            holiday,
            disease_level,
            stock
        ])

columns = [
    "date",
    "medicine_name",
    "category",
    "sales_quantity",
    "price",
    "season",
    "temperature",
    "rainfall",
    "holiday",
    "disease_level",
    "stock_available"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("data/pharmacy_sales.csv", index=False)

print("Dataset generated successfully!")
print("Rows:", len(df))
3. src/preprocess.py
import pandas as pd

def preprocess_data(file_path):

    df = pd.read_csv(file_path)

    # Convert date
    df["date"] = pd.to_datetime(df["date"])

    # Extract date features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek

    # Create previous-day sales
    df = df.sort_values(["medicine_name", "date"])

    df["previous_sales"] = (
        df.groupby("medicine_name")["sales_quantity"]
        .shift(1)
    )

    # Remove first record of each medicine
    df = df.dropna()

    return df
4. src/train.py

This is the main machine-learning code.

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import preprocess_data


# Load and preprocess data
df = preprocess_data("../data/pharmacy_sales.csv")

# Features
X = df[
    [
        "medicine_name",
        "category",
        "price",
        "season",
        "temperature",
        "rainfall",
        "holiday",
        "disease_level",
        "stock_available",
        "year",
        "month",
        "day",
        "day_of_week",
        "previous_sales"
    ]
]

# Target
y = df["sales_quantity"]


# Categorical features
categorical_features = [
    "medicine_name",
    "category",
    "season"
]

# Numerical features
numerical_features = [
    "price",
    "temperature",
    "rainfall",
    "holiday",
    "disease_level",
    "stock_available",
    "year",
    "month",
    "day",
    "day_of_week",
    "previous_sales"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# Transform features
X_processed = preprocessor.fit_transform(X)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=42
)


# Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# Prediction
predictions = model.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("Model Performance")
print("-----------------")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 2))


# Save model and preprocessing
joblib.dump(
    model,
    "../models/demand_model.pkl"
)

joblib.dump(
    preprocessor,
    "../models/preprocessor.pkl"
)

print("\nModel saved successfully!")
5. src/predict.py

This predicts medicine demand.

import pandas as pd
import joblib

model = joblib.load("../models/demand_model.pkl")
preprocessor = joblib.load("../models/preprocessor.pkl")


def predict_demand(
    medicine_name,
    category,
    price,
    season,
    temperature,
    rainfall,
    holiday,
    disease_level,
    stock_available,
    previous_sales
):

    data = pd.DataFrame([{
        "medicine_name": medicine_name,
        "category": category,
        "price": price,
        "season": season,
        "temperature": temperature,
        "rainfall": rainfall,
        "holiday": holiday,
        "disease_level": disease_level,
        "stock_available": stock_available,
        "year": 2026,
        "month": 9,
        "day": 10,
        "day_of_week": 3,
        "previous_sales": previous_sales
    }])

    processed_data = preprocessor.transform(data)

    prediction = model.predict(processed_data)

    return round(float(prediction[0]), 0)


result = predict_demand(
    "Paracetamol",
    "Fever",
    50,
    "Monsoon",
    27,
    15,
    0,
    2,
    150,
    120
)

print("Predicted Demand:", result, "units")
6. app/app.py

This gives you a simple web interface.

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Pharmacy Demand Forecasting",
    page_icon="💊"
)

st.title("💊 Pharmacy Demand Forecasting")

st.write(
    "Predict medicine demand using historical sales, "
    "seasonal patterns and local conditions."
)


# Load model
model = joblib.load("../models/demand_model.pkl")
preprocessor = joblib.load("../models/preprocessor.pkl")


# User inputs
medicine = st.selectbox(
    "Medicine",
    [
        "Paracetamol",
        "Cetirizine",
        "Azithromycin",
        "ORS",
        "Ibuprofen",
        "Vitamin C",
        "Cough Syrup",
        "Antacid"
    ]
)

category = st.selectbox(
    "Category",
    [
        "Fever",
        "Allergy",
        "Antibiotic",
        "Hydration",
        "Painkiller",
        "Vitamin",
        "Cough",
        "Digestive"
    ]
)

price = st.number_input(
    "Medicine Price",
    min_value=1.0,
    value=50.0
)

season = st.selectbox(
    "Season",
    ["Summer", "Monsoon", "Winter"]
)

temperature = st.number_input(
    "Temperature (°C)",
    value=28.0
)

rainfall = st.number_input(
    "Rainfall",
    value=10.0
)

holiday = st.selectbox(
    "Holiday",
    [0, 1]
)

disease_level = st.slider(
    "Local Disease Level",
    1,
    3,
    2
)

stock_available = st.number_input(
    "Current Stock",
    min_value=0,
    value=100
)

previous_sales = st.number_input(
    "Previous Sales",
    min_value=0,
    value=100
)


if st.button("Predict Demand"):

    data = pd.DataFrame([{
        "medicine_name": medicine,
        "category": category,
        "price": price,
        "season": season,
        "temperature": temperature,
        "rainfall": rainfall,
        "holiday": holiday,
        "disease_level": disease_level,
        "stock_available": stock_available,
        "year": 2026,
        "month": 9,
        "day": 10,
        "day_of_week": 3,
        "previous_sales": previous_sales
    }])

    processed_data = preprocessor.transform(data)

    prediction = model.predict(processed_data)[0]

    prediction = max(0, round(prediction))

    st.success(
        f"📦 Expected Demand: {prediction} units"
    )

    if prediction > stock_available:
        st.warning(
            "⚠️ Current stock may not be sufficient."
        )
    else:
        st.info(
            "✅ Current stock appears sufficient."
        )
