import streamlit as st
import pandas as pd
import joblib
import base64
import os

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

# -----------------------------
# Background image (safe load)
# -----------------------------
def add_bg_from_local(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as image:
            encoded_string = base64.b64encode(image.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Optional: only works if you create assets folder
add_bg_from_local(os.path.join("assets", "background.jpg"))

# -----------------------------
# Paths (FIXED for your structure)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "regression")
SCALER_PATH = os.path.join(MODEL_PATH, "standard_scaler.pkl")

# -----------------------------
# Load models
# -----------------------------
models = {
    "Linear Regression": joblib.load(os.path.join(MODEL_PATH, "linear_regression.pkl")),
    "Ridge Regression": joblib.load(os.path.join(MODEL_PATH, "ridge_regression.pkl")),
    "Lasso Regression": joblib.load(os.path.join(MODEL_PATH, "lasso_regression.pkl")),
    "Decision Tree": joblib.load(os.path.join(MODEL_PATH, "decision_tree.pkl")),
    "Gradient Boosting": joblib.load(os.path.join(MODEL_PATH, "gradient_boosting.pkl")),
    "XGBoost": joblib.load(os.path.join(MODEL_PATH, "xgboost.pkl")),
    "LightGBM": joblib.load(os.path.join(MODEL_PATH, "lightgbm.pkl")),
}

scaler = joblib.load(SCALER_PATH)

# -----------------------------
# Title & description
# -----------------------------
st.title("🌾 Crop Yield Prediction System")
st.markdown(
    """
    Predict crop yield using different machine learning models.
    Select agricultural conditions and compare model outputs.
    """
)

# -----------------------------
# Sidebar - Model selection
# -----------------------------
st.sidebar.header("⚙️ Model Selection")
model_name = st.sidebar.selectbox(
    "Choose a prediction model",
    list(models.keys())
)

# -----------------------------
# Input layout
# -----------------------------
st.subheader("🌱 Input Agricultural Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    region = st.selectbox("Region", ["East", "North", "South", "West"])
    soil = st.selectbox("Soil Type", ["Clay", "Loam", "Peaty", "Sandy", "Silt"])
    crop = st.selectbox("Crop", ["Rice", "Wheat", "Maize", "Cotton", "Soybean"])

with col2:
    weather = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Cloudy"])
    rainfall = st.slider("Rainfall (mm)", 0, 300, 100)
    temperature = st.slider("Temperature (°C)", 5, 45, 25)

with col3:
    days = st.slider("Days to Harvest", 60, 150, 110)
    fertilizer = st.checkbox("Fertilizer Used")
    irrigation = st.checkbox("Irrigation Used")

# -----------------------------
# Prepare input data
# -----------------------------
input_dict = {
    "Rainfall_mm": rainfall,
    "Temperature_Celsius": temperature,
    "Fertilizer_Used": int(fertilizer),
    "Irrigation_Used": int(irrigation),
    "Days_to_Harvest": days,
}

# One-hot encoding (must match training)
for r in ["North", "South", "West"]:
    input_dict[f"Region_{r}"] = int(region == r)

for s in ["Clay", "Loam", "Peaty", "Sandy", "Silt"]:
    input_dict[f"Soil_Type_{s}"] = int(soil == s)

for c in ["Cotton", "Maize", "Rice", "Soybean", "Wheat"]:
    input_dict[f"Crop_{c}"] = int(crop == c)

for w in ["Rainy", "Sunny"]:
    input_dict[f"Weather_Condition_{w}"] = int(weather == w)

input_df = pd.DataFrame([input_dict])

# -----------------------------
# Prediction
# -----------------------------
if st.button("🌾 Predict Yield"):
    try:
        model = models[model_name]

        if model_name in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
            prediction = model.predict(scaler.transform(input_df))[0]
        else:
            prediction = model.predict(input_df)[0]

        st.success(f"🌱 Predicted Crop Yield: {prediction:.2f} tons per hectare")

    except Exception as e:
        st.error(f"Error during prediction: {e}")