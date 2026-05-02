import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
from PIL import Image

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# ---------------------------------------------------------
# PATH SETUP (IMPORTANT FIX)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models", "gru")

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "gru_crop_yield_model.keras")

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
@st.cache_resource
def load_preprocessors():
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    # IMPORTANT: keep categories same as training
    encoder.categories_ = [
        ['West', 'South', 'North', 'East'],
        ['Sandy', 'Clay', 'Loam', 'Silt'],
        ['Cotton', 'Rice', 'Barley', 'Soybean', 'Wheat', 'Maize'],
        [False, True],
        [False, True],
        ['Cloudy', 'Sunny', 'Rainy', 'Windy']
    ]
    return scaler, encoder

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# ---------------------------------------------------------
# INIT SESSION
# ---------------------------------------------------------
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Prediction'
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.markdown("## 🥸 Navigation")
st.sidebar.markdown("---")

pages = ['About Project', 'Prediction', 'Model Evaluation']

for page in pages:
    if st.sidebar.button(
        page,
        key=f"nav_{page}",
        use_container_width=True,
        type="primary" if st.session_state.current_page == page else "secondary"
    ):
        st.session_state.current_page = page
        st.rerun()

st.sidebar.markdown("---")

# ---------------------------------------------------------
# ABOUT PAGE
# ---------------------------------------------------------
if st.session_state.current_page == 'About Project':
    st.title("🌾 Crop Yield Prediction System")

    st.markdown("""
    ### 🎯 Project Overview
    Deep learning based GRU model to predict crop yield using environmental and agricultural data.

    ### 🔬 Tech Stack
    - TensorFlow / Keras
    - Streamlit
    - Scikit-learn
    """)

# ---------------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------------
elif st.session_state.current_page == 'Prediction':

    scaler, encoder = load_preprocessors()
    model = load_model()

    st.title("🌾 Crop Yield Prediction")

    progress = (st.session_state.step - 1) / 4
    st.progress(progress, text=f"Step {st.session_state.step} of 5")

    # Step 1
    if st.session_state.step == 1:
        region = st.selectbox('Region', ['West', 'South', 'North', 'East'])
        soil = st.selectbox('Soil Type', ['Sandy', 'Clay', 'Loam', 'Silt'])
        crop = st.selectbox('Crop', ['Cotton', 'Rice', 'Barley', 'Soybean', 'Wheat', 'Maize'])

        st.session_state.form_data.update({
            'region': region,
            'soil_type': soil,
            'crop': crop
        })

    # Step 2
    elif st.session_state.step == 2:
        rainfall = st.number_input('Rainfall (mm)', 0.0, 2000.0, 750.0)
        temperature = st.number_input('Temperature (°C)', 0.0, 50.0, 25.0)

        st.session_state.form_data.update({
            'rainfall_mm': rainfall,
            'temperature_celsius': temperature
        })

    # Step 3
    elif st.session_state.step == 3:
        fertilizer = st.checkbox('Fertilizer Used')
        irrigation = st.checkbox('Irrigation Used')

        st.session_state.form_data.update({
            'fertilizer_used': fertilizer,
            'irrigation_used': irrigation
        })

    # Step 4
    elif st.session_state.step == 4:
        weather = st.selectbox('Weather', ['Cloudy', 'Sunny', 'Rainy', 'Windy'])
        days = st.number_input('Days to Harvest', 1, 365, 100)

        st.session_state.form_data.update({
            'weather_condition': weather,
            'days_to_harvest': days
        })

    # Step 5
    elif st.session_state.step == 5:

        if st.button("🎯 Predict"):
            data = st.session_state.form_data

            new_df = pd.DataFrame({
                'Region': [data['region']],
                'Soil_Type': [data['soil_type']],
                'Crop': [data['crop']],
                'Rainfall_mm': [data['rainfall_mm']],
                'Temperature_Celsius': [data['temperature_celsius']],
                'Fertilizer_Used': [data['fertilizer_used']],
                'Irrigation_Used': [data['irrigation_used']],
                'Weather_Condition': [data['weather_condition']],
                'Days_to_Harvest': [data['days_to_harvest']]
            })

            numerical = ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest']
            categorical = ['Region', 'Soil_Type', 'Crop', 'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition']

            new_df[numerical] = scaler.transform(new_df[numerical])
            encoded = encoder.transform(new_df[categorical])

            final_input = np.concatenate([encoded, new_df[numerical].values], axis=1)
            final_input = final_input.reshape(1, 1, final_input.shape[1])

            pred = model.predict(final_input)[0][0]

            st.success(f"🌱 Predicted Yield: {pred:.2f} tons/hectare")

    # Navigation
    col1, col2 = st.columns(2)

    if col1.button("⬅ Back") and st.session_state.step > 1:
        st.session_state.step -= 1
        st.rerun()

    if col2.button("Next ➡") and st.session_state.step < 5:
        st.session_state.step += 1
        st.rerun()

# ---------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------
elif st.session_state.current_page == 'Model Evaluation':
    st.title("📊 Model Evaluation")

    st.metric("R² Score", "0.9132")
    st.metric("RMSE", "0.5018")
    st.metric("MAE", "0.4005")