import streamlit as st

# ---------------------------------------------------------
# PAGE CONFIG — MUST BE THE FIRST STREAMLIT COMMAND
# ---------------------------------------------------------
st.set_page_config(page_title="Crop Yield Prediction", layout="wide")

# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------
import pandas as pd
import pickle

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
models = {
    "Decision Tree": "decision_tree_model.pkl",
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Lasso Regression": "lasso_model.pkl",
    "LightGBM": "lightgbm_model.pkl",
    "Linear Regression": "linearregression_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Ridge Regression": "ridge_regression_model.pkl",
    "XGBoost": "xgboost_model.pkl",
}

st.write("🔄 Initializing model loading...")

loaded_models = {}
for name, path in models.items():
    try:
        st.write(f"📥 Loading **{name}** from `{path}` ...")
        with open(path, "rb") as f:
            loaded_models[name] = pickle.load(f)
        st.success(f"✔ Successfully loaded: {name}")
    except Exception as e:
        st.error(f"❌ Failed to load {name}: {e}")
        pass

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("crop_yield.csv")

data = load_data()

# ---------------------------------------------------------
# UI START
# ---------------------------------------------------------
st.title("🌾 Crop Yield Prediction & Model Comparison")
st.markdown("### Compare predictions across 8 trained ML regression models.")

# SIDEBAR
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "Predict"])

# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------
if page == "Home":
    st.subheader("📌 Overview")
    st.write(
        """
        This application allows you to compare predictions from multiple
        machine learning models such as Decision Tree, Random Forest,
        Gradient Boosting, LightGBM, XGBoost, and more.
        
        Enter agricultural inputs → Generate predictions → Compare models.
        """
    )

    st.image(
        "https://img.freepik.com/free-vector/organic-flat-farm-landscape_23-2148956787.jpg",
        use_container_width=True
    )

# ---------------------------------------------------------
# DATASET PAGE
# ---------------------------------------------------------
elif page == "Dataset":
    st.subheader("📁 Dataset Preview")
    st.dataframe(data, use_container_width=True)

    st.write("### Column Information")
    st.table(pd.DataFrame({"Column": data.columns, "Type": data.dtypes.astype(str)}))

# ---------------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------------
elif page == "Predict":
    st.subheader("🧮 Enter Input Values for Prediction")

    col1, col2 = st.columns(2)
    inputs = {}

    for i, col in enumerate(data.columns):
        if data[col].dtype in ["int64", "float64"]:
            default_val = float(data[col].median())

            if i % 2 == 0:
                inputs[col] = col1.number_input(col, value=default_val)
            else:
                inputs[col] = col2.number_input(col, value=default_val)

        else:
            unique_vals = data[col].unique()

            if i % 2 == 0:
                inputs[col] = col1.selectbox(col, options=unique_vals)
            else:
                inputs[col] = col2.selectbox(col, options=unique_vals)

    input_df = pd.DataFrame([inputs])

    st.markdown("---")
    st.subheader("🔍 Model Predictions")

    if st.button("Predict Now", use_container_width=True):
        for name, model in loaded_models.items():
            try:
                pred = model.predict(input_df)[0]
                st.success(f"**{name} Prediction:** {pred}")
            except Exception as e:
                st.error(f"{name} failed: {e}")
