# 🌾 Crop Yield Prediction & Optimization

## 📌 Overview

This project is an end-to-end machine learning and deep learning system designed to **predict crop yield** based on environmental and agricultural factors.

It includes:

* 🔹 Multiple Machine Learning models (Regression)
* 🔹 Deep Learning model using GRU (Gated Recurrent Unit)
* 🔹 Interactive web applications built with Streamlit
* 🔹 Separate deployments for modular usage

---

## 🚀 Live Applications

* 🔗 **Regression App:** *(add your deployed link here)*
* 🔗 **GRU Deep Learning App:** *(add your deployed link here)*

---

## 🧠 Models Implemented

### 🔹 Machine Learning (Regression Models)

* Linear Regression
* Ridge Regression
* Lasso Regression
* Decision Tree
* Gradient Boosting
* Random Forest
* XGBoost
* LightGBM

### 🔹 Deep Learning

* GRU (Gated Recurrent Unit) Neural Network using TensorFlow/Keras

---

## 📊 Features

* 📈 Predict crop yield based on real-world agricultural inputs
* 🔍 Compare multiple regression models in one interface
* 🧠 Advanced GRU-based prediction with multi-step input flow
* 🎛️ Interactive and user-friendly UI using Streamlit
* ⚡ Real-time predictions

---

## 📁 Project Structure

```
crop_yield_prediction_and_optimization/

├── apps/
│   ├── gru_app.py              # GRU-based prediction app
│   └── regression_app.py       # ML models comparison app
│
├── models/
│   ├── gru/
│   │   ├── gru_crop_yield_model.keras
│   │   ├── scaler.pkl
│   │   └── encoder.pkl
│   │
│   └── regression/
│       ├── decision_tree.pkl
│       ├── gradient_boosting.pkl
│       ├── lasso_regression.pkl
│       ├── lightgbm.pkl
│       ├── linear_regression.pkl
│       ├── ridge_regression.pkl
│       ├── standard_scaler.pkl
│       └── xgboost.pkl
│
├── data/
│   └── sample_data.csv
│
├── notebooks/
│   ├── gru_model/
│   └── regression_model/
│
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Crop_Yield_Prediction_And_Optimization.git
cd Crop_Yield_Prediction_And_Optimization
```

---

### 2️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Applications

### 🔹 Regression App

```bash
streamlit run apps/regression_app.py
```

---

### 🔹 GRU App

```bash
streamlit run apps/gru_app.py
```

---

## 🌐 Deployment

This project is deployed using **Streamlit Community Cloud**.

### 🔹 Deployment Strategy

* Both apps are deployed **separately**
* Same repository, different entry points

| App        | Entry File               |
| ---------- | ------------------------ |
| Regression | `apps/regression_app.py` |
| GRU        | `apps/gru_app.py`        |

---

## 📊 Dataset

* The dataset contains features like:

  * Region
  * Soil Type
  * Crop Type
  * Rainfall
  * Temperature
  * Fertilizer Usage
  * Irrigation
  * Weather Conditions
  * Days to Harvest

* A sample dataset is included:

```
data/sample_data.csv
```

---

## 🧠 Tech Stack

* **Python**
* **Streamlit**
* **Scikit-learn**
* **TensorFlow / Keras**
* **Pandas, NumPy**
* **XGBoost, LightGBM**

---

## 📈 Model Performance (GRU)

* R² Score: **0.91+**
* RMSE: **~0.50**
* MAE: **~0.40**

---

## 🚧 Future Improvements

* 🔹 Merge both apps into a single unified dashboard
* 🔹 Add API support (FastAPI)
* 🔹 Integrate real-time weather data
* 🔹 Improve UI/UX with advanced visualization
* 🔹 Add model explainability (SHAP, LIME)

---

## ⚠️ Notes

* TensorFlow version is pinned for deployment compatibility
* Large model files may affect deployment performance
* Ensure correct file paths when running locally

---

Give it a ⭐ on GitHub — it helps!
