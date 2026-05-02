# 🌾 Crop Yield Prediction & Optimization

## 📌 Overview

This project predicts crop yield using both **Machine Learning** and **Deep Learning (GRU)** models based on environmental and agricultural factors.

It provides two interactive applications:

* 📊 Regression-based model comparison
* 🧠 GRU-based deep learning prediction

The system helps in understanding how different factors influence crop yield and supports data-driven agricultural decisions.

---

## ✨ Features

* Predict crop yield from user inputs
* Compare multiple ML models in one place
* Deep learning prediction using GRU
* Simple and interactive UI using Streamlit
* Real-time results

---

## 🧠 Models Used

### 🔹 Machine Learning Models

* Linear Regression
* Ridge & Lasso Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM

### 🔹 Deep Learning

* GRU (Gated Recurrent Unit)

---

## 📊 Dataset

The dataset includes important agricultural and environmental features:

* Rainfall
* Temperature
* Soil Type
* Crop Type
* Fertilizer Usage
* Irrigation
* Weather Conditions
* Days to Harvest

🎯 Target: **Crop Yield (tons per hectare)**

---

## 📁 Project Structure

```
crop_yield_prediction_and_optimization/

├── apps/
│   ├── regression_app.py
│   └── gru_app.py
│
├── models/
│   ├── regression/
│   └── gru/
│
├── notebooks/
│   ├── regression_model/
│   └── gru_model/
│
├── dataset/
│   └── sample_data.csv
│
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## ⚙️ Setup Instructions (Run Locally)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/khushaligupta01/Crop_Yield_Prediction_And_Optimization.git
cd Crop_Yield_Prediction_And_Optimization
```

---

### 2️⃣ Create virtual environment

```bash
python -m venv venv
```

#### Activate environment

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run applications

### 🔹 Regression App

```bash
streamlit run apps/regression_app.py
```

### 🔹 GRU App

```bash
streamlit run apps/gru_app.py
```

---

## 📈 Results

* Regression models achieve **~0.91 R² score**
* GRU model performs comparably with strong generalization
* Ensemble models provide stable performance across datasets

---

## 🚧 Future Improvements

* Combine both apps into a single dashboard
* Add real-time weather data integration
* Improve UI/UX
* Add explainability (SHAP, LIME)
* Deploy using scalable cloud infrastructure

---
