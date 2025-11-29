# =====================================================
# 🎓 Intelligent Student Performance Dashboard (Fixed v3)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# ------------------------------
# 1️⃣ Load model & label mappings
# ------------------------------
MODEL_PATH = "student_multi_model.pkl"
MAPPING_PATH = "label_mappings.json"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    label_mappings = json.load(f)

# ------------------------------
# 2️⃣ Streamlit page setup
# ------------------------------
st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("🎓 Student Performance Prediction Dashboard")
st.markdown("Enter student data to predict **Academic Average** and **Overall Performance**.")

# ------------------------------
# 3️⃣ Create input form dynamically
# ------------------------------
user_data = {}

st.header("🧾 Student Information")

# Generate dropdowns for categorical columns
for col, mapping in label_mappings.items():
    options = list(mapping.values())
    user_choice = st.selectbox(f"{col}", options)
    encoded_value = [k for k, v in mapping.items() if v == user_choice][0]
    user_data[col] = int(encoded_value)

# ------------------------------
# 4️⃣ Numeric input features (match training columns)
# ------------------------------
st.header("📊 Academic & Behavioral Data")

user_data["MathScore"] = st.number_input("Math Score", min_value=0, max_value=100, value=75)
user_data["ReadingScore"] = st.number_input("Reading Score", min_value=0, max_value=100, value=80)
user_data["WritingScore"] = st.number_input("Writing Score", min_value=0, max_value=100, value=78)
user_data["AttendanceRate"] = st.slider("Attendance Rate (%)", 0, 100, 90)
user_data["BehaviorIndex"] = st.slider("Behavior Index", 0, 10, 5)
user_data["SocialIndex"] = st.slider("Social Index", 0, 10, 5)
user_data["StudyHours"] = st.slider("Daily Study Hours", 0, 12, 4)
user_data["SleepHours"] = st.slider("Average Sleep Hours", 0, 12, 7)

# ------------------------------
# 5️⃣ Make prediction
# ------------------------------
if st.button("🔮 Predict Performance"):
    input_df = pd.DataFrame([user_data])

    # Align feature names with model training data
    expected_features = model.estimators_[0].feature_names_in_
    for feat in expected_features:
        if feat not in input_df.columns:
            input_df[feat] = 0  # fill missing columns safely
    input_df = input_df[expected_features]

    # Predict
    predictions = model.predict(input_df)

    if isinstance(predictions[0], (list, np.ndarray)):
        academic_pred, overall_pred = predictions[0]
    else:
        academic_pred = predictions[0]
        overall_pred = None

    # ------------------------------
    # 6️⃣ Display Results
    # ------------------------------
    st.success("✅ Prediction Completed!")
    st.metric("Predicted Academic Average", f"{academic_pred:.2f}")

    if overall_pred is not None:
        st.metric("Predicted Overall Performance", f"{overall_pred:.2f}")

    st.progress(int(min(academic_pred, 100)))
    st.balloons()

# ------------------------------
# 7️⃣ Footer
# ------------------------------
st.caption("Developed by Abdullah Ahmed — Student Performance ML Dashboard v3")
