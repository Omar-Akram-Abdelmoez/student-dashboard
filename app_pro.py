# =====================================================
#  Intelligent Student Performance Dashboard 
# =====================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import numpy as np
import json
import requests
import os

# ------------------- Load Data -------------------
@st.cache_data
def load_data():
    return pd.read_csv("DATA.csv")

data = load_data()
student_ids = data['student_id'].unique()

# ------------------- Load ML Model + Label Mappings -------------------
MODEL_PATH = "student_multi_model.pkl"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1PoS263lSD3NU8U83j6Hf90-2LvMZYwbT"

@st.cache_data
def load_model_and_labels():
    # تحميل الموديل من Google Drive لو مش موجود
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model from Google Drive...")
        r = requests.get(MODEL_URL, allow_redirects=True)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        st.write("Model downloaded successfully!")

    # فتح الموديل
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # تحميل الـ label mappings من GitHub repo (ملف صغير)
    with open("label_mappings.json", "r", encoding="utf-8") as f:
        label_mappings = json.load(f)

    return model, label_mappings

model, label_maps = load_model_and_labels()

# ------------------- Sidebar Controls -------------------
st.sidebar.title("Controls")
st.sidebar.subheader("Select Student by ID")
selected_id = st.sidebar.selectbox("Choose Student ID", student_ids)

# ------------------- Tabs / Pages -------------------
tab1, tab2, tab3, tab5, tab6 = st.tabs(
    ["Overview", "Subjects", "Attendance", "Insights", "AI Prediction"]
)

# =================== Tab 1: Overview ===================
with tab1:
    st.header("📊 Overview Dashboard")
    st.markdown("### General Overview of Student Performance Data")

    total_students = len(data)
    avg_overall = data['OverallPerformance'].mean()
    avg_attendance = data['AttendanceRate'].mean()
    avg_math = data['MathScore'].mean()
    avg_reading = data['ReadingScore'].mean()
    avg_writing = data['WritingScore'].mean()

    max_overall = data['OverallPerformance'].max()
    min_overall = data['OverallPerformance'].min()
    max_att = data['AttendanceRate'].max()
    min_att = data['AttendanceRate'].min()

    above_avg = (data['OverallPerformance'] > avg_overall).mean() * 100
    below_avg = (data['OverallPerformance'] < avg_overall).mean() * 100

    st.markdown("#### 🧮 General Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", f"{total_students:,}")
    col2.metric("Average Overall Performance", f"{avg_overall:.2f}")
    col3.metric("Average Attendance", f"{avg_attendance:.2f}%")

    st.markdown("#### 📚 Average Academic Scores")
    col4, col5, col6 = st.columns(3)
    col4.metric("Average Math", f"{avg_math:.2f}")
    col5.metric("Average Reading", f"{avg_reading:.2f}")
    col6.metric("Average Writing", f"{avg_writing:.2f}")

    st.markdown("#### 🏆 Top and Lowest Performance")
    col7, col8, col9, col10 = st.columns(4)
    col7.metric("Highest Overall", f"{max_overall:.2f}")
    col8.metric("Lowest Overall", f"{min_overall:.2f}")
    col9.metric("Highest Attendance", f"{max_att:.2f}%")
    col10.metric("Lowest Attendance", f"{min_att:.2f}%")

    st.markdown("#### 📈 Performance Distribution")
    col11, col12 = st.columns(2)
    col11.metric("Students Above Average", f"{above_avg:.1f}%")
    col12.metric("Students Below Average", f"{below_avg:.1f}%")

    st.markdown("---")
    st.markdown("<p style='text-align:center; color:gray;'>📘 These statistics are automatically generated from the system data</p>", unsafe_allow_html=True)

# =================== Tab 2: Subjects ===================
with tab2:
    st.header(f"Student Academic Details - ID {selected_id}")
    student_data = data[data['student_id'] == selected_id].iloc[0]
    avg_scores = data[['MathScore', 'ReadingScore', 'WritingScore']].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Math", student_data['MathScore'], f"{student_data['MathScore'] - avg_scores['MathScore']:.1f} vs avg")
    col2.metric("Reading", student_data['ReadingScore'], f"{student_data['ReadingScore'] - avg_scores['ReadingScore']:.1f} vs avg")
    col3.metric("Writing", student_data['WritingScore'], f"{student_data['WritingScore'] - avg_scores['WritingScore']:.1f} vs avg")
    avg_overall_local = float(data['OverallPerformance'].mean())
    col4.metric("Overall", float(student_data['OverallPerformance']), f"{float(student_data['OverallPerformance']) - avg_overall_local:.1f} vs avg")

    fig = px.bar(
        x=['Math', 'Reading', 'Writing'],
        y=[student_data['MathScore'], student_data['ReadingScore'], student_data['WritingScore']],
        title="Student Score Breakdown",
        labels={'x': 'Subject', 'y': 'Score'},
        color=['Math', 'Reading', 'Writing']
    )
    st.plotly_chart(fig, use_container_width=True)

# =================== Tab 3: Attendance ===================
with tab3:
    st.header("Attendance Overview")
    fig_att = px.histogram(data, x='AttendanceRate', nbins=20, title="Attendance Distribution")
    st.plotly_chart(fig_att, use_container_width=True)
    st.info("Attendance is a major factor in improving academic performance 🚀")

# ------------------- Tab 5: Insights -------------------
with tab5:
    st.header("📊 Global Insights Summary")

    total_students = len(data)
    avg_perf = data['OverallPerformance'].mean()
    avg_math = data['MathScore'].mean()
    avg_read = data['ReadingScore'].mean()
    avg_write = data['WritingScore'].mean()
    avg_study = data['WklyStudyHours_num'].mean()
    avg_attend = data['AttendanceRate'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", f"{total_students:,}")
    col2.metric("Avg Study Hours / week", f"{avg_study:.1f}")
    col3.metric("Avg Attendance Rate", f"{avg_attend:.1f}%")

    col4, col5, col6, col7 = st.columns(4)
    col4.metric("Avg Math", f"{avg_math:.1f}")
    col5.metric("Avg Reading", f"{avg_read:.1f}")
    col6.metric("Avg Writing", f"{avg_write:.1f}")
    col7.metric("Avg Overall Performance", f"{avg_perf:.1f}")

    st.divider()

    best = data.nlargest(1, 'OverallPerformance')[['student_id', 'OverallPerformance']]
    worst = data.nsmallest(1, 'OverallPerformance')[['student_id', 'OverallPerformance']]
    col8, col9 = st.columns(2)
    col8.metric("Top Student ID", f"{best.iloc[0]['student_id']}", f"{best.iloc[0]['OverallPerformance']:.1f}")
    col9.metric("Lowest Student ID", f"{worst.iloc[0]['student_id']}", f"{worst.iloc[0]['OverallPerformance']:.1f}")

    st.divider()

    score_gap = avg_write - avg_math
    if score_gap > 0:
        st.info(f"🟢 Students perform **{score_gap:.1f} points higher** in Writing than Math on average.")
    else:
        st.warning(f"🔴 Students perform **{abs(score_gap):.1f} points lower** in Writing than Math on average.")

    st.divider()

    high_attendance = (data['AttendanceRate'] >= 85).mean() * 100
    high_study = (data['WklyStudyHours_num'] >= 10).mean() * 100
    high_perf = (data['OverallPerformance'] >= 70).mean() * 100

    col10, col11, col12 = st.columns(3)
    col10.metric("High Attendance Students", f"{high_attendance:.1f}%")
    col11.metric("Active Study (10h+)", f"{high_study:.1f}%")
    col12.metric("High Performers (70+)", f"{high_perf:.1f}%")

    st.divider()

    st.subheader("💬 Key Insights")
    st.markdown(f"""
    - Dataset includes **{total_students:,} students**, with an average performance of **{avg_perf:.1f}%**.
    - Typical student studies around **{avg_study:.1f} hours/week** with **{avg_attend:.1f}% attendance**.
    - Writing skills outperform Math by **{abs(score_gap):.1f} points** on average.
    - Only **{high_perf:.1f}%** of students achieve a high-performance score (≥70%).
    - About **{high_attendance:.1f}%** maintain strong attendance, showing positive study discipline.
    """)
    st.success("✅ Insightful Summary: Attendance and consistent study habits are key success factors.")

# =================== Tab 6: What-If Analysis ===================
with tab6:
    st.header("AI Prediction")
    st.markdown("Modify student data below and see predicted Academic Average and Overall Performance.")

    user_data = {}

    for col, mapping in label_maps.items():
        options = list(mapping.values())
        user_choice = st.selectbox(f"{col}", options, key=f"whatif_{col}")
        encoded_value = [k for k, v in mapping.items() if v == user_choice][0]
        user_data[col] = int(encoded_value)

    user_data["MathScore"] = st.number_input("Math Score", min_value=0, max_value=100, value=75)
    user_data["ReadingScore"] = st.number_input("Reading Score", min_value=0, max_value=100, value=80)
    user_data["WritingScore"] = st.number_input("Writing Score", min_value=0, max_value=100, value=78)
    user_data["AttendanceRate"] = st.slider("Attendance Rate (%)", 0, 100, 90)
    user_data["BehaviorIndex"] = st.slider("Behavior Index", 0, 10, 5)
    user_data["SocialIndex"] = st.slider("Social Index", 0, 10, 5)
    user_data["StudyHours"] = st.slider("Daily Study Hours", 0, 12, 4)
    user_data["SleepHours"] = st.slider("Average Sleep Hours", 0, 12, 7)

    if st.button("🔮 Run AI Prediction", key="whatif_tab6"):
        input_df = pd.DataFrame([user_data])
        expected_features = model.estimators_[0].feature_names_in_
        for feat in expected_features:
            if feat not in input_df.columns:
                input_df[feat] = 0
        input_df = input_df[expected_features]

        predictions = model.predict(input_df)
        if isinstance(predictions[0], (list, np.ndarray)):
            academic_pred, overall_pred = predictions[0]
        else:
            academic_pred = predictions[0]
            overall_pred = None

        st.success("✅ Prediction Completed!")
        st.metric("Predicted Academic Average", f"{academic_pred:.2f}")
        if overall_pred is not None:
            st.metric("Predicted Overall Performance", f"{overall_pred:.2f}")

        st.progress(int(min(academic_pred, 100)))
        st.balloons()

    st.caption("Developed by Abdullah Ahmed — Student Performance ML Dashboard v4")
