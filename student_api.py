from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json

app = Flask(__name__)

# ==============================
# 🧠 تحميل الموديل والـ label mappings
# ==============================
model_path = "student_multi_model.pkl"     # اسم ملف الموديل
labels_path = "label_mappings.json"        # ملف التشفير

# تحميل الموديل
model = joblib.load(model_path)

# تحميل الـ label mappings
with open(labels_path, "r") as f:
    label_mappings = json.load(f)

# ==============================
# 🧩 دالة لتحويل الإدخال إلى DataFrame
# ==============================
def prepare_input(data):
    df = pd.DataFrame([data])
    return df

# ==============================
# 🚀 Route رئيسي لاختبار الاتصال
# ==============================
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "✅ Student Performance API is running successfully!"})

# ==============================
# 🔮 Route التنبؤ
# ==============================
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        # لو Power BI بيبعت GET request، نرجع مثال بسيط
        if request.method == 'GET':
            example = {
                "TransportMeans": 0,
                "ParentEduc": 1,
                "LunchType": 0,
                "TestPrep": 1,
                "ParentMaritalStatus": 2,
                "PracticeSport": 1,
                "IsFirstChild": 1,
                "NrSiblings": 3,
                "MathScore": 90,
                "ReadingScore": 85,
                "WritingScore": 88,
                "AttendanceRate": 95,
                "BehaviorIndex": 8,
                "SocialIndex": 7,
                "DailyStudyHours": 3,
                "AverageSleepHours": 7
            }
            return jsonify({
                "message": "Use POST to send student data for prediction.",
                "example_input": example
            })

        # لو POST request — Power BI أو Streamlit
        data = request.get_json()

        # تحويل الإدخال إلى DataFrame
        input_df = prepare_input(data)

        # عمل التنبؤ
        predictions = model.predict(input_df)

        # استخراج النتائج
        academic_avg = float(predictions[0][0])
        overall_perf = float(predictions[0][1])

        result = {
            "Predicted_Academic_Average": round(academic_avg, 2),
            "Predicted_Overall_Performance": round(overall_perf, 2)
        }

        # نرجع النتيجة على شكل JSON
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

# ==============================
# ▶️ تشغيل السيرفر
# ==============================
if __name__ == '__main__':
    app.run(debug=True)
