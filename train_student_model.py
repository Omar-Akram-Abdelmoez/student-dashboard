# ==============================
# 🤖 Student Performance Multi-Output Model
# ==============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json  # لتخزين mapping الفئات النصية

# ------------------------------
# 1️⃣ Load dataset
# ------------------------------
df = pd.read_csv(r"D:\ML-STD\train_student_model\DATA.csv")

# Drop unnecessary columns
df = df.drop(columns=["student_id", "student_name"], errors="ignore")

# ------------------------------
# 2️⃣ Encode categorical columns
# ------------------------------
categorical_cols = [
    'TransportMeans', 'ParentEduc', 'LunchType', 'TestPrep',
    'ParentMaritalStatus', 'PracticeSport', 'IsFirstChild', 'NrSiblings'
]

label_mappings = {}  # لحفظ القيم الأصلية

le = LabelEncoder()

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str)  # تأكد إنها نصوص
        df[col] = le.fit_transform(df[col])
        label_mappings[col] = {i: cls for i, cls in enumerate(le.classes_)}

# ------------------------------
# 3 Define features and targets
# ------------------------------
targets = ["OverallPerformance", "AcademicIndex"]
features = [col for col in df.columns if col not in targets]

X = df[features]
y = df[targets]

# ------------------------------
# 4 Split dataset
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# 5 Train Multi-output Model
# ------------------------------
base_model = RandomForestRegressor(
    n_estimators=150, random_state=42, max_depth=12
)
model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

# ------------------------------
#  Evaluate Model
# ------------------------------
preds = model.predict(X_test)

r2_overall = r2_score(y_test["OverallPerformance"], preds[:, 0])
r2_academic = r2_score(y_test["AcademicIndex"], preds[:, 1])

mae_overall = mean_absolute_error(y_test["OverallPerformance"], preds[:, 0])
mae_academic = mean_absolute_error(y_test["AcademicIndex"], preds[:, 1])

print("Model Evaluation:")
print(f"Overall Performance: R2 = {r2_overall:.3f}, MAE = {mae_overall:.3f}")
print(f"AcademicIndex : R² = {r2_academic:.3f}, MAE: {mae_academic:.3f}")

# ------------------------------
#  Feature Importance (approximate)
# ------------------------------
importances = np.mean(
    [est.feature_importances_ for est in model.estimators_], axis=0
)
plt.figure(figsize=(10,6))
pd.Series(importances, index=X.columns).nlargest(10).sort_values().plot(
    kind="barh", color="cornflowerblue"
)
plt.title("Top 10 Features Affecting Student Performance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# ------------------------------
#  Save model & label mappings
# ------------------------------
with open("student_multi_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_mappings.json", "w", encoding="utf-8") as f:
    json.dump(label_mappings, f, ensure_ascii=False, indent=4)

print("\n Model and label mappings saved successfully!")

# ------------------------------
#  Example prediction
# ------------------------------
sample = X_test.iloc[0:1]
predicted = model.predict(sample)[0]
print("\n Example Prediction:")
print(f"Predicted OverallPerformance: {predicted[0]:.2f}")
print(f"Predicted AcademicAverage: {predicted[1]:.2f}")

