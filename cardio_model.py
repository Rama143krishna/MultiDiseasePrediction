# cardio_model.py

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

# ============================================================
# LOAD DATASET
# ============================================================
df = pd.read_csv("datasets/cardio_train.csv", sep=";")

# Remove id column if present
if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

# Convert age from days to years
df["age"] = df["age"] / 365

# BMI
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

# Pulse Pressure
df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]

# ============================================================
# SPLIT DATA
# ============================================================
X = df.drop("cardio", axis=1)
y = df["cardio"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# SCALING
# ============================================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# MODEL
# ============================================================
model = XGBClassifier(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.2,
    min_child_weight=3,
    reg_alpha=0.3,
    reg_lambda=1.5,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# ============================================================
# OPTIONAL EVALUATION (runs once on import)
# ============================================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

ACCURACY = accuracy_score(y_test, y_pred)
ROC_AUC = roc_auc_score(y_test, y_prob)

print("Cardio Model Accuracy:", round(ACCURACY, 4))
print("Cardio Model ROC-AUC:", round(ROC_AUC, 4))

# ============================================================
# USER PREDICTION FUNCTION
# ============================================================
def predict_cardio_user(
    age,
    gender,
    height,
    weight,
    ap_hi,
    ap_lo,
    cholesterol,
    gluc,
    smoke,
    alco,
    active
):
    # Feature engineering
    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = ap_hi - ap_lo

    input_data = np.array([[
        age,
        gender,
        height,
        weight,
        ap_hi,
        ap_lo,
        cholesterol,
        gluc,
        smoke,
        alco,
        active,
        bmi,
        pulse_pressure
    ]])

    input_scaled = scaler.transform(input_data)

    prob = model.predict_proba(input_scaled)[0][1]

    if prob > 0.5:
        label = "HIGH RISK (Cardiovascular Disease)"
    else:
        label = "LOW RISK"

    return {
        "label": label,
        "probability": float(prob)
    }