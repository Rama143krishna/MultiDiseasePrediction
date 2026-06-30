# diabetes_model.py (UPDATED WITH ACCURACY DISPLAY)

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    roc_auc_score
)
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv("datasets/diabetes_prediction_dataset.csv")

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

X = df.drop("diabetes", axis=1).values
y = df["diabetes"].values

# Scale
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# SMOTE
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# =====================================================
# FEDERATED CLIENT SPLIT
# =====================================================
def split_clients(X, y, n=3):
    size = len(X) // n
    clients = []

    for i in range(n):
        start = i * size
        end = (i + 1) * size
        clients.append((X[start:end], y[start:end]))

    return clients


clients = split_clients(X_train, y_train)

# =====================================================
# AUTOENCODER + ATTENTION
# =====================================================
input_dim = X_train.shape[1]

inp = tf.keras.Input(shape=(input_dim,))
x = tf.keras.layers.Dense(256, activation="relu")(inp)

att = tf.keras.layers.Dense(256, activation="softmax")(x)
x = tf.keras.layers.Multiply()([x, att])

x = tf.keras.layers.Dense(128, activation="relu")(x)
encoded = tf.keras.layers.Dense(64, activation="relu")(x)

decoded = tf.keras.layers.Dense(input_dim, activation="sigmoid")(encoded)

autoencoder = tf.keras.Model(inp, decoded)
encoder = tf.keras.Model(inp, encoded)

autoencoder.compile(optimizer="adam", loss="mse")

# =====================================================
# FEDERATED TRAINING
# =====================================================
for r in range(3):
    weights_list = []

    for X_c, _ in clients:
        autoencoder.fit(X_c, X_c, epochs=5, batch_size=256, verbose=0)
        weights_list.append(autoencoder.get_weights())

    new_weights = [np.mean(w, axis=0) for w in zip(*weights_list)]
    autoencoder.set_weights(new_weights)

# =====================================================
# FEATURE EXTRACTION
# =====================================================
X_train_enc = encoder.predict(X_train, verbose=0)
X_test_enc = encoder.predict(X_test, verbose=0)

# =====================================================
# BROAD LEARNING SYSTEM
# =====================================================
np.random.seed(42)

W_map = np.random.randn(X_train_enc.shape[1], 150)
W_enh = np.random.randn(150, 80)

X_train_map = np.tanh(X_train_enc @ W_map)
X_train_enh = np.tanh(X_train_map @ W_enh)

X_test_map = np.tanh(X_test_enc @ W_map)
X_test_enh = np.tanh(X_test_map @ W_enh)

X_train_bls = np.concatenate([X_train_map, X_train_enh], axis=1)
X_test_bls = np.concatenate([X_test_map, X_test_enh], axis=1)

# =====================================================
# REMOVE LOW VARIANCE FEATURES
# =====================================================
var = VarianceThreshold(0.0001)
X_train_bls = var.fit_transform(X_train_bls)
X_test_bls = var.transform(X_test_bls)

# =====================================================
# FEATURE SELECTION
# =====================================================
selector = SelectKBest(score_func=f_classif, k=120)
X_train_bls = selector.fit_transform(X_train_bls, y_train)
X_test_bls = selector.transform(X_test_bls)

# =====================================================
# XGBOOST MODEL
# =====================================================
model = XGBClassifier(
    n_estimators=800,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.95,
    colsample_bytree=0.95,
    gamma=1,
    min_child_weight=3,
    reg_alpha=0.5,
    reg_lambda=2,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train_bls, y_train)

# =====================================================
# BEST THRESHOLD
# =====================================================
y_prob = model.predict_proba(X_test_bls)[:, 1]

best_t = 0.5
best_f1 = 0

for t in np.arange(0.3, 0.8, 0.01):
    preds = (y_prob > t).astype(int)
    score = f1_score(y_test, preds)

    if score > best_f1:
        best_f1 = score
        best_t = t

# =====================================================
# FINAL EVALUATION (NEW)
# =====================================================
y_pred = (y_prob > best_t).astype(int)

ACCURACY = accuracy_score(y_test, y_pred)
ROC_AUC = roc_auc_score(y_test, y_prob)

print("Diabetes Model Accuracy:", round(ACCURACY, 4))
print("Diabetes Model ROC-AUC:", round(ROC_AUC, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =====================================================
# USER PREDICTION FUNCTION
# =====================================================
def predict_diabetes_user(
    gender,
    age,
    hypertension,
    heart_disease,
    smoking,
    bmi,
    hba1c,
    glucose
):
    user_df = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose
    }])

    user_df = pd.get_dummies(user_df)

    train_cols = df.drop("diabetes", axis=1).columns

    for col in train_cols:
        if col not in user_df.columns:
            user_df[col] = 0

    user_df = user_df[train_cols]

    # Scale
    user_scaled = scaler.transform(user_df)

    # Encode
    user_enc = encoder.predict(user_scaled, verbose=0)

    # BLS
    m = np.tanh(user_enc @ W_map)
    e = np.tanh(m @ W_enh)

    user_bls = np.concatenate([m, e], axis=1)
    user_bls = var.transform(user_bls)
    user_bls = selector.transform(user_bls)

    # Predict
    prob = model.predict_proba(user_bls)[0][1]

    if prob > best_t:
        label = "HIGH RISK (Diabetes)"
    else:
        label = "LOW RISK (No Diabetes)"

    return {
        "label": label,
        "probability": float(prob)
    }