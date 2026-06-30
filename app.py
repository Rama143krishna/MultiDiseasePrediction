from flask import Flask, render_template, request, jsonify
import traceback

# Import prediction functions from your model files
from diabetes_model import predict_diabetes_user
from cardio_model import predict_cardio_user

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


# =========================
# Diabetes Prediction Route
# =========================
@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():
    try:
        data = request.get_json()

        result = predict_diabetes_user(
            gender=data["gender"],
            age=float(data["age"]),
            hypertension=int(data["hypertension"]),
            heart_disease=int(data["heart_disease"]),
            smoking=data["smoking"],
            bmi=float(data["bmi"]),
            hba1c=float(data["hba1c"]),
            glucose=float(data["glucose"])
        )

        return jsonify({
            "status": "success",
            "prediction": result["label"],
            "probability": round(result["probability"], 2)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        })


# =========================
# Cardio Prediction Route
# =========================
@app.route("/predict_cardio", methods=["POST"])
def predict_cardio():
    try:
        data = request.get_json()

        result = predict_cardio_user(
            age=float(data["age"]),
            gender=int(data["gender"]),
            height=float(data["height"]),
            weight=float(data["weight"]),
            ap_hi=float(data["ap_hi"]),
            ap_lo=float(data["ap_lo"]),
            cholesterol=int(data["cholesterol"]),
            gluc=int(data["gluc"]),
            smoke=int(data["smoke"]),
            alco=int(data["alco"]),
            active=int(data["active"])
        )

        return jsonify({
            "status": "success",
            "prediction": result["label"],
            "probability": round(result["probability"], 2)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        })


import os

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )