import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "fruit_spoilage.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "fruit_encoder.pkl")

# Load ML artifacts
model = None
scaler = None
fruit_encoder = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    elif os.path.exists(os.path.join(BASE_DIR, "output", "M6_Random_Forest", "random_forest_model.pkl")):
        model = joblib.load(os.path.join(BASE_DIR, "output", "M6_Random_Forest", "random_forest_model.pkl"))
        
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        
    if os.path.exists(ENCODER_PATH):
        fruit_encoder = joblib.load(ENCODER_PATH)
except Exception as e:
    print(f"Error loading models: {e}")

# Fruit status mapping
STATUS_MAP = {
    0: {"label": "Spoiled / Critical", "badge": "badge-danger", "color": "#ef4444", "status": "Spoiled"},
    1: {"label": "Deteriorating / At Risk", "badge": "badge-warning", "color": "#f59e0b", "status": "At Risk"},
    2: {"label": "Fresh & Prime Quality", "badge": "badge-success", "color": "#10b981", "status": "Fresh"}
}

def get_recommendation(fruit, status_code, temp, humidity, co2):
    if status_code == 2:
        return "Optimal storage conditions maintained. Produce is fresh and ready for market or cold chain transfer."
    elif status_code == 1:
        advice = []
        if temp > 24:
            advice.append("lower storage temperature to 18-22 C")
        if humidity > 92:
            advice.append("reduce relative humidity to prevent fungal bloom")
        if co2 > 350:
            advice.append("increase ventilation to lower ethylene and CO2 buildup")
        return "Warning: Quality degrading. " + (("Recommended: " + ", ".join(advice) + ".") if advice else "Inspect batch promptly.")
    else:
        return "Critical: Spoilage detected. Isolate batch immediately to prevent cross-contamination and bacterial spread in cold storage."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True) if request.is_json else request.form
        fruit = str(data.get("fruit", "Banana")).strip()
        temp = float(data.get("temp", 22.0))
        humidity = float(data.get("humidity", 90.0))
        light = float(data.get("light", 12.0))
        co2 = float(data.get("co2", 320.0))

        if not model or not scaler or not fruit_encoder:
            return jsonify({"success": False, "error": "Model artifacts not fully loaded"}), 500

        # Encode fruit
        try:
            fruit_encoded = fruit_encoder.transform([fruit])[0]
        except Exception:
            fruit_encoded = 0

        features_df = pd.DataFrame([{
            "Fruit": fruit_encoded,
            "Temp": temp,
            "Humidity": humidity,
            "Light": light,
            "CO2": co2
        }])

        scaled_features = pd.DataFrame(scaler.transform(features_df), columns=features_df.columns)
        prediction = int(model.predict(scaled_features)[0])
        probabilities = model.predict_proba(scaled_features)[0].tolist()

        status_info = STATUS_MAP.get(prediction, STATUS_MAP[1])
        freshness_score = round(probabilities[2] * 100, 1) if len(probabilities) > 2 else (100.0 if prediction == 2 else 15.0)
        recommendation = get_recommendation(fruit, prediction, temp, humidity, co2)

        return jsonify({
            "success": True,
            "prediction": prediction,
            "status": status_info["status"],
            "label": status_info["label"],
            "badge": status_info["badge"],
            "color": status_info["color"],
            "freshness_score": freshness_score,
            "probabilities": {
                "critical": round(probabilities[0] * 100, 1) if len(probabilities) > 0 else 0,
                "at_risk": round(probabilities[1] * 100, 1) if len(probabilities) > 1 else 0,
                "fresh": round(probabilities[2] * 100, 1) if len(probabilities) > 2 else 0
            },
            "recommendation": recommendation,
            "inputs": {
                "fruit": fruit,
                "temp": temp,
                "humidity": humidity,
                "light": light,
                "co2": co2
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/dataset")
def dataset_view():
    fruit_filter = request.args.get("fruit", "")
    status_filter = request.args.get("status", "")

    df = pd.read_csv(DATASET_PATH)
    total_records = len(df)
    
    # Statistics
    stats = {
        "total_records": total_records,
        "fruits": df["Fruit"].unique().tolist(),
        "classes": df["Class"].unique().tolist(),
        "avg_temp": round(df["Temp"].mean(), 2),
        "avg_humidity": round(df["Humid (%)"].mean(), 2),
        "avg_co2": round(df["CO2 (pmm)"].mean(), 2),
    }

    filtered_df = df
    if fruit_filter:
        filtered_df = filtered_df[filtered_df["Fruit"] == fruit_filter]
    if status_filter:
        filtered_df = filtered_df[filtered_df["Class"] == status_filter]

    samples = filtered_df.head(60).to_dict(orient="records")

    return render_template(
        "dataset.html",
        stats=stats,
        samples=samples,
        selected_fruit=fruit_filter,
        selected_status=status_filter,
        filtered_count=len(filtered_df)
    )

@app.route("/models")
def models_view():
    models_metrics = [
        {"name": "Random Forest (Production)", "accuracy": "99.88%", "precision": "99.88%", "recall": "99.88%", "f1": "99.88%", "status": "Active Best"},
        {"name": "Stacking Ensemble", "accuracy": "99.85%", "precision": "99.85%", "recall": "99.85%", "f1": "99.85%", "status": "Ensemble"},
        {"name": "Soft Voting Classifier", "accuracy": "99.82%", "precision": "99.81%", "recall": "99.82%", "f1": "99.81%", "status": "Ensemble"},
        {"name": "Hard Voting Classifier", "accuracy": "99.80%", "precision": "99.80%", "recall": "99.80%", "f1": "99.80%", "status": "Ensemble"},
        {"name": "Decision Tree", "accuracy": "99.50%", "precision": "99.51%", "recall": "99.50%", "f1": "99.50%", "status": "Baseline"},
        {"name": "Support Vector Machine (SVM)", "accuracy": "98.92%", "precision": "98.94%", "recall": "98.92%", "f1": "98.93%", "status": "Tested"},
        {"name": "K-Nearest Neighbors (KNN)", "accuracy": "98.71%", "precision": "98.73%", "recall": "98.71%", "f1": "98.71%", "status": "Tested"},
        {"name": "Logistic Regression", "accuracy": "89.45%", "precision": "89.20%", "recall": "89.45%", "f1": "89.31%", "status": "Baseline"}
    ]
    return render_template("models.html", models=models_metrics)

@app.route("/about")
def about_view():
    return render_template("about.html")

@app.route("/contact", methods=["GET", "POST"])
def contact_view():
    sent = False
    if request.method == "POST":
        sent = True
    return render_template("contact.html", sent=sent)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
