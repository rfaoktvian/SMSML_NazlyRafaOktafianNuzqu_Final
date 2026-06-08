# 3.prometheus_exporter.py
# Kriteria 4 - Level Basic
# Serving model Heart Disease + Prometheus metrics
# Autor: Nazly Rafa Oktafian Nuzqu

import time, os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ── Prometheus Metrics (3 wajib untuk Basic) ──────────────────────────────────
PREDICTION_COUNTER = Counter(
    'heart_prediction_requests_total',
    'Total request prediksi', ['status']
)
PREDICTION_LATENCY = Histogram(
    'heart_prediction_latency_seconds',
    'Latensi prediksi (detik)',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
PREDICTION_CLASS_COUNTER = Counter(
    'heart_prediction_class_total',
    'Prediksi per kelas', ['diagnosis']
)
ACTIVE_REQUESTS = Gauge('heart_active_requests', 'Request aktif saat ini')
MODEL_LOAD_STATUS = Gauge('heart_model_loaded', '1 jika model loaded')

FEATURE_NAMES = [
    'age','sex','cp','trestbps','chol','fbs',
    'restecg','thalach','exang','oldpeak','slope','ca','thal'
]
CLASS_NAMES = {0: 'No Disease', 1: 'Disease'}

# ── Build model dari data lokal ───────────────────────────────────────────────
def build_model():
    csv_path = 'heart_preprocessing/heart_train_preprocessed.csv'
    if os.path.exists(csv_path):
        print("[INFO] Melatih model dari data preprocessed...")
        df = pd.read_csv(csv_path)
        X  = df.drop(columns=['target'])
        y  = df['target']
        model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        model.fit(X, y)
        scaler = StandardScaler().fit(X)
        print("[INFO] Model siap!")
        return model, scaler
    else:
        print("[WARN] Data tidak ditemukan. Membuat dummy model...")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=300, n_features=13, n_classes=2, random_state=42)
        scaler = StandardScaler(); X_s = scaler.fit_transform(X)
        model  = RandomForestClassifier(n_estimators=50, random_state=42); model.fit(X_s, y)
        return model, scaler

model, scaler = build_model()
MODEL_LOAD_STATUS.set(1 if model else 0)

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    ACTIVE_REQUESTS.inc()
    start = time.time()
    try:
        data    = request.get_json(force=True)
        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            PREDICTION_COUNTER.labels(status='error').inc()
            return jsonify({"error": f"Missing: {missing}"}), 400

        X_in   = pd.DataFrame([data])[FEATURE_NAMES]
        X_sc   = scaler.transform(X_in)
        pred   = int(model.predict(X_sc)[0])
        proba  = model.predict_proba(X_sc)[0].tolist()
        label  = CLASS_NAMES[pred]
        lat    = time.time() - start

        PREDICTION_LATENCY.observe(lat)
        PREDICTION_COUNTER.labels(status='success').inc()
        PREDICTION_CLASS_COUNTER.labels(diagnosis=label).inc()

        return jsonify({
            "prediction": pred, "diagnosis": label,
            "probabilities": {"No Disease": round(proba[0],4), "Disease": round(proba[1],4)},
            "latency_seconds": round(lat, 4)
        })
    except Exception as e:
        PREDICTION_COUNTER.labels(status='error').inc()
        return jsonify({"error": str(e)}), 500
    finally:
        ACTIVE_REQUESTS.dec()

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    print("API berjalan di http://localhost:8000")
    app.run(host='0.0.0.0', port=8000, debug=False)
