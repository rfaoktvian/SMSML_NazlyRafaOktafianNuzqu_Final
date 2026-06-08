# modelling.py
# Kriteria 2 - Level Basic
# MLflow Tracking dengan autolog (lokal)
# Autor: Nazly Rafa Oktafian Nuzqu
#
# Cara menjalankan:
#   1. Jalankan MLflow UI dulu: mlflow ui --port 5000
#   2. python modelling.py
#   3. Buka localhost:5000 untuk melihat hasilnya

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Gunakan tracking lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Heart-Disease-Basic")

# Load dataset preprocessed
train_df = pd.read_csv('heart_preprocessing/heart_train_preprocessed.csv')
test_df  = pd.read_csv('heart_preprocessing/heart_test_preprocessed.csv')

X_train = train_df.drop(columns=['target'])
y_train = train_df['target']
X_test  = test_df.drop(columns=['target'])
y_test  = test_df['target']

# Aktifkan autolog — MLflow otomatis mencatat semua parameter & metrik
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RF_autolog_basic"):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[INFO] Accuracy: {acc:.4f}")
    print("[INFO] Run selesai! Buka localhost:5000 untuk melihat hasilnya.")
