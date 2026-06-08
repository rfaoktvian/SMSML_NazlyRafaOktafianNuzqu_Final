# 7.Inference.py
# Autor: Nazly Rafa Oktafian Nuzqu

import requests, time

API_URL = "http://localhost:8000/predict"

samples = [
    {"name": "Pasien 1 - Berisiko Tinggi",
     "data": {"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,
               "restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}},
    {"name": "Pasien 2 - Berisiko Rendah",
     "data": {"age":37,"sex":1,"cp":2,"trestbps":130,"chol":250,"fbs":0,
               "restecg":1,"thalach":187,"exang":0,"oldpeak":3.5,"slope":0,"ca":0,"thal":2}},
    {"name": "Pasien 3 - Perempuan Usia Pertengahan",
     "data": {"age":56,"sex":0,"cp":1,"trestbps":140,"chol":294,"fbs":0,
               "restecg":0,"thalach":153,"exang":0,"oldpeak":1.3,"slope":1,"ca":0,"thal":2}},
]

print("=" * 50)
try:
    h = requests.get("http://localhost:8000/health", timeout=5)
    print(f"Health: {h.json()}\n")
except:
    print("ERROR: API belum berjalan! Jalankan 3.prometheus_exporter.py dulu."); exit()

for s in samples:
    print(f"[TEST] {s['name']}")
    r = requests.post(API_URL, json=s['data'], timeout=10)
    if r.status_code == 200:
        d = r.json()
        print(f"  Diagnosis: {d['diagnosis']} | Disease prob: {d['probabilities']['Disease']:.3f}")
    print()
