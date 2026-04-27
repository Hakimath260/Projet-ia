from flask import Flask, render_template_string, send_from_directory, abort
from pathlib import Path
import time
import json
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
import onnxruntime as ort

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "results" / "figures"
METRICS_DIR = BASE_DIR / "results" / "metrics"
MODELS_DIR = BASE_DIR / "ml" / "models"
ONNX_DIR = BASE_DIR / "ml" / "onnx"

# ===== ThingSpeak config =====
RAW_CHANNEL_ID = 3355057
RAW_READ_KEY = "Q7PT1ZV56Y4FPZ94"

RESULTS_CHANNEL_ID = 3355061
RESULTS_READ_KEY = "5HAYC167EOFNKTIP"

CLASS_MAP = {
    0: "clear",
    1: "cloudy",
    2: "rain"
}

# ===== Load ML assets once =====
keras_model = None
onnx_session = None
onnx_input_name = None
scaler_mean = None
scaler_scale = None

def load_assets():
    global keras_model, onnx_session, onnx_input_name, scaler_mean, scaler_scale

    try:
        keras_model = tf.keras.models.load_model(MODELS_DIR / "weather_model_final.keras")
    except Exception as e:
        print("Erreur chargement modèle Keras:", e)
        keras_model = None

    try:
        onnx_session = ort.InferenceSession(str(ONNX_DIR / "weather_model_final.onnx"))
        onnx_input_name = onnx_session.get_inputs()[0].name
    except Exception as e:
        print("Erreur chargement modèle ONNX:", e)
        onnx_session = None
        onnx_input_name = None

    try:
        scaler_mean = np.load(MODELS_DIR / "scaler_mean.npy")
        scaler_scale = np.load(MODELS_DIR / "scaler_scale.npy")
    except Exception as e:
        print("Erreur chargement scaler:", e)
        scaler_mean = None
        scaler_scale = None

load_assets()

def safe_float(value):
    try:
        if value is None or value == "" or str(value).lower() == "null":
            return None
        return float(value)
    except Exception:
        return None

def fetch_latest_feed(channel_id, read_key, results=10):
    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json"
    params = {"api_key": read_key, "results": results}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    feeds = data.get("feeds", [])
    return feeds[-1] if feeds else None

def latest_raw_data():
    feed = fetch_latest_feed(RAW_CHANNEL_ID, RAW_READ_KEY, results=10)
    if not feed:
        return None

    return {
        "created_at": feed.get("created_at"),
        "temperature": safe_float(feed.get("field1")),
        "humidity": safe_float(feed.get("field2")),
        "pressure": safe_float(feed.get("field3")),
        "windspeed": safe_float(feed.get("field4")),
        "precipitation": safe_float(feed.get("field5")),
        "source": feed.get("field6"),
    }

def latest_cloud_result():
    feed = fetch_latest_feed(RESULTS_CHANNEL_ID, RESULTS_READ_KEY, results=10)
    if not feed:
        return None

    predicted_class = safe_float(feed.get("field1"))
    predicted_class_int = int(predicted_class) if predicted_class is not None else None

    return {
        "created_at": feed.get("created_at"),
        "predicted_class_id": predicted_class_int,
        "predicted_class_name": CLASS_MAP.get(predicted_class_int, "unknown") if predicted_class_int is not None else "unknown",
        "confidence": safe_float(feed.get("field2")),
        "temp_mean": safe_float(feed.get("field3")),
        "humidity_mean": safe_float(feed.get("field4")),
        "pressure_mean": safe_float(feed.get("field5")),
    }

def run_edge_inference_from_raw(raw):
    if raw is None:
        return {"error": "Aucune donnée raw disponible."}

    if scaler_mean is None or scaler_scale is None:
        return {"error": "Scaler non chargé."}

    values = [
        raw["temperature"],
        raw["humidity"],
        raw["pressure"],
        raw["windspeed"],
        raw["precipitation"],
    ]

    if any(v is None for v in values):
        return {"error": "Données raw incomplètes."}

    x = np.array([values], dtype=np.float32)
    x_scaled = (x - scaler_mean) / scaler_scale

    result = {
        "input_raw": values,
        "input_scaled": x_scaled.tolist()[0]
    }

    # Keras inference
    if keras_model is not None:
        t0 = time.perf_counter()
        pred_tf = keras_model.predict(x_scaled, verbose=0)
        tf_ms = (time.perf_counter() - t0) * 1000.0

        tf_label = int(np.argmax(pred_tf, axis=1)[0])
        tf_conf = float(np.max(pred_tf))

        result["tf_label_id"] = tf_label
        result["tf_label_name"] = CLASS_MAP.get(tf_label, "unknown")
        result["tf_confidence"] = tf_conf
        result["tf_time_ms"] = tf_ms
        result["tf_scores"] = pred_tf.tolist()[0]
    else:
        result["tf_error"] = "Modèle Keras non disponible."

    # ONNX inference
    if onnx_session is not None and onnx_input_name is not None:
        t0 = time.perf_counter()
        pred_onnx = onnx_session.run(None, {onnx_input_name: x_scaled.astype(np.float32)})[0]
        onnx_ms = (time.perf_counter() - t0) * 1000.0

        onnx_label = int(np.argmax(pred_onnx, axis=1)[0])
        onnx_conf = float(np.max(pred_onnx))

        result["onnx_label_id"] = onnx_label
        result["onnx_label_name"] = CLASS_MAP.get(onnx_label, "unknown")
        result["onnx_confidence"] = onnx_conf
        result["onnx_time_ms"] = onnx_ms
        result["onnx_scores"] = pred_onnx.tolist()[0]
    else:
        result["onnx_error"] = "Modèle ONNX non disponible."

    if "tf_label_id" in result and "onnx_label_id" in result:
        result["same_prediction"] = (result["tf_label_id"] == result["onnx_label_id"])

    return result

def load_training_table():
    csv_path = METRICS_DIR / "model_comparison.csv"
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")

def load_best_model_info():
    path = METRICS_DIR / "best_model.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

HTML = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Projet IA - Démo locale</title>
  <meta http-equiv="refresh" content="15">
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #f4f6f8; color: #1f2937; }
    header { background: #0f172a; color: white; padding: 18px 24px; }
    h1 { margin: 0; font-size: 28px; }
    .subtitle { margin-top: 6px; opacity: 0.9; }
    main { padding: 20px; max-width: 1400px; margin: auto; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 18px; }
    .card {
      background: white; border-radius: 14px; padding: 18px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }
    .card h2 { margin-top: 0; font-size: 20px; }
    .full { grid-column: 1 / -1; }
    .kpi { font-size: 22px; font-weight: bold; margin: 4px 0 10px 0; }
    .badge {
      display: inline-block; padding: 6px 10px; border-radius: 999px;
      font-size: 12px; font-weight: bold; margin-right: 8px;
      background: #dbeafe; color: #1d4ed8;
    }
    .ok { background: #dcfce7; color: #166534; }
    .warn { background: #fef3c7; color: #92400e; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }
    th, td { border: 1px solid #e5e7eb; padding: 8px; text-align: center; }
    th { background: #f8fafc; }
    .plots { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
    .plots img { width: 100%; border-radius: 12px; border: 1px solid #e5e7eb; background: white; }
    .small { font-size: 13px; color: #475569; }
    .mono { font-family: Consolas, monospace; font-size: 13px; white-space: pre-wrap; }
    .two-col { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  </style>
</head>
<body>
<header>
  <h1>Projet IA - Démo locale</h1>
  <div class="subtitle">STM32 + ThingSpeak + MATLAB + IA (TensorFlow / ONNX)</div>
</header>

<main>
  <div class="grid">
    <section class="card">
      <h2>1. Dernières données STM32 (ThingSpeak Raw)</h2>
      {% if raw %}
        <div class="badge ok">STM32 → ThingSpeak OK</div>
        <p class="small">Dernière mise à jour : <strong>{{ raw.created_at }}</strong></p>
        <table>
          <tr><th>Température</th><th>Humidité</th><th>Pression</th><th>Vent</th><th>Précipitation</th><th>Source</th></tr>
          <tr>
            <td>{{ raw.temperature }}</td>
            <td>{{ raw.humidity }}</td>
            <td>{{ raw.pressure }}</td>
            <td>{{ raw.windspeed }}</td>
            <td>{{ raw.precipitation }}</td>
            <td>{{ raw.source }}</td>
          </tr>
        </table>
      {% else %}
        <p>Aucune donnée Raw disponible.</p>
      {% endif %}
    </section>

    <section class="card">
      <h2>2. Dernier résultat Cloud (ThingSpeak Results)</h2>
      {% if cloud %}
        <div class="badge ok">MATLAB Analysis → Results OK</div>
        <p class="small">Dernière mise à jour : <strong>{{ cloud.created_at }}</strong></p>
        <div class="kpi">Classe cloud : {{ cloud.predicted_class_name }} ({{ cloud.predicted_class_id }})</div>
        <table>
          <tr><th>Confiance</th><th>TempMean</th><th>HumidityMean</th><th>PressureMean</th></tr>
          <tr>
            <td>{{ cloud.confidence }}</td>
            <td>{{ cloud.temp_mean }}</td>
            <td>{{ cloud.humidity_mean }}</td>
            <td>{{ cloud.pressure_mean }}</td>
          </tr>
        </table>
      {% else %}
        <p>Aucun résultat cloud disponible.</p>
      {% endif %}
    </section>

    <section class="card full">
      <h2>3. Comparaison Cloud vs Edge</h2>
      <p class="small">
        Cloud = résultat MATLAB/ThingSpeak. Edge = inférence locale avec le vrai modèle (Keras + ONNX)
        sur la même dernière entrée Raw.
      </p>

      {% if edge.error %}
        <p>{{ edge.error }}</p>
      {% else %}
        <div class="two-col">
          <div>
            <h3>Entrée utilisée</h3>
            <table>
              <tr><th>Temp</th><th>Hum</th><th>Pres</th><th>Wind</th><th>Prcp</th></tr>
              <tr>
                <td>{{ edge.input_raw[0] }}</td>
                <td>{{ edge.input_raw[1] }}</td>
                <td>{{ edge.input_raw[2] }}</td>
                <td>{{ edge.input_raw[3] }}</td>
                <td>{{ edge.input_raw[4] }}</td>
              </tr>
            </table>
          </div>
          <div>
            <h3>Comparaison rapide</h3>
            <table>
              <tr><th>Cloud (MATLAB)</th><th>Edge Keras</th><th>Edge ONNX</th></tr>
              <tr>
                <td>{{ cloud.predicted_class_name if cloud else "N/A" }}</td>
                <td>{{ edge.tf_label_name if edge.tf_label_name is defined else "N/A" }}</td>
                <td>{{ edge.onnx_label_name if edge.onnx_label_name is defined else "N/A" }}</td>
              </tr>
            </table>
          </div>
        </div>

        <table>
          <tr>
            <th>Moteur</th>
            <th>Classe</th>
            <th>Confiance</th>
            <th>Temps d'inférence (ms)</th>
          </tr>
          <tr>
            <td>TensorFlow / Keras</td>
            <td>{{ edge.tf_label_name if edge.tf_label_name is defined else "N/A" }}</td>
            <td>{{ "%.6f"|format(edge.tf_confidence) if edge.tf_confidence is defined else "N/A" }}</td>
            <td>{{ "%.3f"|format(edge.tf_time_ms) if edge.tf_time_ms is defined else "N/A" }}</td>
          </tr>
          <tr>
            <td>ONNX Runtime</td>
            <td>{{ edge.onnx_label_name if edge.onnx_label_name is defined else "N/A" }}</td>
            <td>{{ "%.6f"|format(edge.onnx_confidence) if edge.onnx_confidence is defined else "N/A" }}</td>
            <td>{{ "%.3f"|format(edge.onnx_time_ms) if edge.onnx_time_ms is defined else "N/A" }}</td>
          </tr>
        </table>

        {% if edge.same_prediction is defined %}
          <p><span class="badge ok">TensorFlow et ONNX donnent la même prédiction : {{ edge.same_prediction }}</span></p>
        {% endif %}
      {% endif %}
    </section>

    <section class="card full">
      <h2>4. Comparaison d'entraînement IA</h2>
      {% if best_model %}
        <p><span class="badge ok">Meilleur modèle : {{ best_model.best_model_name }}</span>
        <span class="badge">{{ best_model.best_test_accuracy }}</span></p>
      {% endif %}
      {% if training_rows %}
        <table>
          <tr>
            <th>Model</th>
            <th>Optimizer</th>
            <th>Params</th>
            <th>Best Val Acc</th>
            <th>Test Acc</th>
          </tr>
          {% for row in training_rows %}
          <tr>
            <td>{{ row.model_name }}</td>
            <td>{{ row.optimizer }}</td>
            <td>{{ row.params }}</td>
            <td>{{ row.best_val_accuracy }}</td>
            <td>{{ row.test_accuracy }}</td>
          </tr>
          {% endfor %}
        </table>
      {% else %}
        <p>Tableau de comparaison non trouvé.</p>
      {% endif %}
    </section>

    <section class="card full">
      <h2>5. Graphiques d'entraînement</h2>
      <div class="plots">
        <div>
          <p class="small"><strong>Répartition des classes</strong></p>
          <img src="/plot/class_distribution.png" alt="class_distribution">
        </div>
        <div>
          <p class="small"><strong>Accuracy baseline</strong></p>
          <img src="/plot/baseline_3classes_accuracy.png" alt="baseline_accuracy">
        </div>
        <div>
          <p class="small"><strong>Loss baseline</strong></p>
          <img src="/plot/baseline_3classes_loss.png" alt="baseline_loss">
        </div>
      </div>
    </section>
  </div>
</main>
</body>
</html>
"""

@app.route("/")
def index():
    raw = latest_raw_data()
    cloud = latest_cloud_result()
    edge = run_edge_inference_from_raw(raw)
    training_rows = load_training_table()
    best_model = load_best_model_info()

    return render_template_string(
        HTML,
        raw=raw,
        cloud=cloud,
        edge=edge,
        training_rows=training_rows,
        best_model=best_model
    )

@app.route("/plot/<filename>")
def plot(filename):
    allowed = {
        "class_distribution.png",
        "baseline_3classes_accuracy.png",
        "baseline_3classes_loss.png"
    }
    if filename not in allowed:
        abort(404)
    return send_from_directory(FIG_DIR, filename)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)