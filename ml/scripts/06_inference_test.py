from pathlib import Path
import numpy as np
import tensorflow as tf
import onnxruntime as ort

models_dir = Path("ml/models")
onnx_dir = Path("ml/onnx")

# Chargement scaler
mean = np.load(models_dir / "scaler_mean.npy")
scale = np.load(models_dir / "scaler_scale.npy")

# Exemple météo simulé : temp, rhum, pres, wspd, prcp
sample = np.array([[18.0, 70.0, 1015.0, 12.0, 0.0]], dtype=np.float32)
sample_scaled = (sample - mean) / scale

label_map = {
    0: "clear",
    1: "cloudy",
    2: "rain"
}

print("=== Entrée brute ===")
print(sample)
print()

print("=== Entrée normalisée ===")
print(sample_scaled)
print()

# ----- Test TensorFlow -----
tf_model = tf.keras.models.load_model(models_dir / "weather_model_final.keras")
tf_pred = tf_model.predict(sample_scaled, verbose=0)
tf_label = int(np.argmax(tf_pred, axis=1)[0])
tf_conf = float(np.max(tf_pred))

print("=== Inference TensorFlow ===")
print("Scores :", tf_pred.tolist())
print("Classe :", label_map[tf_label])
print("Confiance :", tf_conf)
print()

# ----- Test ONNX -----
session = ort.InferenceSession(str(onnx_dir / "weather_model_final.onnx"))
input_name = session.get_inputs()[0].name
onnx_pred = session.run(None, {input_name: sample_scaled.astype(np.float32)})[0]
onnx_label = int(np.argmax(onnx_pred, axis=1)[0])
onnx_conf = float(np.max(onnx_pred))

print("=== Inference ONNX ===")
print("Scores :", onnx_pred.tolist())
print("Classe :", label_map[onnx_label])
print("Confiance :", onnx_conf)
print()

print("=== Vérification ===")
print("Même classe TensorFlow / ONNX :", tf_label == onnx_label)
