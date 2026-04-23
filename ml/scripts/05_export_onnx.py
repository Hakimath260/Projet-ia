from pathlib import Path
import tensorflow as tf

models_dir = Path("ml/models")
onnx_dir = Path("ml/onnx")
onnx_dir.mkdir(parents=True, exist_ok=True)

model_path = models_dir / "weather_model_final.keras"
output_path = onnx_dir / "weather_model_final.onnx"

print("Chargement du modèle :", model_path)
model = tf.keras.models.load_model(model_path)

# Appel explicite une première fois pour "builder" le modèle
dummy_input = tf.zeros((1, 5), dtype=tf.float32)
_ = model(dummy_input, training=False)

print("Export ONNX natif Keras 3...")
model.export(
    str(output_path),
    format="onnx",
    input_signature=[tf.TensorSpec(shape=(None, 5), dtype=tf.float32, name="input")]
)

print("\n=== Export ONNX terminé ===")
print("Modèle source :", model_path)
print("Modèle ONNX   :", output_path)