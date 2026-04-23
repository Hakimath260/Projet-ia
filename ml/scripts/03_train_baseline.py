from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

import tensorflow as tf

# Dossiers
models_dir = Path("ml/models")
figures_dir = Path("results/figures")
metrics_dir = Path("results/metrics")
cm_dir = Path("results/confusion_matrices")

models_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)
metrics_dir.mkdir(parents=True, exist_ok=True)
cm_dir.mkdir(parents=True, exist_ok=True)

# Lecture du dataset préparé
df = pd.read_csv("data/processed/weather_4classes.csv")

print("=== Dataset initial ===")
print(df.head())
print()
print("Dimensions initiales :", df.shape)
print()

# On enlève temporairement la classe fog (label = 3) car beaucoup trop rare
df = df[df["label"] != 3].copy()

print("=== Dataset retenu pour entraînement (3 classes) ===")
print(df["label"].value_counts().sort_index())
print()
print("Dimensions retenues :", df.shape)
print()

# Features et labels
X = df[["temp", "rhum", "pres", "wspd", "prcp"]].values.astype(np.float32)
y = df["label"].values.astype(np.int32)

# Split train / validation / test
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Train :", X_train.shape, y_train.shape)
print("Val   :", X_val.shape, y_val.shape)
print("Test  :", X_test.shape, y_test.shape)
print()

# Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Sauvegarde du scaler
np.save(models_dir / "scaler_mean.npy", scaler.mean_)
np.save(models_dir / "scaler_scale.npy", scaler.scale_)

# Modèle baseline simple
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        restore_best_weights=True
    )
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Évaluation
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=["clear", "cloudy", "rain"],
    output_dict=True
)

print()
print("=== Accuracy test ===")
print(acc)
print()
print("=== Confusion Matrix ===")
print(cm)
print()
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["clear", "cloudy", "rain"]))

# Sauvegarde du modèle
model.save(models_dir / "weather_model_3classes.keras")

# Sauvegarde métriques
with open(metrics_dir / "baseline_3classes_report.json", "w") as f:
    json.dump(report, f, indent=2)

with open(metrics_dir / "baseline_3classes_accuracy.txt", "w") as f:
    f.write(f"accuracy={acc:.6f}\n")

np.save(cm_dir / "baseline_3classes_cm.npy", cm)

# Sauvegarde des courbes
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.title("Accuracy - Baseline 3 classes")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(figures_dir / "baseline_3classes_accuracy.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Loss - Baseline 3 classes")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(figures_dir / "baseline_3classes_loss.png")
plt.close()

print()
print("=== Fichiers sauvegardés ===")
print(models_dir / "weather_model_3classes.keras")
print(models_dir / "scaler_mean.npy")
print(models_dir / "scaler_scale.npy")
print(metrics_dir / "baseline_3classes_report.json")
print(metrics_dir / "baseline_3classes_accuracy.txt")
print(cm_dir / "baseline_3classes_cm.npy")
print(figures_dir / "baseline_3classes_accuracy.png")
print(figures_dir / "baseline_3classes_loss.png")
