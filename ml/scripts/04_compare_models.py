from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import tensorflow as tf

# Dossiers
models_dir = Path("ml/models")
metrics_dir = Path("results/metrics")

models_dir.mkdir(parents=True, exist_ok=True)
metrics_dir.mkdir(parents=True, exist_ok=True)

# Lecture du dataset
df = pd.read_csv("data/processed/weather_4classes.csv")

# On retire fog (label=3) pour rester cohérent avec le baseline
df = df[df["label"] != 3].copy()

X = df[["temp", "rhum", "pres", "wspd", "prcp"]].values.astype(np.float32)
y = df["label"].values.astype(np.int32)

# Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Sauvegarde du scaler
np.save(models_dir / "scaler_mean.npy", scaler.mean_)
np.save(models_dir / "scaler_scale.npy", scaler.scale_)

# Définition des architectures à comparer
def build_model(model_name):
    if model_name == "small_relu":
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(5,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax")
        ])

    elif model_name == "baseline_relu":
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(5,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax")
        ])

    elif model_name == "bigger_relu":
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(5,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax")
        ])

    elif model_name == "tanh_model":
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(5,)),
            tf.keras.layers.Dense(32, activation="tanh"),
            tf.keras.layers.Dense(16, activation="tanh"),
            tf.keras.layers.Dense(3, activation="softmax")
        ])

    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    return model

experiments = [
    {"name": "small_relu", "optimizer": "adam"},
    {"name": "baseline_relu", "optimizer": "adam"},
    {"name": "bigger_relu", "optimizer": "adam"},
    {"name": "tanh_model", "optimizer": "adam"},
    {"name": "baseline_relu_rmsprop", "optimizer": "rmsprop"},
]

results = []
best_model = None
best_name = None
best_test_acc = -1

for exp in experiments:
    exp_name = exp["name"]
    optimizer = exp["optimizer"]

    # Cas spécial pour baseline_relu_rmsprop
    if exp_name == "baseline_relu_rmsprop":
        model = build_model("baseline_relu")
        save_name = "baseline_relu_rmsprop"
    else:
        model = build_model(exp_name)
        save_name = exp_name

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

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
        verbose=0
    )

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    test_acc = accuracy_score(y_test, y_pred)

    val_best = max(history.history["val_accuracy"])
    params = model.count_params()

    results.append({
        "model_name": save_name,
        "optimizer": optimizer,
        "params": int(params),
        "best_val_accuracy": float(val_best),
        "test_accuracy": float(test_acc)
    })

    print(f"{save_name} | optimizer={optimizer} | params={params} | "
          f"best_val_acc={val_best:.4f} | test_acc={test_acc:.4f}")

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_model = model
        best_name = save_name

# Sauvegarde tableau comparaison
results_df = pd.DataFrame(results).sort_values(by="test_accuracy", ascending=False)
results_df.to_csv(metrics_dir / "model_comparison.csv", index=False)

# Sauvegarde du meilleur modèle
best_model.save(models_dir / "weather_model_final.keras")

with open(metrics_dir / "best_model.json", "w") as f:
    json.dump({
        "best_model_name": best_name,
        "best_test_accuracy": float(best_test_acc)
    }, f, indent=2)

print()
print("=== Résultats triés ===")
print(results_df)
print()
print(f"Meilleur modèle : {best_name}")
print(f"Accuracy test   : {best_test_acc:.4f}")
print(f"Modèle sauvegardé : {models_dir / 'weather_model_final.keras'}")
print(f"CSV sauvegardé    : {metrics_dir / 'model_comparison.csv'}")
