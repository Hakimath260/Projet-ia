from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fichiers
input_file = Path("data/raw/meteostat_chambery_hourly.csv")
output_dir = Path("data/processed")
figures_dir = Path("results/figures")

output_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

# Lecture du CSV
df = pd.read_csv(input_file)

print("=== Colonnes brutes ===")
print(df.columns.tolist())
print()

# Si la colonne temporelle existe, on la convertit proprement
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"])

# On garde seulement les colonnes utiles si elles existent
wanted_cols = ["time", "temp", "rhum", "pres", "wspd", "prcp", "coco"]
available_cols = [col for col in wanted_cols if col in df.columns]
df = df[available_cols].copy()

print("=== Colonnes retenues ===")
print(df.columns.tolist())
print()

# Remplacement éventuel des NA pandas par vrai NaN numérique
df = df.replace({pd.NA: np.nan})

# Conversion en numérique si besoin
for col in ["temp", "rhum", "pres", "wspd", "prcp", "coco"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Remplacer les précipitations manquantes par 0
if "prcp" in df.columns:
    df["prcp"] = df["prcp"].fillna(0.0)

# Fonction de création des labels météo à partir de coco + quelques règles simples
def map_weather(row):
    coco = row["coco"] if "coco" in row else np.nan
    prcp = row["prcp"] if "prcp" in row else 0.0
    rhum = row["rhum"] if "rhum" in row else np.nan

    # Règles principales basées sur le code météo Meteostat
    if pd.notna(coco):
        coco = int(coco)

        # clear / mostly clear
        if coco in [1, 2]:
            return 0  # clear

        # cloudy / overcast
        if coco in [3, 4, 5]:
            return 1  # cloudy

        # rain / showers / mixed precipitation
        if coco in [6, 7, 8, 9, 10, 11, 12, 13, 14]:
            return 2  # rain

        # fog / mist
        if coco in [15, 16]:
            return 3  # fog

    # règles de secours si coco est absent
    if pd.notna(prcp) and prcp > 0:
        return 2  # rain

    if pd.notna(rhum) and rhum >= 95:
        return 3  # fog

    return 1  # cloudy par défaut

# Création du label
df["label"] = df.apply(map_weather, axis=1)

# On garde uniquement les features du modèle + label
final_cols = ["temp", "rhum", "pres", "wspd", "prcp", "label"]
df = df[final_cols].copy()

# Suppression des lignes incomplètes
df = df.dropna()

# Conversion propre des types
df["label"] = df["label"].astype(int)

# Sauvegarde du dataset final
output_file = output_dir / "weather_4classes.csv"
df.to_csv(output_file, index=False)

print("=== Aperçu du dataset préparé ===")
print(df.head())
print()
print("Dimensions :", df.shape)
print()
print("Répartition des classes :")
print(df["label"].value_counts().sort_index())
print()
print("Répartition normalisée :")
print(df["label"].value_counts(normalize=True).sort_index())

# Petit graphique de répartition
label_names = {
    0: "clear",
    1: "cloudy",
    2: "rain",
    3: "fog"
}

counts = df["label"].value_counts().sort_index()
labels = [label_names[i] for i in counts.index]

plt.figure(figsize=(8, 5))
plt.bar(labels, counts.values)
plt.title("Répartition des classes météo")
plt.xlabel("Classe")
plt.ylabel("Nombre d'exemples")
plt.tight_layout()
plt.savefig(figures_dir / "class_distribution.png")
plt.close()

print()
print(f"Dataset sauvegardé : {output_file}")
print(f"Figure sauvegardée : {figures_dir / 'class_distribution.png'}")
