from datetime import datetime
from pathlib import Path
import meteostat as ms
import pandas as pd

# Dossier de sortie
out_dir = Path("data/raw")
out_dir.mkdir(parents=True, exist_ok=True)

# Position : Chambéry
point = ms.Point(45.5646, 5.9178)

# Période
start = datetime(2022, 1, 1)
end = datetime(2024, 12, 31, 23, 59)

print("=== Recherche des stations proches ===")
stations = ms.stations.nearby(point, limit=5)
print(stations)

if stations is None or len(stations) == 0:
    raise RuntimeError("Aucune station proche trouvée.")

# Selon la doc, nearby retourne les stations triées par distance
# On prend la première station
first_station_id = stations.index[0]
print(f"\nStation choisie : {first_station_id}")

print("\n=== Vérification de l'inventaire ===")
inventory = ms.stations.inventory(first_station_id)
print(f"Début dispo : {inventory.start}")
print(f"Fin dispo   : {inventory.end}")
print(f"Paramètres  : {inventory.parameters}")

print("\n=== Téléchargement des données horaires ===")
ts = ms.hourly(first_station_id, start, end)
df = ts.fetch()

if df is None:
    raise RuntimeError(
        "fetch() a renvoyé None. Essaie une autre station proche ou une période plus courte."
    )

if df.empty:
    raise RuntimeError(
        "Le DataFrame est vide. Essaie une autre station proche ou une période plus courte."
    )

output_file = out_dir / "meteostat_chambery_hourly.csv"
df.to_csv(output_file)

print("\n=== Succès ===")
print("Fichier créé :", output_file)
print("\nColonnes disponibles :")
print(df.columns.tolist())
print("\nAperçu :")
print(df.head())
print("\nDimensions :", df.shape)