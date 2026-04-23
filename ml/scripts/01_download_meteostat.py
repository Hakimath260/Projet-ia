pip install meteostat pandas numpy scikit-learn matplotlib tensorflow tf2onnx onnx
from meteostat import Point, Hourly
from datetime import datetime
import pandas as pd
from pathlib import Path

out_dir = Path("data/raw")
out_dir.mkdir(parents=True, exist_ok=True)

# Exemple : Chambéry
location = Point(45.5646, 5.9178)

start = datetime(2022, 1, 1)
end = datetime(2024, 12, 31)

df = Hourly(location, start, end).fetch()
df.to_csv(out_dir / "meteostat_chambery_hourly.csv")

print(df.head())
print(df.columns.tolist())
print(df.shape)
