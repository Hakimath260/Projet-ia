import time
import requests
import pandas as pd
from config import WRITE_API_KEY_RAW

URL = "https://api.thingspeak.com/update.json"

df = pd.read_csv("/workspaces/Projet-ia/data/processed/weather_4classes.csv").head(5)

for i, row in df.iterrows():
    payload = {
        "api_key": WRITE_API_KEY_RAW,
        "field1": float(row["temp"]),
        "field2": float(row["rhum"]),
        "field3": float(row["pres"]),
        "field4": float(row["wspd"]),
        "field5": float(row["prcp"]),
        "field6": "meteostat_csv"
    }

    r = requests.post(URL, data=payload, timeout=20)
    print(f"Ligne {i} -> status={r.status_code}, reponse={r.text}")
    time.sleep(16)
