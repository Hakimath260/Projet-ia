import time
import random
import requests
from config import WRITE_API_KEY_RAW

URL = "https://api.thingspeak.com/update.json"

for i in range(5):
    payload = {
        "api_key": WRITE_API_KEY_RAW,
        "field1": round(random.uniform(15, 28), 2),      # Temperature
        "field2": round(random.uniform(40, 85), 2),      # Humidity
        "field3": round(random.uniform(1005, 1022), 2),  # Pressure
        "field4": round(random.uniform(0, 20), 2),       # WindSpeed
        "field5": round(random.choice([0.0, 0.0, 0.2, 1.1]), 2),  # Precipitation
        "field6": "simulated"
    }

    r = requests.post(URL, data=payload, timeout=20)
    print(f"Envoi {i+1}/5 -> status={r.status_code}, reponse={r.text}")
    time.sleep(16)
