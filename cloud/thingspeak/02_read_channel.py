import requests
import pandas as pd
from config import CHANNEL_ID_RAW, READ_API_KEY_RAW

url = f"https://api.thingspeak.com/channels/{CHANNEL_ID_RAW}/feeds.json"
params = {
    "api_key": READ_API_KEY_RAW,
    "results": 10
}

r = requests.get(url, params=params, timeout=20)
data = r.json()

feeds = data.get("feeds", [])
df = pd.DataFrame(feeds)

print("=== 10 dernieres lignes du channel RAW ===")
print(df[["created_at", "field1", "field2", "field3", "field4", "field5", "field6"]])
