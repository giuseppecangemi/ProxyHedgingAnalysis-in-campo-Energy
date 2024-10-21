import requests

# Esempio di richiesta API per ottenere dati del TTF spot da TradingEconomics
url = "https://api.tradingeconomics.com/commodity/ttf"
api_key = "d874463ced0846f:42s0g15s89g55vn"
response = requests.get(f"{url}?c={api_key}")
data = response.json()

# Manipolazione e analisi dei dati
