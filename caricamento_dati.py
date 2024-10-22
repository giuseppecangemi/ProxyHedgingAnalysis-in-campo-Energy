import pandas as pd
import numpy as np
import yfinance as yf

# Funzione per scaricare dati storici da Yahoo Finance
def get_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close']

def variabili():
    # Definizione delle date
    start_date = "2023-01-01"
    end_date = "2024-01-01"

    # Scarica i dati storici per diversi future
    gas_future_ttf = get_data('TTF=F', start_date, end_date)
    gas_future = get_data('NG=F', start_date, end_date)
    brent_future = get_data('BZ=F', start_date, end_date)
    wti_future = get_data('CL=F', start_date, end_date)
    heating_oil = get_data('HO=F', start_date, end_date)
    gasoline = get_data('RB=F', start_date, end_date)
    etanolo_future = get_data('EH=F', start_date, end_date)

    # Carica i dati storici del gas naturale UK da un file Excel
    prezzo_spot = pd.read_excel(r"C:\Users\giucan03\OneDrive - Robert Half\Documents\Project\RISK\PROXY HEDGING\Py\project2\data.xlsx")
    prezzo_spot = pd.Series(prezzo_spot['psv_eur'].values, index=prezzo_spot['data'])
    uk_naturalgas = pd.read_excel(r"C:\Users\giucan03\OneDrive - Robert Half\Documents\Project\RISK\PROXY HEDGING\Py\project2\Natural Gas Futures Historical Data UK.xlsx")
    uk_naturalgas = pd.Series(uk_naturalgas['price'].values, index=uk_naturalgas['data'])

    # Crea il DataFrame combinando i dati disponibili
    data = pd.DataFrame({
        'Gas_Spot': prezzo_spot,
        'Future_TTF': gas_future_ttf,
        'UK_Future': uk_naturalgas,
        'NG_Future': gas_future,
        'Brent_Future': brent_future,
        'WTI_Future': wti_future,
        'Heating_Oil': heating_oil,
        'Gasoline': gasoline,
        
    })

    # Gestione dei dati mancanti (opzionale)
    data = data.dropna()  # Rimuove le righe con valori mancanti

    return data
