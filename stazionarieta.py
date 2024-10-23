import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from caricamento_dati import variabili  # Importa la funzione per ottenere i dati
from statsmodels.tsa.stattools import adfuller

def adf_test(series, title=''):
    print(f'Resultati del Test di Dickey-Fuller Aumentato per {title}:')
    result = adfuller(series.dropna(), autolag='AIC')
    labels = ['ADF Test Statistic', 'p-value', '# Lags Used', '# Observations Used']
    for value, label in zip(result, labels):
        print(f'{label}: {value}')
    if result[1] <= 0.05:
        print("La serie è stazionaria (ipotesi nulla rifiutata): si suggerisce di usare i prezzi in log!")
    else:
        print("La serie non è stazionaria (non posso rifiutare l'ipotesi nulla)")
    print('')


def stazionarieta(data):
    diff_gas_spot = data.Gas_Spot.diff().dropna()
    diff_gas_future = data.UK_Future.diff().dropna()

    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.plot(diff_gas_spot, label="Differenza Prima Prezzo Spot Gas", color='blue')
    plt.title('Differenza Prima: Prezzo Spot Gas Naturale')
    plt.grid(True)
    plt.subplot(212)
    plt.plot(diff_gas_future, label="Differenza Prima Prezzo Future Gas", color='green')
    plt.title('Differenza Prima: Prezzo Future Gas Naturale')
    plt.grid(True)
    plt.show()

    adf_test(diff_gas_spot, "Differenza Prima Log Prezzo Spot Gas Naturale")
    adf_test(diff_gas_future, "Differenza Prima Log Prezzo Future Gas Naturale")
  

#if __name__ == "__main__":
#    dati = variabili()
#    stazionarieta(dati)  




















