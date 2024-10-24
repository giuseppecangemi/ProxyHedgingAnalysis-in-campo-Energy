import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from mvgarch.mgarch import DCCGARCH
from mvgarch.ugarch import UGARCH
import seaborn as sns
from caricamento_dati import variabili

def calcola_rendimenti(spot, future):
    dati = variabili()
    dati['Return_Spot'] = 10 * np.log(dati[spot].pct_change() + 1)  
    dati['Return_Future'] = 10 * np.log(dati[future].pct_change() + 1) 
    rendimenti = dati.dropna()
    return rendimenti

def dcc_garch_hedge_ratio(spot, future):
    # Carico i dati di rendimenti
    dati = calcola_rendimenti(spot, future)
    returns = dati[['Return_Spot', 'Return_Future']].copy()  # Creazione del DataFrame dei rendimenti

    # Imposto il modello GARCH(1,1) per entrambe le serie (spot e future)
    garch_specs = [UGARCH(order=(1, 1)) for _ in range(2)]  # Due asset

    # Fit DCCGARCH al DataFrame dei rendimenti
    dcc = DCCGARCH()
    dcc.spec(ugarch_objs=garch_specs, returns=returns)
    dcc_fit = dcc.fit()

    # Ottenere la correlazione dinamica stimata dal DCC
    correlations = dcc_fit.dynamic_corr  # A seconda della versione, potrebbe essere 'corr' o 'dynamic_corr'

    # Estrai le volatilità condizionali (deviazioni standard condizionali)
    h_spot = dcc_fit.ugarch_objs[0].conditional_volatility
    h_future = dcc_fit.ugarch_objs[1].conditional_volatility

    # Calcolo dell'OHR dinamico utilizzando la correlazione stimata
    OHR_dcc = correlations[:, 0, 1] * (h_spot / h_future)
    print(f"Hedge Ratio (DCC-GARCH): {OHR_dcc.mean():.4f}")

    # Plot dell'hedge ratio dinamico nel tempo
    plt.figure(figsize=(12, 8))
    
    # Plot delle volatilità condizionali
    plt.subplot(2, 1, 1)
    plt.plot(h_future, label='Volatilità Future', color='blue')
    plt.plot(h_spot, label='Volatilità Spot', color='green')
    plt.title('Volatilità Condizionale (GARCH(1,1))')
    plt.ylabel('Volatilità Condizionale')
    plt.legend()
    plt.grid(True)

    # Plot dell'Optimal Hedge Ratio dinamico
    plt.subplot(2, 1, 2)
    plt.plot(OHR_dcc, label='Optimal Hedge Ratio (DCC-GARCH)', color='red')
    plt.title('Optimal Hedge Ratio (DCC-GARCH) nel tempo')
    plt.xlabel('Data')
    plt.ylabel('Hedge Ratio')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Calcolo della riduzione della varianza come nel modello CCGARCH
    varianza_no_hedge = returns['Return_Spot'].var()

    # Rendimenti coperti con il modello DCC-GARCH
    rendimenti_coperti_dcc = returns['Return_Spot'] - (OHR_dcc * returns['Return_Future'])

    # Varianza copertura DCC-GARCH
    varianza_hedge_dcc = rendimenti_coperti_dcc.var()
    
    # Riduzione della varianza
    riduzione_varianza_dcc = (varianza_no_hedge - varianza_hedge_dcc) / varianza_no_hedge * 100
    print(f"Riduzione della varianza con DCC-GARCH: {riduzione_varianza_dcc:.2f}%")

    return OHR_dcc.mean(), OHR_dcc
