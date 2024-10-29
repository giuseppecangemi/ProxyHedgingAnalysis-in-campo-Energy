import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from caricamento_dati import variabili  

def conditional_OLS_hedge_ratio(spot, future, log):
    # Prendo i dati dalla funzione scritta sullo script caricamento_dati
    data = variabili()

    #------------------------------------------------------------------------------------------------------------------------------------#
    # OLS con Return
    if log == "si":  # da selezionare sul main in base all'analisi sulla stazionarietà 
        # Differenza prima (Return)
        data['Gas_Spot_returns'] = np.log(data[spot]).pct_change() 
        data['Future_returns'] = np.log(data[future]).pct_change() 
    else:
        data['Gas_Spot_returns'] = data[spot].pct_change() 
        data['Future_returns'] = data[future].pct_change() 


    # Creazione delle variabili ritardate
    data['Future_returns_lag'] = data['Future_returns'].shift(1)
    data['Base_lag'] = (data['Gas_Spot_returns'] - data['Future_returns']).shift(1)  # Calcola la base e crea lag

    # Interazione tra Future Return e variabili laggate
    data['Interaction_lag'] = data['Future_returns'] * (data['Future_returns_lag'] + data['Base_lag'])

    # Pulizia dei dati
    data_cleaned = data.dropna()

    # Definisco le variabili per la regressione lineare
    Y = data_cleaned['Gas_Spot_returns']  # Variabile dipendente (spot)
    # Variabili indipendenti includendo il Future Return, le variabili laggate e le interazioni
    X = data_cleaned[['Future_returns', 'Interaction_lag']]
    #Siccome posso anche creare beta1 e beta2 con i lag (paper Miffre) provo a vedere se cambia qualcosa come stima
    X = data_cleaned[['Future_returns', 'Future_returns_lag', 'Base_lag', 'P_Base_lag']]


    # Aggiungo la costante nel modello di regressione
    X = sm.add_constant(X)

    # Fitting
    #model = sm.OLS(Y, X).fit()
    model = sm.OLS(Y, X).fit(cov_type='HC0')


    # Stampa dei risultati
    print(model.summary())

    # Estrazione dei parametri della regressione
    alpha = model.params[0]  
    beta_0 = model.params[1]  # Coefficiente per il return del future
    beta_1 = model.params[2]  # Coefficiente per l'interazione laggata (Future_returns * (Future_returns_lag + Base_lag))
    residui = model.resid

    # Stampa dei parametri
    print(f"Intercept (Alpha): {alpha:.4f}")
    print(f"Hedge Ratio (Beta_0 - Future Return): {beta_0:.4f}")
    print(f"Hedge Ratio (Beta_1 - Interaction Lag): {beta_1:.4f}")
    
    # TEST ETEROSCHEDASTICITA'
    print("TEST ETEROSCHEDASTICITA'... ")
    # utilizzo il test breusch-pagan
    test_results = sm.stats.diagnostic.het_breuschpagan(model.resid, model.model.exog)
    labels = ['LM Stat', 'LM p-value', 'F-stat', 'F p-value']
    
    # stampo i risultati del test
    print(dict(zip(labels, test_results)))
    if test_results[1] < 0.05:
        print(f"Il P-value {test_results[1]:.4f} è minore di 0.05: si rifiuta l'ipotesi nulla -> c'è eteroschedasticità!")
    else:
        print(f"Il P-value {test_results[1]:.4f} è maggiore di 0.05: si rifiuta l'ipotesi alternativa -> omoschedasticità!!!")

    importo_spot = float(input("Inserisci l'importo da investire nel mercato spot: "))
    # calcolo l'importo da investire nei futures per coprirmi dal rischio
    if log == "si":
        importo_futures = importo_spot * np.exp(beta_0)
        print(f"Importo da investire nei futures: {importo_futures:.2f}€")
    else:
        importo_futures = importo_spot * beta_0
        print(f"Importo da investire nei futures: {importo_futures:.2f}€")

    return model

#if __name__ == "__main__":
#    calcola_hedge_ratio()  # Salva il modello restituito
   # Esegui il test di eteroschedasticità
