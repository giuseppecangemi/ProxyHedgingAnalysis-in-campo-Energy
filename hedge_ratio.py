import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from caricamento_dati import variabili  

def calcola_hedge_ratio(spot, future, log):
    #prendo i dati dalla funzione scritta sullo script caricamento_dati
    data = variabili()

    if log == "no": #da selezionare sul main in base all'analisi sulla stazionarietà 
        #log dei prezzi
        data['Gas_Spot'] = np.log(data[spot])
        data['Future'] = np.log(data[future])
    else:
        data['Gas_Spot'] = data[spot]
        data['Future'] = data[future]  

    #definisco le variabili per la regressione lineare
    X = data['Future']  #variabile indipendente (future)
    Y = data['Gas_Spot']    #variabile dipendente (spot)

    #aggiungo la costante nel modello di reg
    X = sm.add_constant(X)

    #fitting
    model = sm.OLS(Y, X).fit()
    print(model.summary())

    #estraggo i parametri della regressione
    intercept = model.params[0]  
    beta = model.params[1]        # Hedge Ratio (STATICO)

    print(f"Hedge Ratio (Beta): {beta:.4f}")
    print(f"Intercept: {intercept:.4f}")

    #plotto scatterplot della regressione
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Future'], data['Gas_Spot'], color='blue', label='Dati')
    plt.plot(data['Future'], model.fittedvalues, color='red', label='Regr. Lineare')
    plt.title('Regressione Lineare: Prezzo Spot vs Prezzo Future')
    plt.xlabel('Log(Prezzo Future NG=F)')
    plt.ylabel('Log(Prezzo Spot TTF=F)')
    plt.legend()
    plt.grid(True)
    plt.show()

    importo_spot = float(input("Inserisci l'importo da investire nel mercato spot: "))
    #calcolo l'importo da investire nei futures per coprirmi dal rischio
    if log == "si":
        importo_futures = importo_spot * np.exp(beta)
        print(f"Importo da investire nei futures: {importo_futures:.2f}€")
    else:
        importo_futures = importo_spot * beta
        print(f"Importo da investire nei futures: {importo_futures:.2f}€")

    #i residui del modello di reg sono omoschedastici?
    #TEST ETEROSCHEDASTICITA'
    print("TEST ETEROSCHEDASTICITA'... ")

    #utilizzo il test breusch-pagan
    test_results = sm.stats.diagnostic.het_breuschpagan(model.resid, model.model.exog)
    labels = ['LM Stat', 'LM p-value', 'F-stat', 'F p-value']
    
    #stampo i risultati del test
    print(dict(zip(labels, test_results)))
    if test_results[1] < 0.05:
        print(f"Il P-value {test_results[1]:.4f} è minore di 0.05: si rifiuta l'ipotesi nulla -> c'è eteroschedasticità!")
    else:
        print(f"Il P-value {test_results[1]:.4f} è maggiore di 0.05: si rifiuta l'ipotesi alternativa -> omoschedasticità!!!")
    
    return model

#if __name__ == "__main__":
#    calcola_hedge_ratio()  # Salva il modello restituito
   # Esegui il test di eteroschedasticità
