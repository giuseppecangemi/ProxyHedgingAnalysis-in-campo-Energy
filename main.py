import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.api as sms
#---------------------------------------------------------------------------------------------------------------#
from caricamento_dati import variabili  # Importa la funzione variabili
from correlazione import correlation  # Importa la funzione correlation
from basis_risk import rischio_base
from condizioni_mercato import condizioni_mercato
from hedge_ratio import calcola_hedge_ratio    # Importa la funzione per calcolare il hedge ratio
#from condizioni_mercato import market_conditions 

def main():
    # Esegui la funzione variabili per ottenere i dati combinati
    dati = variabili()
    # Visualizza i dati per confermare che tutto funzioni correttamente
    print(dati)
    # Calcola e visualizza la matrice di correlazione passando i dati
    correlation_matrix = correlation(dati)
    # Mostra la matrice di correlazione calcolata
    print(correlation_matrix)
    #calcolo volatilita basis Risk:
    rischio_base(dati)
    # calcolo condizioni di mercato
    condizioni_mercato(dati) 
    #hedge ratio e test eteroschedasticità
    #calcola_hedge_ratio()
    calcola_hedge_ratio()


if __name__ == "__main__":
    main()
   


#---------------------------------------------------------------------------------------------------------------#

























#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
#effettivamente vedo un'ottima correlazione!!!!
#RISSUNTO SCELTE VARIABILI:
    #UTILIZZO IL TTF EUROPEO COME PROXY DEL PREZZO SPOT DEL GAS NATURALE ITALIAN0. MI ASPETTO CHE SIANO ALTAMENTE CORRELATI
    #PER COPRIRMI DA UN ACQUISTO DI GAS NATURALE ITALIANO VADO SHORT SUL FUTURE DEL GAS NATURALE UK:
        #DALLE ANALISI EFFETTUATE SI HA UNA CORR. 0.92 - SUFFRAGATA ANCHE DALL'ANALISI SU TRADING ECONOMICS!!
        #FACCIO QUESTO PROXY HEDGING POICHé NON TROVO STRUMENTI LIQUIDI PER LA COMMODITY GAS NAT ITA -> DEVIO SU GAS NAT UK FUTURE!
#richiamo le variabili da usare:
#      - gas_spot     (FUTURE USATO COME SPOT)
#      - uk_naturalgas (FUTURE)
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------------------------#
#TEST Breusch-Pagan
import statsmodels.stats.api as sms
#il test viene eseguito sui residui, quindi fitto il modello e ottengo i residui
fitted_values = model.fittedvalues
residuals = model.resid
# scatterplot: già si nota eteroschedasticità
plt.figure(figsize=(10, 6))
plt.scatter(fitted_values, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Grafico dei Residui vs Valori Predetti')
plt.xlabel('Valori Predetti')
plt.ylabel('Residui')
plt.grid(True)
plt.show()
#test
test_stat, p_value, _, _ = sms.het_breuschpagan(residuals, model.model.exog)
print(f"Statistiche del test di Breusch-Pagan: {test_stat}, P-value: {p_value}")
#se p-value < 0.05: rifiuto l'ipotesi nulla di non etoeroschedasticità
#in questo caso c'è eteroschedasticità
#se procedessi l'analisi, utilizzando la pendenza della retta di regressione come hedge ratio la mia stima sarebbe biased!
#rimane eteroschedastico anche usando standard error robusti come di seguito:
#model_robust = sm.OLS(Y, X).fit(cov_type='HC3') 











