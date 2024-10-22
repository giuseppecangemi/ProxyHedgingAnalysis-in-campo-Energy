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
from hedge_ratio import calcola_hedge_ratio  # Importa la funzione per calcolare il hedge ratio
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

    #hedge ratio
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
# Regressione + Hedge Ratio sulle variabili maggiormente correlate
#devo fare una regressione cercando di spiegare il movimento del prezzo spot con quello del future
# Variabile indipendente 
X = data['uk_naturalgas']
# Variabile dipendente 
Y = data['Gas_Spot']
#aggiiungo costante
X = sm.add_constant(X)
# Fit del modello di regressione
model = sm.OLS(Y, X).fit()
print(model.summary())
intercept = model.params[0]
slope = model.params[1]
#analisi della regressione:
plt.figure(figsize=(10, 6))
sns.regplot(x='Gas_Spot', y='uk_naturalgas', data=data, line_kws={"color": "red"})
plt.title(f"Correlazione e Regressione tra NG=F e TTF=F (Correlazione: {correlation:.2f})")
plt.xlabel('Prezzo Future NG=F')
plt.ylabel('Prezzo Future TTF=F')
plt.grid(True)
plt.show()
#osserviamo una possibile eteroskedasticità
#sviluppo i test per verificarla
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













correlation_matrix = data.corr()

# Stampa la matrice di correlazione
print(correlation_matrix)

# Visualizza la matrice di correlazione con una heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={"shrink": .8})
plt.title('Matrice di Correlazione')
plt.show()






# Calcolo dell'hedge ratio (slope della regressione)
hedge_ratio = slope

# Quantità monetaria del contratto di vendita
contract_value = 10000  # Esempio di 10k euro

# Valore di future da comprare per hedging
hedge_value = contract_value * hedge_ratio

print(f"Correlazione tra gas spot e future: {correlation}")
print(f"Hedge Ratio: {hedge_ratio}")
print(f"Valore del future da comprare per hedging: {hedge_value}")
