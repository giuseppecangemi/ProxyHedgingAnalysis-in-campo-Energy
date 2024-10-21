import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.api as sms

# Funzione per scaricare dati storici da Yahoo Finance
def get_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close']

# Scarica dati storici per il GAS Naturale e il Future TTF
start_date = "2023-01-01"
end_date = "2024-01-01"
gas_spot = get_data('TTF=F', start_date, end_date)  # Gas naturale: utilizzo il future del gas naturale europeo come proxy per il gas naturale commodity ita
gas_future = get_data('NG=F', start_date, end_date) # Future: utilizzo il future americano maggiormente liquido
gas_spot = np.log(gas_spot)
gas_future = np.log(gas_future)
#grafiici
plt.plot(gas_spot, label="Prezzo SPOT Gas Naturale")
plt.plot(gas_future, label="Prezzo FUTURE sul Gas Naturale")
plt.legend()
plt.show()

# Unisci i dataset per la correlazione
data = pd.DataFrame({'Gas_Spot': gas_spot, 'TTF_Future': gas_future}).dropna()
#---------------------------------------------------------------------------------------------------------------#
# Calcolo della correlazione e dell'hedge ratio
correlation = data.corr().iloc[0, 1] #correlazione non buonissima
# Variabile indipendente 
X = data['TTF_Future']
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
sns.regplot(x='Gas_Spot', y='TTF_Future', data=data, line_kws={"color": "red"})
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
