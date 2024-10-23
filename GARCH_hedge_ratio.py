import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
import statsmodels.api as sm
from caricamento_dati import variabili
import seaborn as sns


#calcolo dei rendimenti logaritmici
#per sviluppare i modelli garch abbiamo bisogno che la serie sia stazionaria. il log ci aiuta ad "appiattire" la serie.
def calcola_rendimenti(spot, future):
    dati = variabili()
    dati['Return_Spot'] = np.log(dati[spot].pct_change() + 1)
    dati['Return_Future'] = np.log(dati[future].pct_change() + 1)
    #i rendimenti sono molto piccoli quindi andrebbero riscalati:
    dati['Return_Spot'] = 10 * np.log(dati[spot].pct_change() + 1)  # Riscalamento dei ritorni spot
    dati['Return_Future'] = 10 * np.log(dati[future].pct_change() + 1)  # Riscalamento dei ritorni future

    rendimenti = dati.dropna()
    print(f"Questi sono i rendimenti:\n{rendimenti}")
    return rendimenti


#il modello naive è un approccio base: se vado long sul gas di 100 allora devo andare short sul future sempre di 100. In questo modo mi copro dal rischio.
def naive_hedge_ratio():
    return 1  # Hedge ratio costante di 1:1

#VaR per valutazione successiva!
def calcola_var(rendimenti, alpha=0.05):
    return np.percentile(rendimenti, 100 * alpha)

#sviluppo modello Constant Correlation GARCH (CCGARCH)
def ccgarch_hedge_ratio(spot, future):
    dati = calcola_rendimenti(spot, future) 

    # Modello GARCH(1,1) sui rendimenti dello spot
    model_spot = arch_model(dati['Return_Spot'].dropna(), vol='Garch', p=1, q=1, dist='normal')
    res_spot = model_spot.fit(disp='off') 
    # Modello GARCH(1,1) sui rendimenti del future
    model_future = arch_model(dati['Return_Future'].dropna(), vol='Garch', p=1, q=1, dist='normal')
    res_future = model_future.fit(disp='off')

    #ottengo le varianze condizionali
    h_future = res_future.conditional_volatility
    h_spot = res_spot.conditional_volatility

    #qui si assume che la correlazione tra spot e future sia costante nel tempo
    correlation = dati['Return_Spot'].corr(dati['Return_Future'])  
    print(correlation)
    #calcolo del OHR come Cov(condizionale) / Var(condizionale)
    OHR = correlation * (h_spot / h_future)
    print(f"Hedge Ratio (CCGARCH): {OHR.mean():.4f}")

    #plot per visualizzare la volatilità condizionale e l'OHR
    plt.figure(figsize=(12, 8))
    
    #plot volatilità condizionale
    plt.subplot(2, 1, 1)
    plt.plot(h_future, label='Volatilità Future', color='blue')
    plt.plot(h_spot, label='Volatilità Spot', color='green')
    plt.title('Volatilità Condizionale (GARCH(1,1))')
    plt.ylabel('Volatilità Condizionale')
    plt.legend()
    plt.grid(True)
    
    #plot del rapporto di copertura OHR nel tempo
    plt.subplot(2, 1, 2)
    plt.plot(OHR, label='Optimal Hedge Ratio (OHR)', color='red')
    plt.title('Optimal Hedge Ratio (OHR) nel tempo')
    plt.xlabel('Data')
    plt.ylabel('Hedge Ratio')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    #qui printo la volatilità NON condizionata (dovrebbe essere alpha + beta)
    print("VOLATILITà NON CONDIZIONATA:")
    omega_future = res_future.params['omega']
    alpha_future = res_future.params['alpha[1]']
    beta_future = res_future.params['beta[1]']

    omega_spot = res_spot.params['omega']
    alpha_spot = res_spot.params['alpha[1]']
    beta_spot = res_spot.params['beta[1]']

    #dal paper "Risk management and hedging approaches in energy markets" la volatilità non condizionata si stima come:
    unconditional_volatility_future = np.sqrt(omega_future / (1 - alpha_future - beta_future))
    unconditional_volatility_spot = np.sqrt(omega_spot / (1 - alpha_spot - beta_spot))

    print(f"Volatilità non condizionata del Future: {unconditional_volatility_future:.4f}")
    print(f"Volatilità non condizionata dello Spot: {unconditional_volatility_spot:.4f}")

    #VALUTAZIONE HEDGING 
    #uno degli elementi di rischio sui mercati è la volatilità. Pertanto vogliamo capire se l'utilizzo di questi hedge ratio siano in grado di ridurla.
    varianza_no_hedge = dati['Return_Spot'].var()
    
    importo = 100000  #assumo che l'investimento sia di 100k -> meglio se passare un argomento dal main!!!!!!!!

    #calcolo i rendimenti coperti con il modello GARCH, quindi i rendimenti dello spot meno la quota (h) dei rendimenti future
    rendimenti_coperti_ccgarch = dati['Return_Spot'] - (OHR * dati['Return_Future'])
    
    #calcolo la varianza della copertura con il modello GARCH
    varianza_hedge_ccgarch = rendimenti_coperti_ccgarch.var()
    
    #per comprendere se il modello GARCH ci riduce la varianza, faccio la differenza percentuale tra la varianza dei rendimenti spot - varianza della copertura
    riduzione_varianza_ccgarch = (varianza_no_hedge - varianza_hedge_ccgarch) / varianza_no_hedge * 100
    print(f"Varianza NO hedge: {varianza_no_hedge:.2f}%")
    print(f"Varianza hedge CCGARCH: {varianza_hedge_ccgarch:.2f}%")
    print(f"Riduzione della varianza con CCGARCH: {riduzione_varianza_ccgarch:.2f}%")
   
    #allo stesso modo calcolo i rendimenti del modello naive: cioè investo la stessa quota dello spot sul future 
    OHR_naive = naive_hedge_ratio()
    rendimenti_coperti_naive = dati['Return_Spot'] - (OHR_naive * dati['Return_Future'])

    #calcolo quindi la varianza del modello naive
    varianza_hedge_naive = rendimenti_coperti_naive.var()

    #calcolo la riduzione della varianza per il modello naive
    riduzione_varianza_naive = (varianza_no_hedge - varianza_hedge_naive) / varianza_no_hedge * 100
    print(f"Riduzione della varianza (Naïve): {riduzione_varianza_naive:.2f}%")

    #qui calcolo la differenza della riduzione della varianza tra naive e garch
    differenza_modelli_riduzione_varianza = riduzione_varianza_ccgarch - riduzione_varianza_naive
    print(f"Differenza nella riduzione della varianza (CCGARCH - Naïve): {differenza_modelli_riduzione_varianza:.2f}%")

    var_no_hedge = calcola_var(dati['Return_Spot'])
    var_hedged = calcola_var(dati['Return_Spot'] - (OHR * dati['Return_Future']))
    var_naive = calcola_var( dati['Return_Spot'] - (OHR_naive * dati['Return_Future']))

    print(f"Value at Risk (No Hedge): {var_no_hedge:.4f}")
    print(f"Value at Risk (Hedged GARCH): {var_hedged:.4f}")
    print(f"Value at Risk (Hedged Naive): {var_naive:.4f}")

    #plot delle distribuzioni
    plt.figure(figsize=(14, 6))

    #distribuzione dei rendimenti No Hedge
    plt.subplot(1, 3, 1)
    sns.histplot(dati['Return_Spot'], bins=30, kde=True, color='blue', stat='density')
    plt.axvline(var_no_hedge, color='red', linestyle='--', label=f'VaR No Hedge: {var_no_hedge:.4f}')
    plt.title('Distribuzione dei Rendimenti - No Hedge')
    plt.xlabel('Rendimenti')
    plt.ylabel('Densità')
    plt.legend()
    plt.grid(True)

    #distribuzione dei rendimenti Hedged GARCH
    plt.subplot(1, 3, 2)
    sns.histplot(rendimenti_coperti_ccgarch, bins=30, kde=True, color='green', stat='density')
    plt.axvline(var_hedged, color='red', linestyle='--', label=f'VaR Hedged GARCH: {var_hedged:.4f}')
    plt.title('Distribuzione dei Rendimenti - Hedged CCGARCH')
    plt.xlabel('Rendimenti')
    plt.ylabel('Densità')
    plt.legend()
    plt.grid(True)

    #distribuzione dei rendimenti Naive
    plt.subplot(1, 3, 3)
    sns.histplot(rendimenti_coperti_naive, bins=30, kde=True, color='red', stat='density')
    plt.axvline(var_naive, color='red', linestyle='--', label=f'VaR Hedged Naive: {var_naive:.4f}')
    plt.title('Distribuzione dei Rendimenti - Hedged Naive')
    plt.xlabel('Rendimenti')
    plt.ylabel('Densità')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    #faccio restituire l'OHR medio e la serie storica di hedge ratio
    return OHR.mean(), OHR

