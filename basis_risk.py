import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from caricamento_dati import variabili  

def rischio_base(data, spot, future):  
    #da modificare per renderlo dinamico: in questo caso devo usare esattamente le stesse stringhe
    data.columns = ['Gas_Spot', 'Future_TTF', 'UK_Future', 'NG_Future', 'Brent_Future', 'WTI_Future', 'Heating_Oil', 'Gasoline']

    #calcolo la volatilità annualizzata
    volatilità_spot = data[spot].pct_change().std() #* np.sqrt(252) 
    volatilità_future = data[future].pct_change().std() #* np.sqrt(252) 

    print("ANALISI BASIS RISK | " f"Volatilità Spot: {volatilità_spot:.4f}, Volatilità Future: {volatilità_future:.4f}")

    #calcolo la "base" come differenza tra spot e future
    data['Basis'] = data[spot] - data[future]

    #stimo la volatilità della base
    basis_volatility = data['Basis'].std()
    print(f"Volatilità del Basis: {basis_volatility:.2f}")
    #questa formula calcola solo la deviazione standard del differenziale
    #per calcolare la varianza del basis considerando anche la correlazione e le deviazioni standard individuali dei prezzi spot e future devo modificarla come segue:
    sigma_S = data[spot].std()
    sigma_F = data[future].std()

    #calcolo la correlazione tra spot e future
    rho = data[spot].corr(data[future])

    # Calcolo della varianza del basis
    basis_variance = sigma_S**2 + sigma_F**2 - 2 * rho * sigma_S * sigma_F

    print(f"++++++++++++Varianza del Basis: {basis_variance:.4f}")


    #plotto la base nel tempo -> più bassa è meglio è
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Basis'], label='Basis (Prezzo Spot - Prezzo Future)', color='blue')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)  # Linea di riferimento a 0
    plt.title('Andamento del Basis nel Tempo')
    plt.xlabel('Data')
    plt.ylabel('Basis')
    plt.legend()
    plt.grid()
    plt.show()

    #plotto la distribuzione della base. valori vicino lo zero sono
    plt.figure(figsize=(12, 6))
    plt.hist(data['Basis'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Distribuzione del Basis')
    plt.xlabel('Valore del Basis')
    plt.ylabel('Frequenza')
    plt.grid()
    plt.show()
