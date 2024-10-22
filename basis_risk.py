import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from caricamento_dati import variabili  # Importa la funzione per ottenere i dati

def rischio_base(data, spot, future):  # Modifica per accettare un argomento
    # Assicurati che i nomi delle colonne siano corretti
    # Sostituisci 'Gas_Spot' e 'NG_Future' con i nomi appropriati se sono diversi
    data.columns = ['Gas_Spot', 'Future_TTF', 'UK_Future', 'NG_Future', 'Brent_Future', 'WTI_Future', 'Heating_Oil', 'Gasoline']

    # Calcola la volatilità annualizzata
    volatilità_spot = data[spot].pct_change().std() * np.sqrt(252)  # Annualizzata
    volatilità_future = data[future].pct_change().std() * np.sqrt(252)  # Annualizzata

    print("ANALISI BASIS RISK | " f"Volatilità Spot: {volatilità_spot:.4f}, Volatilità Future: {volatilità_future:.4f}")

    # Calcolo della base
    data['Basis'] = data[spot] - data[future]

    # Volatilità della base
    basis_volatility = data['Basis'].std()

    print(f"Volatilità del Basis: {basis_volatility:.2f}")

    # Tracciare il basis nel tempo
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Basis'], label='Basis (Prezzo Spot - Prezzo Future)', color='blue')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)  # Linea di riferimento a 0
    plt.title('Andamento del Basis nel Tempo')
    plt.xlabel('Data')
    plt.ylabel('Basis')
    plt.legend()
    plt.grid()
    plt.show()

    # Analizza la distribuzione del basis
    plt.figure(figsize=(12, 6))
    plt.hist(data['Basis'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Distribuzione del Basis')
    plt.xlabel('Valore del Basis')
    plt.ylabel('Frequenza')
    plt.grid()
    plt.show()
