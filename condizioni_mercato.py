import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def condizioni_mercato(data, spot, future):  
    #calcolo la base (lo faccio gia nello script basis_risk -> capire come collegarlo)
    data['Basis'] = data[spot] - data[future]

    #logiche condizioni di mercato
    def market_conditions(row):
        if row['Basis'] > 0:
            return 'Backwardation'
        elif row['Basis'] < 0:
            return 'Contango'
        else:
            return 'Flat'

    data['Market_Condition'] = data.apply(market_conditions, axis=1)

    #stampo il numero di giorni in contango e backwardation
    contango_count = (data['Market_Condition'] == 'Contango').sum()
    backwardation_count = (data['Market_Condition'] == 'Backwardation').sum()

    print(f"Giorni in Contango: {contango_count}")
    print(f"Giorni in Backwardation: {backwardation_count}")

    #plotto la base con le condizioni di mercato
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Basis'], label='Basis (Prezzo Spot - Prezzo Future)', color='blue')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title('Andamento del Basis nel Tempo')
    plt.xlabel('Data')
    plt.ylabel('Basis')
    plt.fill_between(data.index, 0, data['Basis'], where=(data['Market_Condition'] == 'Contango'), color='orange', alpha=0.3, label='Contango')
    plt.fill_between(data.index, 0, data['Basis'], where=(data['Market_Condition'] == 'Backwardation'), color='green', alpha=0.3, label='Backwardation')
    plt.legend()
    plt.grid()
    plt.show()

    if contango_count > backwardation_count:
        print("La maggior parte del periodo analizzato è in contango.")
        print("Questo può aumentare il costo di mantenere una posizione a lungo termine e influire negativamente sull'efficacia del proxy hedging.")
    else:
        print("La maggior parte del periodo analizzato è in backwardation.")
        print("Questo può rendere più conveniente il mantenimento della posizione di copertura nel tempo.")
