from caricamento_dati import variabili  # Importa la funzione variabili
from correlazione import correlation  # Importa la funzione correlation
from basis_risk import rischio_base
from condizioni_mercato import condizioni_mercato
from hedge_ratio import calcola_hedge_ratio    # Importa la funzione per calcolare il hedge ratio
#from condizioni_mercato import market_conditions 

spot = 'Gas_Spot'
future = 'UK_Future'
log = "no"
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
    rischio_base(dati, spot, future)
    # calcolo condizioni di mercato
    condizioni_mercato(dati, spot, future) 
    #hedge ratio e test eteroschedasticità
    #calcola_hedge_ratio()
    calcola_hedge_ratio(spot, future, log)

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
