from caricamento_dati import variabili  # Importa la funzione variabili
from correlazione import correlation  # Importa la funzione correlation
from basis_risk import rischio_base
from condizioni_mercato import condizioni_mercato
from hedge_ratio import calcola_hedge_ratio    # Importa la funzione per calcolare il hedge ratio
#from condizioni_mercato import market_conditions 
from stazionarieta import stazionarieta
from GARCH_hedge_ratio import calcola_rendimenti, ccgarch_hedge_ratio

spot = 'Gas_Spot'
future = 'UK_Future'
log = "si"
def main():
    # Esegui la funzione variabili per ottenere i dati combinati
    dati = variabili()
    # Visualizza i dati per confermare che tutto funzioni correttamente
    print(dati)
    # Calcola e visualizza la matrice di correlazione passando i dati
    correlation_matrix = correlation(dati)
    # Mostra la matrice di correlazione calcolata
    print(correlation_matrix)
    #stzionarietà
    stazionarieta(dati)  
    #calcolo volatilita basis Risk:
    rischio_base(dati, spot, future)
    # calcolo condizioni di mercato
    condizioni_mercato(dati, spot, future) 
    #hedge ratio e test eteroschedasticità
    #calcola_hedge_ratio()
    calcola_hedge_ratio(spot, future, log)

    #GARCH
    rendimenti = calcola_rendimenti()
    print(rendimenti)

    ccgarch_hedge_ratio()

if __name__ == "__main__":
    main()
   
#---------------------------------------------------------------------------------------------------------------#
