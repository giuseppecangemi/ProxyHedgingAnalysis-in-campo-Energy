from caricamento_dati import variabili 
from correlazione import correlation  
from basis_risk import rischio_base
from condizioni_mercato import condizioni_mercato
from hedge_ratio import calcola_hedge_ratio    
from stazionarieta import stazionarieta
from GARCH_hedge_ratio import calcola_rendimenti, ccgarch_hedge_ratio

spot = 'Gas_Spot'
future = 'Future_TTF'
log = "si"
def main():
    dati = variabili()
    print(dati)
    #calcolo e view matrice correlazione
    correlation_matrix = correlation(dati)
    #sstampo la matrice
    print(correlation_matrix)
    #stzionarietà
    stazionarieta(dati, spot, future)  
    #calcolo volatilita basis Risk:
    rischio_base(dati, spot, future)
    # calcolo condizioni di mercato
    condizioni_mercato(dati, spot, future) 
    #hedge ratio e test eteroschedasticità
    calcola_hedge_ratio(spot, future, log)

    #GARCH
    rendimenti = calcola_rendimenti(spot, future)
    print(rendimenti)
    ccgarch_hedge_ratio(spot, future)

if __name__ == "__main__":
    main()
   
#---------------------------------------------------------------------------------------------------------------#
