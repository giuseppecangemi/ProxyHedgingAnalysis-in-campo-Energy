from caricamento_dati import variabili 
from caricamento_dati_returns import variabili_returns 
from correlazione import correlation  
from basis_risk import rischio_base
from condizioni_mercato import condizioni_mercato
from OLS_hedge_ratio import OLS_hedge_ratio    
from stazionarieta import stazionarieta
from CCGARCH_hedge_ratio import calcola_rendimenti, ccgarch_hedge_ratio
from DCC_GARCH_hedge_ratio import dcc_garch_hedge_ratio
from Conditional_OLS_hedge_ratio import conditional_OLS_hedge_ratio

spot = 'Gas_Spot'
future = 'Future_TTF'
log = "no"
def main():
    #attivare le funzioni in base alla tipologia di dato: in level o differenze % (returns)
    dati = variabili()
    #print(dati)
    #dati = variabili_returns()
    #calcolo e view matrice correlazione
    correlation_matrix = correlation(dati)
    #sstampo la matrice
    print(correlation_matrix)
    #calcolo volatilita basis Risk:
    rischio_base(dati, spot, future)
    # calcolo condizioni di mercato
    condizioni_mercato(dati, spot, future) 
    #OLS hedge ratio e test eteroschedasticità
    OLS_hedge_ratio(spot, future, log)
    #CONDITIONAL OLS HEDGE RATIO
    conditional_OLS_hedge_ratio(spot, future, log)
    #stzionarietà
    stazionarieta(dati, spot, future) 
    #Constant Correlation GARCH
    rendimenti = calcola_rendimenti(spot, future)
    print(rendimenti)
    ccgarch_hedge_ratio(spot, future)

    #Dynamic Correlation GARCH
    #dcc_garch_hedge_ratio(spot, future)


if __name__ == "__main__":
    main()
   
#---------------------------------------------------------------------------------------------------------------#
