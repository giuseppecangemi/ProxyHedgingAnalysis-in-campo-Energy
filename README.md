# Analisi della Correlazione e del Rischio Base nelle Commodity

Questo progetto si concentra sull'analisi della correlazione tra asset nel contesto delle commodity. Mentre altre analisi includono anche la liquidità, la mia attenzione è rivolta esclusivamente alla correlazione.

## 1. Analisi del Rischio Base

Una volta identificato l'asset più appropriato e il contratto future da utilizzare per coprirsi dal rischio di prezzo della commodity, si procede all'analisi del rischio base. Il *basis risk* rappresenta la differenza di prezzo tra il prezzo spot e il prezzo future. Un valore di basis vicino a zero indica un rischio base ridotto. Durante questa fase, viene anche esaminata la volatilità del basis e la sua distribuzione nel tempo.

## 2. Ciclo di Mercato: Contango e Backwardation

Successivamente, si analizzano le condizioni di mercato per determinare se ci si trova in una situazione di contango o backwardation.

### Contango

- **Definizione**: Il contango si verifica quando i prezzi dei futures superano il prezzo spot attuale della commodity. Questa condizione può derivare da diverse ragioni, tra cui aspettative di un aumento della domanda, costi di stoccaggio elevati o una visione ottimistica delle condizioni di mercato future.
  
- **Impatto sul Trading**: In un mercato in contango, gli investitori che desiderano mantenere una posizione a lungo termine affrontano costi maggiori al momento della scadenza dei contratti futures. Questo è dovuto al processo di *rollover*, che implica la chiusura di una posizione in un contratto in scadenza e l'apertura di un nuovo contratto con scadenza futura. Poiché il nuovo contratto è a un prezzo più elevato, ciò può comportare perdite aggiuntive.

### Backwardation

- **Definizione**: Il backwardation si verifica quando i prezzi dei futures sono inferiori al prezzo spot. Questa situazione può manifestarsi in presenza di una forte domanda immediata rispetto all'offerta disponibile o in previsione di un abbassamento dei prezzi futuri.

- **Impatto sul Trading**: In un mercato in backwardation, mantenere una posizione di copertura risulta più conveniente, poiché i contratti futures costano meno rispetto al prezzo spot. In questo contesto, i trader possono realizzare guadagni al momento della scadenza dei contratti.

## 3. Analisi della Stazionarietà

Un altro aspetto fondamentale dell'analisi pre-modellistica è la verifica della stazionarietà delle serie temporali. La stazionarietà è una condizione in cui le proprietà statistiche di una serie temporale, come la media e la varianza, rimangono costanti nel tempo. Questa analisi è cruciale per l'applicazione di modelli statistici, inclusi i modelli autoregressivi come il GARCH.

### Test ADF (Augmented Dickey-Fuller)

Il test ADF è uno strumento utilizzato per verificare la presenza di una radice unitaria in una serie temporale. I risultati del test forniscono informazioni sulla stazionarietà della serie:

- **Ipotesi nulla**: La serie ha una radice unitaria (non è stazionaria).
- **Ipotesi alternativa**: La serie non ha una radice unitaria (è stazionaria).

Se il valore del p-value è inferiore a 0.05, si rifiuta l'ipotesi nulla, indicando che la serie è stazionaria e suggerendo l'uso dei prezzi in log. In caso contrario, non si può rifiutare l'ipotesi nulla, e la serie è considerata non stazionaria.

Questa analisi è essenziale per garantire la validità dei modelli statistici e per una corretta interpretazione dei risultati.

## 4. Analisi OLS (Ordinary Least Squares)

L'analisi OLS è una tecnica fondamentale per stimare i parametri di un modello di regressione lineare. In questo contesto, si utilizza l'OLS per analizzare la relazione tra i prezzi spot e i prezzi futures delle commodity. L'OLS minimizza la somma dei quadrati degli errori tra i valori osservati e quelli previsti, fornendo così una stima efficiente e non distorta dei coefficienti.

### Come Funziona l'OLS

1. **Definizione delle Variabili**: Nel modello, i prezzi futures fungono da variabile indipendente (X), mentre i prezzi spot rappresentano la variabile dipendente (Y). Questo approccio consente di esplorare come i cambiamenti nei prezzi futures influenzano i prezzi spot.

2. **Fitting del Modello**: Utilizzando la libreria `statsmodels`, il modello OLS viene adattato ai dati. La funzione `sm.OLS()` viene utilizzata per calcolare i parametri del modello, inclusi l'intercetta e il coefficiente angolare, che rappresenta l'hedge ratio. Il coefficiente angolare (beta) indica quanto varia il prezzo spot in risposta a un cambiamento nel prezzo futures.

3. **Output del Modello**: Il metodo `fit()` genera un sommario dettagliato del modello, evidenziando statistiche importanti come il R-quadrato e i p-value, che forniscono indicazioni sulla bontà dell'adattamento del modello.

4. **Interpretazione dei Risultati**: L'hedge ratio calcolato (beta) viene utilizzato per determinare l'importo da investire nei futures per coprirsi dal rischio di prezzo della commodity. Un beta maggiore di 1 indica una maggiore volatilità nel prezzo spot rispetto al prezzo futures, suggerendo che è necessario investire di più nei futures per una copertura efficace.

### Residui OLS

Una parte importante dell'analisi OLS è l'analisi dei residui. I residui rappresentano la differenza tra i valori osservati e quelli previsti dal modello. Un'analisi accurata dei residui può rivelare informazioni sulla validità del modello e sulla presenza di eteroschedasticità o autocorrelazione, che possono compromettere l'affidabilità delle stime.

Per garantire la correttezza dei risultati, è fondamentale effettuare test sui residui e, se necessario, adottare misure correttive, come la trasformazione dei dati o l'utilizzo di modelli più complessi, come il GARCH, per catturare la dinamica della volatilità.

## Conclusioni

Questa analisi fornisce una visione approfondita della correlazione, del rischio base, delle condizioni di mercato e della stazionarietà nelle commodity, offrendo strumenti utili per prendere decisioni informate nel trading.
