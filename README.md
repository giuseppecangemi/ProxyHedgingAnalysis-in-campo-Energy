# Proxy Hedging


Nel settore delle commodity energetiche, è frequente l'assenza di futures specifici per ogni commodity, una situazione che può derivare da una bassa liquidità di mercato o da discrepanze temporali tra gli strumenti finanziari disponibili e le esigenze di copertura. La gestione del rischio di prezzo è cruciale per le aziende operanti in questo ambito, poiché le fluttuazioni dei prezzi possono avere un impatto significativo sui costi operativi e sui margini di profitto. In assenza di strategie di copertura adeguate, le aziende possono affrontare rischi considerevoli, inclusi quelli legati al delivery, ovvero la necessità di ricevere o consegnare fisicamente una commodity a un prezzo superiore rispetto a quello di mercato al momento della scadenza del contratto, il che può comportare perdite finanziarie rilevanti.

In tali contesti, viene impiegato il proxy hedging, una strategia di copertura che utilizza futures di commodity correlate. Questo approccio metodologico comprende diverse fasi: analisi della correlazione tra le commodity, valutazione delle condizioni di mercato e analisi del rischio base, finalizzate all'individuazione degli strumenti più appropriati per la gestione del rischio di prezzo. Un aspetto fondamentale della strategia di copertura, proposto in questa analisi, è rappresentato dall’hedge ratio, che indica il rapporto ottimale tra la posizione nella commodity e quella nel future, al fine di minimizzare la volatilità del portafoglio. Si precisa che, per il presente progetto, è stata temporaneamente esclusa l'analisi della liquidità.

Questa analisi si basa su dati dal 1° gennaio 2023 al 1° gennaio 2024 sul prezzo spot del gas in Italia, analizzando possibili futures quali: TTF, UK_Future, NG, brent, WTI, heating oil, gasoline e etanolo.

## Analisi della Correlazione

L'analisi della correlazione riveste un'importanza fondamentale per comprendere le interrelazioni tra diverse commodity e i rispettivi contratti futures. Essa consente di identificare i legami che possono influenzare le strategie di copertura e di investimento. A tal proposito, ho incluso un grafico che presenta la matrice di correlazione tra i prezzi degli strumenti esaminati.

Dalla matrice si può notare una chiara e significativa correlazione tra il prezzo del GAS e il TTF. Le correlazioni osservate sono in linea con quanto riportato in letteratura, suggerendo che le dinamiche di mercato delle commodity siano influenzate da fattori simili. Comprendere queste correlazioni è cruciale per le aziende che cercano di gestire il rischio di prezzo, poiché consente di scegliere gli strumenti più appropriati per le strategie di hedging.

![Descrizione Immagine](img/corr_matrix.png)

<!--

### Matrice di Correlazione

|                | Gas_Spot | Future_TTF | UK_Future | NG_Future | Brent_Future | WTI_Future | Heating_Oil | Gasoline |
|----------------|----------|------------|-----------|-----------|--------------|------------|-------------|----------|
| **Gas_Spot**      | 1.000000 |            |           |           |              |            |             |          |
| **Future_TTF**    | 0.669480 | 1.000000   |           |           |              |            |             |          |
| **UK_Future**     | 0.251176 | 0.456784   | 1.000000  |           |              |            |             |          |
| **NG_Future**     | 0.311908 | 0.553174   | 0.919424  | 1.000000  |              |            |             |          |
| **Brent_Future**  | 0.248799 | 0.283622   | 0.515228  | 0.354178  | 1.000000     |            |             |          |
| **WTI_Future**    | 0.205299 | 0.182846   | 0.531196  | 0.342093  | 0.982599     | 1.000000   |             |          |
| **Heating_Oil**   | 0.332338 | 0.449748   | 0.603096  | 0.561489  | 0.851139     | 0.809809   | 1.000000    |          |
| **Gasoline**      | 0.329363 | -0.292603  | -0.297367 | -0.336401 | 0.209285     | 0.224240   | 0.131426    | 1.000000 |
-->

## Analisi della Base e del relativo Rischio Base

L'analisi della **base** e del **basis risk** è cruciale, poiché il basis risk indica il rischio che i prezzi forward e spot non si muovano in modo sincronizzato nel tempo, causando discrepanze nella copertura (Geman, 2005). Comprendere questo rischio è essenziale per valutare l'efficacia delle strategie di hedging e per garantire una protezione adeguata contro le fluttuazioni dei prezzi.

La base è la differenza tra i prezzi spot e future, e può essere rappresentata con la seguente formula:

$$
\text{Basis} = \text{Prezzo Spot} - \text{Prezzo Forward}
$$

Un valore della base vicino a zero indica un rischio base ridotto, ovvero una ridotta differenza tra il prezzo spot della commodity e il prezzo del future scelto come proxy per la copertura. Durante questa fase, oltre al livello della base, esamino anche la volatilità della stessa, che rappresenta l'incertezza associata alla differenza tra i due prezzi, e la sua distribuzione nel tempo per comprendere eventuali pattern o variazioni significative. Una base stabile e con bassa volatilità rende la copertura più efficiente perché riduce l'incertezza legata alla differenza tra prezzo spot e future. In altre parole, se la base è meno volatile, significa che il rapporto tra il prezzo spot e il prezzo future è più prevedibile, e questo migliora la precisione della copertura, rendendola meno soggetta a variazioni impreviste. Così, il proxy hedging è più efficace perché, in condizioni di base stabile, la quantità di future necessaria per neutralizzare il rischio di prezzo è calcolata con maggiore affidabilità, riducendo il rischio base e migliorando l’accuratezza dell’hedging stesso.

L'analisi della volatilità della base presenta aspetti significativi che la differenziano dall'analisi della correlazione. Infatti, è possibile che la base, ovvero la differenza tra il prezzo spot e il prezzo forward, non segua in modo rigoroso la correlazione tra i due prezzi. Questo fenomeno si verifica poiché la correlazione misura esclusivamente la direzione e la forza della relazione lineare tra due serie temporali, senza fornire indicazioni su quanto un prezzo si muova in relazione all'altro. Tale comportamento è particolarmente comune nei mercati energetici, dove i prezzi spot tendono a rispondere in modo più immediato a eventi di breve termine, come shock di domanda o offerta, rispetto ai prezzi forward, che sono più influenzati da fattori di medio-lungo termine.

Questo rischio emerge nel momento in cui un *hedger* apre una posizione sul mercato finanziario al tempo $t_0$ e persiste fino alla chiusura della stessa, ossia in un momento $t'$ tale che $t_0 \leq t' < T$, dove $T$ indica la maturità del contratto derivato. Durante l’intervallo in cui la posizione rimane aperta, è possibile che il prezzo spot e il prezzo forward non seguano perfettamente lo stesso andamento, manifestata in diversi istanti temporali durante la vita residua del future.

Questo scarto nel prezzo contribuisce al *basis risk*, influenzando l’efficacia della copertura se la volatilità della base risulta elevata.
Una misura comune del **basis risk** si ottiene calcolando la varianza della base. Indicando con $B_{t_0,T}, B_{t_0+1,T}, \ldots, B_{t',T}$ i valori della base ai vari istanti temporali, il basis risk è dato dalla formula:

$$
BR_T = \text{Var}(B_{t_0,T}, B_{t_0+1,T}, \ldots, B_{t',T}) = \sigma_S^2 + \sigma_F^2 - 2 \rho \sigma_S \sigma_F
$$

Dove:
- $S(t)$ rappresenta il prezzo spot al tempo $t$,
- $F_T(t)$ indica il prezzo forward al tempo $t$ per il derivato con scadenza $T$,
- $\sigma_S$ e $\sigma_F$ sono le deviazioni standard dei prezzi spot e forward,
- $\rho$ è il coefficiente di correlazione tra i prezzi spot e forward.

Questi valori sono considerati per $t = t_0, t_0 + 1, \ldots, t'$, con $t_0 \leq t' < T$. È importante notare che il basis risk è nullo in presenza di una correlazione perfetta tra il prezzo spot e quello forward. Dal **basis risk** si può ottenere l'hedge ratio, ma non sarà sviluppato in questa analisi.

Di seguito riporto le stime sulla volatilità della base e dei prezzi spot e future.

| **Parametro**                | **Valore**   |
|------------------------------|--------------|
| Volatilità Spot              | 0.0347       |
| Volatilità Future            | 0.0628       |
| Volatilità della Base        | 7.62         |

Come vediamo, la volatilità dello spot è circa $1/2$ di quella del future. La volatilità della base è piuttosto elevata, suggerendo una notevole incertezza nella differenza tra i due prezzi. Probabilmente questa dinamica può essere compresa meglio analizzando il seguente grafico della base. 

![Descrizione Immagine](img/andamento_basis.png)

Inoltre, la distribuzione della base nel tempo, come da grafico sottostante, presenta una forma bimodale ma una tendenza delle basi vicino lo zero.

![Descrizione Immagine](img/distribuzione_basis.png)

<!--
**Alla luce di questi risultati dovremmo provare ad analizzare ulteriori combinazioni tra asset**

Poiché intendo esaminare la varianza della base per valutare il **basis risk**, in conformità con l'equazione del $BR_T$, la varianza del Basis risulta essere **58.0693**. Nel paragrafo dell'Hedge Ratio svilupperò un modello per ottenere l'OHR attraverso una revisitazione dell'equazione del rischio base ($BR_T$).

-->

## Ciclo di Mercato: Contango e Backwardation

Successivamente, ho analizzato le condizioni di mercato per determinare se ci troviamo in una fase di contango o backwardation. Questi due scenari illustrano le dinamiche tra il mercato spot e quello dei futures, influenzando significativamente le strategie di copertura e la gestione del rischio. È essenziale che ogni partecipante al mercato comprenda la struttura attuale dei prezzi. Un mercato è in contango quando i prezzi dei futures superano i prezzi spot, mentre è in backwardation quando i prezzi dei futures sono inferiori ai prezzi spot (Schofield, 2007). Ad esempio, il petrolio greggio è frequentemente soggetto a backwardation a causa dei costi relativi all'immagazzinamento, che scoraggiano la conservazione. In situazioni di aumento della domanda, le scorte disponibili possono rivelarsi insufficienti, creando un divario temporale tra i prezzi spot e quelli dei futures. Questo porta a un incremento dei prezzi spot rispetto ai futures, facendo sì che il mercato entri in backwardation.

Tipicamente, i mercati in **backwardation** sono caratterizzati da una scarsità della commodity, scorte basse, prezzi volatili a causa delle scorte limitate e un aumento dei prezzi. Al contrario, i mercati in **contango** si caratterizzano per un'abbondanza della commodity e per scorte elevate. In queste condizioni, i prezzi spot tendono a essere inferiori rispetto ai prezzi dei futures, poiché i partecipanti al mercato anticipano che le scorte attuali potranno soddisfare la domanda futura. Questa situazione può verificarsi, ad esempio, in periodi di stabilità nella produzione e di bassa volatilità dei prezzi. In un mercato in contango, gli investitori possono essere incentivati a comprare e immagazzinare la commodity, aspettandosi di rivenderla a un prezzo più elevato in futuro.

- **Impatto sul Trading**: In un mercato in **contango**, gli investitori che desiderano mantenere una posizione a lungo termine affrontano costi maggiori al momento della scadenza dei contratti futures. Questo è dovuto al processo di *rollover*, che implica la chiusura di una posizione in un contratto in scadenza e l'apertura di un nuovo contratto con scadenza futura. Poiché il nuovo contratto è a un prezzo più elevato, ciò può comportare perdite aggiuntive.

- **Impatto sul Trading**: In un mercato in **backwardation**, mantenere una posizione di copertura risulta più conveniente, poiché i contratti futures costano meno rispetto al prezzo spot. In questo contesto, i trader possono realizzare guadagni al momento della scadenza dei contratti.

Questo aspetto è particolarmente rilevante nel nostro contesto, poiché diversi momenti temporali possono comportare un roll over del future. Ad esempio, se mi copro nel mercato dei futures, ma la mia copertura scade sei mesi prima della scadenza del contratto spot, ciò mi espone a un rischio di prezzo, poiché resterò scoperto per sei mesi. Di conseguenza, dovrò acquistare (o vendere) un ulteriore contratto future per mantenere la copertura. Tuttavia, se il prezzo forward aumenta in questo intervallo, dovrò affrontare costi più elevati, il che potrebbe tradursi in perdite.

Nel nostro campione, la totalità dei giorni (250) nel campione mostrano una condizione di **backwardation**. Questo **può rendere più conveniente** il mantenimento della posizione di copertura nel tempo. 

### Conclusioni analisi preliminari
L'analisi della correlazione ha rivelato una discreta interconnessione tra i diversi strumenti finanziari nel contesto delle commodity. Allo stesso tempo, l'analisi delle condizioni di mercato ha evidenziato una preferenza nell'utilizzo di questo strumento, grazie alla sua capacità di adattarsi alle fluttuazioni dei prezzi. Pertanto, nonostante l'elevata volatilità del basis, questa viene considerata secondaria rispetto ai vantaggi potenziali derivanti dall'adozione di strategie di copertura basate su tali strumenti. In questo contesto, la combinazione di una correlazione soddisfacente e delle condizioni di mercato favorevoli può rendere l'approccio del proxy hedging particolarmente interessante. Rimaniamo in attesa dell'analisi sulla liquidità, che potrà fornire ulteriori spunti per valutare l'efficacia di queste strategie.

## Stima Hedge Ratio
Un aspetto fondamentale, una volta scelto lo strumento finanziario, è la stima della quantità ottima di futures da acquistare per ridurre il rischio di portafoglio. Questo può essere realizzato attraverso l'utilizzo di diversi modelli. In questo lavoro, presenterò tre approcci principali: OLS, CC-GARCH e DCC-GARCH. Questi modelli offrono diverse prospettive e metodi per ottimizzare la copertura e minimizzare il rischio associato all'oscillazione dei prezzi delle commodity.

## Analisi OLS (Ordinary Least Squares)

In questo contesto, utilizziamo il modello OLS per analizzare la relazione tra i prezzi spot e i prezzi futures delle commodity. In particolare, l'equazione OLS si presenta come una semplice relazione del tipo $Y = X + \epsilon$, dove $Y$ rappresenta i prezzi spot, $X$ i prezzi futures, e $\epsilon$ è l'errore. I risultati di questa analisi sono riportati di seguito:

### OLS Regression Results

| **Dep. Variable:**   **Gas_Spot**      | **R-squared:**         | 0.448   |
|---------------------------------------|---------------------|----------|
| **Model:**           OLS              | **Adj. R-squared:**   | 0.446   |


| **Coefficients**              | **coef**  | **std err** | **z**    | **P>t** | **[0.025** | **0.975]** |
|-------------------------------|-----------|-------------|----------|-----------|-------------|-------------|
| const                         | 32.8455   | 1.434       | 22.905   | 0.000     | 30.021      | 35.670      |
| Future                        | 0.4786    | 0.034       | 14.193   | 0.000     | 0.412       | 0.545       |

**Notes:**  
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
**Aggiustati per Standard Error:** nessun miglioramento

![Descrizione Immagine](img/reg_OLS.png)

### Residui OLS
Per garantire la correttezza dei risultati, è fondamentale effettuare test sui residui e, se necessario, adottare misure correttive, come la trasformazione dei dati o l'utilizzo di modelli più complessi, come il GARCH, per catturare la dinamica della volatilità.
In questo caso, i residui non seguono una distribuzione normale e il test di Breusch-Pagan fornisce una chiara evidenza della presenza di eteroschedasticità. L'eteroschedasticità si riferisce a una situazione in cui gli errori non sono distribuiti uniformemente. Questo comporta che la varianza degli errori non rimane costante e può influenzare l'affidabilità delle stime e dei test statistici, rendendo la stima biased e la consecutiva necessità di un approccio alternativo.

### Test eteroschedasticità

| **LM Stat** | **LM p-value** | **F-stat** | **F p-value** |
|-------------|----------------|------------|---------------|
| 5.0065      | 0.0253         | 5.0679     | 0.0252        |

**Conclusione**:  
Il P-value di 0.0253 è inferiore a 0.05, quindi si rifiuta l'ipotesi nulla.  
C'è **eteroschedasticità**.

### Trasformazione dati in log

Ho effettuato la trasformazione dei dati utilizzando i logaritmi e ho successivamente eseguito nuovamente la regressione. Tuttavia, non ho osservato alcun miglioramento nei risultati. Di seguito, presento i risultati ottenuti.

### test eteroschedasticità

| **LM Stat** | **LM p-value** | **F-stat** | **F p-value** |
|-------------|----------------|------------|---------------|
| 5.9757      | 0.0145         | 6.0730     | 0.0144        |

**Conclusione**:  
Il P-value di 0.0145 è inferiore a 0.05, quindi si rifiuta l'ipotesi nulla.  
C'è **eteroschedasticità**.

### OLS con i Returns
Nella regressione precedente, ho utilizzato il prezzo per entrambi gli asset (spot e future). Ora, seguendo la letteratura, procedo a regredire i rendimenti del future selezionato rispetto ai rendimenti dello spot gas. I miglioramenti ottenuti nella regressione sono evidenti, con un $R^2$ significativamente più elevato, che indica un migliore adattamento ai dati, come mostrato nel seguente grafico. 

![Descrizione Immagine](img/reg_OLS_returns.png)

Il p-value associato alla variabile $Y$ indica una stima statisticamente significativa, mentre la costante non risulta significativa. Di seguito sono riportati i risultati della regressione, aggiustando gli standard error per l'eteroschedasticità. 

| **Dep. Variable:**   **Gas_Spot**      | **R-squared:**         | 0.742   |
|---------------------------------------|---------------------|----------|
| **Model:**           OLS              | **Adj. R-squared:**   | 0.741   |


| **Coefficients**              | **coef**  | **std err** | **z**    | **P>t** | **[0.025** | **0.975]** |
|-------------------------------|-----------|-------------|----------|-----------|-------------|-------------|
| const                         | -0.0020   | 0.001       | -1.770   | 0.077     | -0.004      | 0.000      |
| Future_returns                | 0.4761    | 0.030       | 16.016   | 0.000     | 0.418       | 0.534       |


L'**Hedge Ratio**, dunque, non cambia particolarmente rispetto il primo modello OLS. Questo potrebbe essere dovuto al fatto che la componente principale nella stima del rapporto è la correlazione tra gli asset.

A questo punto, mi aspetto di osservare i residui distribuiti uniformemente attorno allo zero, o comunque un miglioramento rispetto al modello OLS basato sui prezzi. Il grafico seguente illustra questa distribuzione.

![Descrizione Immagine](img/residui_returns.png)

Infine, ho condotto test sull'eteroschedasticità e ho riscontrato la sua presenza nei dati. Pertanto, è fondamentale valutare con attenzione i risultati di questo modello.


## Analisi Conditional OLS 

Sviluppato da Miffre (2004), il modello di regressione lineare condizionato unisce la semplicità computazionale e interpretativa della regressione lineare con una stima dinamica dell'hedge ratio. Il principio fondamentale di questo modello si basa sulla prevedibilità dei rendimenti, particolarmente valida per periodi prolungati: il rendimento è proporzionale al rischio a cui gli operatori di mercato sono esposti e, nel contesto finanziario, compensa il ritardo tra la stipula del contratto e la sua scadenza (Miffre, 2004). Ciò consente di spiegare la variabilità del fenomeno introducendo opportuni regressori nel modello presentato in precedenza.
Miffre, considera un insieme di variabili, ritardate di un lag, selezionate sulla base di considerazioni sia teoriche che empiriche. Tra le variabili ritardate di -1, due componenti di particolare rilevanza per questa analisi sono il rendimento del future e la base. L'inserimento di variabili ritardate anziché contemporanee permette di aggiornare la stima dell'hedge ratio non appena nuove informazioni diventano disponibili, consentendo di modificare periodicamente la posizione sul mercato per garantire una protezione adeguata dal rischio di prezzo anche in contesti altamente incerti.

La componente temporale viene modellata da Miffre assumendo una relazione lineare tra \$\beta_t$, il coefficiente che rappresenta l'hedge ratio in diversi istanti temporali, e le variabili $z$ ritardate di un lag:

$$
(\beta_t | z_{t-1}) = \beta_0 + \beta_1 z_{t-1}
$$

dove:

- $\beta_0$ rappresenta l'hedge ratio medio,
- $\beta_1$ è un vettore di parametri associato a $z_{t-1}$.

Il modello che ne deriva è il modello lineare condizionato con base costante:

$$
RS_t = \alpha + \beta\_0 RF_t + \beta\_1 RF_t z_{t-1} + \epsilon_t \quad 
$$

Sarebbe opportuno sviluppare ulteriormente il modello di Miffre per testare in modo conclusivo la possibilità di impiegare il modello OLS condizionato nella stima dell'OHR.
Infatti, anche il parametro $\alpha$ può essere modellato affinché risulti temporaneamente dipendente:

$$
(\alpha_t | z_{t-1}) = \alpha_0 + \alpha_1 z_{t-1} \quad 
$$

Pertanto, il modello si esprime come:

$$
RS_t = \alpha\_0 + \alpha\_1 z_{t-1} + \beta\_0 RF_t + \beta\_1 RF_t z_{t-1} + \epsilon_t
$$

I risultati mostrati nella tabella seguente non evidenziano miglioramenti, nemmeno dopo aver corretto gli errori standard per l'eteroschedasticità. Inoltre, si osserva una non significatività statistica sia della costante che dell'interazione tra i rendimenti del Future ($RF_t$) e il set di variabili ritardate $z_t{-1}$. L'hedge ratio (0.4729) sarebbe comunque vicino a quello stimato con gli altri due modelli OLS.

| **Dep. Variable:**   **Gas_Spot**      | **R-squared:**         | 0.740   |
|---------------------------------------|---------------------|----------|
| **Model:**     Conditional OLS              | **Adj. R-squared:**   | 0.737   |


| **Coefficients**              | **coef**  | **std err** | **z**    | **P>t** | **[0.025** | **0.975]** |
|-------------------------------|-----------|-------------|----------|-----------|-------------|-------------|
| const                         | -0.0019   | 0.001       | -1.597   | 0.110     | -0.004      | 0.000      |
| Future_returns                | 0.4729    | 0.030       | 15.855   | 0.000     |  0.414       | 0.531       |
| Interaction_lag               | 0.1493    | 0.546       | 0.274   | 0.784     | -0.920       | 1.219       |


La regressione con $\alpha$ dipendente dal tempo non ci restituisce degli output statisticamente significativi, pertanto non presento i risultati.


## Analisi della Stazionarietà

Poiché i modelli OLS non possono essere utilizzati a causa della presenza di bias, è necessario adottare un modello alternativo tra quelli disponibili in letteratura. Il modello GARCH rappresenta sicuramente un'opzione valida, ma per poter applicare questa classe di modelli è fondamentale testare la stazionarietà dei dati. La stazionarietà è una condizione in cui le proprietà statistiche di una serie temporale, come la media e la varianza, rimangono costanti nel tempo. Nello specifico, le serie temporali sono raramente stazionarie, ma la loro differenza prima $diff = P_t - P_{t-1}$ è spesso stazionaria, come osservato nel grafico seguente.

![Descrizione Immagine](img/stazionarieta.png)

### Test ADF (Augmented Dickey-Fuller)

Il test ADF è uno strumento utilizzato per verificare la presenza di una radice unitaria in una serie temporale. I risultati del test forniscono informazioni sulla stazionarietà della serie:

- **Ipotesi nulla**: La serie ha una radice unitaria (non è stazionaria).
- **Ipotesi alternativa**: La serie non ha una radice unitaria (è stazionaria).

Se il valore del p-value è inferiore a 0.05, si rifiuta l'ipotesi nulla, indicando che la serie è stazionaria e suggerendo l'uso dei prezzi in log. In caso contrario, non si può rifiutare l'ipotesi nulla, e la serie è considerata non stazionaria.

### Risultati del Test di Dickey-Fuller Aumentato

| **Serie**                               | **ADF Test Statistic** | **p-value**         | **Lags Used** | **Observations Used** | **Conclusione**                        |
|-----------------------------------------|------------------------|---------------------|---------------|-----------------------|----------------------------------------|
| Differenza Prima Log Prezzo Spot Gas     | -16.1787               | 4.2893e-29          | 0             | 248                   | Serie stazionaria (ipotesi nulla rifiutata), si suggerisce di usare i prezzi in log |
| Differenza Prima Log Prezzo Future Gas   | -8.2667                | 4.9151e-13          | 3             | 245                   | Serie stazionaria (ipotesi nulla rifiutata), si suggerisce di usare i prezzi in log |


## Modello CC-GARCH (Constant Correlation GARCH)

### Introduzione
Il modello CCGARCH (Constant Correlation Generalized Autoregressive Conditional Heteroskedasticity) è una generalizzazione dei modelli GARCH tradizionali, progettato per analizzare le relazioni di volatilità tra più serie temporali. Questo modello è particolarmente utile nel contesto della gestione del rischio e della copertura in mercati finanziari come quelli delle materie prime, dove la volatilità dei prezzi può influenzare significativamente profitti e perdite.

### Funzionalità e Applicazione
Il CCGARCH assume che la correlazione tra le serie dei Returns (in questo caso, i rendimenti del gas naturale spot e del future) rimanga **costante nel tempo**. Attraverso il modello, possiamo calcolare l'OHR come il rapporto tra la covarianza condizionale e la varianza condizionale dei rendimenti, fornendo così un'indicazione della quantità di future da acquistare o vendere per coprire il rischio associato a una posizione spot.

**formalmente abbiamo:**

$$OHR = correlation \times \frac{h_{spot}}{h_{future}}$$

Dove:

- $$h_{\text{future}}$$ rappresenta la volatilità condizionale del future,
- $$h_{\text{spot}}$$ rappresenta la volatilità condizionale dello spot,
- $$\text{correlation}$$ è la correlazione tra i rendimenti spot e future,
- $$\text{OHR}$$ è il rapporto di copertura ottimale (Optimal Hedge Ratio).


### Codice per il Modello CCGARCH
Di seguito è riportato una brevissima parte di codice che permette di sviluppare il modello CCGARCH e calcolare l'Optimal Hedge Ratio (OHR):

```python
# Calcolo dei rendimenti logaritmici
def calcola_rendimenti(spot, future):
    dati = variabili()
    dati['Return_Spot'] = np.log(dati[spot].pct_change() + 1)
    dati['Return_Future'] = np.log(dati[future].pct_change() + 1)
    #i rendimenti sono molto piccoli quindi andrebbero riscalati:
    dati['Return_Spot'] = 10 * np.log(dati[spot].pct_change() + 1)
    dati['Return_Future'] = 10 * np.log(dati[future].pct_change() + 1)

    rendimenti = dati.dropna()
    print(f"Questi sono i rendimenti:\n{rendimenti}")
    return rendimenti


def ccgarch_hedge_ratio(spot, future):
    dati = calcola_rendimenti(spot, future) 

    # Modello GARCH(1,1) sui rendimenti dello spot
    model_spot = arch_model(dati['Return_Spot'].dropna(), vol='Garch', p=1, q=1, dist='t') #ho scelto la distribuzione t percheé fitta meglio
    res_spot = model_spot.fit(disp='off') 
    # Modello GARCH(1,1) sui rendimenti del future
    model_future = arch_model(dati['Return_Future'].dropna(), vol='Garch', p=1, q=1, dist='t') #ho scelto la distribuzione t percheé fitta meglio
    res_future = model_future.fit(disp='off')

    # Ottengo le varianze condizionali
    h_future = res_future.conditional_volatility
    h_spot = res_spot.conditional_volatility
    correlation = dati['Return_Spot'].corr(dati['Return_Future'])  
    OHR = correlation * (h_spot / h_future)

```
### Utilizzando modelli GARCH la stima dell'OHR è calcolata come nella formula seguente:

$$
\text{OHR} = \text{correlation} \times \left( \frac{h_{\text{spot}}}{h_{\text{future}}} \right)
$$

dove la correlazione è considerata **costante**:

$$
\rho = (Return\_{Spot}, Return\_{Future})
$$

Di seguito riporto il grafico della volatilità condizionata e dell'OHR dinamico.
![Descrizione Immagine](img/garch.png)


## Effetti Hedging

### Effetti sulla varianza CC-GARCH

Uno degli elementi di rischio sui mercati è la volatilità. Pertanto vogliamo capire se l'utilizzo di questi hedge ratio siano in grado di ridurla.
La varianza senza copertura è calcolata come segue:

$$
\sigma^{2}_{\text{no hedge}} = \text{Var}(Return\_{Spot})
$$

Per calcolare i rendimenti coperti con il modello GARCH, si sottrae la quota (\(OHR\)) dei rendimenti futuri dai rendimenti dello spot:

$$Return\_{coperti}^{ccgarch} = Return\_{Spot} - (OHR \times Return\_{Future})$$

La varianza della copertura con il modello GARCH è calcolata come:

$$
\sigma^{2}{hedge}^{ccgarch} = \text{Var}(Return\_{coperti}^{ccgarch})
$$


Una delle metodologie utilizzate per valutare l'efficacia della copertura è il "minimum variance", che si riferisce alla riduzione della varianza rispetto alla posizione non coperta. Questo approccio calcola la differenza percentuale tra la varianza dei rendimenti spot e la varianza della copertura.

$$
\text{riduzione della varianza} = \frac{\text{varianza no hedge} - \text{varianza hedge CCGARCH}}{\text{varianza no hedge}} \times 100
$$

### Naive

Considero anche la metodologia Naïve, che consiste nel prendere la stessa quantità dello spot per andare short nel future. Questa strategia, pur essendo semplice, spesso porta a risultati inferiori rispetto alla copertura ottenuta tramite il modello CC-GARCH. Nei risultati che presenterò, sarà evidente come la metodologia Naïve non riesca a ridurre il rischio in modo altrettanto efficace rispetto alla strategia di copertura più sofisticata.

### VaR

Nel contesto del mio progetto, la valutazione dell'efficacia della copertura non si basa solo sulla riduzione della volatilità, ma anche sulla diminuzione del Value at Risk (VaR). Il VaR è una misura statistica che quantifica il potenziale di perdita di un investimento in un dato intervallo di tempo con un certo livello di confidenza. Pertanto, un'analisi approfondita del VaR consente di comprendere meglio i rischi associati e l'impatto della copertura sul profilo di rischio complessivo.
Si osserva una riduzione del VaR per il portafoglio coperto con hedge ratio stimato tramite CC-GARCH e un aumento del rischio per il metodo Naive (1:1). Di seguito il grafico con i risultati.

![Descrizione Immagine](img/var_garch.png)

In generale i risultati sono presentati nella tabella sottostante.

### Di seguito i risultati 

| **Metriche**                                          | **Valore**    |
|------------------------------------------------------|---------------|
| Varianza NO hedge                                    | 0.12%         |
| Varianza hedge CCGARCH                               | 0.03%         |
| Riduzione della varianza con CCGARCH                 | 78.46%        |
| Riduzione della varianza (Naïve)                    | -9.18%        |
| Differenza nella riduzione della varianza (CCGARCH - Naïve) | 87.65%  |
| Value at Risk (No Hedge)                             | -0.5889       |
| Value at Risk (Hedged GARCH)                         | -0.2678       |
| Value at Risk (Hedged Naive)                         | -0.5954       |


## Effetti Hedging OLS

Svolgo la stessa analisi per il modello OLS. 
Di seguito il grafico del VaR.
![Descrizione Immagine](img/var_OLS.png)

Di seguito presento i risultati della riduzione di volatilità del portafoglio grazie alla copertura stimata tramite modello Naive e OLS. Il modello OLS performa molto meglio del Naive, con riduzioni del VaR non indifferenti. 

| **Metriche**                                          | **Valore**    |
|------------------------------------------------------|---------------|
| Varianza NO hedge                                    | 0.00%         |
| Varianza hedge CCGARCH                               | 0.00%         |
| Riduzione della varianza con CCGARCH                 | 74.20%        |
| Riduzione della varianza (Naïve)                    | -15.66%        |
| Differenza nella riduzione della varianza (CCGARCH - Naïve) | 89.86% |
| Value at Risk (No Hedge)                             | -0.0572       |
| Value at Risk (Hedged GARCH)                         | -0.0287      |
| Value at Risk (Hedged Naive)                         | -0.0625       |



## Conclusioni
Dai risultati preliminari emerge una significativa riduzione della volatilità del portafoglio utilizzando il modello CCGARCH. Al contrario, il modello Naive si comporta peggio rispetto alla situazione in cui non ci si copre affatto dal rischio. Questo fenomeno può essere attribuito al fatto che, come discusso nel paragrafo sul basis risk, i movimenti tra il prezzo spot e il prezzo future sono spesso ampiamente discordanti. Di conseguenza, una copertura 1:1 non riesce a sterilizzare il rischio in modo efficace; al contrario, tende ad aumentarlo.
Tali risultati sono ulteriormente avvalorati dall'analisi del VaR, che contribuisce a rendere la nostra valutazione più robusta. La combinazione di questi modelli e delle loro implicazioni suggerisce che, per una gestione efficace del rischio, è fondamentale considerare non solo la correlazione, ma anche la volatilità e il comportamento dei prezzi nel tempo.

Infine, gli stessi risultati si ottengono per il modello OLS, il quale è capace di generare una riduzione della volatilità non indifferente (74.20%). Allo stesso modo si osserva una buona riduzione del rischio tramite stima del VaR. In generale, comunque, rimangono maggiormente attendibili i risultati ottenuti dal modello GARCH.





