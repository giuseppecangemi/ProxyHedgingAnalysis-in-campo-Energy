# Analisi della Correlazione e del Rischio Base nelle Commodity

Questo progetto si concentra sull'analisi della correlazione tra asset nel contesto delle commodity. Mentre altre analisi includono anche la liquidità, la mia attenzione è rivolta esclusivamente alla correlazione.

## 1. Analisi della Correlazione

L'analisi della correlazione è fondamentale per comprendere le relazioni tra le diverse commodity e i loro contratti futures. Di seguito è riportato un grafico che illustra la matrice di correlazione tra i prezzi delle commodity analizzate.

Dalla matrice sottostante, si osserva che la correlazione più elevata tra la commodity GAS e i futures è quella con il TTF:

![Descrizione Immagine](img/corr_matrix.png)

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


## 2. Analisi del Rischio Base

Una volta identificato l'asset più appropriato e il contratto future da utilizzare per coprirsi dal rischio di prezzo della commodity, si procede all'analisi del rischio base. Il *basis risk* rappresenta la differenza di prezzo tra il prezzo spot e il prezzo future. Un valore di basis vicino a zero indica un rischio base ridotto. Durante questa fase, viene anche esaminata la volatilità del basis e la sua distribuzione nel tempo.

| **Parametro**                | **Valore**   |
|------------------------------|--------------|
| Volatilità Spot              | 0.5508       |
| Volatilità Future            | 0.9966       |
| Volatilità del Basis         | 7.62         |


## 3. Ciclo di Mercato: Contango e Backwardation

Successivamente, si analizzano le condizioni di mercato per determinare se ci si trova in una situazione di contango o backwardation.

### Contango

- **Definizione**: Il contango si verifica quando i prezzi dei futures superano il prezzo spot attuale della commodity. Questa condizione può derivare da diverse ragioni, tra cui aspettative di un aumento della domanda, costi di stoccaggio elevati o una visione ottimistica delle condizioni di mercato future.
  
- **Impatto sul Trading**: In un mercato in contango, gli investitori che desiderano mantenere una posizione a lungo termine affrontano costi maggiori al momento della scadenza dei contratti futures. Questo è dovuto al processo di *rollover*, che implica la chiusura di una posizione in un contratto in scadenza e l'apertura di un nuovo contratto con scadenza futura. Poiché il nuovo contratto è a un prezzo più elevato, ciò può comportare perdite aggiuntive.

### Backwardation

- **Definizione**: Il backwardation si verifica quando i prezzi dei futures sono inferiori al prezzo spot. Questa situazione può manifestarsi in presenza di una forte domanda immediata rispetto all'offerta disponibile o in previsione di un abbassamento dei prezzi futuri.

- **Impatto sul Trading**: In un mercato in backwardation, mantenere una posizione di copertura risulta più conveniente, poiché i contratti futures costano meno rispetto al prezzo spot. In questo contesto, i trader possono realizzare guadagni al momento della scadenza dei contratti.

| **Parametro**                | **Valore**   |
|------------------------------|--------------|
| Giorni in Contango           | 0            |
| Giorni in Backwardation      | 250          |

La maggior parte del periodo analizzato è in backwardation. Questo può rendere più conveniente il mantenimento della posizione di copertura nel tempo. 

## 4. Analisi OLS (Ordinary Least Squares)

In questo contesto, si utilizza l'OLS per analizzare la relazione tra i prezzi spot e i prezzi futures delle commodity. Nello specifico l'equazione OLS è una equaizone semplice del tipo $Y = X + \epsilon$. I risultati sono presentati di seguito:

### OLS Regression Results

| **Dep. Variable:**            | **Gas_Spot**      | **R-squared:**         | 0.448   |
|-------------------------------|------------------|---------------------|----------|
| **Model:**                    | OLS              | **Adj. R-squared:**   | 0.446   |


| **Coefficients**              | **coef**  | **std err** | **t**    | **P>t** | **[0.025** | **0.975]** |
|-------------------------------|-----------|-------------|----------|-----------|-------------|-------------|
| const                         | 32.8455   | 1.434       | 22.905   | 0.000     | 30.021      | 35.670      |
| Future                        | 0.4786    | 0.034       | 14.193   | 0.000     | 0.412       | 0.545       |

**Notes:**  
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
**Aggiustati per Standard Error:** nessun miglioramento

### Residui OLS
Per garantire la correttezza dei risultati, è fondamentale effettuare test sui residui e, se necessario, adottare misure correttive, come la trasformazione dei dati o l'utilizzo di modelli più complessi, come il GARCH, per catturare la dinamica della volatilità.

In questo caso, i residui non si distribuiscono secondo una normale e il test sviluppato di breusch-pagan ci fornisce una chiara evidenza sulla presenza di eteroschedasticità.

### Test eteroschedasticità

| **LM Stat** | **LM p-value** | **F-stat** | **F p-value** |
|-------------|----------------|------------|---------------|
| 5.0065      | 0.0253         | 5.0679     | 0.0252        |

**Conclusione**:  
Il P-value di 0.0253 è inferiore a 0.05, quindi si rifiuta l'ipotesi nulla.  
C'è **eteroschedasticità**.

### Trasformazione dati in log
Ho trasformato i dati in logaritmo e lanciato nuovamente la regressione, senza però osservare alcun miglioramento. Presento i risultati di seguito.

### test eteroschedasticità

| **LM Stat** | **LM p-value** | **F-stat** | **F p-value** |
|-------------|----------------|------------|---------------|
| 5.9757      | 0.0145         | 6.0730     | 0.0144        |

**Conclusione**:  
Il P-value di 0.0145 è inferiore a 0.05, quindi si rifiuta l'ipotesi nulla.  
C'è **eteroschedasticità**.


## 5. Analisi della Stazionarietà

Dato che i modelli OLS non possono essere usati per la presenza di bias, devo utilizare un modello alternativo tra quelli presenti in letteratura. Il GARCH è sicuramente una scelta valida ma per usare questa classe di modelli dobbiamo testare la presenza di stazionarietà nei dati. La stazionarietà è una condizione in cui le proprietà statistiche di una serie temporale, come la media e la varianza, rimangono costanti nel tempo. Nello specifico, le serie temporali sono raramente stazionarie, ma la loro differenza prima $diff = P_t - P_{t-1}$
 è spesso stazionaria.

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


## 6. Modello CCGARCH (Constant Correlation GARCH)

### Introduzione
Il modello CCGARCH (Constant Correlation Generalized Autoregressive Conditional Heteroskedasticity) è una generalizzazione dei modelli GARCH tradizionali, progettato per analizzare le relazioni di volatilità tra più serie temporali. Questo modello è particolarmente utile nel contesto della gestione del rischio e della copertura in mercati finanziari come quelli delle materie prime, dove la volatilità dei prezzi può influenzare significativamente i profitti e le perdite.

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
Di seguito è riportato il codice per implementare il modello CCGARCH e calcolare l'Optimal Hedge Ratio (OHR):

```python
# Calcolo dei rendimenti logaritmici
def calcola_rendimenti(spot, future):
    dati = variabili()
    dati['Return_Spot'] = np.log(dati[spot].pct_change() + 1)
    dati['Return_Future'] = np.log(dati[future].pct_change() + 1)
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


### Effetti Hedging: effetti sulla varianza

**Uno degli elementi di rischio sui mercati è la volatilità. Pertanto vogliamo capire se l'utilizzo di questi hedge ratio siano in grado di ridurla.**

La varianza senza copertura è calcolata come segue:

$$
\sigma^{2}_{\text{no hedge}} = \text{Var}(Return\_{Spot})
$$

Per calcolare i rendimenti coperti con il modello GARCH, si sottrae la quota (\(OHR\)) dei rendimenti futuri dai rendimenti dello spot:

$$Rendimenti\_{coperti}^{ccgarch} = Rendimenti\_{Spot} - (OHR \times Rendimenti\_{Future})$$

La varianza della copertura con il modello GARCH è calcolata come:

$$
\sigma^{2}_{\text{hedge}}^{\text{ccgarch}} = \text{Var}(R_{\text{coperti}}^{\text{ccgarch}})
$$


Per comprendere se il modello GARCH riduce la varianza, si calcola la differenza percentuale tra la varianza dei rendimenti spot e la varianza della copertura:

\[
\text{riduzione\_varianza\_ccgarch} = \frac{(\text{varianza\_no\_hedge} - \text{varianza\_hedge\_ccgarch})}{\text{varianza\_no\_hedge}} \times 100
\]

