import seaborn as sns  # Importa seaborn per la heatmap
import matplotlib.pyplot as plt  # Importa matplotlib per i grafici

# Funzione per calcolare e visualizzare la matrice di correlazione
def correlation(data):  # Accetta il DataFrame 'data' come argomento
    # Calcola la matrice di correlazione
    correlation_matrix = data.corr()

    # Crea la heatmap della matrice di correlazione
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Matrice di Correlazione tra Futures Energetici')
    plt.show()

    # Restituisce la matrice di correlazione
    return correlation_matrix
