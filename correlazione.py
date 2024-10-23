import seaborn as sns  
import matplotlib.pyplot as plt  

#funzione per calcolare e visualizzare la matrice di correlazione
def correlation(data): 
    correlation_matrix = data.corr()

    #plotto la heatmap della matrice di correlazione
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Matrice di Correlazione tra Futures Energetici')
    plt.show()
    return correlation_matrix
