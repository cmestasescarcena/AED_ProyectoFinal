import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import matplotlib.pyplot as plt
import time

class KNNFuerzaBruta:
    def __init__(self, k=3):
        self.k = k

    def entrenar(self, X_entrenamiento, y_entrenamiento):
        self.X_entrenamiento = np.array(X_entrenamiento)
        self.y_entrenamiento = np.array(y_entrenamiento)

    def distancia_euclidiana(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predecir(self, X_prueba):
        y_prediccion = []
        for x in X_prueba:
            distancias = [self.distancia_euclidiana(x, x_entrenamiento) for x_entrenamiento in self.X_entrenamiento]
            k_indices = np.argsort(distancias)[:self.k]
            k_etiquetas_cercanas = [self.y_entrenamiento[i] for i in k_indices]
            etiqueta_predominante = Counter(k_etiquetas_cercanas).most_common(1)[0][0]
            y_prediccion.append(etiqueta_predominante)
        return y_prediccion

def cargar_datos_desde_csv(archivo):
    X = []
    y = []
    with open(archivo, 'r') as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
        next(lector_csv)  
        for fila in lector_csv:
            x = [float(valor) for valor in fila[:-1]]
            etiqueta = int(fila[-1])
            X.append(x)
            y.append(etiqueta)
    return X, y

if __name__ == "__main__":
    X, y = cargar_datos_desde_csv('iris.csv')

    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

    eficiencias = []
    
    tiempos = []

    for k in range(3, 11):
        knn = KNNFuerzaBruta(k=k)
        
        inicio_tiempo = time.time()
        
        knn.entrenar(X_entrenamiento, y_entrenamiento)

        y_prediccion = knn.predecir(X_prueba)

        precision = accuracy_score(y_prueba, y_prediccion)
        eficiencias.append(precision)
        
        fin_tiempo = time.time()
        
        tiempo_procesamiento = (fin_tiempo - inicio_tiempo) * 1000
        tiempos.append(tiempo_procesamiento)

    for k, eficiencia in enumerate(eficiencias, start=3):
        print(f'Eficiencia para k={k}: {eficiencia * 100:.2f}% - Tiempo de procesamiento: {tiempos[k-3]:.2f} ms')

    plt.figure(figsize=(10, 6))
    plt.bar(range(3, 11), eficiencias)
    plt.xlabel('Número de Vecinos (k)')
    plt.ylabel('Eficiencia')
    plt.title('Eficiencia del Clasificador K-NN para Diferentes Valores de k')
    plt.show()