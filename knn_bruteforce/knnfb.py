import os

ruta_absoluta = os.path.abspath('')
#print (ruta_absoluta)
"""
import sys
#print(sys.path)
sys.path.append(ruta_absoluta)
#print(sys.path)
"""
ruta_absoluta = os.path.abspath('')
#print (ruta_absoluta)
from pathlib import Path
ruta_absoluta = str(Path(ruta_absoluta).parents[1]) 
#print (ruta_absoluta+'/AED_ProyectoFinal')

import sys
sys.path.append(ruta_absoluta+'/AED_ProyectoFinal')


import csv
from knn_kdtree import dataSet
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
    #X, y = cargar_datos_desde_csv('heart.csv')

    #X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

    eficiencias = []
    
    tiempos = []

    X_train_array = np.array(dataSet.X_train_list)
    X_test_array = np.array(dataSet.X_test)
    y_train_array = np.array(dataSet.y_train)
    y_test_array = np.array(dataSet.y_test)
    
    for k in range(1, 16):
        knn = KNNFuerzaBruta(k=k)
        
        inicio_tiempo = time.time()
        
        knn.entrenar(X_train_array, y_train_array)

        y_prediccion = knn.predecir(X_test_array)

        precision = accuracy_score(y_test_array, y_prediccion)
        eficiencias.append(precision)
        
        fin_tiempo = time.time()
        
        tiempo_procesamiento = (fin_tiempo - inicio_tiempo) * 1000
        tiempos.append(tiempo_procesamiento)

    for k, eficiencia in enumerate(eficiencias, start=1):
        print(f'Eficiencia para k={k}: {eficiencia * 100:.2f}% - Tiempo de procesamiento: {tiempos[k-3]:.2f} ms')

    """plt.figure(figsize=(10, 6))
    plt.scatter(range(1, 16), eficiencias, marker='o', c='blue', label='Eficiencia')
    plt.xlabel('Número de Vecinos (k)')
    plt.ylabel('EfDiciencia')
    plt.title('Eficiencia del Clasificador K-NN para Diferentes Valores de k')
    plt.legend()
    plt.grid(True)
    plt.show()"""
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test_array[:, 0], X_test_array[:, 1], c=y_prediccion, cmap='coolwarm', marker='.', s=100, label='Predicción')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.title('Visualización de Datos de Prueba y Predicciones (k=3)')
    plt.legend()
    plt.show()