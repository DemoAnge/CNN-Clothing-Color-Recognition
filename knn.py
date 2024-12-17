import numpy as np
import pandas as pd

# Cargar el conjunto de datos de colores desde un archivo CSV
data = pd.read_csv('colores.csv')
X = data[['Red', 'Green', 'Blue']].values  # Obtener los valores RGB
y = data['Color'].values  # Obtener los nombres de los colores

# Definir el valor de K para KNN
K = 5

# Función para calcular la distancia euclidiana entre dos puntos
def distancia_euclidiana(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Función para predecir el color usando KNN manual
def predecir_color_KNN(rgb):
    distancias = []
    # Calcular la distancia entre el color dado y todos los colores del dataset
    for i in range(len(X)):
        distancia = distancia_euclidiana(X[i], rgb)
        distancias.append((distancia, y[i]))
    distancias.sort(key=lambda x: x[0])  # Ordenar distancias
    vecinos = distancias[:K]  # Obtener los K vecinos más cercanos
    
    # Contar las ocurrencias de cada color en los K vecinos más cercanos
    conteos = {}
    for _, etiqueta in vecinos:
        if etiqueta in conteos:
            conteos[etiqueta] += 1
        else:
            conteos[etiqueta] = 1
    
    # Devolver el color con más ocurrencias
    return max(conteos, key=conteos.get)
