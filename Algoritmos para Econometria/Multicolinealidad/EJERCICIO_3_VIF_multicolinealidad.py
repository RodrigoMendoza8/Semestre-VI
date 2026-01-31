import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

'''
Modelo para explicar el precio de vivienda.
'''

df = pd.DataFrame({
    'Tamano': [80, 100, 120, 150, 180, 200],
    'Habitaciones': [2, 3, 3, 4, 4, 5],
    'Banos': [1, 2, 2, 3, 3, 4]
})

X = df[['Tamano', 'Habitaciones', 'Banos']]
matriz_corr = X.corr()
print(matriz_corr)

X_const = sm.add_constant(X)
vifs = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
print("\nVIFs (Const, Tamano, Habitaciones, Banos):", vifs)

plt.figure(figsize=(6, 5))
plt.imshow(matriz_corr.values, cmap='coolwarm', vmin=-1, vmax=1)
plt.xticks(range(len(matriz_corr.columns)), matriz_corr.columns)
plt.yticks(range(len(matriz_corr.index)), matriz_corr.index)
plt.colorbar()
for i in range(matriz_corr.shape[0]):
    for j in range(matriz_corr.shape[1]):
        plt.text(j, i, f"{matriz_corr.values[i, j]:.2f}", ha="center", va="center")
plt.title("Matriz de Correlación - Ejercicio 3")
plt.show()

'''
RESPUESTAS EJERCICIO 3:

1. Calcula el VIF de cada X: 
   - Tamaño: 12.92
   - Habitaciones: Infinito
   - Baños: Infinito

2. ¿Qué variable presenta mayor VIF?: Habitaciones y Baños.

3. ¿Existe multicolinealidad? Justifica. 
Sí. Existe multicolinealidad perfecta entre Habitaciones y Baños (correlación de 1.0). 
También hay multicolinealidad alta con Tamaño (VIF > 10).
'''