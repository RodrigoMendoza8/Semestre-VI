import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pandas.plotting import scatter_matrix

'''
Contexto: Un analista estima un modelo para explicar el consumo usando dos variables.
'''

df = pd.DataFrame({
    "Ingreso": [10, 12, 14, 16, 18, 20],
    "Gasto": [9, 11, 13, 15, 17, 19]
})

X_cols = ["Ingreso", "Gasto"]
data = df[X_cols].copy()
X = data

corr = X.corr()
print("\nMatriz de correlación (X):")
print(corr.round(3))

X_vif = sm.add_constant(X)
vif_table = pd.DataFrame({
    "Variable": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
print("\nVIF (incluye const):")
print(vif_table)

plt.figure()
plt.imshow(corr.values, aspect="auto", cmap="coolwarm")
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.index)), corr.index)
plt.colorbar(label="Correlación")
plt.title("Heatmap de correlación entre variables X")

for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        plt.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center")

plt.tight_layout()
plt.show()

scatter_matrix(X, diagonal="hist", figsize=(8, 8))
plt.suptitle("Scatter Matrix (relación entre X)")
plt.tight_layout()
plt.show()

'''
1. Calcula la correlación entre Ingreso y Gasto.
Es de 1.0. Existe una relación lineal positiva perfecta.

2. ¿Qué observas?
Las variables se mueven en la misma proporción y dirección siempre.

3. ¿Crees que ambas variables aportan información distinta? 
No. Al ser una relación perfecta, una variable explica totalmente a la otra.

4. ¿Hay riesgo de multicolinealidad?
Sí, hay multicolinealidad perfecta. El VIF resulta en infinito.
'''