import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

'''
Contexto: Se quiere explicar el rendimiento de una acción con variables financieras.
'''
df = pd.DataFrame({
    "Rend": [0.02, 0.01, -0.01, 0.03, 0.02],
    "Tasa": [7.0, 7.2, 7.4, 7.1, 7.3],
    "CETES": [6.9, 7.1, 7.3, 7.0, 7.2],
    "Inflacion": [4.1, 4.3, 4.5, 4.2, 4.4]
})

matriz_corr = df.corr()
print(matriz_corr)

X = df[["Tasa", "CETES", "Inflacion"]]
X_vif = sm.add_constant(X)
vifs = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print("\nVIFs (Const, Tasa, CETES, Inflacion):", vifs)

plt.figure(figsize=(7, 6))
plt.imshow(matriz_corr.values, cmap='coolwarm', vmin=-1, vmax=1)
plt.xticks(range(len(matriz_corr.columns)), matriz_corr.columns)
plt.yticks(range(len(matriz_corr.index)), matriz_corr.index)
plt.colorbar()
for i in range(matriz_corr.shape[0]):
    for j in range(matriz_corr.shape[1]):
        plt.text(j, i, f"{matriz_corr.values[i, j]:.2f}", ha="center", va="center")
plt.title("Matriz de Correlación (Incluye Rendimiento)")
plt.tight_layout()
plt.show()

'''
1. Identifica pares con correlación alta.
Tasa, CETES e Inflación tienen una correlación de 1.0.

2. ¿Qué variables podrían causar multicolinealidad?
Tasa, CETES e Inflación son linealmente dependientes.

3. ¿Cuál eliminarías y por qué?: Eliminaría CETES e Inflación. 
Porque no aportan información nueva al modelo; al estar perfectamente correlacionadas con la Tasa, 
solo generan redundancia y hacen que el VIF sea infinito.
'''