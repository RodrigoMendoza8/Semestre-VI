import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.DataFrame({
    'Crecimiento': [2.1, 2.3, 2.5, 2.4, 2.6],
    'Inversion': [18, 19, 20, 21, 22],
    'Ahorro': [17, 18, 19, 20, 21]
})

y = df['Crecimiento']
X = df[['Inversion', 'Ahorro']]
X_const = sm.add_constant(X)

modelo = sm.OLS(y, X_const).fit()
print(modelo.summary())

print("\n--- P-VALUES ---")
print(modelo.pvalues.round(4))

vifs = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
print("\n--- VIFs ---")
for i, col in enumerate(X_const.columns):
    print(f"{col}: {round(vifs[i], 2)}")

'''
1. Observa los p-values.
Los p-values de Inversión y Ahorro som muy altos, ninguna variable es estadísticamente significativa de forma individual.

2. ¿Es posible que ninguna beta sea significativa? 
Sí.El modelo sabe que alg" explica el crecimiento, pero no puede distinguir si es la Inversión o el Ahorro porque se mueven igual.

3. ¿Cómo lo relacionas con la multicolinealidad?
La multicolinealidad perfecta infla los errores estándar de los coeficient, escondiendo la importancia real de las variables.
'''