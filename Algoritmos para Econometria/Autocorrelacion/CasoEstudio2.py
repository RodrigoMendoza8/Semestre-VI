import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

np.random.seed(42)
n = 200
educacion = np.random.randint(6, 22, n)
experiencia = np.random.randint(0, 30, n)
error = np.random.normal(0, 1, n) * (educacion ** 2) * 0.5
salario = 1000 + 200 * educacion + 50 * experiencia + error

df = pd.DataFrame({'Salario': salario, 'Educacion': educacion, 'Experiencia': experiencia})

X = sm.add_constant(df[['Educacion', 'Experiencia']])
y = df['Salario']
modelo = sm.OLS(y, X).fit()
residuos = modelo.resid

print("Estimación")
print(modelo.summary().tables[1])

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(modelo.fittedvalues, residuos)
plt.title("Residuos vs Ajustados")
plt.axhline(0, c='r', ls='--')

plt.subplot(1, 2, 2)
plt.scatter(df['Educacion'], residuos)
plt.title("Residuos vs Educacion")
plt.axhline(0, c='r', ls='--')
plt.tight_layout()
plt.show()

bp_test = het_breuschpagan(residuos, modelo.model.exog)
white_test = het_white(residuos, modelo.model.exog)

print(f"\nBreusch-Pagan p-value: {bp_test[1]:.6f}")
print(f"White Test p-value:    {white_test[1]:.6f}")

if bp_test[1] < 0.05 or white_test[1] < 0.05:
    print("\n¿Varianza constante?: NO (Heterocedasticidad)")
else:
    print("\n¿Varianza constante?: SI")

modelo_robusto = sm.OLS(y, X).fit(cov_type='HC1')
print("\nErrores robustos (White)")
print(modelo_robusto.summary().tables[1])

y_log = np.log(df['Salario'])
modelo_log = sm.OLS(y_log, X).fit()
print("\nModelo logarítmico")
print(modelo_log.summary().tables[1])