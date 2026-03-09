import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox

np.random.seed(42)
n = 60 
fechas = pd.date_range(start='2019-01-01', periods=n, freq='M')
ingreso = np.linspace(10, 20, n) + np.random.normal(0, 0.5, n)

u = np.zeros(n)
error = np.random.normal(0, 0.5, n)
for t in range(1, n):
    u[t] = 0.7 * u[t-1] + error[t]

consumo = 5 + 0.8 * ingreso + u
df = pd.DataFrame({'Fecha': fechas, 'Consumo': consumo, 'Ingreso': ingreso})

X = sm.add_constant(df['Ingreso'])
y = df['Consumo']
modelo = sm.OLS(y, X).fit()
resid = modelo.resid

print("MODELO")
print(modelo.summary().tables[1])

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(df['Fecha'], resid)
plt.title("Residuos vs Tiempo")
plt.axhline(0, c='r', ls='--')

plt.subplot(1, 2, 2)
plot_acf(resid, ax=plt.gca(), lags=15)
plt.title("ACF Residuos")
plt.tight_layout()
plt.show()

dw = durbin_watson(resid)
lb = acorr_ljungbox(resid, lags=[10])['lb_pvalue'].values[0]

print(f"\nDurbin-Watson: {dw:.4f}")
print(f"Ljung-Box p-value: {lb:.4f}")

if dw < 1.5 or lb < 0.05:
    print("\n¿Existe autocorrelación?: SÍ")
else:
    print("\n¿Existe autocorrelación?: NO")

glsar = sm.GLSAR(y, X, rho=1)
res_glsar = glsar.iterative_fit(maxiter=10)
print("\nREESTIMACIÓN A: GLSAR (AR1)")
print(res_glsar.summary().tables[1])

df['Lag_Y'] = df['Consumo'].shift(1)
df_din = df.dropna()
X_din = sm.add_constant(df_din[['Ingreso', 'Lag_Y']])
mod_din = sm.OLS(df_din['Consumo'], X_din).fit()

print("\nREESTIMACIÓN B: MODELO DINÁMICO")
print(mod_din.summary().tables[1])