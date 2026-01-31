'''
Contexto : Se analiza el rendimiento de un índice bursátil en función del riesgo.
'''
import numpy as np 
import statsmodels.api as sm 
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
import matplotlib.pyplot as plt

def heterocesidad(p_value):
    if p_value < 0.05:
        print("Se rechaza la hipótesis nula de homocedasticidad. Hay evidencia de heterocedasticidad.")
    else:
        print("No se rechaza la hipótesis nula de homocedasticidad. No hay evidencia de heterocedasticidad.")

riesgo = [0.5, 0.6, 0.7, 0.8, 0.9, 1.5, 1.7, 2.0, 2.3, 2.8]
rendimiento = [0.01, 0.015, 0.018, 0.020, 0.025, -0.10, 0.15, -0.20, 0.30, -0.40]
 
Y = rendimiento
X = riesgo
X = sm.add_constant(X)
 
model = sm.OLS(Y, X).fit()
print(model.summary())

yhat = model.fittedvalues
resid = model.resid

plt.figure()
plt.scatter(yhat, resid)
plt.axhline(0)
plt.xlabel("Valores ajustados (Ŷ)")
plt.ylabel("Residuos (e)")
plt.title("Residuos vs Ajustados")
plt.show()

bp = het_breuschpagan(resid, model.model.exog)
print(f"\nP-valor de Breusch-Pagan: {bp[1]:.4f}")
heterocesidad(bp[1])

'''
¿Qué sucede con la varianza del error cuando el riesgo aumenta?
Se normaliza, al inicio es cuando es muy volatil.
¿Este comportamiento es típico en finanzas?
No creo, a mayor riesgo mayor volatilidad.
'''