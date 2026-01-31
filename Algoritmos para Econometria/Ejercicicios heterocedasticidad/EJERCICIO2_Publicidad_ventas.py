'''
Contexto : Una empresa analiza cómo la publicidad afecta las ventas mensuales.
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

publicidad = [2, 3, 4, 5, 6, 10, 12, 15, 18, 20]
ventas = [10, 13, 15, 18, 21, 30, 40, 55, 80, 110]
 
Y = ventas
X = publicidad
X = sm.add_constant(X)
 
model = sm.OLS(Y, X).fit()
print(model.summary())

#=========================
# A) Visual: residuos vs ajustados
# =========================
yhat = model.fittedvalues
resid = model.resid
 
plt.figure()
plt.scatter(yhat, resid)
plt.axhline(0)
plt.xlabel("Valores ajustados (Ŷ)")
plt.ylabel("Residuos (e)")
plt.title("Residuos vs Ajustados")
plt.show()

# =========================
# B) Prueba Breusch–Pagan
# =========================
bp = het_breuschpagan(resid, model.model.exog)
print(f"\nP-valor de Breusch-Pagan: {bp[1]:.4f}")
heterocesidad(bp[1])

'''
Describe el patrón de los residuos.
Sigue un patron de abanico, a bajos valores ajustados los residuos se quedan cerca de 0, aumenta y se alejan de la linea del 0.

¿Por qué es razonable esperar heterocedasticidad en este caso?
Por la forma del grafico, no sigue una distribucion constante.
'''