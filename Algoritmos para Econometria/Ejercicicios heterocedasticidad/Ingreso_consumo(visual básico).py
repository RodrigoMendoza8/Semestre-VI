'''
Contexto : Un economista estima un modelo simple para explicar el consumo mensual a partir del ingreso.
'''
import pandas as pd 
import numpy as np 
import statsmodels.api as sm 
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
import matplotlib.pyplot as plt

def heterocesidad(p_value):
    if p_value < 0.05:
        print("Se rechaza la hipótesis nula de homocedasticidad. Hay evidencia de heterocedasticidad.")
    else:
        print("No se rechaza la hipótesis nula de homocedasticidad. No hay evidencia de heterocedasticidad.")

ingreso = [5, 6, 7, 8, 9, 15, 18, 22, 25, 30]
consumo = [3.2, 3.8, 4.1, 4.5, 5.0, 7.5, 8.9, 11.2, 13.8, 18.5]
 
Y = consumo
X = ingreso
X = sm.add_constant(X)
 
model = sm.OLS(Y, X).fit()
print(model.summary())

#=========================
# A) Visual: residuos vs ingresos
# =========================
ingresos = ingreso
resid = model.resid
 
plt.figure()
plt.scatter(ingresos, resid)
plt.axhline(0)
plt.xlabel("Ingresos")
plt.ylabel("Residuos (e)")
plt.title("Residuos vs Ingresos")
plt.show()

# =========================
# B) Prueba Breusch–Pagan
# =========================
bp = het_breuschpagan(resid, model.model.exog)
print(f"\nP-valor de Breusch-Pagan: {bp[1]:.4f}")
heterocesidad(bp[1])

'''
¿La dispersión del error es constante?
No, la dispersión no es constante. 
Al observar la gráfica, se aprecia que los residuos no se distribuyen de manera uniforme alrededor de la línea de cero.

¿Existe heterocedasticidad? Explica.
Si, de acuerdo a la prueba de Breusch-Pagan, el p-valor es menor a 0.05, lo que indica heterocedasticidad
'''