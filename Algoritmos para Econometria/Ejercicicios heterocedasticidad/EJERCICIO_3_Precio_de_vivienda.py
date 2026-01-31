'''
Contexto : Un analista inmobiliario modela el precio de la vivienda según su tamaño.
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

tamanio = [50, 60, 70, 80, 90, 150, 180, 220, 260, 300]
precio = [800, 950, 1100, 1300, 1550, 2800, 3500, 4800, 6200, 8000]
 
Y = precio
X = tamanio
X = sm.add_constant(X)
 
model = sm.OLS(Y, X).fit()
print(model.summary())

yhat = model.fittedvalues
resid = model.resid

bp = het_breuschpagan(resid, model.model.exog)
print(f"\nP-valor de Breusch-Pagan: {bp[1]:.4f}")
heterocesidad(bp[1])