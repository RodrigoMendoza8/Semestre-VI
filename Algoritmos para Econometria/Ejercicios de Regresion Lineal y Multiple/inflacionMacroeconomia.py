'''
Contexto:
La inflación es uno de los principales objetivos de política económica y puede verse influida por factores monetarios y externos.
'''
import numpy as np
from sklearn.linear_model import LinearRegression

inflacion_y = [3.2, 3.8, 4.5, 5.1, 4.0, 3.6]
tipo_cambio_x1 = [18.5, 19.0, 19.8, 20.5, 20.0, 19.5]
tasa_interes_x2 = [6.0, 6.5, 7.0, 7.5, 7.0, 6.5]
crec_pib_x3 = [2.5, 2.0, 3.0, 1.5, 2.8, 3.2]

x_n = np.column_stack((tipo_cambio_x1, tasa_interes_x2, crec_pib_x3))
model = LinearRegression()
model.fit(x_n, inflacion_y)
print("Intercepto:", model.intercept_)
print("Coeficientes:", model.coef_)

'''
1. Interpreta el efecto del tipo de cambio.
El coeficiente es de -0.70, indicando que un aumento en el tipo de cambio reduce la inflación en este modelo.

2. ¿Qué variable controla la demanda agregada?
La tasa de interes, por su coeficiente positivo y grande.

3. ¿Qué problemas econométricos pueden surgir?
Falta de datos, variables omitidas, etc.
'''
