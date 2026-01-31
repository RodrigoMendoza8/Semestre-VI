'''
Contexto:
Un economista quiere analizar cómo cambia el consumo mensual de un hogar cuando aumenta su ingreso.
'''
import numpy as np
from sklearn.linear_model import LinearRegression

ingreso_x = [8, 10, 12, 14, 16]
consumo_y = [4.5, 5.2, 6.0, 6.8, 7.5]

ingreso_x = np.array(ingreso_x).reshape(-1, 1)
model = LinearRegression()
model.fit(ingreso_x, consumo_y)
print("Intercepto:", model.intercept_)
print("Coeficiente:", model.coef_)
print('Consumo =', round(model.intercept_, 2), '+', round(model.coef_[0], 2), 'Ingreso')

'''
1. ¿Qué variable es Y y cuál es X?
Consumo es Y porque es la variable que queremos predecir, ingreso es X porque es la variable que usamos para predecir Y.

2. ¿Esperas que β1 sea positiva o negativa?
Positiva, porque un mayor ingreso debería llevar a un mayor consumo.

3. Interpreta β1 en palabras.
Por cada unidad de ingreso, el consumo aumenta en 0.38.

4. ¿Qué factor importante no está incluido en el modelo?
Podria ser el ahorro.
'''
