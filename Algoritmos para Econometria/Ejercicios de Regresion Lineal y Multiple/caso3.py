'''
Contexto:
Una empresa analiza si aumentar el gasto en publicidad incrementa sus ventas mensuales.
'''
import numpy as np
from sklearn.linear_model import LinearRegression

publicidad_x = [5, 7, 9, 11, 13]
ventas_y = [20, 24, 28, 32, 36]

publicidad_x = np.array(publicidad_x).reshape(-1, 1)
model = LinearRegression()
model.fit(publicidad_x, ventas_y)
print("Intercepto:", model.intercept_)
print("Coeficiente:", model.coef_)
print('Ventas =', round(model.intercept_, 2), '+', round(model.coef_[0], 2), 'Gasto en publicidad')

'''
1. ¿Qué representa β1 para la empresa?
El aumento en ventas por cada unidad adicional gastada en publicidad.

2. ¿Puede existir un punto donde la publicidad ya no funcione igual?
Hablando en la vida real, sí. Pero en el modelo no hay tal punto.

3. ¿Qué riesgos hay al usar solo una variable?
No considerar otros factores que afectan las ventas, como la calidad del producto o la competencia.
'''
