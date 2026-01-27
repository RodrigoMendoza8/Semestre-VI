'''
Contexto:
Una empresa analiza los determinantes de sus ventas mensuales.
'''
import numpy as np
from sklearn.linear_model import LinearRegression

ventas_y = [120, 135, 140, 150, 145, 160]
publicidad_x1 = [10, 12, 14, 15, 13, 16]
precio_x2 = [20, 19, 18, 18, 19, 17]
promociones_x3 = [1, 1, 2, 2, 1, 3]

x_n = np.column_stack((publicidad_x1, precio_x2, promociones_x3))
model = LinearRegression()
model.fit(x_n, ventas_y)
print("Intercepto:", model.intercept_)
print("Coeficientes:", model.coef_)
print('Ventas =', round(model.intercept_, 2), '+', round(model.coef_[0], 2), 'Publicidad +',
      round(model.coef_[1], 2), 'Precio', round(model.coef_[2], 2), 'Promociones')

'''
1. Interpreta el efecto del precio.
Por cada unidad que aumente el precio, va a aumentar en 2 las ventas.

2. ¿Qué variable representa una decisión estratégica?
Publicidad, por el gran impacto que tiene en las ventas.

3. ¿Qué signo esperas para la publicidad?
Positivo, más publicidad debería aumentar las ventas.

4. ¿Qué pasa si publicidad y promociones están correlacionadas?
Tendran un impacto en las ventas.
'''
