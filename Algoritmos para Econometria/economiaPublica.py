'''
Contexto:
Las autoridades energéticas analizan los factores que influyen en el consumo eléctrico.
'''
import numpy as np
from sklearn.linear_model import LinearRegression

consumo_y = [100, 102, 105, 98, 108, 110]
pib_x1 = [2.5, 2.0, 3.0, 1.5, 3.2, 3.5]
temperatura_x2 = [22, 23, 24, 21, 25, 26]
precio_energia_x3 = [1.8, 1.9, 1.7, 2.0, 1.6, 1.5]

x_n = np.column_stack((pib_x1, temperatura_x2, precio_energia_x3))
model = LinearRegression()
model.fit(x_n, consumo_y)
print("Intercepto:", model.intercept_)
print("Coeficientes:", model.coef_)
print('Consumo =', round(model.intercept_, 2), round(model.coef_[0], 2), 'PIB +',
      round(model.coef_[1], 2), 'Temperatura', round(model.coef_[2], 2), 'Precio de Energía')

'''
1. Interpreta el efecto de la temperatura.
Por cada aumento de temperatura, el consumo aumenta en 2.18.

2. ¿Qué variable representa ingreso?
PIB, ya que refleja la actividad económica.

3. ¿Qué signo esperas para el precio?
Negativo, ya que un aumento en el precio debería reducir el consumo.

4. ¿Qué problema podría surgir al usar datos anuales?
No capturar variaciones estacionales o cambios a corto plazo en el consumo.
'''
