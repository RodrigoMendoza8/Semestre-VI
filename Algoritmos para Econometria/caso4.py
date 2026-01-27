'''
Contexto:
Se estudia si la depreciación del peso afecta la inflación anual.
'''
import numpy as np
from sklearn.linear_model import LinearRegression

tipo_cambio_x = [18.5, 19.0, 19.8, 20.5, 21.0]
inflacion_y = [3.2, 3.8, 4.5, 5.1, 5.6]

tipo_cambio_x = np.array(tipo_cambio_x).reshape(-1, 1)
model = LinearRegression()
model.fit(tipo_cambio_x, inflacion_y)
print("Intercepto:", model.intercept_)
print("Coeficiente:", model.coef_)
print('Inflación =', round(model.intercept_, 2), '+', round(model.coef_[0], 2), 'Tipo de cambio')

'''
1. ¿Cómo interpretas una pendiente positiva?
Que la variable independiente (tipo de cambio) y la dependiente (inflación) se mueven en la misma dirección.

2. ¿Qué significa que el peso se deprecie?
Significa un aumento en la tasa de cambio y poor lo tanto un aumento en la inflación según el modelo.

3. ¿Es correcto usar solo el tipo de cambio para explicar la inflación?
No, hay otros factores que afectan la inflación como la política, precios internacionales, etc.
'''
