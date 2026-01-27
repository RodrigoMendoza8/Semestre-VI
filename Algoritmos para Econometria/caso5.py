'''
Contexto:
Un analista inmobiliario estudia la relación entre el tamaño de una casa y su precio. 
'''
import numpy as np
from sklearn.linear_model import LinearRegression

tamano_x = [60, 80, 100, 120, 140]
precio_y = [900, 1150, 1400, 1650, 1900]

tamano_x = np.array(tamano_x).reshape(-1, 1)
model = LinearRegression()
model.fit(tamano_x, precio_y)
print("Intercepto:", model.intercept_)
print("Coeficiente:", model.coef_)
print('Precio =', round(model.intercept_, 2), '+', round(model.coef_[0], 2), 'Tamaño')

'''
1. ¿Qué indica β1?
Cuanto aumenta el precio por cada metro cuadrado adicional.

2. ¿Es razonable que dos casas del mismo tamaño tengan distinto precio?
Si, por factores como ubicación, estado de la casa, etc.

3.¿Qué variables faltan?
Estado de la casa, ubicación, número de habitaciones, antigüedad, etc.
'''
