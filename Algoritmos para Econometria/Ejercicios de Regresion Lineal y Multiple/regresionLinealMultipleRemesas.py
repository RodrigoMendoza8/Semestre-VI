import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

crec_pib_y = [2.1, 1.8, 2.5, 1.2, 2.8, 3.0]
remesas_x1 = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
inversion_x2 = [20, 19, 21, 18, 22, 23]
apertura_x3 = [60, 58, 62, 57, 63, 65]

x_n = np.column_stack((remesas_x1, inversion_x2, apertura_x3))
model = LinearRegression()
model.fit(x_n, crec_pib_y)
print("Intercepto:", model.intercept_)
print("Coeficientes:", model.coef_)
print('Crecimiento PIB =', model.intercept_, '+', model.coef_[0], 'Remesas +', model.coef_[1], 'Inversion +', model.coef_[2], 'Apertura')

'''
1. Interpreta el coeficiente de las remesas ceteris paribus.
La inversion tiene un impacto positivo y grande en el crecimiento del PIB. Mientras que las 
remesas y la apertura tienen un impacto negativo y menor.

2. ¿Cuál variable parece más relevante para el crecimiento?
La inversion

3. ¿Qué problema econométrico podría existir entre remesas y crecimiento?
En este caso con estos datos las remesas estan afectando de menera negativa al crecimiento del PIB.
'''