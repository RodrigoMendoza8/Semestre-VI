import numpy as np
from sklearn.linear_model import LinearRegression

ingreso_x = [6, 8, 10, 12, 14, 16]
consumo_y = [5.1, 5.7, 6.4, 6.9, 7.7, 8.0]

ingreso_x = np.array(ingreso_x).reshape(-1, 1)
model = LinearRegression()
model.fit(ingreso_x, consumo_y)
print("Intercepto:", model.intercept_)
print("Coeficiente:", model.coef_)
print('Consumo =', round(model.intercept_, 2), '+', round(model.coef_[0], 2), 'Ingreso')

print('Consumo para ingreso de 11:', round(model.intercept_ + model.coef_[0] * 11, 3))
print('Residuo con ingreso de 10:', 6.4 - round(model.intercept_ + model.coef_[0] * 10, 3))