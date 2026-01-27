'''
Contexto:
Un profesor quiere saber si más horas de estudio mejoran la calificación final.
'''
import numpy as np
from sklearn.linear_model import LinearRegression

horas_estudio_x = [2, 4, 6, 8, 10]
calificacion_y = [60, 68, 75, 82, 88]

horas_estudio_x = np.array(horas_estudio_x).reshape(-1, 1)
model = LinearRegression()
model.fit(horas_estudio_x, calificacion_y)
print("Intercepto:", model.intercept_)
print("Coeficiente:", model.coef_)
print('Calificación =', round(model.intercept_, 2), '+', round(model.coef_[0], 2), 'Horas de estudio')

'''
1. ¿Qué significa el intercepto β0?
El valor cuando las horas de estudio son cero. En este caso si no estudiaran nada, la calificación sería 53.6.

2. ¿Es realista pensar que solo estudiar explica la calificación?
No.

3. ¿Qué otras variables podrían influir?
Asistencia a clases, horas de sueño, etc.
'''
