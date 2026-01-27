'''
Contexto:
Una universidad quiere analizar qué factores influyen en el promedio final de sus estudiantes.
'''
import numpy as np
from sklearn.linear_model import LinearRegression

promedio_y = [78, 82, 75, 88, 90, 85]
horas_estudio_x1 = [10, 12, 8, 15, 16, 14]
asistencia_porcentaje_x2 = [80, 85, 78, 90, 92, 88]
horas_trabajo_x3 = [20, 15, 25, 10, 8, 12]

x_n = np.column_stack((horas_estudio_x1, asistencia_porcentaje_x2, horas_trabajo_x3))
model = LinearRegression()
model.fit(x_n, promedio_y)
print("Intercepto:", model.intercept_)
print("Coeficientes:", model.coef_)
print('Promedio =', round(model.intercept_, 2), '+', round(model.coef_[0], 2), 'Horas de estudio +',
      round(model.coef_[1], 2), 'Asistencia (%) +', round(model.coef_[2], 2), 'Horas de trabajo')

'''
1. Interpreta cada coeficiente.
Todos los coeficientes son positivos, indicando que tienen un impacto bueno en el promedio final.

2. ¿Qué variable esperas que tenga efecto negativo?
Horas de trabajo, ya que reduce el tiempo disponible para estudiar, Aunque en el modelo es positivo

3. ¿Es razonable una relación lineal?
No del todo, puede haber variables que afecten pero no se puedan modelar linealmente.

4. ¿Qué variable faltante podría mejorar el modelo?
Motivación del estudiante, calidad de enseñanza, recursos disponibles, etc.
'''
