'''
Contexto:
Un analista estudia si el gasto en salud mejora la esperanza de vida.
'''
import numpy as np
from sklearn.linear_model import LinearRegression

esperanza_vida_y = [72, 75, 78, 80, 82, 83]
gasto_salud_x1 = [5.0, 6.5, 7.0, 8.0, 9.0, 9.5]
pib_per_capita_x2 = [9, 11, 14, 18, 22, 25]
medicos_por_1000_x3 = [1.8, 2.1, 2.5, 3.0, 3.5, 3.8]

x_n = np.column_stack((gasto_salud_x1, pib_per_capita_x2, medicos_por_1000_x3))
model = LinearRegression()
model.fit(x_n, esperanza_vida_y)
print("Intercepto:", model.intercept_)
print("Coeficientes:", model.coef_)
print('Esperanza de Vida =', round(model.intercept_, 2), '+', round(model.coef_[0], 2), 'Gasto en Salud',
      round(model.coef_[1], 2), 'PIB per cápita +', round(model.coef_[2], 2), 'Médicos')

'''
1. ¿Qué variable controla el nivel de desarrollo?
PIB per cápita, ya que refleja el ingreso promedio de la población.

2. Interpreta el coeficiente del gasto en salud.
Por cada aumento en el gasto en salud, la esperanza de vida aumenta en 0.85.

3. ¿Qué problema de endogeneidad puede existir?
No analizar variable con correlacion en la esperanza de vida como los medicos.

4. ¿Qué otra variable social incluirías?
Pobreza, educación, acceso a servicios básicos, etc.
'''
