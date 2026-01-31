'''
Contexto:
Un hotel de los suburbios obtiene su ingreso bruto de la renta de sus instalaciones y de su restaurante. 
Los propietarios tienen interés en conocer la relación entre el número de habitaciones ocupadas por noche 
y el ingreso por día en el restaurante.
'''
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm

ingreso = [
    1452, 1361, 1426, 1470, 1456, 1430, 1354, 1442, 1394, 1459, 
    1399, 1458, 1537, 1425, 1445, 1439, 1348, 1450, 1431, 1446, 
    1485, 1405, 1461, 1490, 1426
]

habitaciones_ocupadas = [
    23, 47, 21, 39, 37, 29, 23, 44, 45, 16, 
    30, 42, 54, 27, 34, 15, 19, 38, 44, 47, 
    43, 38, 51, 61, 39
]

habitaciones_ocupadas = np.array(habitaciones_ocupadas).reshape(-1, 1)
model = LinearRegression()
model.fit(habitaciones_ocupadas, ingreso)
print("Intercepto:", model.intercept_)
print("Coeficiente:", model.coef_)
print('Ingreso =', round(model.intercept_, 2), '+', round(model.coef_[0], 2), 'Habitaciones Ocupadas')

X = sm.add_constant(habitaciones_ocupadas) 
y = ingreso
model_stats = sm.OLS(y, X).fit()
p_valor = model_stats.pvalues[1]
print(f"El p-valor es: {p_valor:.4f}")

r2 = model.score(habitaciones_ocupadas, ingreso)
print(f"Coeficiente de determinación (R²): {r2:.4f}")

plt.figure(figsize=(8,6))
plt.scatter(habitaciones_ocupadas, ingreso, color='blue', label='Datos')
plt.plot(habitaciones_ocupadas, model.predict(habitaciones_ocupadas), color='red', label='Línea de Regresión')
plt.xlabel('Habitaciones Ocupadas')
plt.ylabel('Ingreso por Restaurante')
plt.grid()
plt.show()

'''
a) ¿Parece que aumenta el ingreso por desayunos a medida que aumenta el número de habitaciones ocupadas? Trace un diagrama de dispersión para apoyar su conclusión.
Si, la pendiente positiva lo indica, aunque no aumenta de gran manera porque vale 1.48.

b) Determine el coeficiente de correlación entre las dos variables. Interprete el valor.
La pendiente es 1.48, lo que indica que por cada habitación ocupada adicional, el ingreso aumenta en 1.48.

c) ¿Es razonable concluir que hay una relación positiva entre ingreso y habitaciones ocupadas? Utilice el nivel de significancia 0.10.
El p-valor es de 0.035, con lo que podemos concluir que hay una relación positiva entre las variables.

d) ¿Qué porcentaje de la variación de los ingresos del restaurante se contabilizan por el número
de habitaciones ocupadas?
Un 17.89%
'''