import matplotlib.pyplot as plt
import numpy as np

def predecir(x):
    return (m * x) + b

def error(y, x):
    y_gorro = (m*x) + b
    error = y - y_gorro
    return error

consumo = [4.9, 5.6, 6.3, 7.0, 7.6]
ingresos = [6, 8, 10, 12, 14]

promedio_x = sum(ingresos) / len(ingresos)
promedio_y = sum(consumo) / len(consumo)

numerador = 0
denominador = 0

for i in range(len(ingresos)):
    numerador += (ingresos[i] - promedio_x) * (consumo[i] - promedio_y)
    denominador += (ingresos[i] - promedio_x) ** 2

m = numerador / denominador
b = promedio_y - (m * promedio_x)

print(f"La ecuación del modelo es: y = {b:.2f} + {m:.2f}x ")
print(f"Para un ingreso de 50, el consumo va a ser de: {predecir(50):.2f}")
print(f"Error para X = 10 (y = {consumo[3]}) (y gorro = {predecir(ingresos[3]):.2f}): {error(consumo[3], ingresos[3]):.2f}")

'''
Al calcular el error para X = 10, obtenemos un valor de 0.04, lo que indica que el modelo predice el consumo con bastante precisión.
'''

plt.figure()
plt.scatter(ingresos, consumo, label='Datos')
x = np.array(ingresos)
y = m * x + b
plt.plot(x, y, color='red', label='Línea de regresión')
plt.xlabel('Ingresos')
plt.ylabel('Consumo')
plt.legend()
plt.grid()
plt.show()

'''
Cuando x sea 0 (el ingreso sea 0), se va a consumir 2.88. 
Por cada unidad que aumente el ingreso, el consumo aumenta en 0.34 unidades.
'''