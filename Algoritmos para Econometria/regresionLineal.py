import matplotlib.pyplot as plt
import numpy as np
consumo = [4.9, 5.6, 6.3, 7.0, 7.6]
ingresos = [6, 8, 10, 12, 14]

mean_x = sum(ingresos) / len(ingresos)
mean_y = sum(consumo) / len(consumo)

numerador = 0
denominador = 0

for i in range(len(ingresos)):
    numerador += (ingresos[i] - mean_x) * (consumo[i] - mean_y)
    denominador += (ingresos[i] - mean_x) ** 2

m = numerador / denominador
b = mean_y - (m * mean_x)

def predecir(x):
    return (m * x) + b

def error(y, x):
    y_gorro = (m*x) + b
    error = y - y_gorro
    return error

print(f"La ecuación del modelo es: y = {b:.2f} + {m:.2f}x ")
print(f"Predicción para un ingreso de 50: {predecir(50):.2f}")
print(f"Error para el primer dato (y = {consumo[0]}) (y gorro = {predecir(ingresos[0]):.2f}): {error(consumo[0], ingresos[0]):.2f}")

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