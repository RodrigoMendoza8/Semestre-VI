consumo_y = [52, 54, 55, 56, 57, 58, 59, 60]
ingreso_x1 = [80, 82, 84, 86, 88, 90, 92, 94]
inflacion_x2 = [4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4]
tasa_interes_x3 = [6.0, 6.2, 6.3, 6.5, 6.7, 6.9, 7.1, 7.3]
b0 = 10
b1 = 0.6
b2 = -1.20
b3 = -0.80

r2_auxiliares = {
    "X1": 0.94,
    "X2": 0.95,
    "X3": 0.96
}

def predecir(x1, x2, x3):
    return b0 + b1 * x1 + b2 * x2 + b3 * x3

vif = []
for var in r2_auxiliares:
    r2 = r2_auxiliares[var]
    vif.append(round(1 / (1 - r2), 3))

print("VIFs:", vif)

consumo = predecir(86, 4.6, 6.5)
print(f"Consumo estimado para X1=86, X2=4.6, X3=6.5: {consumo:.2f}")


'''
Interpreta los coeficientes (ceteris paribus).
El ingreso es la unica variable que tiene un efecto positivo sobre el consumo, mientras que la inflación 
y la tasa de interés tienen un efecto negativo.

Calcula VIF de cada X y concluye si hay multicolinealidad grave.
En cada caso, el VIF es mayor a 10, lo que indica una multicolinealidad grave.

Propón 1 acción correctiva.
Elimina una variable entre inflacion y tasa de interes.
'''