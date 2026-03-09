import numpy as np
residuos = np.array([0.40, 0.35, 0.30, 0.25, 0.20, 0.10, 0.05, 0.00, -0.05, -0.10])

def durbin_watson(residuos):
    diferencias = np.diff(residuos)
    numerador = np.sum(diferencias**2)
    denominador = np.sum(residuos**2)
    
    return numerador / denominador

dw_stat = durbin_watson(residuos)
p_value_bg = 0.003  

print(f"Estadístico Durbin-Watson:", dw_stat)
print(f"P-value Breusch-Godfrey:", p_value_bg)

if dw_stat < 1:
    print("Autocorrelacion Possitiva.")
else:
    print("Autocorrelacion Negativa.")

if p_value_bg < 0.05:
    print("BG: Hay autocorrelacion.")
else:
    print("BG: No hay autocorrelacion.")
