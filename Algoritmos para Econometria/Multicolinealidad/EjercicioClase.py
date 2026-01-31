import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib 
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
 
PATH_FILE = pathlib.Path(__file__).resolve().parent.parent
PATH = PATH_FILE/ 'Ejercicicios heterocedasticidad' / "mex_fin.csv"

# =========================
# A) Cargar datos
# =========================
df = pd.read_csv(PATH)
 
# Si tienes fecha, la ordenamos (opcional)
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
 
# =========================
# B) Preparar variables (ejemplo finanzas)
#    Ajusta nombres según tu CSV
# =========================
# Si ya tienes r_ipc, r_tc, rate, vol_ipc en tu archivo, omite esta sección.
if {"ipc", "fix"}.issubset(df.columns):
    df["r_ipc"] = np.log(df["ipc"] / df["ipc"].shift(1))
    df["r_tc"]  = np.log(df["fix"] / df["fix"].shift(1))
 
# Volatilidad móvil (12) si no existe
if "r_ipc" in df.columns and "vol_ipc" not in df.columns:
    df["vol_ipc"] = df["r_ipc"].rolling(12).std()
 
# =========================
# C) Seleccionar X (explicativas)
#    Cambia esta lista a tus columnas
# =========================
X_cols = ["r_tc", "rate", "vol_ipc"]  # <- AJUSTA AQUÍ
data = df.dropna(subset=X_cols).copy()
 
X = data[X_cols]
 
# =========================
# D) 1) Matriz de correlación
# =========================
corr = X.corr()
 
print("\nMatriz de correlación (X):")
print(corr.round(3))
 
# =========================
# E) 2) Calcular VIF
#    (ojo: VIF se calcula SIN la constante en el DataFrame final, pero
#    statsmodels lo pide con constante para consistencia en exog)
# =========================
X_vif = sm.add_constant(X)
 
vif_table = pd.DataFrame({
    "Variable": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
 
print("\nVIF (incluye const):")
print(vif_table)
 
# =========================
# F) Gráfico 1: Heatmap de correlación (con matplotlib)
# =========================
plt.figure()
plt.imshow(corr.values, aspect="auto")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.index)), corr.index)
plt.colorbar(label="Correlación")
plt.title("Heatmap de correlación entre variables X")
 
# Anotar valores en cada celda (opcional, útil en clase)
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        plt.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center")
 
plt.tight_layout()
plt.show()
 
# =========================
# G) Gráfico 2: Scatter matrix (pares) para ver relaciones lineales
# =========================
from pandas.plotting import scatter_matrix
 
scatter_matrix(X, diagonal="hist", figsize=(8, 8))
plt.suptitle("Scatter Matrix (relación entre X)")
plt.tight_layout()
plt.show()
 
# =========================
# H) (Opcional) Gráfico 3: Barras de VIF sin la constante
# =========================
vif_no_const = vif_table[vif_table["Variable"] != "const"].copy()
 
plt.figure()
plt.bar(vif_no_const["Variable"], vif_no_const["VIF"])
plt.axhline(5, linestyle="--")
plt.axhline(10, linestyle="--")
plt.title("VIF por variable (líneas guía: 5 y 10)")
plt.ylabel("VIF")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
