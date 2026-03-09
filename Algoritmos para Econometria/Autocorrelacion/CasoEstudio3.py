import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

np.random.seed(42)
n = 100

activos = np.random.normal(1000, 200, n)
ventas = activos * 0.9 + np.random.normal(0, 50, n)
cap_trabajo = activos * 0.3 + np.random.normal(0, 20, n)

roa = 10 + 0.05 * activos - 0.04 * ventas + 0.1 * cap_trabajo + np.random.normal(0, 2, n)

df = pd.DataFrame({
    'ROA': roa,
    'Activos': activos,
    'Ventas': ventas,
    'Cap_Trabajo': cap_trabajo
})

X = sm.add_constant(df[['Activos', 'Ventas', 'Cap_Trabajo']])
y = df['ROA']

modelo = sm.OLS(y, X).fit()
print("Modelo")
print(modelo.summary().tables[1])

print("\nMatriz de Correlaciones")
print(df[['Activos', 'Ventas', 'Cap_Trabajo']].corr())

vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\nFactor de Inflación de Varianza (VIF)")
print(vif_data)

max_vif = vif_data.iloc[1:]['VIF'].max()
if max_vif > 10:
    print(f"¿Existe Multicolinealidad Severa?: SÍ (VIF Máximo = {max_vif:.2f})")
else:
    print(f"¿Existe Multicolinealidad Severa?: NO (VIF Máximo = {max_vif:.2f})")

print("\nSolución A: Eliminar Variable Redundante (Ventas)")
X_red = sm.add_constant(df[['Activos', 'Cap_Trabajo']])
modelo_red = sm.OLS(y, X_red).fit()
print(modelo_red.summary().tables[1])

vif_red = pd.DataFrame()
vif_red["Variable"] = X_red.columns
vif_red["VIF"] = [variance_inflation_factor(X_red.values, i) for i in range(X_red.shape[1])]
print("\nVIF Nuevo (Eliminando variable):")
print(vif_red)

print("\nSolución B: Ratios Financieros (Rotación de Activos)")
df['Rotacion_Activos'] = df['Ventas'] / df['Activos']
X_ratio = sm.add_constant(df[['Rotacion_Activos', 'Cap_Trabajo']])
modelo_ratio = sm.OLS(y, X_ratio).fit()
print(modelo_ratio.summary().tables[1])