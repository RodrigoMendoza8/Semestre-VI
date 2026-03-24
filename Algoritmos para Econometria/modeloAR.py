import pandas as pd
from sklearn.linear_model import LinearRegression

class AR1Forecaster:
    def __init__(self, target_col: str):
        self.target_col = target_col
        self.model = LinearRegression()
        self.coef_ = None
        self.intercept_ = None

    def _prepare_data(self, df):
        df_calc = df.copy()
        df_calc['lag_1'] = df_calc[self.target_col].shift(1)
        return df_calc.dropna()

    def fit(self, df):
        df_train = self._prepare_data(df)
        X = df_train[['lag_1']]
        y = df_train[self.target_col]
        
        self.model.fit(X, y)
        self.coef_ = self.model.coef_[0]
        self.intercept_ = self.model.intercept_
        return self

    def get_equation(self):
        return f"y_t = {self.intercept_:.4f} + {self.coef_:.4f}(y_t-1)"

    def forecast_next(self, current_value):
        if self.coef_ is None:
            raise ValueError("El modelo no se ha entrenado.")
        return self.intercept_ + (self.coef_ * current_value)

data = {'Mes': ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep'],
            'Ventas': [120, 128, 135, 142, 150, 158, 165, 170, 178]}
df_sales = pd.DataFrame(data)

forecaster = AR1Forecaster(target_col='Ventas')
forecaster.fit(df_sales)
    
print(f"Modelo: {forecaster.get_equation()}")
    
last_sale = df_sales['Ventas'].iloc[-1]
next_forecast = forecaster.forecast_next(last_sale)
    
print(f"Pronóstico Octubre: {next_forecast:.2f}")