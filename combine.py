import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

original = pd.read_csv('Wind Time Series Dataset(10min).csv')
original = original.iloc[:100]

original_power = original['Power']

forecast_trend = pd.read_csv('forecast_trend.csv')
forecast_seasonal = pd.read_csv('forecast_seasonal.csv')
forecast_resid = pd.read_csv('forecast_resid')

# mae = mean_absolute_error(y_test, y_predict)
# mse = mean_squared_error(y_test, y_predict)
# rmse = np.sqrt(mean_squared_error(y_test, y_predict))
# mape = (abs(y_predict - y_test) / y_test).mean()
# r_2 = r2_score(y_test, y_predict)

plt.figure(figsize=(12,8))
plt.plot(original.Time,original_power,color = 'blue',label = 'Original')
plt.show()



