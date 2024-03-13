import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

original = pd.read_csv('Wind Time Series Dataset(10min).csv')
original = original.iloc[800:1000]

original_power = original['Power']

forecast_trend = pd.read_csv('forecast_trend.csv')
forecast_seasonal = pd.read_csv('forecast_seasonal.csv')
forecast_resid = pd.read_csv('forecast_resid.csv')

# 合并预测值数据并设置时间戳列为索引
merged_forecast = pd.merge(forecast_trend, forecast_seasonal, on='Time')
merged_forecast = pd.merge(merged_forecast, forecast_resid, on='Time')

# 将缺失值填充为0（如果有的话）
merged_forecast.fillna(0, inplace=True)

# 执行相加操作
merged_forecast['Combined'] = merged_forecast['Regression'] + merged_forecast['seasonal'] + merged_forecast['resid']

print(merged_forecast)

plt.figure(figsize=(12,8))
plt.plot(original.Time,original_power,color = 'blue',label = 'Original')
plt.plot(original.Time, merged_forecast['Combined'], color='red', label='Forecast')
plt.show()



