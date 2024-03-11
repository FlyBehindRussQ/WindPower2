import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 截取数据
df = pd.read_csv('seasonal.csv')
df = df.iloc[:1000]
df['Time'] = pd.to_datetime(df['Time'])  # 将时间列转换为日期时间类型
df.set_index('Time', inplace=True)  # 将时间列设置为索引

# 绘制自相关函数（ACF）图
plot_acf(df['seasonal'].dropna())
plt.title('Autocorrelation Function (ACF)')
plt.show()

# 绘制偏自相关函数（PACF）图
plot_pacf(df['seasonal'].dropna())
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# 划分训练集和测试集
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# 定义滑动窗口大小
window_size = 10

# 初始化预测结果列表
predictions_test = []

# 滑动窗口预测（测试集）
for i in range(len(test)):
    # 获取当前窗口的训练数据
    window_train = df.iloc[i:i+window_size]

    # 拟合ARIMA模型
    try:
        model = sm.tsa.arima.ARIMA(window_train['seasonal'], order=(1, 1, 1))  # 使用季节性数据列拟合模型
        arima_res = model.fit()

        # 预测下一个时间步
        next_time_step = arima_res.forecast()[0]

        # 检查预测结果的有效性
        if np.isnan(next_time_step):
            # 无效预测结果，用缺失值替换
            predictions_test.append(np.nan)
        else:
            # 有效预测结果，将其添加到列表中
            predictions_test.append(next_time_step)
    except:
        # 拟合模型失败，用缺失值替换
        predictions_test.append(np.nan)

# 将测试集预测结果转换为Series，索引与测试集相同
predictions_series_test = pd.Series(predictions_test, index=test.index)

# 对预测数据进行线性插值填充缺失值
predictions_series_test = predictions_series_test.interpolate(method='linear')

# 绘制测试集预测结果
plt.plot(test.index, test['seasonal'], label='True (Test)')  # 使用日期时间索引和季节性数据列
plt.plot(predictions_series_test.index, predictions_series_test, label='Predicted (Test) (Shifted)')

plt.legend()
plt.show()

# 计算移动后的测试集预测数据的残差
forecast_residuals_shifted = test['seasonal'] - predictions_series_test

# 计算移动后的MAE（排除缺失值）
mae_shifted = np.mean(np.abs(forecast_residuals_shifted.dropna()))
print("Mean Absolute Error (Shifted):", mae_shifted)

