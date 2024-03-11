from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# 多元非线性回归
data = pd.read_csv('trend.csv')
# 定义一个非线性函数
def nonlinear_function(x, a, b, c):
    return a * x**2 + b * x + c

# 提取趋势数据
x_data = np.arange(len(data.trend))
y_data = data.trend

# 使用 curve_fit 函数拟合趋势数据
popt, pcov = curve_fit(nonlinear_function, x_data, y_data)

# 提取拟合的参数值
a_fit, b_fit, c_fit = popt

# 绘制原始数据和拟合结果
regression = nonlinear_function(x_data, a_fit, b_fit, c_fit)

plt.scatter(x_data, y_data, label='Original Data')
plt.plot(x_data, regression, color='red', label='Fitted Curve')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Nonlinear Regression of Trend Data')
plt.show()

# 打印拟合的参数值
print("Fitted Parameters:")
print("a =", a_fit)
print("b =", b_fit)
print("c =", c_fit)

# 将时间和回归值合并成一个DataFrame
regression_data = pd.DataFrame({'Time': data.Time, 'Regression': regression})

# 保存数据到CSV文件
regression_data.to_csv('forecast_trend.csv', index=False)
