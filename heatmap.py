import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data_average_delete.csv')
# data = data.iloc[:1000]

# 删除前三列
data = data.iloc[:, 3:]

# 计算相关系数矩阵
correlation_matrix = data.corr()

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

