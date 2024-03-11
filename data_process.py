import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Wind Time Series Dataset(10min).csv')
data = data.iloc[:137]

plt.plot(data['Time'],data['Power'])
plt.show()
