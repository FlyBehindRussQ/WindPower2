from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


######################################################################
# STL分解
plt.rc("figure", figsize=(10, 6))
 
df=pd.read_csv("Wind Time Series Dataset(10min).csv")
df['Time']=pd.to_datetime(df['Time'])
df.set_index('Time',inplace=True)
 
res = STL(df['Power'], period=24*6*90).fit()  #period确定：10分钟一个数据，以天为周期就是24*6
res.plot()
 
df['trend']=res.trend
df['seasonal']=res.seasonal
df['resid']=res.resid

df['trend'].to_csv('trend.csv')
df['seasonal'].to_csv('seasonal.csv')
df['resid'].to_csv('resid.csv')

plt.show()

print('residual mean:',df.resid.mean())
df.resid.hist()

#从数据中删除趋势项
df['detrend']=df['Power']-df.trend
#从数据中删除季节项
df['deseasonal']=df['Power']-df.seasonal

trend_strength=max(0,1-df.resid.var()/df.deseasonal.var())
seasonal_strength=max(0,1-df.resid.var()/df.detrend.var())
print('trend_strength:',trend_strength)
print('seasonal_strength:',seasonal_strength)





