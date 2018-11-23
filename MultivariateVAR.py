from datetime import datetime
import pandas as pd 
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import os 
os.chdir("C:/Users/Reet Barik/Desktop/Machine Learning/Project/ml-timeseries-project")

def parser(x):
    return datetime.strptime(x, '%d-%b-%Y')


nifty_data = pd.read_csv('./ml-project-data/NIFTY-50/2014.csv', header=0, 
                                                              parse_dates=['Date'], index_col=0, date_parser=parser)

scaler = MinMaxScaler(feature_range=(-1, 1))
nifty_data['Average'] = nifty_data.loc[: , "High":"Low"].mean(axis=1)
nifty = scaler.fit_transform(nifty_data[['Average', 'Shares Traded', 'Turnover (Rs. Cr)']])

nifty15 = pd.read_csv('./ml-project-data/NIFTY-50/2015.csv', header=0, 
                                                              parse_dates=['Date'], index_col=0, date_parser=parser)
nifty15['Average'] = nifty15.loc[: , "High":"Low"].mean(axis=1)
nifty2015 = scaler.fit_transform(nifty15[['Average', 'Shares Traded', 'Turnover (Rs. Cr)']])


coint_johansen(nifty,-1,1).lr1

model = VAR(endog=nifty)

model = model.fit()

prediction = model.forecast(model.y, steps=int(len(nifty)/4))

print('RMSE: ', np.sqrt(mean_squared_error(prediction[:,0], nifty2015[:int(len(nifty)/4),0])))

print('MAPE: ', np.mean(np.abs((nifty2015[:int(len(nifty)/4),0] - prediction[:,0])/len(nifty2015[:int(len(nifty)/4),0])) * 100 ))

plt.plot(prediction[:,0], color = 'blue')
plt.plot(nifty2015[:int(len(nifty)/4),0], color = 'red')

plt.show()