#Import Libraries
import psycopg2
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.arima_model import ARIMA
from sqlalchemy import create_engine

#Configure PostgreSQL Database and Define Variables
engine = create_engine('postgresql+psycopg2://postgres:rainbow@localhost/datapoints')
table_name = 'series'
df = pd.read_sql_table(table_name, engine)
date=df['date']
value=df['value']

#Autocorrelation and Partial Autocorrelation Plots
value
#plt.plot(value)
#plt.show()
acf_1 =  acf(value)[1:20]
test_df = pd.DataFrame([acf_1]).T
test_df.columns = ['Pandas Autocorrelation']
test_df.index += 1
test_df.plot(kind='bar')
plt.plot(acf_1)
plt.show()

pacf_1 =  pacf(value)[1:20]
test_df = pd.DataFrame([pacf_1]).T
test_df.columns = ['Pandas Partial Autocorrelation']
test_df.index += 1
test_df.plot(kind='bar')
plt.plot(pacf_1)
plt.show()

result = ts.adfuller(value, 1)
result

#ARIMA
value_matrix=value.as_matrix()
model = ARIMA(value_matrix, order=(0,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
predictions=model_fit.predict(633, 731, typ='levels')
predictions=np.array(predictions)
last100days=value[632:731]
last100days=np.array(last100days)
len(predictions)
len(last100days)
deviation=(predictions-last100days)/last100days
meandeviation=np.mean(deviation)
meandeviation*100
plt.plot(predictions)
plt.legend(['Predicted'], loc='upper left')
plt.title(r'ARIMA (0,1,0)')
plt.show()

#Monte Carlo Simulation
mu=np.mean(value)
sigma=np.std(value)
x = mu + sigma * np.random.randn(10000)
num_bins = 50

# Histogram
n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='blue', alpha=0.5)
# add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')
plt.xlabel('Values')
plt.ylabel('Probability')
plt.title(r'Histogram of Value')

# Spacing Adjustment
plt.subplots_adjust(left=0.15)
plt.show()
