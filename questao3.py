#%%
%matplotlib inline
%load_ext autoreload
#%%
# -*- coding: utf-8 -*-
%autoreload 2 
import EspaceState as ee
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import statsmodels.api as sm
from scipy import optimize
import pandas
import LocalTrend
import csv
import pandas as pd
import Seasonal
import SeasonMQO
#%%
seas = Seasonal.Seasonal()
y = seas.simulate(20)
plt.plot(y)
#%%
reg = SeasonMQO.MQOSeasonal(5)
dummie = reg.filter_dummie(y)
dummie =[dummie.item(i) for i in range(len(y))]
plt.plot(dummie)
plt.plot(y)
plt.savefig('dummie.png', bbox_inches='tight')
#%%
reg = SeasonMQO.MQOSeasonal(5)
dummie = reg.filter_trig(y)
dummie =[dummie.item(i) for i in range(len(y))]
plt.plot(dummie)
plt.plot(y)
plt.savefig('trig.png', bbox_inches='tight')
