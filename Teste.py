#%%
# %matplotlib inline
# %load_ext autoreload
#%%
# -*- coding: utf-8 -*-
# %autoreload 2 
import EspaceState as ee
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import LocalLevel
from scipy.optimize import minimize
import statsmodels.api as sm
from scipy import optimize
import pandas
import LocalTrend
#%%
llv = LocalLevel.LocalLevel()
fu = LocalTrend.LocalLevel()
y = llv.simulate(20)
sns.tsplot(y)
#%%
llv.fit(y)
#%%
#%%
alpha = llv.filter(y)
level = [value.item(0) for value in alpha['level']]
#phi   = [value.item(1) for value in alpha['level']]
plt.plot(y, color = (0,0,1),label = 'serie')
plt.plot(level, color = (1,0,0),label = 'nivel')
#plt.plot(phi, color = (0.7,0.3,0),label = 'phi')
#plt.plot(teste, color = (0,0,0),label = 'teste')
plt.legend()
#%%
beta = fu.smooth(y)
smt_level = [value.item(0) for value in beta['smth_level'][1:]]
plt.plot(y, color = (0,0,1),label = 'serie')
plt.plot(smt_level, color = (1,0,0),label = 'nivel')
plt.legend()