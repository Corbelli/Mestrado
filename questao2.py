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
#%%
series = pd.read_csv('ipeadata[20-06-2017-11-57].csv')
teste = series.as_matrix()
teste = teste[:,1]
plt.plot(teste)
#%%
fu = LocalTrend.LocalTrend()
fu.fit(teste)
#%%
alpha = fu.smooth(teste)
level = [value.item(0) for value in alpha['level']]
plt.plot(teste, color = (0,0,1),label = 'serie')
plt.plot(level, color = (1,0,0),label = 'nivel')
plt.legend()
plt.savefig('pib_level.png', bbox_inches='tight')
#%%
phi   = [value.item(1) for value in alpha['level']]
plt.plot(phi, color = (0.7,0.3,0),label = 'phi')
plt.savefig('pib_arma.png', bbox_inches='tight')
plt.legeng()


