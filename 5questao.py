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
import Meb
import csv
import pandas as pd
#%%
#1970-2014
series = pd.read_csv('ipeadata[21-06-2017-01-26].csv')
BalancaComercial = series.as_matrix()
BalancaComercial = np.float64(BalancaComercial[:,1])
BalancaComercial = BalancaComercial[150:190])
BalancaComercial = (BalancaComercial - min(BalancaComercial) + 1)/10
print(BalancaComercial)
BalancaComercial = np.log(BalancaComercial )
#plt.plot(BalancaComercial)
#%%
meb = Meb.Meb()
#meb.fit(BalancaComercial)
#%%
alpha = meb.smooth(BalancaComercial)
level = [value.item(0) for value in alpha['level']]
plt.plot(BalancaComercial, color = (0,0,1),label = 'serie')
plt.plot(level, color = (1,0,0),label = 'nivel')
plt.legend()
plt.savefig('Balanca_level.png', bbox_inches='tight')
#%%
season   = [value.item(2) for value in alpha['level']]
plt.plot(season, color = (0.7,0.3,0),label = 'season')
plt.savefig('Balanca_Season.png', bbox_inches='tight')
plt.legend()
#%%
inov   = [value.item(0) for value in alpha['inovations']]
plt.plot(inov, color = (0.7,0.3,0),label = 'inov')
plt.savefig('Balança_errors.png', bbox_inches='tight')
plt.legeng()