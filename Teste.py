#%%
%matplotlib inline
%load_ext autoreload
#%%
# -*- coding: utf-8 -*-
%autoreload 2 
import numpy as np
import matplotlib.pyplot as plt
import MEB12 as mm
import Meb as m
import pandas as pd
#%%
mebm = mm.MultMEB()
meb = m.Meb()
ap = pd.read_csv('../R_analisys/Airpassengers.csv')
ts = np.array(ap['x'].tolist())/100












#%%
plt.plot(ts*100)
plt.title('Airline Series')
plt.savefig('../ExtendedKalman/images/airlines.png', bbox_inches='tight')
#%%
plt.plot(np.log(ts))
plt.title('Log(Airline Series)')
plt.savefig('../ExtendedKalman/images/log-airlines.png', bbox_inches='tight')
#%%
mebm.fit(ts)
#%%
params = mebm.get_params()
print(params)
p = [np.exp(params[0]),np.exp(params[1]),np.exp(params[2]),np.exp(params[3]),params[4],params[5]]
print(p)
#%%
mebm.set_params([np.log(6.5e-6),np.log(0.0043),np.log(2e-6),np.log(0.265),-4.11,0.408])
alpha = mebm.filter(ts)
level = [value.item(0) for value in alpha['level']]
phi   = [value.item(2) for value in alpha['level']]
trend = [value.item(1) for value in alpha['level']]
params = mebm.get_params()
c0 = params[4]
c_mu = params[5]
s_phi   = [value.item(2)*np.exp(c0 +c_mu*value.item(0)) for value in alpha['level']]
#%%

plt.plot(ts, color = (0,0,1),label = 'series')
plt.plot(level, color = (1,0,0),label = 'level')
#plt.title('')
plt.legend()
plt.savefig('../ExtendedKalman/images/level.png', bbox_inches='tight')
#%%
plt.plot(phi, color = (0.7,0.3,0),label = 'phi')
plt.title('Seasonal Component')
plt.savefig('../ExtendedKalman/images/seasonal.png', bbox_inches='tight')
#%%
plt.plot(s_phi, color = (0.4,0.5,0),label = 's_phi')
plt.title('Seasonal additivity')
plt.savefig('../ExtendedKalman/images/seasonaladd.png', bbox_inches='tight')
plt.legend()
#%%
from statsmodels.tsa.stattools import acf 
from scipy.stats import histogram
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox

#%%
inov = [value.item(0) for value in alpha['inovations']]
s2 = sum(np.array(inov)**2)/len(inov)
inov = inov/np.sqrt(s2)
#%%
plt.plot(inov)
plt.savefig('../ExtendedKalman/images/residuals.png', bbox_inches='tight')
#%%
plt.hist(inov)
plt.title('Residuals histogram')
plt.savefig('../ExtendedKalman/images/residualshist.png', bbox_inches='tight')
jarque_bera(inov)
#%%
plot_acf(acf(inov))
plt.title('ACF')
plt.savefig('../ExtendedKalman/images/residualsACF.png', bbox_inches='tight')
t, p_value =acorr_ljungbox(inov,15)
print(p_value)