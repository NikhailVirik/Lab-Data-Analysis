# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:15:03 2023

@author: Nikhail
"""

import numpy as np
import matplotlib.pyplot as plt
data=np.loadtxt(r'C:\Users\Nikhail\Downloads\sine_wave2.csv', skiprows=1, delimiter=',')
freq=data[:,0]
ratio=data[:,1]
uncert=data[:,2]

plt.scatter(freq,ratio, marker='x')
plt.errorbar(freq, ratio, yerr=None, xerr=uncert, ls='none', ecolor='orange')
weights = np.where(uncert != 0, 1 / uncert, 0)

fit,cov=np.polyfit(freq,ratio,1,w=weights, cov='unscaled')
pfreq=np.poly1d(fit)
plt.plot(freq,pfreq(freq), color='blue')
plt.title('amplitude ratio vs frequency')
plt.xlabel('amplitude ratio:Vout/Vin')
plt.ylabel('frequency/kHz')
plt.show()
sig_0 = np.sqrt(cov[0,0]) 
sig_1 = np.sqrt(cov[1,1]) 

print('Slope = %.3e +/- %.3e' %(fit[0],sig_0))
print('Intercept = %.3e +/- %.3e' %(fit[1],sig_1))
inte=-fit[1]/fit[0]
delta=inte*((sig_0/fit[0])+(sig_1/fit[1]))
print('Intercept is',inte,'+-',delta)
