# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 21:07:25 2022

@author: Nikhail
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
data=np.loadtxt(r'C:\Users\Nikhail\Downloads\book2.csv', skiprows=1, delimiter=',', unpack=True)
x=data[0]
y=data[1]
def fit_func(x,a,b):
    quad=1.239194*((1+(a*x)+(b*x*x)))
    return quad
ig=[0,0]
po,po_cov=curve_fit(fit_func,x,y,ig)

plt.plot(x,fit_func(x,po[0],po[1]))
plt.ylabel('Period(s)')
plt.xlabel('Theta 0(degree)')
plt.errorbar(x,y,xerr=0.5, yerr=5E-4, ls="none")
plt.show()
c=po[0]
d=po[1]
print("a=",c,"+/-",np.sqrt(po_cov[0,0]))
print("b=",d,"+/-",np.sqrt(po_cov[1,1]) )


