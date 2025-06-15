# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 13:46:19 2023

@author: Nikhail
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
data=np.loadtxt(r'C:\Users\Nikhail\Downloads\Book5.csv', delimiter=',', skiprows=1)
x=data[:,0]*1e-6
y=data[:,1]
plt.scatter(x,y)
plt.xlabel('Wavelength of Light (m)')
plt.ylabel('Refractive Index')
plt.title('Book Value Dispersion')

def fit_func(x,b1,b2,b3,c1,c2,c3):
    p1=(b1*(x**2))/((x**2)-c1)
    p2=(b2*(x**2))/((x**2)-c2)
    p3=(b3*(x**2))/((x**2)-c3)
    # p4=(b4*(x**2))/((x**2)-c4)
    return np.sqrt(1+p1+p2+p3)

ig=[0.696166300,0.407942600,0.897479400,4.67914826e-15,1.35120631e-3,097.9340025]

params,covaraince= curve_fit(fit_func, x, y,p0=ig)
b1_est,b2_est,b3_est,c1_est,c2_est,c3_est=params
#print(f'a estimate: {a_est}')
print(f'b1 estimate: {b1_est}')
print(f'b2 estimate: {b2_est}')
print(f'b3 estimate: {b3_est}')
# print(f'b4 estimate: {b4_est}')
print(f'c1 estimate: {c1_est}')
print(f'c2 estimate: {c2_est}')
print(f'c3 estimate: {c3_est}')
# print(f'c4 estimate: {c4_est}')

plt.scatter(x,y,label='Data')
plt.plot(x,fit_func(x, b1_est,b2_est,b3_est,c1_est,c2_est,c3_est),label='Fitted Function', color='green')
plt.legend()
plt.show()

print(mean_squared_error(y, fit_func(x, b1_est,b2_est,b3_est,c1_est,c2_est,c3_est)))
# po,po_cov=curve_fit(fit_func,x,y,ig)
# plt.plot(x,fit_func(x,po[0],po[1],po[2],po[3],po[4],po[5]))
# plt.xlabel('Wavelength of Light (m)')
# plt.ylabel('Refractive Index')
# plt.show()
