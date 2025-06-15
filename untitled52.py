# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 20:33:32 2023

@author: Nikhail
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
x=[459e-9,520e-9,580e-9,633e-9]
y=[1.441976,1.415170966,1.38746,1.438049]

xer=[25e-9,13e-9,1e-9,0.2e-9]
yer=[0.6339099076513773,0.267482593,0.610197552,0.2568815685335232]

plt.errorbar(x,y,yerr=0,xerr=0, ls='none',marker='o',color="green", label='Data')
plt.xlabel('Wavelength (m)')
plt.ylabel('Refractive Index')
plt.title('Dispersion relation for glass slide')



x=np.array(x)
def fit_func(x,b1,b2,c1,c2):
    p1=(b1*(x**2))/((x**2)-c1)
    p2=(b2*(x**2))/((x**2)-c2)
    #p3=(b3*(x**2))/((x**2)-c3)
    # p4=(b4*(x**2))/((x**2)-c4)
    return np.sqrt(1+p1+p2)

ig=[0.19688763926804523,368954348282.7125,1.5137278693680247e-13,-0.2030796430512917]

params,covaraince= curve_fit(fit_func, x, y,p0=ig)
b1_est,b2_est,c1_est,c2_est=params
#print(f'a estimate: {a_est}')
print(f'b1 estimate: {b1_est}')
print(f'b2 estimate: {b2_est}')
#print(f'b3 estimate: {b3_est}')
# print(f'b4 estimate: {b4_est}')
print(f'c1 estimate: {c1_est}')
print(f'c2 estimate: {c2_est}')
#print(f'c3 estimate: {c3_est}')
# print(f'c4 estimate: {c4_est}')






yer=np.array(yer)
fit,cov = np.polyfit(x,y,3,w=1/yer,cov='unscaled')
fit_values=np.poly1d(fit)
print(fit)

#plt.plot(x,fit_values(x),label='3rd order polynimal', color='blue')


fit2,cov2 = np.polyfit(x,y,2,w=1/yer,cov='unscaled')
fit_values2=np.poly1d(fit2)
print(fit2)

#plt.plot(x,fit_values2(x),label='2nd order polynimal', color='hotpink')


def f(x):
    val=(7.3100447e19*(x**3))+(-1.141488752e14*(x**2))+(5.86974227e7*(x))-8.52015986
    return val
xnew=np.linspace(4.6e-7,6.33e-7,2000)
plt.plot(xnew,f(xnew), color="blue",label='3rd order polynimal')

def f2(x):
    val=(4.84855649e12*(x**2))+(-5.36601141e06*(x))+2.89054121
    return val
plt.plot(xnew,f2(xnew), color="hotpink",label='2nd order polynimal')

plt.plot(xnew,fit_func(xnew, b1_est,b2_est,c1_est,c2_est),label='Sellemeier Curve', color='red')
print('err',mean_squared_error(y, fit_func(x, b1_est,b2_est,c1_est,c2_est)))
plt.legend()
plt.show()
print('err2d',mean_squared_error(y,f2(x)))

print('err3d',mean_squared_error(y,f(x)))