# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:36:07 2023

@author: Nikhail
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(r'C:\Users\Nikhail\Downloads\Definitive.csv', skiprows=1, delimiter=',')
uncert=np.loadtxt(r'C:\Users\Nikhail\Downloads\new_uncert.csv', skiprows=1, delimiter=',')
freq=data[:,1]
L = 40
lambda_n=[]
for i in range(1,13):
    sam = 2*L/(2*i +1)
    bruce = L/i
    lambda_n.append(sam)
    lambda_n.append(bruce)

lambda_n=np.array(lambda_n)
print(lambda_n)
#lambda_n=np.sort(lambda_n)

k = 2*np.pi/lambda_n
print(k)
k=np.sort(k)
k=np.delete(k,[16])


w = 2*np.pi*freq

plt.scatter(k, w, marker="x")
plt.errorbar(k,w,yerr=uncert,xerr=None, ls='none', ecolor='orange')
plt.grid()
plt.xlabel('k/section^-1')
plt.ylabel('w x 10^3/rad s^-1')
plt.title('dispersion relation of transmission line')
best_k=k[0:12]
best_w=w[0:12]
end_k =k[-8:-1]
end_w=w[-8:-1]
uncert1=uncert[0:12]
uncert2=uncert[-8:-1]

zero_uncertainty_indices = uncert2 == 0
uncert2[zero_uncertainty_indices] = 0.0125

fit,cov=np.polyfit(best_k,best_w,1,w=1/uncert1,cov='unscaled')
pbest_k=np.poly1d(fit)
sig_0 = np.sqrt(cov[0,0]) 
sig_1 = np.sqrt(cov[1,1]) 
plt.plot(k,pbest_k(k), color='blue')
print('Slope = %.3e +/- %.3e' %(fit[0],sig_0))
significant_figures = 6  # Adjust as needed

formatted_coefficients = [f"{coeff:.{significant_figures}g}" for coeff in fit]
print(formatted_coefficients)
fit2,cov2=np.polyfit(end_k,end_w,1,w=1/uncert2, cov='unscaled')
p2=np.poly1d(fit2)
sig_2 = np.sqrt(cov2[0,0]) 
sig_3 = np.sqrt(cov2[1,1]) 
plt.plot(k,p2(k), color="green")
print('Slope = %.3e +/- %.3e' %(fit2[0],sig_2))
formatted_coefficients2 = [f"{coeff:.{significant_figures}g}" for coeff in fit2]
print(formatted_coefficients2)
y_err=np.zeros(len(w))
y_err[16]=2*np.pi*(3.205)
plt.errorbar(k, w, xerr=None, yerr=y_err,ls="none",color="orange")
plt.show()
plt.show()
