# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 18:56:25 2023

@author: Nikhail
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

freq = np.loadtxt(r'C:\Users\Nikhail\Downloads\meanfreq.csv', skiprows=1, delimiter=',')
uncert = np.loadtxt(r'C:\Users\Nikhail\Downloads\ori_uncert.csv', skiprows=1, delimiter=',')
L = 40
lambda_n=[]
for i in range(1,15):
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


w = 2*np.pi*freq

plt.scatter(k, w)
plt.grid()
plt.xlabel('k/section^-1')
plt.ylabel('w x 10^3/rad s^-1')
plt.title('dispersion relation of transmission line')



end_k=k[-6:-1]
end_w=w[-6:-1]
uncert=uncert[-6:-1]
zero_uncertainty_indices = uncert == 0
uncert[zero_uncertainty_indices] = 0.0125
fit_k,cov_k=np.polyfit(end_k,end_w,1,w=1/uncert,cov='unscaled')
pend=np.poly1d(fit_k)
plt.plot(k,pend(k), color="green")
plt.show()

sig_0 = np.sqrt(cov_k[0,0]) 
sig_1 = np.sqrt(cov_k[1,1]) 
print('Slope = %.3e +/- %.3e' %(fit_k[0],sig_0))
significant_figures = 6  # Adjust as needed

formatted_coefficients = [f"{coeff:.{significant_figures}g}" for coeff in fit_k]
print(formatted_coefficients)



freq_new_2 = freq[1:]
vg = []
for i in range(0, 26):
    delta_freq = freq_new_2[i+1] - freq_new_2[i]
    vg.append(delta_freq*2*L)

freq_new = freq_new_2[0:len(freq_new_2)-1]

vp = []
for i in range(0,27):
    v_phase = (2*L*freq_new_2[i])/(i+1)
    vp.append(v_phase)
#vp2=(2*np.pi*freq)/k   

plt.scatter(freq_new, vg)
plt.scatter(freq_new_2, vp)
#plt.scatter(freq,vp2)
plt.grid()
plt.xlabel('frequency/kHz')
plt.ylabel('Velocity/sections s^-1')
plt.title('group and phase velocties vs frequency')
plt.legend(['group velocity', 'phase velocity'])
plt.show()

ratio = np.loadtxt(r'C:\Users\Nikhail\Downloads\Book10.csv', skiprows = 1, delimiter = ',')
plt.scatter(freq, ratio)
plt.grid()
plt.xlabel('frequency/kHz')
plt.ylabel('amplitude ratio:Vout/Vin')
plt.title('amplitude ratios vs frequency')

fit, cov = np.polyfit(freq, ratio, deg = 1)
pratio = np.poly1d(np.polyfit(freq, ratio, 1))
plt.plot(freq, pratio(freq))
print('ratio =', fit, 'x frequency +', cov)
print('x-intercept =', cov/(-1*fit), 'kHz' )
