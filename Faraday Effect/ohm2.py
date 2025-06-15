# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:57:32 2023

@author: Nikhail
"""

import numpy as np
import matplotlib.pyplot as plt
raw_data_1=np.loadtxt(r'C:\Users\Nikhail\OneDrive\Documents\2ohms.csv', skiprows=1, delimiter=',', unpack=True,)
from scipy.optimize import curve_fit
from scipy.optimize import brentq
vall=raw_data_1[0:8,]
V_1=vall[3]
frq_1=vall[0]
phase_1=vall[4]

print(len(frq_1),len(phase_1))
V_1=np.array(V_1)
frq_1=np.array(frq_1)
phase_1=np.array(phase_1)
phase_1=[i*(-1) for i in phase_1]
w0_1=1/(2*np.pi*np.sqrt((1E-3)*(1E-7)))

plt.plot(frq_1,V_1)


plt.show()
plt.plot(frq_1,phase_1)
plt.show()
plt.plot(frq_1,V_1)
def fit_func(frq_1,a,mu,sig,m,c):
    gauss=a*np.exp(-((frq_1)-mu)**2/(2*sig**2)) 
    line=m*(frq_1)+c 
    return gauss + line
grad_guess=((V_1[-1]-V_1[0])/(frq_1[-1]-frq_1[0]))
rang=np.max(V_1)-grad_guess
ig=[rang,np.mean(frq_1),np.std(frq_1, ddof=1),grad_guess,V_1[0]]
po,po_cov=curve_fit(fit_func,frq_1,V_1,ig)

lst3=list()
lst3.append(fit_func(frq_1,po[0],po[1],po[2],po[3],po[4]))
plt.xlabel("frequency(kHz)")
plt.ylabel("Vout/Vin")
plt.plot(frq_1,fit_func(frq_1,po[0],po[1],po[2],po[3],po[4])) 
VQ=np.max(V_1)/np.sqrt(2)


#x_interp=np.interp(VQ,V_1,frq_1)
#plt.plot(x_interp, VQ, 'o')

plt.show()
w0=po[1]
print(po[1])
lst1=list()
lst2=list()
lst3=list()

for i in range(0,len(frq_1)):
    if frq_1[i]<w0:
        lst1.append(frq_1[i])
    else:
        lst2.append(frq_1[i])
print(len(frq_1),len(lst1),len(lst2))
#def pha(lst1,lst2,m1,m2,c):
 #   y1=m1(lst1)+c
  #  y2=m2(lst2)
   # return y1 + y2
#m1_guess=(phase_1[len(lst1)]-phase_1[0])/(lst1[-1]-lst1[0])
#m2_guess=(phase_1[13]-phase_1[7])/(lst1[-1]-lst1[0])
#ig=[m1_guess,m2_guess,0]
#l,l_cov=curve_fit(pha, frq_1, phase_1)
plt.plot(frq_1,phase_1)
def pha(frq_1,u):
    psi=np.arctan((-1*(frq_1)*(2E3))/(((u)**2)-((frq_1)**2)))
    return psi
ig=[16.3]
l,l_cov=curve_fit(pha,frq_1,phase_1,ig)
plt.xlabel("frequency(kHz)")
plt.ylabel("phase difference(rad)")
plt.plot(frq_1,pha(frq_1,l[0]))
plt.show()
print(l[0])

plt.plot(frq_1,V_1)
#def res(frq_1,N,v):
 #   A=N/np.sqrt(((((frq_1)**2)-(v**2))**2)+(((frq_1)*2E3)**2))
  #  return A
#ig2=[50,16.3]
#k,k_cov=curve_fit(res,frq_1,V_1,ig2)
#plt.plot(frq_1,res(frq_1,k[0],k[1]))
VQ=np.max(V_1)/np.sqrt(2)
print(VQ)
x_interp=np.interp(VQ,V_1,frq_1)
plt.plot(x_interp, VQ, 'o')
plt.show()
#print(k[0],k[1])
print(x_interp)
for i in range(0,len(lst3)):
    if lst3[i]==VQ:
        print(frq_1[i])
    else:
        continue
print(np.sqrt(po_cov[0,0]))
print(np.sqrt(l_cov[0,0]))
