# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 20:42:05 2023

@author: Nikhail
"""
import numpy as np
import matplotlib.pyplot as plt
data=np.loadtxt(r'C:\Users\Nikhail\Downloads\ii.txt')
Fs=50
tstep= 1/Fs
y=np.array(data[:,0])
f0=(3E8)/(217E-9)
N=len(y)
t=np.linspace(0,(N-1)*tstep, N)
fstep=Fs/N
f=np.linspace(0,(N-1)*fstep,N)
X=np.fft.fft(y)
X_mag=np.abs(X)/N
f_plot=f[0:int(N/2+1)]
X_mag_plot=2*X_mag[0:int(N/2+1)]
X_mag_plot[0]=X_mag_plot[0]/2
plt.plot(t,y)
plt.show()
plt.plot(f_plot,X_mag_plot,'.-')