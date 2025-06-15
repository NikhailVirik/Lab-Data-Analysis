# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 18:12:27 2023

@author: Nikhail
"""


import numpy as np
import matplotlib.pyplot as plt
port=np.loadtxt(r'C:\Users\Nikhail\Downloads\faradays_data.csv', skiprows=6, delimiter=',', unpack=True)
row1=port[:,7]
freq=row1[0]
rIp=row1[3]
rIs=row1[5]
re=row1[12]

A=row1[0]
w=freq*2*np.pi
t=np.linspace(0,(4E-4)*np.pi,40)
e=re/100
n1=100
n2=50
Is=e*(1/18)*(n1/n2)*A*np.cos(w*t)   
Ip=(1/18)*A*np.cos(w*t)

plt.xlabel("time(s)")
plt.ylabel("Current(A)")
plt.plot(t,Is,color='red')
plt.plot(t,Ip)
plt.legend(["Secondary Coil","Primary Coil"], loc="lower right")
plt.show()
frIp=rIp*np.cos(2*np.pi*freq*t)
frIs=rIs*np.cos(2*np.pi*freq*t)

plt.xlabel("time(s)")
plt.ylabel("Current(A)")
plt.plot(t,frIp)
plt.plot(t,frIs,color='red')
plt.legend(["Primary Coil","Secondary Coil"], loc="lower right")
plt.show()