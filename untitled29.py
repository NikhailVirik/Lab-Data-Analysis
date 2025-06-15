# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 13:15:37 2023

@author: Nikhail
"""
import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt(r'C:\Users\Nikhail\Downloads\W_LED_6.txt')
x=np.array(data[:,5])
y=np.array(data[:,0])
print(x[np.argmax(y)])
plt.plot(x,y)
plt.ylabel('Signal 1')
plt.xlabel('Position musteps')
plt.title('White LED Interferogram (zoomed in)')
plt.xlim(0.7e7,0.81e7)
posxlow=[]
posxhigh=[]
for i in range(0,len(x)):
    if x[i]>=7.3e7:
        posxlow.append(x[i])
    elif x[i]>=7.7e7:
        posxhigh.append(x[i])
    else:
        continue

posx=np.array[posxlow,posxhigh]
posx=np.concatenate(posxlow,posxhigh)

        