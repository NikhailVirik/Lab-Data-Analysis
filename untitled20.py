# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 17:43:42 2023

@author: Nikhail
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(r'C:\Users\Nikhail\Downloads\Book7.csv', skiprows=1, delimiter=',')
freq=data[:,1]
L = 40
lambda_n=[]
for i in range(1,17):
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
k=np.delete(k,[15,17,19,21,23,25,27,29,31])


w = 2*np.pi*freq

print(len(k),len(w))
plt.scatter(k, w)
plt.grid()
plt.xlabel('k/section^-1')
plt.ylabel('w x 10^3/rad s^-1')
plt.title('dispersion relation of transmission line')




 

