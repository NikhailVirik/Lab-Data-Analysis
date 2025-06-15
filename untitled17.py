# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:52:32 2023

@author: Nikhail
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:36:07 2023

@author: Nikhail
"""

import numpy as np
import matplotlib.pyplot as plt

freq = np.loadtxt(r'C:\Users\Nikhail\Downloads\meanfreq.csv', skiprows=1, delimiter=',')

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

plt.show()