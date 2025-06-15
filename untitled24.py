# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:43:41 2023

@author: Nikhail
"""

import numpy as np
import matplotlib.pyplot as plt
data=np.loadtxt(r'C:\Users\Nikhail\Downloads\ii.txt')
x=np.array(data[:,5])
y=np.array(data[:,0])
print(x[np.argmax(y)])
posy=[]
for i in range(0,len(y)):
    if y[i]>=150000:
        posy.append(y[i])
    else:
        continue
print(np.where(y==min(posy)))
print(x[5686])
plt.plot(x,y)
plt.show()

################################
################################

xnew=[]
indexes=[]
for i in range(0,len(x)):
    if -2e6<=x[i]<=6e6:
        xnew.append(x[i])
        indexes.append(i)
    else: continue
ynew=y[indexes]
posynew=[]
for i in range(0,len(ynew)):
    if ynew[i]>=140000:
        posynew.append(ynew[i])
    else: 
        continue
print('lengths', len(xnew),len(ynew))
xnew=np.sort(xnew)
posy1=posynew[0:round((len(posynew))/2)]
posy2=posynew[round((len(posynew))/2):len(posynew)]

print(np.where(ynew==min(posy1)))
print(np.where(ynew==min(posy2)))
print(posy2[0])
print(np.where(ynew==217663))
print(xnew[588], xnew[5614])

plt.plot(xnew,ynew)
plt.show()

