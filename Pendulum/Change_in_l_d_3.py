# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 14:50:03 2023

@author: Nikhail
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
l_data=np.loadtxt(r'C:\Users\Nikhail\Downloads\Change_In_L_2.csv', skiprows=2, delimiter=',', unpack=True, usecols=range(6))
l_10=l_data[0:2,]
l_10_first=l_10[0]
l_10_second=l_10[1]
l_15=l_data[2:4,]
l_15_first=l_15[0]
l_15_second=l_15[1]
l_20=l_data[4:6,]
l_20_first=l_20[0]
l_20_second=l_20[1]
lst=list()
lst_10=list()
for i in range(0,len(l_10_second)):
    if l_10_second[i]==np.max(l_10_second):
        lst_10.append(i)
    else:
        continue
lst.append(l_10_first[lst_10[len(lst_10)//2]])
lst_15=list()
for i in range(0,(len(l_15_second))):
    if l_15_second[i]==np.max(l_15_second):
        lst_15.append(i)
    else:
        continue


lst.append(l_15_first[lst_15[len(lst_15)//2]])
lst_20=list()
for i in range(0,len(l_20_second)):
    if l_20_second[i]==np.max(l_20_second):
        lst_20.append(i)
    else:
        continue
lst.append(l_20_first[lst_20[len(lst_20)//2]])

l_lst=list()
l_lst.append(10E-3)
l_lst.append(15E-3)
l_lst.append(20E-3)
l_lst=np.array(l_lst)
lst=np.array(lst)
print(lst)
plt.plot(l_lst,lst, color="red")
def fit_func(l_lst,a,b,c):
    lg=(a*np.exp(-1*b*(l_lst)))+c
    return lg
ig=[2600,1,1600]
po,po_cov=curve_fit(fit_func,l_lst,lst,ig)
plt.plot(l_lst,fit_func(l_lst,po[0],po[1],po[2]), color="green")

amp=po[0]
in_off=po[1]
out_off=po[2]
new_lest=list()
new_lest.append(amp)
new_lest.append(in_off)
new_lest.append(out_off)
print(new_lest)


