# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 15:02:16 2023

@author: Nikhail
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 22:53:31 2023

@author: Nikhail
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
d_data=np.loadtxt(r'C:\Users\Nikhail\Downloads\DeltaD1.csv', skiprows=2, delimiter=',', unpack=True, usecols=range(8))
lest_2_5=list()
freq_lst=list()
spl_lst=list()
d_2_5=d_data[0:2,]

d_2_5_first=d_2_5[0]
d_2_5_second=d_2_5[1]
d_3=d_data[2:4,]
d_3_first=d_3[0]
d_3_second=d_3[1]
d_3_5=d_data[4:6,]
d_3_5_first=d_3_5[0]
d_3_5_second=d_3_5[1]
d_4=d_data[6:8,]
d_4_first=d_4[0]
d_4_second=d_4[1]
for i in range(0,len(d_2_5_second)):
    if d_2_5_second[i]==np.max(d_2_5_second):
        lest_2_5.append(i)
    else:
        continue

id_2_5=lest_2_5[(len(lest_2_5)//2)]
freq_lst.append(d_2_5_first[id_2_5])
spl_lst.append(d_2_5_second[id_2_5])
lest_3=list()
for i in range(0,len(d_3_second)):
    if d_3_second[i]==np.max(d_3_second):
        lest_3.append(i)
    else:
        continue
id_3=lest_3[(len(lest_3)//2)]
freq_lst.append(d_3_first[id_3])
spl_lst.append(d_3_second[id_3])

lest_3_5=list()
for i in range(0,len(d_3_5_second)):
    if d_3_5_second[i]==np.max(d_3_5_second):
        lest_3_5.append(i)
    else:
        continue
id_3_5=lest_3_5[(len(lest_3_5)//2)]
freq_lst.append(d_3_5_first[id_3_5])
spl_lst.append(d_3_5_second[id_3_5])

lest_4=list()
for i in range(0,len(d_4_second)):
    if d_4_second[i]==np.max(d_4_second):
        lest_4.append(i)
    else:
        continue
id_4=lest_4[(len(lest_4)//2)]
freq_lst.append(d_4_first[id_4])
spl_lst.append(d_4_second[id_4])

print(freq_lst,spl_lst)

#plt.plot(freq_lst,spl_lst)



lest_lst=list()
lest_lst.append(2.5E-3)
lest_lst.append(3E-3)
lest_lst.append(3.5E-3)
lest_lst.append(4E-3)
lest_lst=np.array(lest_lst)
spl_lst=np.array(spl_lst)
plt.plot(lest_lst,spl_lst)
def fit_func(lest_lst,a,b,c):
    lg=(-1*a*np.exp(-1*b*(lest_lst)))+c
    return lg
ig=[20,5,120]
po,po_cov=curve_fit(fit_func,lest_lst,spl_lst,ig)
plt.plot(lest_lst,fit_func(lest_lst,po[0],po[1],po[2]))
amp=po[0]
offset=po[1]
out_off=po[2]
new_lst=list()
new_lst.append(amp)
new_lst.append(offset)
new_lst.append(out_off)
print('new_lst params:', new_lst)