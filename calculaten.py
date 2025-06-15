# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 15:01:48 2023

@author: Nikhail
"""
p1s=[]
p2s=[]
p3s=[]
vals=[]
import numpy as np
def find_n(t1,t2,N,lamb):
    lamb=lamb*1e-9
    t=1.07e-3
    for i in range(0,len(t1)):
        
        p1= ((2*t)-(N[i]*lamb))*(1-np.cos((np.pi/180)*(t1[i]-t2[i])))
        p1s.append(p1)
        p2=((int(N[i])**2)*(lamb**2))/(4*t)
        p2s.append(p2)
        p3=(2*t*(1-np.cos((np.pi/180)*(t1[i]-t2[i]))))-(N[i]*lamb)
        p3s.append(p3)
    for i in range(0,len(p1s)):
        val=(p1s[i]+p2s[i])/p3s[i]
        print(val)
        vals.append(val)
    return vals

    
t1green=[138,138,138,140,138,152,150,150]
t2green=[106,108,108,108,107,112,107,110]
Ngreen=[225,244,254,177,177,320,397,355]
lambgreen=532
print('Greens')
find_n(t1green,t2green,Ngreen,lambgreen)

t1yellow=[138,150,150,150]
t2yellow=[120,132,132,129]
Nyellow=[57,52,54,86]
lambyellow=580
print('Yellows')
#find_n(t1yellow,t2yellow,Nyellow,lambyellow)

t1red=[150,150]
t2red=[108,108]
Nred=[314,310]
lambred=633
print('Reds')
#find_n(t1red,t2red,Nred,lambred)

t1blue=[150,150,150]
t2blue=[130,130,132]
Nblue=[82,80,72]
lambblue=459
print('Blues')
#find_n(t1blue,t2blue,Nblue,lambblue)
