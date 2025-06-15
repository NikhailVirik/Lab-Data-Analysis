# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 03:03:49 2023

@author: Nikhail
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 00:06:07 2023

@author: Nikhail
"""

import numpy as np
diffs=[]
true_ns=[]
def find_n(t1,t2,N,lamb,n):
    d=1.07e-3
    lamb=lamb*1e-9
    for i in range(0,len(t1)):
        ti=t1[i]-t2[i]
        ti=(np.pi/180)*ti
        for j in range(0,len(n)):
            tr=np.arcsin(np.sin(ti)/n[j])
            p1=np.cos(ti-tr)/np.cos(ti)
            p2=n[j]*((1/np.cos(ti))-1)
            p3=(2*d/lamb)*(p2+1-p1)
            diff=np.abs(N[i]-p3)
            diffs.append(diff)
        print(diffs)
        true_ns.append(n[np.argmin(diffs)])
    return true_ns

t1green=[138]
t2green=[106]
Ngreen=[225]
lambgreen=532
ns=np.linspace(1.3, 1.6,3000)

greens=find_n(t1green,t2green,Ngreen,lambgreen,ns)
print('Greens',greens)
            