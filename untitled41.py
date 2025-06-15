# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 03:20:22 2023

@author: Nikhail
"""

import numpy as np
ns=np.linspace(1.3,1.6,3000)
lamb=532e-9
d=1.07e-3
ti=(152-112)*((np.pi)/180)
diffs=[]
for i in range(0,len(ns)):
    
    tr=np.arcsin(np.sin(ti)/ns[i])
    p1=np.cos(ti-tr)/np.cos(ti)
    p2=ns[i]*((1/np.cos(ti))-1)
    p3=(2*d/lamb)*(p2+1-p1)
    diff=np.abs(320-p3)
    diffs.append(diff)
    
print(diffs) 
    
    