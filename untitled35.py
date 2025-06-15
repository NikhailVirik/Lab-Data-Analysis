# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 00:06:07 2023

@author: Nikhail
"""

import numpy as np
true_ns=[]
diffs=[]
l_diffs=[]

def steph(n,N,t1,t2,lamb):
    t=1.27e-3
    lamb=lamb*1e-9
    #for i in range(0,len(t1)):
    for j in range(0,len(n)):
            t1=(t1/180)*np.pi
            t2=(t2/180)*np.pi
            f1_1=n[j]-np.cos((t1-np.arccos(np.sqrt(1-(((np.sin(t1))**2)/(n[j]**2))))))
            f1_2=2*t/(np.sqrt(1-(((np.sin(t1))**2)/(n[j]**2))))
            f2_1=n[j]-np.cos((t2-np.arccos(np.sqrt(1-(((np.sin(t2))**2)/(n[j]**2))))))
            f2_2=2*t/(np.sqrt(1-(((np.sin(t2))**2)/(n[j]**2))))
            f1=f1_1*f1_2
            f2=f2_1*f2_2
            d=np.abs(f1-f2)
            diff=np.abs(d-(N*lamb))
            diffs.append(diff)
            
            
           # if -1e-4<=(d-(N[i]*lamb))<=1e-4:
            #    true_ns.append(n[j])
           # else:
              #  continue
    l_diffs.append(min(diffs))
    print(len(diffs))
    print(np.argmin(diffs))
    print(n[np.argmin(diffs)])
    true_ns.append(n[np.argmin(diffs)])
        
    return   l_diffs, true_ns
            
ns=np.linspace(1.3, 1.8,3000)
t1green=29
t2green=9
Ngreen=117
lambgreen=633
Greens=steph(ns,Ngreen,t1green,t2green,lambgreen)
print(Greens)


                
            
            
        