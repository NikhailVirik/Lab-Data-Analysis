# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 21:44:17 2023

@author: Nikhail
"""
import numpy as np
from sklearn.metrics import mean_squared_error
def un(t1,t2,N,lamb,dl):
    t=1.07e-3
    dl=dl*1e-9
    lamb=lamb*1e-9
    th=(np.pi/180)*(t1-t2)
    rad=(2/180)*np.pi
    dth=(np.cos(th+rad)-np.cos(th-rad))
    uth=dth/np.cos(th)
    dt=0.005e-3
    ut=dt/t
    ul=dl/lamb
    de1=np.sqrt((dt**2)+(dl**2))
    de2=dth
    e1=(2*t)-(N*lamb)
    e2=1-np.cos(th)
    ue1e2=np.sqrt(((de1/e1)**2)+((de2/np.cos(th))**2))
    dce1e2=ue1e2*(e1*e2)
    
    e3=((N**2)*(lamb**2))/(4*t)
    ue3=np.sqrt((2*(ul**2))+(ut**2))
    de3=e3*ue3
    d_num=np.sqrt((dce1e2**2)+(de3**2))
    num=(e1*e2)+e3
    
    e4=2*t*(1-np.cos(th))
    ue4=np.sqrt((ut**2)+((de2/np.cos(th))**2))
    de4=ue4*e4
    e5=N*lamb
    d_dem=np.sqrt((de4**2)+(dt**2))
    dem=e4-e5
    
    u_final=np.sqrt(((d_num/num)**2)+((d_dem/dem)**2))
    d_final=(num/dem)*u_final
    
    
    return d_final, (num/dem)

print('Green',un(152,112,320,532,1))
print('Yellow', un(150,132,52,580,13))
print('Red', un(150,108,310,633,0.2))
print('Blue',un(150,132,72,459,25))
def mean_squared_error2(x,y):
    val=(np.abs(x-y))/((x+y)/2)
    return val
print(mean_squared_error2(1.4067, 1.4151709655024687))
print(mean_squared_error2(1.4587, 1.387459908516103))
print(mean_squared_error2(1.4570,1.4380490483002761))
print(mean_squared_error2(1.4649,1.4419755807939398))
    
    