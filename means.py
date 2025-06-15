# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 16:43:09 2023

@author: Nikhail
"""
import numpy as np
import matplotlib.pyplot as plt
def t_uncert(N,lamb,t1,t2):
    t=1.07e-3
    lamb=lamb*1e-9
    th=(t1-t2)*(np.pi/180)
    p1=-8*N*(t**2)*lamb*np.cos(th)
    p2=8*N*(t**2)*lamb*((np.cos(th)**2))
    p3=-4*(N**2)*t*(lamb**2)
    p4=4*(N**2)*t*(lamb**2)*np.cos(th)
    p5=(N**3)*(lamb**3)
    p6=4*(t**2)*(((2*t)-(2*t*np.cos(th))-(N*lamb))**2)
    return (p1+p2+p3+p4+p5)/p6

def lamb_uncert(N,lamb,t1,t2):
    t=1.07e-3
    lamb=lamb*1e-9
    th=(t1-t2)*(np.pi/180)
    p1=8*N*(t**2)*np.cos(th)
    p2=-8*N*(t**2)*((np.cos(th)**2))
    p3=4*(N**2)*t*(lamb)
    p4=-4*(N**2)*t*(lamb)*np.cos(th)
    p5=-(N**3)*(lamb**2)
    p6=4*(t**2)*(((2*t)-(2*t*np.cos(th))-(N*lamb))**2)
    return (p1+p2+p3+p4+p5)/p6
def th_uncert(N,lamb,t1,t2):
    t=1.07e-3
    lamb=lamb*1e-9
    th=(t1-t2)*(np.pi/180)
    p1=-4*N*t*lamb*np.sin(th)
    p2=(N**2)*(lamb**2)*np.sin(th)
    p3=2*(((2*t)-(2*t*np.cos(th))-(N*lamb))**2)
    
    return (p1+p2)/p3
dth=0.0339
dt=0.005e-3
#########################
#GREEN
dgreen=30
un1=np.sqrt(((t_uncert(320,520,152,112)**2)*(dt**2))+((lamb_uncert(320,520,152,112)**2)*(dgreen**2))+((th_uncert(320,520,152,112)**2)*(dth**2)))
un2=np.sqrt(((t_uncert(397,520,150,107)**2)*(dt**2))+((lamb_uncert(397,520,150,107)**2)*(dgreen**2))+((th_uncert(397,520,150,107)**2)*(dth**2)))
un3=np.sqrt(((t_uncert(355,520,150,110)**2)*(dt**2))+((lamb_uncert(355,520,150,110)**2)*(dgreen**2))+((th_uncert(355,520,150,110)**2)*(dth**2)))
s_g=np.sqrt((un1**2)+(un2**2)+(un3**2))
err_g=s_g/np.sqrt(3)

dyellow=64
uy1=np.sqrt(((t_uncert(52,580,150,132)**2)*(dt**2))+((lamb_uncert(52,580,150,132)**2)*(dyellow**2))+((th_uncert(52,580,150,132)**2)*(dth**2)))
uy2=np.sqrt(((t_uncert(54,580,150,132)**2)*(dt**2))+((lamb_uncert(54,580,150,132)**2)*(dyellow**2))+((th_uncert(54,580,150,132)**2)*(dth**2)))
uy3=np.sqrt(((t_uncert(86,580,150,129)**2)*(dt**2))+((lamb_uncert(86,580,150,129)**2)*(dyellow**2))+((th_uncert(86,580,150,129)**2)*(dth**2)))
s_y=np.sqrt((uy1**2)+(uy2**2)+(uy3**2))
err_y=s_y/np.sqrt(3)

dred=0.2
ur1=np.sqrt(((t_uncert(314,633,150,108)**2)*(dt**2))+((lamb_uncert(314,633,150,108)**2)*(dred**2))+((th_uncert(314,633,150,108)**2)*(dth**2)))
ur2=np.sqrt(((t_uncert(310,633,150,108)**2)*(dt**2))+((lamb_uncert(310,633,150,108)**2)*(dred**2))+((th_uncert(310,633,150,108)**2)*(dth**2)))
s_r=np.sqrt((ur1**2)+(ur2**2))
err_r=s_r/np.sqrt(2)

dblue=21
ub1=np.sqrt(((t_uncert(82,459,150,130)**2)*(dt**2))+((lamb_uncert(82,459,150,130)**2)*(dblue**2))+((th_uncert(82,459,150,130)**2)*(dth**2)))
ub2=np.sqrt(((t_uncert(80,459,150,130)**2)*(dt**2))+((lamb_uncert(80,459,150,130)**2)*(dblue**2))+((th_uncert(80,459,150,130)**2)*(dth**2)))
ub3=np.sqrt(((t_uncert(72,459,150,132)**2)*(dt**2))+((lamb_uncert(72,459,150,132)**2)*(dblue**2))+((th_uncert(72,459,150,132)**2)*(dth**2)))
s_b=np.sqrt((ub1**2)+(ub2**2)+(ub3**2))
err_b=s_b/np.sqrt(3)



print(uy3)
x=[459,520,580,633]
y=[1.402914136,1.436394788,1.434783524,1.442636177]
yer=[err_b,err_g,err_y,err_r]
print(yer)
xer=[21e-9,30e-9,64e-9,0.2e-9]
plt.errorbar(x,y,yerr=yer,xerr=xer)
plt.show()
