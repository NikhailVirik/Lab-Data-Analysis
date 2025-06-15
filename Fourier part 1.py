# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 08:50:21 2023

@author: Nikhail
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
high=np.loadtxt(r'C:\Users\Nikhail\Downloads\Task1.3_Semicircle_high.txt', delimiter='\t', skiprows=1)
low=np.loadtxt(r'C:\Users\Nikhail\Downloads\Task1.3_Semicircle_low.txt', delimiter='\t', skiprows=1)
b=np.linspace(0,1200,1200)
amplitude=50
frequency=5
offset=50
x=(amplitude*signal.square((2 * np.pi * frequency * b)))+offset
plt.plot(b, x)


def fourier(t):
    #print(((2/(3*np.pi))*np.sin(t*(np.pi/40))))
    return 50+(100*(((2/np.pi)*np.sin(t*(np.pi/120)))+((2/(3*np.pi))*np.sin(t*(np.pi/40)))))
    
plt.plot(b,fourier(b))
plt.title('Fig 1.2: Fourier Series Aprroximation to Sqaure Wave')
plt.legend(["Theoretical","Fourier Expansions"],loc='upper right')
plt.show()

y_high=high[:,1]
x_high=high[:,0]
plt.plot(x_high,y_high)



a=0
for i in range(0,len(y_high)-1):
    a=a+((1/2)*(y_high[i]+y_high[i+1])*(x_high[i+1]-x_high[i]))
    x_vals=[x_high[i],x_high[i+1]]
    y_vals=[y_high[i],y_high[i+1]]
    X_val=[x_high[i],x_high[i]]
    Y_val=[y_high[i],0]
    plt.plot(x_vals,y_vals)
    plt.plot(X_val,Y_val)
    
#print(a)
plt.title('Fig 1.3a: Trapezium numerical integration; high resolution')
plt.show()

y_low=low[:,1]
x_low=low[:,0]

c=0
for i in range(0,len(y_low)-1):
    c=c+((1/2)*(y_low[i]+y_low[i+1])*(x_low[i+1]-x_low[i]))
    x_val2=[x_low[i],x_low[i+1]]
    y_val2=[y_low[i],y_low[i+1]]
    X_val2=[x_low[i],x_low[i]]
    Y_val2=[y_low[i],0]
    
    plt.plot(X_val2,Y_val2)
    plt.plot(x_val2,y_val2)
#print(c) 
plt.title('Fig 1.3b: Trapezium numerical integration; low resolution') 
plt.show()



#this is the real oscillation frequency of the system, not a harmonic. All other harmonics are simply multiples of this fundemental frequency
#It will in the limit of large n. for small n, cosine nature of wave is preserved and oscialltion is not flat because there are only two waves interfereing, you cannot cancel the oscillations at many points
    


a4=np.loadtxt(r'C:\Users\Nikhail\Downloads\thermal_4min_a.txt', delimiter='\t', skiprows=3)
b4=np.loadtxt(r'C:\Users\Nikhail\Downloads\thermal_4min_b.txt', delimiter='\t', skiprows=3)

ta4=a4[:,0]*0.1
tempa4=a4[:,1]
#tb4=b4[:,0]*0.1
#tempb4=b4[:,1]

plt.plot(ta4,tempa4)

#plt.plot(tb4,tempb4)

p=(amplitude*signal.square(((2 * np.pi * frequency * b))))+offset

plt.plot(b,p)
plt.title('Fig 2.3: 4 min period vs theretical square wave')
plt.legend(["T=4min Experimental","Theoretical"],loc='upper right')
plt.xlabel("Period/s")
plt.ylabel('Temperature/oC')
plt.show()
#experimental data has much lower temperature gradient than theratical vqalue, and the two are in antiphase
    


tempa4_1=tempa4[0:2400]
tempa4_2=tempa4[2401:4800]
tempa4_3=tempa4[4801:7200]
tempa4_4=tempa4[7201:9600]
# print(tempa4_1)
# print(tempa4_2)
# print(tempa4_3)
# print(tempa4_4)

max1=max(tempa4_1)
max2=max(tempa4_2)
max3=max(tempa4_3)
max4=max(tempa4_4)
maxes=[max1,max2,max3,max4]
err_max=((max(maxes)-min(tempa4))-(min(maxes)-min(tempa4)))/2
# print(np.argmax(tempa4_1))
# print(np.argmax(tempa4_2))
# print(np.argmax(tempa4_3))
# print(np.argmax(tempa4_4))


rat=(max(tempa4)-min(tempa4))/100
# print(rat)
# print((max(tempa4)-min(tempa4))/2)

phase_lag=((300-(0.1*np.argmax(tempa4_1)))/240)*2*np.pi
# print(phase_lag)

w=(2*np.pi)/240
D_4_gam=(w*((2.5E-3)**2))/(2*((np.log(rat))**2))
err_gam=((2*(0.05/2.5))+(2*err_max/(0.5*(max(tempa4)-min(tempa4)))))*D_4_gam
D_4_phase=(w*((2.5E-3)**2))/(2*(phase_lag**2))
err_phase=2*(0.05/2.5)*D_4_phase

#print('Ds',D_4_gam,'+-',err_gam, D_4_phase,'+-',err_phase)


a1=np.loadtxt(r'C:\Users\Nikhail\Downloads\thermal_1min_a.txt', delimiter='\t', skiprows=3)
ta1=a1[:,0]*0.1
tempa1=a1[:,1]
b1=np.loadtxt(r'C:\Users\Nikhail\Downloads\thermal_1min_b.txt', delimiter='\t', skiprows=3)
tb1=b1[:,0]*0.1
tempb1=b1[:,1]
a2=np.loadtxt(r'C:\Users\Nikhail\Downloads\thermal_2min_a.txt', delimiter='\t', skiprows=3)
ta2=a2[:,0]*0.1
tempa2=a2[:,1]
b2=np.loadtxt(r'C:\Users\Nikhail\Downloads\thermal_2min_b.txt', delimiter='\t', skiprows=3)
tb2=b2[:,0]*0.1
tempb2=b2[:,1]
a6=np.loadtxt(r'C:\Users\Nikhail\Downloads\thermal_6min.txt', delimiter='\t', skiprows=3)
ta6=a6[:,0]*0.1
tempa6=a6[:,1]
a8=np.loadtxt(r'C:\Users\Nikhail\Downloads\thermal_8min.txt', delimiter='\t', skiprows=3)
ta8=a8[:,0]*0.1
tempa8=a8[:,1]
a16=np.loadtxt(r'C:\Users\Nikhail\Downloads\thermal_16min.txt', delimiter='\t', skiprows=3)
ta16=a16[:,0]*0.1
tempa16=a16[:,1]

plt.plot(ta1,tempa1)
plt.title('a1')
plt.xlabel("Period/s")
plt.ylabel('Temperature/oC')
plt.show()

plt.plot(tb1,tempb1)
plt.title('b1')
plt.xlabel("Period/s")
plt.ylabel('Temperature/oC')
plt.show()


plt.plot(ta2,tempa2)
plt.title('a2')
plt.xlabel("Period/s")
plt.ylabel('Temperature/oC')
plt.show()

plt.plot(tb2,tempb2)
plt.title('b2')
plt.xlabel("Period/s")
plt.ylabel('Temperature/oC')
plt.show()

plt.plot(ta6,tempa6)
plt.title('a6')
plt.xlabel("Period/s")
plt.ylabel('Temperature/oC')
plt.show()


plt.plot(ta8,tempa8)
plt.title('a8')
plt.xlabel("Period/s")
plt.ylabel('Temperature/oC')
plt.show()


plt.plot(ta16,tempa16)
plt.title('a16')
plt.xlabel("Period/s")
plt.ylabel('Temperature/oC')
plt.show()

# all temperature waves are sine waves as in each case, it takes time for the thermal energy to diffuse across the cylinder. Before the centre can reach 100oC, cooling commences. This process goes on continuously such that temperatures of 100 or 0oC are never reached
 #impacted by transient=b1,b2
 #a1,a2,a6 are sinusoidal. only fundemental frequency contributions present
 #a8,a16 is not purely sinusoidal. shape runs as harmonic contribution is seen
 #a1,a2 have increasing period
 #a6 has slightly decreseing period
 #a8 period is approximately const
 #a16
 #fundamental mode approx works better for short periods, while fourier analysis is preferred for long periods. Therefore, approximation is justified for a1, a2, a6 while fourier analysis is preferred for a8,a16












# anss=list()
# def a_n_coef(period,t,n):
#     for i in range(0,n):
#         coefs=(2/period)*np.trapz((10*np.sin(((np.pi/120)*t)-((np.pi/3)+phase_lag))+50)*np.cos(i*np.pi*t/120),x=None,dx=1.0,axis=-1)
#         anss.append(coefs)
#     return anss
# l=np.linspace(0,960,9600)
# vals=a_n_coef(240,l,3)
# print(vals)

# anss_b=list()
# def b_n_coef(period,t,n):
#     for i in range(0,n):
#         coefs=(2/period)*np.trapz((10*np.sin(((np.pi/120)*t)-((np.pi/3)+phase_lag))+50)*np.sin(i*np.pi*t/120),x=None,dx=1.0,axis=-1)
#         anss_b.append(coefs)
#     return anss_b


# vals_b=b_n_coef(240,l,3)
# print(vals_b)

# totals=list()
# for i in range(0,len(anss)):
    

def intigrasi_a(x,y,n,period):
    randi=list()
    c=0
    for j in range(0,n):
        for i in range(0,period-1):
            y_vals=y[i]*np.cos((j*np.pi*x[i])/(period/2))
            y_next=y[i+1]*np.cos((j*np.pi*x[i+1])/(period/2))
            c=c+((1/2)*(y_vals+y_next)*(x[i+1]-x[i]))
        c=(1/(period/2))*c
        randi.append(c)
    return randi
        
anss=intigrasi_a(ta4,tempa4,6,2400)
# print(anss)

def intigrasi_b(x,y,n,period):
    
    randi_2=list()
    c=0
    for j in range(0,n):
        for i in range(0,period-1):
            y_vals=y[i]*np.sin((j*np.pi*x[i])/(period/2))
            y_next=y[i+1]*np.sin((j*np.pi*x[i+1])/(period/2))
            c=c+((1/2)*(y_vals+y_next)*(x[i+1]-x[i]))
        c=(1/(period/2))*c
        randi_2.append(c)
    return randi_2


anss_b=intigrasi_b(ta4,tempa4,6,2400)
# print(anss_b)
# def real(a_ns,b_ns,period,t):
#     return a_ns[0]/2+a_ns[1]*np.cos((np.pi/120)*t)+b_ns[0]*np.sin((np.pi/120)*t)+a_ns[2]*np.cos(2*(np.pi/120)*t)+b_ns[1]*np.sin(2*(np.pi/120)*t)+a_ns[3]*np.cos(3*(np.pi/120)*t)+b_ns[2]*np.sin(3*(np.pi/120)*t)

# x=np.linspace=(0,960,9600)
# plt.plot(x,real(anss,anss_b,240,x,))

amps2=list()
for i in range(0,len(anss)):
    total=np.sqrt((anss[i]**2)+(anss_b[i]**2))
    amps2.append(total)

amps=list()  
for i in range(0,len(amps2)):
    ampc=amps2[i]/100
    amps.append(ampc)

d_t=list()
for i in range(0,len(amps)):
    ds=(((2*np.pi*(i+1))/240)*((2.5E-3)**2))/(2*(np.log(amps[i])**2))
    d_t.append(ds)
   

d_phi=list()
for i in range(0,len(anss)):
    phis=-np.arctan2(anss[i],anss_b[i])
    d_phi.append(phis)
    
d_p=list()
for i in range(0,len(d_phi)):
    ds=(((2*np.pi*(i+1))/240)*((2.5E-3)**2))/(2*(d_phi[i]**2))
    d_p.append(ds)

# print('Ds',d_t,d_p)
    
#For the first harmonic, Dtf is the same order of magnitude in both, but Dpl for fourier analysis is an order of magnitude higher than for back o'envolope calculation, bringing it closer to the value of Dtf, which is what was initially expected




def quad(a,b):
    ampsf=list()
    for i in range(0,len(a)):
        total=np.sqrt((a[i]**2)+(b[i]**2))
        ampsf.append(total)
    ampsg=list()
    for i in range(0,len(ampsf)):
        ampc=ampsf[i]/100
        ampsg.append(ampc)
    return ampsg

def dtf(period,inp):
    d_new=list()
    for i in range(0,len(inp)):
        ds_new=(((2*np.pi*(i+1))/period)*((2.5E-3)**2))/(2*(np.log(inp[i])**2))
        d_new.append(ds_new)
    return d_new
    
def dphi(period,a,b):
    d_phi_new=list()
    for i in range(0,len(a)):
        phis=-np.arctan2(a[i],b[i])
        d_phi_new.append(phis)
    d_p_new_2=list()
    for i in range(0,len(d_phi_new)):
        ds_phi=(((2*np.pi*(i+1))/period)*((2.5E-3)**2))/(2*(d_phi_new[i]**2))
        d_p_new_2.append(ds_phi)
    return d_p_new_2
        

ansa_a1=intigrasi_a(ta4,tempa4,11,600)
ansb_a1=intigrasi_b(ta4,tempa4,11,600) 
amps_a1=quad(ansa_a1,ansb_a1)
dtf_a1=dtf(60,amps_a1)
dphi_a1=dphi(60,ansa_a1,ansb_a1)
# print('amps_a1', amps_a1)
# print('dtf_a1', dtf_a1)
# print('dphi_a1', dphi_a1)

ansa_a2=intigrasi_a(ta2,tempa2,11,1200)
ansb_a2=intigrasi_b(ta2,tempa2,11,1200)
amps_a2=quad(ansa_a2,ansb_a2)
dtf_a2=dtf(120,amps_a2)
dphi_a2=dphi(120,ansa_a2,ansb_a2)
# print('amps_a2', amps_a2)
# print('dtf_a2', dtf_a2)
# print('dphi_a2', dphi_a2)

ansa_a4=intigrasi_a(ta4,tempa4,11,4*60*10)
ansb_a4=intigrasi_b(ta4,tempa4,11,4*60*10)
amps_a4=quad(ansa_a4,ansb_a4)
dtf_a4=dtf(240,amps_a4)
dphi_a4=dphi(240,ansa_a4,ansb_a4)
# print('amps_a4', amps_a4)
# print('dtf_a4', dtf_a4)
# print('dphi_a4', dphi_a4)

ansa_a6=intigrasi_a(ta6,tempa6,11,10*6*60)
ansb_a6=intigrasi_b(ta6,tempa6,11,10*6*60)
amps_a6=quad(ansa_a6,ansb_a6)
dtf_a6=dtf(60*6,amps_a6)
dphi_a6=dphi(60*6,ansa_a6,ansb_a6)
# print('amps_a6', amps_a6)
# print('dtf_a6', dtf_a6)
# print('dphi_a6', dphi_a6)

ansa_a8=intigrasi_a(ta8,tempa8,11,10*8*60)
ansb_a8=intigrasi_b(ta8,tempa8,11,10*8*60)
amps_a8=quad(ansa_a8,ansb_a8)
dtf_a8=dtf(60*8,amps_a8)
dphi_a8=dphi(60*8,ansa_a8,ansb_a8)
# print('amps_a8', amps_a8)
# print('dtf_a8', dtf_a8)
# print('dphi_a8', dphi_a8)

ansa_a16=intigrasi_a(ta16,tempa16,11,10*16*60)
ansb_a16=intigrasi_b(ta16,tempa16,11,10*16*60)
amps_a16=quad(ansa_a16,ansb_a16)
dtf_a16=dtf(60*16,amps_a16)
dphi_a16=dphi(60*16,ansa_a16,ansb_a16)
# print('amps_a16', amps_a16)
# print('dtf_a16', dtf_a16)
# print('dphi_a16', dphi_a16)

dtfs=[dtf_a1,dtf_a2,dtf_a4,dtf_a6,dtf_a8,dtf_a16]
dphis=[dphi_a1,dphi_a2,dphi_a4,dphi_a6,dtf_a8,dtf_a16]
dphis_f=[]
for i in range(0,len(dphis)):
    for j in range(0,len(dphi_a1)):
        if dphis[i][j]<=0.05:
            dphis_f.append(dphis[i][j])
        else:
            continue
# print('dtfs',dtfs)
# print('dphis',dphis)
xdtf=np.linspace(0,max(dtfs),len(dtfs))
xdphi=np.linspace(0,max(dphis_f),len(dphis_f))
plt.scatter(xdtf,dtfs, color='green')
plt.scatter(xdphi,dphis_f, color='hotpink')
plt.show()
dtfs1=[dtf_a1[0],dtf_a2[0],dtf_a4[0],dtf_a6[0],dtf_a8[0],dtf_a16[0]]
dphis1=[dphi_a1[0],dphi_a2[0],dphi_a4[0],dphi_a6[0],dphi_a8[0],dphi_a16[0]]
xd=np.linspace(0,max(dphis1),len(dtfs1))
plt.scatter(xd,dtfs1)
plt.scatter(xd,dphis1)
plt.show()


dtfs10=[dtf_a1[5],dtf_a2[5],dtf_a4[5],dtf_a6[5],dtf_a8[5],dtf_a16[5]]
dphis10=[dphi_a1[5],dphi_a2[5],dphi_a4[5],dphi_a6[5],dphi_a8[5],dphi_a16[5]]
xd2=np.linspace(0,max(dphis10),len(dtfs1))
plt.scatter(xd2,dtfs10,color='green')
plt.scatter(xd2,dphis10)
plt.show()
print(dphis10)

#Unique as D_TFs and D_Phis represent the same thing, unique value of D can be taken as the central value for all the datapoints


dphis_u=np.concatenate(dphis,axis=0)
dtfs_u=np.concatenate(dtfs,axis=0)
sekaligus=[dtfs_u,dphis_f]
sekaligus=np.concatenate(sekaligus,axis=0)
D_unique=np.mean(sekaligus)
print(D_unique)


#This is order of magnitude 6 larger than the back of envelope method. That method only considered the first harmonic, fourier analysis considers up to 6 harmonics. As D is directly proportional to the angular frequency of the harmonic, it is expected that fourer D is larger than back of envelope
#In this sense, a unique D can only be specified for a specific harmonic, not the entire curve
##################################
##################################
##################################
#Finding Transmission factors and phase lags

p_a1=60*1
tempa1_1=tempa1[10*p_a1:20*p_a1]
wa1=(2*np.pi)/(p_a1)
rat_a1=(max(tempa1)-min(tempa1))/100
pl_a1=(((p_a1*1.5)-(0.1*np.argmax(tempa1_1)))/p_a1)*2*np.pi

p_a2=60*2
tempa2_1=tempa2[0:10*p_a2]
wa2=(2*np.pi)/(p_a2)
rat_a2=(max(tempa2)-min(tempa2))/100
pl_a2=(((p_a2*1.5)-(0.1*np.argmax(tempa2_1)))/p_a2)*2*np.pi


p_a4=60*4
tempa4_1=tempa4[0:10*p_a4]
wa4=(2*np.pi)/(p_a4)
rat_a4=(max(tempa4)-min(tempa4))/100
pl_a4=(((p_a4*1.5)-(0.1*np.argmax(tempa4_1)))/p_a4)*2*np.pi


p_a6=60*6
tempa6_1=tempa6[0:10*p_a6]
wa6=(2*np.pi)/(p_a6)
rat_a6=(max(tempa6)-min(tempa6))/100
pl_a6=(((p_a6*1.5)-(0.1*np.argmax(tempa6_1)))/p_a6)*2*np.pi


p_a8=60*8
tempa8_1=tempa8[0:10*p_a8]
wa8=(2*np.pi)/(p_a8)
rat_a8=(max(tempa8)-min(tempa8))/100
pl_a8=(((p_a8*1.5)-(0.1*np.argmax(tempa8_1)))/p_a8)*2*np.pi


p_a16=60*16
tempa16_1=tempa16[0:10*p_a16]
wa16=(2*np.pi)/(p_a16)
rat_a16=(max(tempa16)-min(tempa16))/100
pl_a16=(((p_a16*1.5)-(0.1*np.argmax(tempa16_1)))/p_a16)*2*np.pi

gamms=[rat_a1,rat_a2,rat_a4,rat_a6,rat_a8,rat_a16]
delphis=[pl_a1,pl_a2,pl_a4,pl_a6,pl_a8,pl_a16]

# print('gamms',gamms)
# print('delphis',delphis)


dtfs_1=[]
for i in range(0,len(dtfs)):
    dtfs_1.append(dtfs[i][0])
# print('dtfs_1',dtfs_1)
dphis_1=[]
for i in range(0,len(dphis)):
    dphis_1.append(dphis[i][0])
# print('dphis_1',dphis_1)


rat_a1_2=amps_a1[0]*rat_a1
rat_a2_2=amps_a2[0]*rat_a2
rat_a4_2=amps_a4[0]*rat_a4
rat_a6_2=amps_a6[0]*rat_a6
rat_a8_2=amps_a8[0]*rat_a8
rat_a16_2=amps_a16[0]*rat_a16

gamms2=[rat_a1_2,rat_a2_2,rat_a4_2,rat_a6_2,rat_a8_2,rat_a16_2]
print(gamms2)
dtfs_2=[]
for i in range(0,len(dtfs)):
    dtfs_2.append(dtfs[i][1])
print('dtfs_2', dtfs_2)