# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:48:11 2024

@author: nsv22
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:45:25 2024

@author: nsv22
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
dist=[]
vals=[]
vals_adj=[]
two=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\RadLab\2cm_run_3')
two=np.mean(two)
vals.append(two)
dist.append(0.02)
two_adj=two*(0.02**2)
vals_adj.append(two_adj)
print('2cm', two, two_adj)
four=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\RadLab\4cm_run_2')
four=np.mean(four)
vals.append(four)
dist.append(0.04)
four_adj=four*(0.04**2)
vals_adj.append(four_adj)

six=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\RadLab\6cm_run_2')
six=np.mean(six)
vals.append(six)
dist.append(0.06)
six_adj=six*(0.06**2)
vals_adj.append(six_adj )
eight=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\RadLab\8cm_run_2')
eight=np.mean(eight)
vals.append(eight)
dist.append(0.08)
eight_adj=eight*(0.08**2)
vals_adj.append(eight_adj)
print('8cm',eight,eight_adj)

sixteen=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\RadLab\16cm_run_2')
sixteen=np.mean(sixteen)
vals.append(sixteen)
dist.append(0.16)
sixteen_adj=sixteen*(0.16**2)
vals_adj.append(sixteen_adj)
print('16cm', sixteen,sixteen_adj)

twelve=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\RadLab\12cm_run_2')
twelve=np.mean(twelve)
vals.append(twelve)
dist.append(0.12)
twelve_adj=twelve*(0.12**2)
vals_adj.append(twelve_adj)
print('12cm', twelve, twelve_adj)

twotwo=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\RadLab\22cm_run_2')
twotwo=np.mean(twotwo)
vals.append(twotwo)
dist.append(0.22)
twotwo_adj=twotwo*(0.22**2)
vals_adj.append(twotwo_adj)
print('22cm', twotwo, twotwo_adj)

threetwo=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\RadLab\32cm_run_2')
threetwo=np.mean(threetwo)
vals.append(threetwo)
dist.append(0.32)
threetwo_adj=threetwo*(0.32**2)
vals_adj.append(threetwo_adj)
print('32cm', threetwo, threetwo_adj)

fourtwo=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\RadLab\42cm_run_2')
fourtwo=np.mean(fourtwo)
vals.append(fourtwo)
dist.append(0.42)
fourtwo_adj=fourtwo*(0.42**2)
vals_adj.append(fourtwo_adj)
print('42cm', fourtwo, fourtwo_adj)

fivetwo=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\RadLab\52cm_run_2')
fivetwo=np.mean(fivetwo)
vals.append(fivetwo)
dist.append(0.52)
fivetwo_adj=fivetwo*(0.52**2)
vals_adj.append(fivetwo_adj)
print('52cm', fivetwo,fivetwo_adj)

sixtwo=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\RadLab\62cm_run_2')
sixtwo=np.mean(sixtwo)
vals.append(sixtwo)
dist.append(0.62)
sixtwo_adj=sixtwo*(0.62**2)
vals_adj.append(sixtwo_adj)
print('62cm', sixtwo,sixtwo_adj)

seventwo=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\RadLab\72cm_run_2')
seventwo=np.mean(seventwo)
vals.append(seventwo)
dist.append(0.72)
seventwo_adj=seventwo*(0.72**2)
vals_adj.append(seventwo_adj)
print('72cm', seventwo,seventwo_adj)

eighttwo=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\RadLab\82cm_run_2')
eighttwo=np.mean(eighttwo)
vals.append(eighttwo)
dist.append(0.82)
eighttwo_adj=eighttwo*(0.82**2)
vals_adj.append(eighttwo_adj)
print('82cm', eighttwo,eighttwo_adj)

ninetwo=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\RadLab\92cm_run_2')
ninetwo=np.mean(ninetwo)
vals.append(ninetwo)
dist.append(0.92)
ninetwo_adj=ninetwo*(0.92**2)
vals_adj.append(ninetwo_adj)
print('92cm', ninetwo,ninetwo_adj)


print(vals[0])





#### ERR0R PR0P

def err(n, d):
    d_n =np.sqrt(n)
    d_d = 2e-3
    return np.sqrt((((d**2)/1)*d_n)**2 + (((2*n*d)/1)*(d_d))**2)

erry = []
for i in range(0, len(vals)):
    err1 = err(vals[i], dist[i])
    erry.append(err1)
print(erry)  

plt.plot(dist,vals)
plt.xlabel('Distance/m')
plt.ylabel('Mean Count Rate/s-1')
plt.title('Mean Count Rate Raw')
plt.show()


    

plt.errorbar(dist, vals_adj, yerr= erry, capsize = 2)
plt.scatter(dist,vals_adj)

plt.xlabel('Distance/m')
plt.ylabel('Mean Count Rate*Distance^2/m^2s-1')
plt.title('Mean Count Rate Distance Adj')
plt.ylim(0,20)

points=np.array(vals_adj[4:])
x=dist[4:]


erry2=erry[4:]
def line(x,a,b):
    return (a*x)+b
params,cov=curve_fit(line,x,points,sigma=erry2,absolute_sigma=True)
a,b=params
#plt.plot(x,line(x,a,b))
xext=np.array([0,0.5,0.96])
yext=line(xext,a,b)
plt.plot(xext,yext)
plt.xlim(0,1)
plt.show()

print(b)
print('err',np.sqrt(cov[0][0]))


def line2(x,c,d):
    return (c*x)+d
x2=dist[3:]
points2=np.array(vals_adj[3:])
params2,cov2=curve_fit(line2,x2,points2)
c,d=params2

def line3(x,f,g):
    return (x*f)+g
x3=dist[5:]
points3=np.array(vals_adj[5:])
params3,cov3=curve_fit(line3,x3,points3)
f,g=params3
inter=b/0.021
print('syserr', np.abs(d-g))

print('loss t 10 cm',(line(0.12,a,b)/(0.12**2)))
print('loss from max thikness of Al',(1.69e7)-(line(0.11941,a,b)/(0.11941**2)))
print('loss from max thickness Cu',(1.69e7)-(line(0.09891584,a,b)/(0.09891584**2)))

cm10loss=(line(0.125,a,b)/(0.125**2))
Alloss=(line(0.11941,a,b)/(0.11941**2))
Culoss=(1.69e7)-(line(0.09891584,a,b)/(0.09891584**2))
print((line(0.125,a,b)/(0.125**2)) - (line(0.11941,a,b)/(0.11941**2)))
print('extra loss from change in thickess for Al',cm10loss-Alloss)
print('extra loss from change in thickness of Cu',cm10loss-Culoss)

Difference=[]
ratio=[]
distances=np.array([0.49e-3,0.7e-3,0.99e-3,1.2e-3,1.5e-3,1.71e-3,1.89e-3,1.98e-3,2.19e-3,2.47e-3,2.96e-3,3.45e-3,3.61e-3,3.94e-3,4.35e-3,4.76e-3,5.18e-3,5.59e-3])
for i in range(len(distances)):
    distances[i]=0.125-distances[i]
proval=line(0.125,a,b)/(0.125**2)
Difference.append((line(distances[0],a,b)/(distances[0]**2))-proval)
ratio.append(539.75/((line(distances[0],a,b)/(distances[0]**2))))

vals=np.array([ 539.75, 427.82608695652175, 265.2, 194.76, 115.475, 79.725, 50.62, 44.425, 27.266666666666666, 13.02375, 3.505, 1.02, 0.8525, 0.65, 0.695, 0.81, 0.86, 0.63])
for i in range(0,len(distances)-1):
    #print(distances[i],vals[i])
    Difference.append((line(distances[i+1],a,b)/(distances[i+1]**2))-(line(distances[i],a,b)/(distances[i]**2)))
    ratio.append(vals[i+1]/(line(distances[i+1],a,b)/(distances[i+1]**2)))


err=[]
for i in range(len(ratio)):
    err.append(ratio[i]*Difference[i])

err2put=[]
for i in range(len(err)):
    err2put.append(np.log(vals[i]+err[i])-np.log(vals[i]))
print(err2put)


cudiff=[]
curatio=[]
cudist=np.array([0.1e-3,0.18e-3,0.29e-3,0.41e-3,0.49e-3,0.59e-3,0.64e-3,0.72e-3,0.82e-3,0.96e-3,1.04e-3,1.12e-3,1.21e-3])
cuvals=np.array([599.0588235294117, 378.44, 221.51111111111112, 81.61666666666666, 47.69, 24.026315789473685, 14.172, 6.265, 2.6225, 0.8675, 0.62, 0.5275, 0.4875])
for i in range(len(cudist)):
    cudist[i]=0.125-cudist[i]
cudiff.append((line(cudist[0],a,b)/(cudist[0]**2))-proval)
curatio.append(599.0588235294117/((line(cudist[0],a,b)/(cudist[0]**2))))
for i in range(len(cudist)-1):
    cudiff.append((line(cudist[i+1],a,b)/(cudist[i+1]**2))-(line(cudist[i],a,b)/(cudist[i]**2)))
    curatio.append(cuvals[i+1]/(line(cudist[i+1],a,b)/(cudist[i+1]**2)))

errcu=[]
for i in range(len(curatio)):
    errcu.append(curatio[i]*cudiff[i])
errincu=[]
for i in range(len(errcu)):
    errincu.append(np.log(cuvals[i]+errcu[i])-np.log(cuvals[i]))
print(errincu)