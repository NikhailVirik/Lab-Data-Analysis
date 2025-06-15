# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:35:59 2024

@author: nsv22
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
import statistics as stats
from scipy import stats
dist=[]
count=[]
err_n=[]
lens=[]
err_dist=0.005

nums=[0,1,3,4,5,5,4,5,5,6,7,8,9,8,10,12,14,16]

mm_0=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\0mm_run_4')

lens.append(len(mm_0))
#print(sum(mm_0))
mm_0=np.mean(mm_0)
dist.append(0)
count.append(mm_0)


mm_0_49=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\0.49mm_run_1')
lens.append(len(mm_0_49))
mm_0_49=np.mean(mm_0_49)
dist.append(0.49)
count.append(mm_0_49)

mm_0_7=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\0.7mm_run_1')
#lens.append(len(mm_0_7))
mm_0_7=np.mean(mm_0_7)
#dist.append(0.7)
#count.append(mm_0_7)

mm_0_99=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\0.99mm_run_1')
lens.append(len(mm_0_99))

mm_0_99=np.mean(mm_0_99)
dist.append(0.99)
count.append(mm_0_99)

mm_1_20=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\1.2mm_run_1')
lens.append(len(mm_1_20))

mm_1_20=np.mean(mm_1_20)
dist.append(1.2)
count.append(mm_1_20)   

mm_1_5=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\1.5mm_run_1')
lens.append(len(mm_1_5))

mm_1_5=np.mean(mm_1_5)
dist.append(1.5)
count.append(mm_1_5)

mm_1_71=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\1.71mm_run_1')

lens.append(len(mm_1_71))
mm_1_71=np.mean(mm_1_71)
dist.append(1.71)
count.append(mm_1_71)

mm_1_89=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\1.89mm_run_1')
lens.append(len(mm_1_89))
mm_1_89=np.mean(mm_1_89)
dist.append(1.89)
count.append(mm_1_89)

mm_1_98=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\1.98mm_run_1')
lens.append(len(mm_1_98))

mm_1_98=np.mean(mm_1_98)
dist.append(1.98)
count.append(mm_1_98)

mm_2_19=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\2.19mm-run_1')
lens.append(len(mm_2_19))
mm_2_19=np.mean(mm_2_19)
dist.append(2.19)
count.append(mm_2_19)

mm_2_47=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\2.47mm_run_1')
lens.append(len(mm_2_47))
mm_2_47=np.mean(mm_2_47)
dist.append(2.47)
count.append(mm_2_47)



mm_2_96=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\2.96mm_run_1')
lens.append(len(mm_2_96))
mm_2_96=np.mean(mm_2_96)
dist.append(2.96)
count.append(mm_2_96)



mm_3_45=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\3.45mm_run_1')
lens.append(len(mm_3_45))
mm_3_45=np.mean(mm_3_45)
dist.append(3.45)
count.append(mm_3_45)

mm_3_61=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\3.61mm_run_1')
lens.append(len(mm_3_61))
mm_3_61=np.mean(mm_3_61)
dist.append(3.61)
count.append(mm_3_61)

mm_3_74=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\3.74mm_run_1')
lens.append(len(mm_3_74))
mm_3_74=np.mean(mm_3_74)
#dist.append(3.74)
#count.append(mm_3_74)

mm_3_94=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\3.94mm_run_1')
lens.append(len(mm_3_94))
mm_3_94=np.mean(mm_3_94)
dist.append(3.94)
count.append(mm_3_94)



mm_4_35=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\4.35mm_run_1')
lens.append(len(mm_4_35))
mm_4_35=np.mean(mm_4_35)
dist.append(4.35)
count.append(mm_4_35)

mm_4_76=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\4.76mm_run_1')
lens.append(len(mm_4_76))
mm_4_76=np.mean(mm_4_76)
dist.append(4.76)
count.append(mm_4_76)


mm_5_18=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\5.18mm_run_1')
lens.append(len(mm_5_18))
mm_5_18=np.mean(mm_5_18)
dist.append(5.18)
count.append(mm_5_18)


mm_5_59=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\5.59mm_run_1')
lens.append(len(mm_5_59))
mm_5_59=np.mean(mm_5_59)
dist.append(5.59)
count.append(mm_5_59)

err_c=[]
err_d=[]
for i in range(len(count)):
    err_d.append(0.005*nums[i])
air_al=[]
for i in range(len(dist)):
    air_al.append(125-dist[i])
#print('air_al',air_al)
for i in range(0,len(count)):
    err=np.sqrt(count[i])*(1/np.sqrt(lens[i]))
    err_c.append(err)
plt.scatter(dist,count)
plt.xlabel('Thickness of Aluminium/mm')
plt.ylabel('Count Rate/s-1')
plt.title('Linear Scaling Count Rate vs Thickness')
plt.errorbar(dist,count,xerr=err_d,yerr=err_c,linestyle='')
#plt.xlim(2,6)
#plt.ylim(0,10)
plt.show()

logcount=np.log(count)
for i in range(len(dist)):
    dist[i]=2.7*0.1*dist[i]
for i in range(len(err_d)):
    err_d[i]=2.7*0.1*err_d[i]
err_c_up=[]
err_c2_down=[]
for i in range(0,len(count)):
    err2=np.log(count[i]+(np.sqrt(count[i]))*(1/np.sqrt(lens[i])))-np.log(count[i])
    err_c_up.append(err2)
err_c_down=[]   
for i in range(0,len(count)):
    err4=np.log(count[i])-np.log(count[i]-(np.sqrt(count[i])*1/np.sqrt(lens[i])))
    err_c2_down.append(err4)
huat=np.array([0,0.007975511190418061, 0.004757843219970859, 0.0034566331201570932, 0.004940621372162113, 0.0034705310645550824, 0.0029808016515642244, 0.0014942214833260081, 0.003483716597881159, 0.004648574107967374, 0.008131467136911041, 0.008163152527888337, 0.002687235914232783, 0.005537495681636306, 0.0068911015828967415, 0.006913856940265639, 0.00710504165396067, 0.006960397474250424])
#err_c_down=err_c_down[:-5]

#for i in range(0,len(count)):
 #   if count[i]-np.sqrt(count[i])<0:
  #      err_c_down[i]=np.abs(np.log(count[i])-np.log(np.sqrt(count[i])*(1/np.sqrt(lens[i]))))
        #err_c_down.append(np.log(count[i])-np.log(np.sqrt(count[i])))
for i in range(len(err_c2_down)):
    err_c_down.append(err_c2_down[i]+huat[i])
push=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,]
plt.errorbar(dist,logcount,yerr=[err_c_down,push],linestyle='',color='pink',capsize=4, label='Systematic Errorbar')
plt.errorbar(dist,logcount,xerr=err_d,yerr=[err_c2_down,err_c_up],linestyle='',color='orange', capsize=2, label='Statistical Errorbar')

plt.scatter(dist,np.log(count),marker='x',color='red',label='Al data')
plt.xlabel('Areal Density/ gcm^-2')
plt.ylabel('Log(Count rate)')
plt.title('Log Scaling Count Rate vs Areal Density')
#plt.ylim(4.3,4.5)
#
#plt.xlim(1,3)

def line1(x,a,b):
    return (a*x)+b
def line2(x,c,d):
    return (c*x)+d
def line3(x,f,g):
    return (f*x)+g

x1=np.array(dist[0:5])
points1=np.array(logcount[0:5])
x2=np.array(dist[5:12])
points2=np.array(logcount[5:12])
x3=np.array(dist[12:])
points3=np.array(logcount[12:])

params1,cov1=curve_fit(line1,x1,points1)
a,b=params1
xext1=np.array([0,0.1,0.3,0.42])
yext1=line1(xext1,a,b)
plt.plot(xext1,yext1,color='blue',label='Al fit')

sigma_x=err_d
sigma_y_u=err_c_up[5:12]
sigma_y_d=err_c_down[5:12]
weights=[]
for i in range(0,len(sigma_y_u)):
    weights.append(np.sqrt(((sigma_y_u[i]**2 + sigma_y_d[i]**2))))
#weights.append(sigma_x**2)

params2,cov2=curve_fit(line2,x2,points2,sigma=weights, absolute_sigma=True)
c,d=params2
xext2=np.array([0.45,0.6,0.8,0.96])
yext2=line2(xext2,c,d)
plt.plot(xext2,yext2,color='blue')
xext222=np.array([0.9,0.93,1.01])
yext222=line2(xext222,c,d)
plt.plot(xext222,yext222,linestyle='dashed',color='blue', label='Al extrapolation')
background=np.mean(logcount[-6:])
print('background',background)

#plt.xlim(-0.01,0.45)
#plt.ylim(4,7.5)

x_i=(background-d)/c 
print('x-int',x_i)

#
#def weighted_residuals(x,c,d,y, weights):
#
 #   y_pred = line2(x, c, d)
  #  return (y - y_pred) * weights

#initial_guess=[c,d]
#popt,pcov=curve_fit(weighted_residuals(x2,c,d,points2,weights),x2,points2,p0=initial_guess, sigma=weights, absolute_sigma=True)
#param_errors=np.sqrt(np.diag(pcov))
weights_bg=[]
sigma_bg_u=err_c_up[-6:]
sigma_bg_d=err_c_down[-6:]
for i in range(len(sigma_bg_d)):
    weights_bg.append(np.sqrt((sigma_bg_u[i]**2)+(sigma_bg_d[i]**2)))
err_bg=0
for i in range(len(sigma_bg_d)):
    err_bg+=(weights_bg[i]/len(weights_bg))**2
err_bg=np.sqrt(err_bg)


d_d=np.sqrt(cov2[0][0])
d_c=np.sqrt(cov2[1][1])
print('d_d',d_d,d_c,d,c)
err_top=d_d+err_bg
top=background-d
err_intercept=(np.abs((err_top/top))+np.abs((d_c/c)))*x_i
print('err_int',err_intercept)
print('bgerr',err_bg)
xext3=np.array([0.97,1.6])
yext3=np.array([background,background])
plt.plot(xext3,yext3,color='blue')
x4=[0,0.3,1.6]
y4=[0,0,0]
#plt.plot(x4,y4,color='black')


def line12(x,l,m):
    return (l*x)+m
x12=np.array(dist[4:13])
points12=np.array(logcount[4:13])
params12,cov12=curve_fit(line12,x12,points12)
l,m=params12
newbg=np.mean(logcount[-7:])
x_n=(newbg-m)/l
err_sys=np.abs(x_n-x_i)
print('system err', np.abs(x_n-x_i))
print('err total', err_sys+err_intercept)
R=x_i
E_max=np.sqrt((1/22.4)*((((R/0.11)+1))**2)-1)
del_emax=E_max*((err_sys+err_intercept)/(x_i))
print('Emax=', E_max, '+-', del_emax)


####################################

fudge_num3=np.sqrt((err_c_up[-3]**2)+(err_c_down[-3]**2))
fudge_num2=np.sqrt((err_c_up[-2]**2)+(err_c_down[-2]**2))
fudge_num1=np.sqrt((err_c_up[-1]**2)+(err_c_down[-1]**2))
fudge_num4=np.sqrt((err_c_up[-4]**2)+(err_c_down[-4]**2))
fudge_num5=np.sqrt((err_c_up[-5]**2)+(err_c_down[-5]**2))
fudge_num6=np.sqrt((err_c_up[-6]**2)+(err_c_down[-6]**2))
fudge_num7=np.sqrt((err_c_up[-7]**2)+(err_c_down[-7]**2))
fudge_num8=np.sqrt((err_c_up[-8]**2)+(err_c_down[-8]**2))
obsy=logcount[-6:]
errs=np.array([fudge_num1,fudge_num2,fudge_num3,fudge_num4,fudge_num5,fudge_num6])
chi_square=np.sum((obsy-np.mean(obsy))**2/errs**2)
df=len(obsy)-1
critical_value=chi2.ppf(0.95,df)
pback=1-stats.chi2.cdf(chi_square,df)
print('p-vlaue of background',pback)
if chi_square > critical_value:
    print("Reject null hypothesis: Observed y values are not consistent with being around a constant value.")
else:
    print("Fail to reject null hypothesis: Observed y values are consistent with being around a constant value.")

#####################################
def line4(x,p,q):
    return (p*x)+q
x5=dist[0:7]
points5=logcount[0:7]
params5,cov5=curve_fit(line4,x5,points5)
p,q=params5
residuals5=0
for i in range(len(x5)):
    residuals5+=np.abs(logcount[i]-line4(dist[i],p,q))**2
residuals5=np.sqrt(residuals5)/5

x6=dist[0:6]
points6=logcount[0:6]
params6,cov6=curve_fit(line4,x6,points6)
p,q=params6
residuals6=0
for i in range(len(x6)):
    residuals6+=np.abs(logcount[i]-line4(dist[i],p,q))**2
residuals6=np.sqrt(residuals6)/6

x7=dist[0:7]
points7=logcount[0:7]
params7,cov7=curve_fit(line4,x7,points7)
p,q=params7
residuals7=0
for i in range(len(x7)):
    residuals7+=np.abs(logcount[i]-line4(dist[i],p,q))**2
residuals7=np.sqrt(residuals7)/7

x8=dist[0:8]
points8=logcount[0:8]
params8,cov8=curve_fit(line4,x8,points8)
p,q=params8
residuals8=0
for i in range(len(x8)):
    residuals8+=np.abs(logcount[i]-line4(dist[i],p,q))**2
residuals8=np.sqrt(residuals8)/8

x9=dist[0:9]
points9=logcount[0:9]
params9,cov9=curve_fit(line4,x9,points9)
p,q=params9
residuals9=0
for i in range(len(x9)):
    residuals9+=np.abs(logcount[i]-line4(dist[i],p,q))**2
residuals9=np.sqrt(residuals6)/9

x10=dist[0:10]
points10=logcount[0:10]
params10,cov10=curve_fit(line4,x10,points10)
p,q=params10
residuals10=0
for i in range(len(x10)):
    residuals10+=np.abs(logcount[i]-line4(dist[i],p,q))**2
residuals10=np.sqrt(residuals10)/10

print('Residuals',residuals5,residuals6,residuals7,residuals8,residuals9,residuals10)

fudge_num52=np.sqrt((err_c_up[5]**2)+(err_c_down[5]**2)+(err_d[5]**2))
fudge_num62=np.sqrt((err_c_up[6]**2)+(err_c_down[6]**2)+(err_d[6]**2))
fudge_num72=np.sqrt((err_c_up[7]**2)+(err_c_down[7]**2)+(err_d[7]**2))
fudge_num82=np.sqrt((err_c_up[8]**2)+(err_c_down[8]**2)+(err_d[8]**2))
fudge_num9=np.sqrt((err_c_up[9]**2)+(err_c_down[9]**2)+(err_d[9]**2))
fudge_num10=np.sqrt((err_c_up[10]**2)+(err_c_down[10]**2)+(err_d[10]**2))
fudge_num11=np.sqrt((err_c_up[11]**2)+(err_c_down[11]**2)+(err_d[11]**2))
fudge_num12=np.sqrt((err_c_up[12]**2)+(err_c_down[12]**2)+(err_d[12]**2))
fudge_num13=np.sqrt((err_c_up[13]**2)+(err_c_down[13]**2)+(err_d[13]**2))
fudge_num14=np.sqrt((err_c_up[14]**2)+(err_c_down[14]**2)+(err_d[14]**2))
fudge_num15=np.sqrt((err_c_up[15]**2)+(err_c_down[15]**2)+(err_d[15]**2))
obsy_2=logcount[7:13]
errz=np.array([fudge_num52,fudge_num62,fudge_num72, fudge_num82,fudge_num9,fudge_num10,fudge_num11,fudge_num12,fudge_num13,fudge_num14,fudge_num15])
obsy_bcu=[]
for i in range(len(obsy_2)):
    obsy_bcu.append(line2(dist[i+7],c,d))
chi_bcu=0
for i in range(len(obsy_2)):
    chi_top=(obsy_2[i]-obsy_bcu[i])**2
    chi_btm=errz[i]**2
    chi_bcu+=(chi_top/chi_btm)
df2=len(obsy_2)-2
crit_2=chi2.ppf(0.95,df2)

p_value=1-chi2.cdf(chi_bcu,df2)
print('p-val', p_value)

chi_bcu2=0
for i in range(len(obsy_2)):
    chi_top=(obsy_2[i]-obsy_bcu[i])**2
    chi_btm=obsy_bcu[i]
    chi_bcu+=(chi_top/chi_btm)
pv_new=1-chi2.cdf(chi_bcu2,df2)
print('p-val no uncertainty',pv_new)

observedx=dist[5:12]
expectedx=[]
for i in range(len(observedx)):
    expectedx.append((logcount[i+5]-d)/c)
observed_errs=err_d[5:12]
free2=len(observedx)-2
chix2=0


for i in range(len(observedx)):
    deviations=observedx[i]-expectedx[i]
    #print(deviations)
    scaled_deviations=deviations/observed_errs[i]
    chix2+=scaled_deviations**2
    #print(chix2)

kk=1-stats.chi2.cdf(chix2,free2)
print('p val!!!!', kk)

observed1=dist[0:5]
expected1=[]

for i in range(len(observed1)):
    expected1.append((logcount[i]-b)/a)

observed_errs1=err_d[0:5]
dfkkj=len(observed1)-2
chi17=0
#chi17+=((observed1[0]-expected1[0])/expected1[0])**2
for i in range(len(observed1)-1):
    deviations=observed1[i+1]-expected1[i+1]
    #print(deviations)
    scaled_deviations=deviations/observed_errs1[i+1]
    chi17+=scaled_deviations**2
    #print(chix2)

kk1=1-stats.chi2.cdf(chi17,dfkkj)
print('pval top line',kk1)


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import statistics as stats
from scipy.stats import chi2
from scipy.stats import t
from scipy import stats
count2=[]
dist2=[]
lens2=[]

nums2=[0,1,2,3,2,3,4,2,3,4,3,4,3,4]

mm_0_2=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\cp_0mm_run_1')
lens2.append(len(mm_0_2))
mm_0_2=np.mean(mm_0_2)
dist2.append(0)
count2.append(mm_0_2)
print(count2)

#mm_0_2_49=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\cp_0.49mm_run_1')
#mm_0_2_49=np.mean(mm_0_2_49)
#dist2.append(0.49)
#count2.append(mm_0_2_49)

mm_0_2_1=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\cp_0.1mm_run_1')
lens2.append(len(mm_0_2_1))
mm_0_2_1=np.mean(mm_0_2_1)
dist2.append(0.1)
count2.append(mm_0_2_1)

mm_0_2_18=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\0.18mm_run_1')
lens2.append(len(mm_0_2_18))
mm_0_2_18=np.mean(mm_0_2_18)
dist2.append(0.18)
count2.append(mm_0_2_18)

mm_0_2_29=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\0.29mm_run_1')
lens2.append(len(mm_0_2_29))
mm_0_2_29=np.mean(mm_0_2_29)
dist2.append(0.29)
count2.append(mm_0_2_29)

mm_0_2_31=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\cp_0.31mm_run_1')
#lens2.append(len(mm_0_2_31))
mm_0_2_31=np.mean(mm_0_2_31)
#dist2.append(0.31)
#count2.append(mm_0_2_31)

mm_0_2_41=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\0.41mm_run_1')
lens2.append(len(mm_0_2_41))
mm_0_2_41=np.mean(mm_0_2_41)
dist2.append(0.41)
count2.append(mm_0_2_41)

mm_0_2_49=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\0.48mm_run_1')
lens2.append(len(mm_0_2_49))
mm_0_2_49=np.mean(mm_0_2_49)
count2.append(mm_0_2_49)
dist2.append(0.49)

mm_0_2_59=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\cp_0.59mm_run1')
lens2.append(len(mm_0_2_59))
print(len(mm_0_2_59))
mm_0_2_59=np.mean(mm_0_2_59)
dist2.append(0.59)
count2.append(mm_0_2_59)

mm_0_2_64=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\cp_0.64mm_run1')
lens2.append(len(mm_0_2_64))
mm_0_2_64=np.mean(mm_0_2_64)
dist2.append(0.64)
count2.append(mm_0_2_64)

mm_0_2_72=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\cp_0.72_mm')
lens2.append(len(mm_0_2_72))
mm_0_2_72=np.mean(mm_0_2_72)
dist2.append(0.72)
count2.append(mm_0_2_72)

mm_0_2_82=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\cp_0.82mm_run_1')
lens2.append(len(mm_0_2_82))
mm_0_2_82=np.mean(mm_0_2_82)
dist2.append(0.82)
count2.append(mm_0_2_82)

mm_0_2_96=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\cp_0.96_run_1')
lens2.append(len(mm_0_2_96))
mm_0_2_96=np.mean(mm_0_2_96)
dist2.append(0.96)
count2.append(mm_0_2_96)


mm_1_04=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\cp_1.04mm_run1')
lens2.append(len(mm_1_04))
mm_1_04=np.mean(mm_1_04)
dist2.append(1.04)
count2.append(mm_1_04)

mm_1_12=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\cp_1.12mm_run_1')
lens2.append(len(mm_1_12))
mm_1_12=np.mean(mm_1_12)
dist2.append(1.12)
count2.append(mm_1_12)

mm_1_21=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\cp_1.21mm_run1')
lens2.append(len(mm_1_21))
mm_1_21=np.mean(mm_1_21)
dist2.append(1.21)
count2.append(mm_1_21)
print('lens2',lens2)
err_d_2=[]
for i in range(len(count2)):
    err_d_2.append(0.005*nums2[i])
err_c_2=[]
for i in range(0,len(count2)):
    err_c_2.append(np.sqrt(count2[i])*(1/np.sqrt(lens2[i])))

#plt.scatter(dist2,count2)
#plt.xlim(0.9,1.22)
#plt.ylim(0,10)
#plt.errorbar(dist2,count2,xerr=err_d_2,yerr=err_c_2,linestyle='')
#plt.show()

logcount2s=np.log(count2)
for i in range(len(dist2)):
    dist2[i]=8.96*0.1*dist2[i]
for i in range(len(err_d_2)):
    err_d_2[i]=8.96*0.1*err_d_2[i]
plt.scatter(dist2,logcount2s,marker='x',label='Cu data')

err_c_2_up=[]
err_c_22_down=[]

for i in range(0,len(count2)):
    err=np.log(count2[i]+(np.sqrt(count2[i])*(1/np.sqrt(lens2[i]))))-np.log(count2[i])
    err_c_2_up.append(err)
err_c_2_down=[]
for i in range(0,len(count2)):
    err=np.log(count2[i])-np.log(count2[i]-(np.sqrt(count2[i])*(1/np.sqrt(lens2[i]))))
    err_c_22_down.append(err)

heng=np.array([0,0.0016355606013931379, 0.0013097961782078116, 0.0018014209664292125, 0.0019666305383498894, 0.0013129737639143713, 0.0016418347959787916, 0.0008220749660834414, 0.0013153415665325685, 0.0016447968868619212, 0.0023033686124728225, 0.0013186505105191038, 0.00131948041634955, 0.0014851636805655044])
for i in range(len(err_c_22_down)):
    err_c_2_down.append(err_c_22_down[i]+heng[i])

bono=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
plt.errorbar(dist2,logcount2s,yerr=[err_c_2_down,bono],linestyle='',color='pink',capsize=4)
plt.errorbar(dist2,logcount2s, xerr=err_d_2, yerr=[err_c_22_down,err_c_2_up],linestyle='',color='orange',capsize=2)
#plt.xlim(0.4,0.6)
#plt.ylim(3.8,4)
def line1_2(x,a_2,b_2):
    return (a_2*x)+b_2
x1_2=np.array(dist2[0:3])
points1_2=np.array(logcount2s[0:3])
params1_2,cov1_2=curve_fit(line1_2,x1_2,points1_2)
a_2,b_2=params1_2
xext1_2=np.array([0,0.18])
yext1_2=line1_2(xext1_2,a_2,b_2)
plt.plot(xext1_2,yext1_2,color='brown',label='Cu fit')

def line2_2(x,c_2,d_2):
    return (c_2*x)+d_2
sigma_x_2=err_d_2[3:12]
sigma_y_u_2=err_c_2_up[3:12]
sigma_y_d_2=err_c_2_down[3:12]
weights_2=[]
for i in range(len(sigma_x_2)):
    weights_2.append(np.sqrt((sigma_y_u_2[i]**2)+(sigma_y_d_2[i]**2)))
weights_2=np.array(weights_2)
#weights_2=weights_2**2
x2_2=np.array(dist2[3:12])
points2_2=np.array(logcount2s[3:12])
params2_2,cov2_2=curve_fit(line2_2,x2_2,points2_2, sigma=weights_2, absolute_sigma=True)
c_2,d_2=params2_2
xext2_2=np.array([0.25,0.58,0.7,0.92])
xext2_22=np.array([0.8,0.95])
yext2_2=line2_2(xext2_2,c_2,d_2)
yext2_22=line2_2(xext2_22,c_2,d_2)
plt.plot(xext2_2,yext2_2,color='brown')
plt.plot(xext2_22,yext2_22, linestyle='dashed',color='brown',label= 'Cu extrapolation')
background_2=np.mean(logcount2s[12:14])
print('Bgbgbg',background_2)
x3_2=dist2[12:14]
points3_2=logcount2s[12:14]
xext3_2=np.array([1,1.1])
yext3_2=np.array([np.mean(logcount2s[12:14]),np.mean(logcount2s[12:14])])
plt.plot(xext3_2,yext3_2,color='brown')
xext3_22=np.array([0.9,1])
yext3_22=np.array([np.mean(logcount2s[12:14]),np.mean(logcount2s[12:14])])
plt.plot(xext3_22,yext3_22,linestyle='dashed', color='brown')

weights_2_bg=[]
sigma_bg_u_2=err_c_2_up[3:12]
sigma_bg_d_2=err_c_2_down[3:12]
for i in range(len(x3_2)):
    weights_2_bg.append(np.sqrt((sigma_bg_u_2[i]**2)+(sigma_bg_d_2[i]**2)))
err_bg_2=0
for i in range(len(x3_2)):
    err_bg_2+=(weights_2_bg[i]/len(weights_2_bg))**2
err_bg_2=np.sqrt(err_bg_2)
print('err_bg_2',err_bg_2)
plt.legend()
plt.grid()
#plt.xlim(0.5,1.6)
#plt.ylim(-1,2)
plt.show()
x_i2=(background_2-d_2)/c_2 
print('x-int',x_i2)
d_d2=np.sqrt(cov2_2[0][0])
d_c2=np.sqrt(cov2_2[1][1])
print('d_d2',d_d2,d_c2)
err_top2=d_d2+err_bg_2
top2=background_2-d_2
err_intercept2=(np.abs((err_top2/top2))+np.abs((d_c2/c_2)))*x_i2
print('err_int',err_intercept2)

def line7_2(x,p_2,q_2):
    return (p_2*x)+q_2
x7_2=np.array(dist2[2:13])
points7_2=np.array(logcount2s[2:13])
params7_2,cov7_2=curve_fit(line7_2,x7_2,points7_2)
p,q=params7_2
newbg_2=np.mean(logcount2s[11:14])
x_n2=(newbg_2-q)/p
err_sys2=np.abs(x_n2-x_i2)
print('system err', np.abs(x_n2-x_i2))

total_err2=err_intercept2+err_sys2
print('total err',total_err2)
R2=x_i2
E_max2=np.sqrt((1/22.4)*((((R2/0.11)+1))**2)-1)
del_emax2=E_max2*((total_err2)/x_i2)
print('Emax=', E_max2,'+-', del_emax2)

################################
def line4_2(x,p_3,q_3):
    return (p_3*x)+q_3
x6_2=dist2[0:6]
points6_2=logcount2s[0:6]
params6_2,cov6_2=curve_fit(line4_2,x6_2,points6_2)
p_3,q_3=params6_2
residuals6_2=0
for i in range(len(x6_2)):
    residuals6_2+=np.abs(logcount2s[i]-line4_2(dist2[i],p_3,q_3))**2
residuals6_2=np.sqrt(residuals6_2)/6

x5_2=dist2[0:5]
points5_2=dist2[0:5]
points5_2=logcount2s[0:5]
params5_2,cov5_2=curve_fit(line4_2,x5_2,points5_2)
p_3,q_3=params5_2
residuals5_2=0
for i in range(len(x5_2)):
    residuals5_2+=np.abs(logcount2s[i]-line4_2(dist2[i],p_3,q_3))**2
residuals5_2=np.sqrt(residuals5_2)/5

x7_3=dist2[0:7]
points7_3=dist2[0:7]
points7_3=logcount2s[0:7]
params7_3,cov7_3=curve_fit(line4_2,x7_3,points7_3)
p_3,q_3=params7_3
residuals7_3=0
for i in range(len(x7_3)):
    residuals7_3+=np.abs(logcount2s[i]-line4_2(dist2[i],p_3,q_3))**2
residuals7_3=np.sqrt(residuals7_3)/7

x8_2=dist2[0:8]
points8_2=dist2[0:8]
points8_2=logcount2s[0:8]
params8_2,cov8_2=curve_fit(line4_2,x8_2,points8_2)
p_3,q_3=params8_2
residuals8_2=0
for i in range(len(x8_2)):
    residuals8_2+=np.abs(logcount2s[i]-line4_2(dist2[i],p_3,q_3))**2
residuals8_2=np.sqrt(residuals8_2)/8

x9_2=dist2[0:9]
points9_2=dist2[0:9]
points9_2=logcount2s[0:9]
params9_2,cov9_2=curve_fit(line4_2,x9_2,points9_2)
p_3,q_3=params9_2
residuals9_2=0
for i in range(len(x9_2)):
    residuals9_2+=np.abs(logcount2s[i]-line4_2(dist2[i],p_3,q_3))**2
residuals9_2=np.sqrt(residuals9_2)/9

x10_2=dist2[0:10]
points1_20=dist2[0:10]
points1_20=logcount2s[0:10]
params1_20,cov1_20=curve_fit(line4_2,x10_2,points1_20)
p_3,q_3=params1_20
residuals10_2=0
for i in range(len(x10_2)):
    residuals10_2+=np.abs(logcount2s[i]-line4_2(dist2[i],p_3,q_3))**2
residuals10_2=np.sqrt(residuals10_2)/10
print('Residuals',residuals5_2,residuals6_2,residuals7_3,residuals8_2,residuals9_2,residuals10_2)

fudge_num3=np.sqrt((err_c_2_up[-3]**2)+(err_c_2_down[-3]**2))
fudge_num2=np.sqrt((err_c_2_up[-2]**2)+(err_c_2_down[-2]**2))
fudge_num1=np.sqrt((err_c_2_up[-1]**2)+(err_c_2_down[-1]**2))
fudge_num4=np.sqrt((err_c_2_up[-4]**2)+(err_c_2_down[-4]**2))
obsy_2=logcount2s[-2:]
errs_2=np.array([fudge_num1,fudge_num2])
chi_square2=np.sum((obsy_2-np.mean(obsy_2))**2/errs_2**2)
df_2=len(obsy_2)-1
print('backgorundp',1-stats.chi2.cdf(chi_square2,df_2))
critical_value2=chi2.ppf(0.95,df_2)

if chi_square2 > critical_value2:
    print("Reject null hypothesis: Observed y values are not consistent with being around a constant value.")
else:
    print("Fail to reject null hypothesis: Observed y values are consistent with being around a constant value.")

fudge_num8=np.sqrt((err_c_2_up[8]**2)+(err_c_2_down[8]**2)+(err_d_2[8]**2))
fudge_num9=np.sqrt((err_c_2_up[9]**2)+(err_c_2_down[9]**2)+(err_d_2[9]**2))
fudge_num10=np.sqrt((err_c_2_up[10]**2)+(err_c_2_down[10]**2)+(err_d_2[10]**2))
fudge_num11=np.sqrt((err_c_2_up[11]**2)+(err_c_2_down[11]**2)+(err_d_2[11]**2))
obsy_2_2=logcount2s[3:12]
errz2=np.array([fudge_num8,fudge_num9,fudge_num10,fudge_num11])
obsy_2_bcu=[]
for i in range(len(obsy_2_2)):
    obsy_2_bcu.append(line2_2(dist2[i+3],c_2,d_2))
#chi_bcu2=0
#for i in range(len(obsy_2_2)):
#    chi_top=(obsy_2_2[i]-obsy_2_bcu[i])**2
 #   chi_btm=errz2[i]**2
  #  chi_bcu2+=(chi_top/chi_btm)
df_22=len(obsy_2_2)-2
#crit_22=chi2.ppf(0.95,df_22)

#_value2=1-chi2.cdf(chi_bcu2,df_22)
#print('p-val', p_value2)

#chi_bcu22=0
#for i in range(len(obsy_2_2)):
  #  chi_top=(obsy_2_2[i]-obsy_2_bcu[i])**2
  # chi_btm=obsy_2_bcu[i]
   # chi_bcu2+=(chi_top/chi_btm)
#pv_new2=1-chi2.cdf(chi_bcu22,df_22)
#print('p-val no uncertainty',pv_new2)
#print(obsy_2_bcu)
obsy_2_bcu=np.array(obsy_2_bcu)

observedx2=dist2[3:12]
expectedx2=[]
for i in range(len(observedx2)):
    expectedx2.append((logcount2s[i+3]-d_2)/c_2)


free=len(observedx2)-2 
  
#print(p_plsbethelasttime2)

observed_errs_2=err_d_2[3:12]
chix_22=0
for i in range(len(observedx2)):
    deviations=observedx2[i]-expectedx2[i]
    scaled_d2eviations=deviations/observed_errs_2[i]
    chix_22+=scaled_d2eviations**2

kk2=1-stats.chi2.cdf(chix_22,free)
print('pvalue!!!',kk2)

chixtop=0
observedx21=dist2[0:3]
expectedx21=[]
for i in range(len(observedx21)):
    expectedx21.append((logcount2s[i]-b_2)/a_2)
observed_errs_21=err_d_2[0:3]
chixtop+=((observedx21[0]-expectedx21[0])/(expectedx21[0]))**2
for i in range(len(observedx21)-1):
    deviations=observedx21[i+1]-expectedx21[i+1]
    scaled_d2eviations=deviations/observed_errs_21[i+1]
    chixtop+=scaled_d2eviations**2
free2=len(observedx21)-2
kk2top=1-stats.chi2.cdf(chixtop,free2)
print('p val top!!!',kk2top)
#################
intercept_estimate=E_max
book_value=2.27
standard_error=del_emax
n=len(dist)
t_statistic=-1*(intercept_estimate - book_value) / standard_error
degrees_of_freedom = n - 1
p15 = 1-stats.t.sf(abs(t_statistic), degrees_of_freedom) * 2
print('Alunimum p compare book', p15)
print(t_statistic,degrees_of_freedom)


intercept_estimate2=E_max2
book_value=2.27
standard_error2=del_emax2
n2=len(dist2)
t_statistic2=-1*(intercept_estimate2 - book_value) / standard_error2
degrees_of_freedom2 = n2 - 1
p16 = 1-stats.t.sf(abs(t_statistic2), degrees_of_freedom2) * 2
print('copper p compare book', p16)

print('Aluminium No. of Cycle',lens)
print('Copper No. of Cycles',lens2)

air_al=[]
for i in range(len(dist)):
    air_al.append(125-dist[i])
air_cu=[]
for i in range(len(dist2)):
    air_cu.append(125-dist2[i])
print('lenghts of air column for Al',air_al)
print('lenghts of air coulmn for Cu',air_cu)
print(count2)

print(lens2)
