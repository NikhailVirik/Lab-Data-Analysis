# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 10:53:04 2023

@author: Nikhail
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 10:15:47 2023

@author: Nikhail
"""

#!/usr/bin/python

import sys
#import read_data_results3 as rd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import iminuit as im
from scipy import signal
import scipy.fftpack as spf
import scipy.signal as sps
import scipy.interpolate as spi
from scipy.optimize import curve_fit

#Step 1 get the data and the x position
#file='%s'%(sys.argv[1]) #this is the data
results = np.loadtxt(r'C:\Users\Nikhail\Downloads\Filter_w_1.txt')

# Describe the global calibration used
metres_per_microstep = 1.874e-11 # metres

# We are only going to need data from one detector
# make sure the index is the right one for your data!
y1 = np.array(results[:,0])

# get x-axis data from the results
# factor of 2 arrives because we assume conversion does not include path difference to motor distance factor
x = np.array(results[:,5])*metres_per_microstep*2.0

# centre the y-axis on zero by either subtracting the mean
# or using the Butterworth filter
y1 = y1 - y1.mean()

# Butterworth filter to correct for offset
#filter_order = 2
#freq = 1 #cutoff frequency
#sampling = 50 # sampling frequency
#sos = signal.butter(filter_order, freq, 'hp', fs=sampling, output='sos')
#filtered = signal.sosfilt(sos, y1)
#y1 = filtered



# Cubic Spline part - the FFT requires a regular grid on the x-axis
N = 100000 # these are the number of points that you will resample - try changing this and look how well the resampling follows the data.
xs = np.linspace(x[0], x[-1], N) # x-axis to resample onto
y = y1[:len(x)] # make sure y axis has same length as x 
cs = spi.CubicSpline(x, y) # construct the cubic spline function


# step 5 FFT to extract spectra
yf1=spf.fft(cs(xs))
xf1=spf.fftfreq(len(xs)) # setting the correct x-axis for the fourier transform. Osciallations/step  
xf1=spf.fftshift(xf1) #shifts to make it easier (google if interested)
yf1=spf.fftshift(yf1)
xx1=xf1[int(len(xf1)/2+1):len(xf1)]

# convert x-axis to meaningful units - wavelength
distance = xs[1:]-xs[:-1]
# rather than the amplitude
repx1 = distance.mean()/xx1  

plt.figure("Spectrum using global calibration FFT")
#plt.title('Data from: \n%s'%file)
plt.plot(abs(repx1),abs(yf1[int(len(xf1)/2+1):len(xf1)]), label='Real Spectrum')
plt.xlim(300e-9,800e-9)
plt.ylabel('Intensity (a.u.)')
plt.xlabel('Wavelength(m)')
plt.title('White LED, Yellow Filter Wavelength Spectrum')
plt.xlim(5.4e-7,6.1e-7)

#plt.savefig("figures/spectrum_from_global_calibration.png")
newy=abs(yf1[int(len(xf1)/2+1):len(xf1)])
print(repx1[np.argmax(newy)])
newnewy=[]
newyi=[]
print(max(newy)/2)
for i in range(0,len(newy)):
    if newy[i]>= 19074454.214023843:
        newnewy.append(newy[i])
        newyi.append(i)

print(np.where(newy == newnewy[0]),np.where(newy==newnewy[22]))
print(repx1[1225]-repx1[1247])
print(repx1[1260])
print(repx1[1190])
repx2=repx1[1190:1260]
newy2=newy[1190:1260]
def fit_func(repx2,a,mu,sig):
    gauss=a*np.exp(-((repx2)-mu)**2/(2*sig**2)) # formula for a gaussian curve; based on the frequency data, a is the amplitude of the gaussian, mu is the mean value of the gaussian, sig is the std deviation width
     # formula for the straight line, m is the gradient, c is the y-int
    return gauss
ig=[4e7,5.8e-7,np.std(repx2, ddof=1),]
params,covariance= curve_fit(fit_func,repx2,newy2,ig)

a_est,mu_est,sig_est=params
#print(f'a estimate: {a_est}')
print(f'a estimate: {a_est}')
print(f'mu estimate: {mu_est}')
#print(f'b3 estimate: {b3_est}')
# print(f'b4 estimate: {b4_est}')
print(f'sig estimate: {sig_est}')
plt.plot(repx2,fit_func(repx2, a_est,mu_est,sig_est),label='Fitted Gauss', color='red')
plt.legend()
plt.xlim(5.65e-7, 6.1e-7)
plt.show()
FWHM=sig_est*2*np.sqrt(2*np.log(2))
print(FWHM)



