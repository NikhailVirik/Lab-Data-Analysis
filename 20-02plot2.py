# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:32:13 2024

@author: nsv22
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy, math 
from scipy.special import factorial
from scipy.optimize import curve_fit
from numpy import trapz
from scipy import stats
data=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\ICT from old H Drive\RadLab\Mean_3', delimiter='\t', skiprows=0)
#error=np.sqrt(np.sum(data))/100

#error=np.full(20,error)

counts, bins, bars=plt.hist(data, bins=20)

#y,binEdges=np.histogram(data,bins=10)
#bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#plt.errorbar(bincenters, y, yerr=error, linestyle='', lolims=0)
y,binEdges=np.histogram(data,bins=20)   
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])         
counts=np.array(counts)
bin_heights=counts   
area=0
for i in range(0,len(counts)):
    area=area+(3.1*counts[i])
#adjusted_error = np.minimum(bin_heights, error)
err=[]
for i in range(0,len(counts)):
    err.append(np.sqrt(counts[i]))
    

plt.errorbar(bincenters, y, yerr=err, linestyle='', capsize=2) 
def poisson(n):
    m=np.mean(data)
    p=area*((m**n)*np.exp(-m))/(factorial(n, exact=False))
    return p
def Gauss(n):
    m=np.mean(data)
    sig=np.std(data)
    return area*(1/(sig*np.sqrt(2*np.pi)))*np.exp(-1*((n-m)**2)/(2*(sig**2)))

x=np.linspace(82,144,num=62)

plt.plot(x,Gauss(x), label='Gaussian')  


 
#prob = [300*poisson(x_, data) for x_ in x]
#plt.plot(x,prob)
#plt.plot(x,300*Gauss(x))
plt.plot(x,poisson(x), label='Poisson')
plt.legend()
plt.xlabel('Counts in 1 second')
plt.ylabel('Number of Occurances')
plt.title('Sr-90 100counts/s Counts')

plt.show()

print(len(counts),len(bincenters))
def chi_square_test(observed, expected):
    chi_square_statistic = np.sum((observed - expected)**2 / expected)
    
    # Calculate the degrees of freedom
    degrees_of_freedom = len(observed) - 1
    
    # Calculate the p-value using the chi-square distribution
    p_value = 1 - stats.chi2.cdf(chi_square_statistic, degrees_of_freedom)
    
    return chi_square_statistic, p_value
observed_counts = counts  # Observed counts from histogram
bin_centers = bincenters  # Bin centers
mean = np.mean(data)  # Mean of the Gaussian
std_dev = np.std(data) # Standard deviation of the Gaussian
expected_counts = np.array([len(bin_centers) * (stats.norm.cdf(bin_centers[i] + 0.5, mean, std_dev) - stats.norm.cdf(bin_centers[i] - 0.5, mean, std_dev)) for i in range(len(bin_centers))])
chi_square_statistic, p_value = chi_square_test(observed_counts, expected_counts)

print("Chi-square statistic:", chi_square_statistic)
print("p-value:", p_value)

least=0
for i in range(0,len(bincenters)):
    add=counts[i]-Gauss((bincenters[i]))
    least=least+(np.sqrt(add**2))
print('Least-Squares Gaussian', least/len(counts))

least2=0
for i in range(0,len(bincenters)):
    add=counts[i]-poisson((bincenters[i]))
    least2=least2+(np.sqrt(add**2))
print('Least-Squares Poisson', least2/len(counts))
print(max(counts),min(counts))
print(np.mean(data))
    





