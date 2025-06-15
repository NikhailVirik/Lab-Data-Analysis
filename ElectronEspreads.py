# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:56:19 2024

@author: nsv22
"""

import numpy as np
import matplotlib.pyplot as plt
data2M=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\RadLab\data_IdealE2m.txt', delimiter=',', skiprows=1)
data300k=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\RadLab\data_IdealE300.txt', delimiter=',', skiprows=1)
energy2m=data2M[0:,0]
energy300=data300k[:,0]
plt.hist(energy2m)

plt.show()
plt.hist(energy300)
plt.show()
plt.hist(energy2m, label='2MeV')
plt.hist(energy300, label='300keV')
plt.ylabel('No. of Electrons')
plt.xlabel('Energy Deposited/keV')
plt.title('2MeV vs 300keV Electron Energy Spread')
plt.legend()
plt.show()