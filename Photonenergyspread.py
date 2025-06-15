# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
data2=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\RadLab\2MevIdealPhoton.txt', delimiter=',', skiprows=1)
data60=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\RadLab\60kevphoton.txt', delimiter=',', skiprows=1)
energy2=data2[0:,0]
energy60=data60[0:,0]

plt.hist(energy2, label='2MeV photon')
plt.hist(energy60, label='60keV photon')
plt.xlabel('Photon energy /keV')
plt.ylabel('No. of photons')
plt.title('Photon energy spread')
plt.legend()
plt.show()