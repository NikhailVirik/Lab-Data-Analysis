# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 22:15:42 2023

@author: Nikhail
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
x=[459e-9,532e-9,580e-9,633e-9]
y=[1.441976,1.415170966,1.38746,1.438049]
xer=[25e-9,1e-9,13e-9,0.2e-9]
yer=[0.11648872510620363,0.15171017200646358,0.10809826668912487,0.16784968542820702]
y_real=[1.4649,1.4607,1.4587,1.457]

plt.plot(x,y, color="orange", ls="none", marker='o', label='Data')
plt.errorbar(x,y,yerr=yer,xerr=xer, ls='none')
plt.xlabel('Wavelength (m)')
plt.ylabel('Refractive Index')
plt.title('Dispersion relation for glass slide')
plt.plot(x,y_real,ls='none', color='hotpink',marker='o', label='Book Vals')
plt.legend()
yer=np.array(yer)
