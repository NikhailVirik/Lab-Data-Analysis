# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
data=np.loadtxt(r'C:\Users\nsv22\OneDrive - Imperial College London\ICT from old H Drive\downloads\Data_Session1\Book1.csv', skiprows=1)
plt.hist(data)
plt.xlabel('Energy/keV')
plt.ylabel('No. of particles')
plt.title('50k electron energy spread')
plt.show()

