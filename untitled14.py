# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 23:01:04 2023

@author: Nikhail
"""
import numpy as np
import matplotlib.pyplot as plt
D_trans=[1.8613859620777235e-07, 1.2149763785477178e-07, 1.2747822439875666e-07, 1.476092103656636e-07, 1.9617435030388902e-07, 1.9874547785204035e-07]
D_trans_x=[60,120,240,360,480,960]
D_phase=[3.7584920145338866e-08, 1.9282399961807693e-08, 1.1978190863479046e-08, 5.4629919698775726e-09]
D_phase_x=[240,360,480,960]

plt.scatter(D_trans_x,D_trans)
plt.scatter(D_phase_x,D_phase)
plt.show()
D_all=[D_trans,D_phase]
D_all=np.concatenate(D_all,axis=0)
print(np.mean(D_all))