# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:11:20 2023

@author: Nikhail
"""

import numpy as np
def R(w,c,l):
    resistor=np.sqrt((1/2))*np.sqrt((((1/w*c)**2)-(w*l))+np.sqrt((((w**6)*(l**2)*(c**4))-1+4*((l**2)*(w**4)*(c**2)))/((c**4)*(w**4))))
    return resistor

c=3*(0.015E-6)
l=3*(330E-6)
w=2*np.pi*(34.8E3)
print(R(w,c,l))