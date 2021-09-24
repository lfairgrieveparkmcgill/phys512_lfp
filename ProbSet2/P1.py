#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 21:35:36 2021

@author: lfairgrievepark12
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

# Will assign some values to constants because i don't want to normalize things
# Radius of sphere is 1m surface charge density is 100nC/m^2
e0 = 8.85e-12
R=1
c=100e-9

# Will consider from 0 to 3R
zrange=np.linspace(0,3,64)

# This is our function to integrate, taken from
# https://www.physicsisbeautiful.com/resources/introduction-to-electrodynamics/problems/2.7/solutions/2.7-2.8.pdf/gTkMTWTatStZcqFYqG3RFb/
func = lambda u,z: R**2*c/e0*(z-R*u)/(R**2+z**2-2*R*z*u)**(3/2)

E = []
for i in zrange:
    a,b = quad(func,-1,1,args=(i))
    E.append(a)


plt.clf()
plt.figure(figsize=(8,6))
plt.plot(zrange,E,'*')
plt.title('E field of spherical shell conductor')
plt.xlabel('Position')
plt.ylabel('E-field (N/C)')
plt.xticks([0,1,2,3],[0,'R','2R','3R'])

# There is a singularity in the interval at z=R where the integral is undefined
# quad doesn't care and seems to calculate the value to be about 50% between 
# the value to its left and right

# The integrator only seems to care when the denominator is evaluated at zero
# ie. u = (R**2+z**2)/(2Rz)
print(func(1,1))