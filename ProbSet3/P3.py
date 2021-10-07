#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 10:29:50 2021

@author: lfairgrievepark12
"""

import numpy as np
import matplotlib.pyplot as plt

dat=np.loadtxt('dish_zenith.txt')
x=np.array(dat[:,0])
y=np.array(dat[:,1])
z=np.array(dat[:,2])

# We have four parameters to fit for and therefore we have 4 equations.
# See the pdf for this
A = np.zeros([len(x),4])
A[:,0] = 1
A[:,1] = A[:,0]*x
A[:,2] = A[:,0]*y
A[:,3] = A[:,1]*x+A[:,2]*y

# Same fitting procedure as Sievers example code from pdf
A = np.matrix(A)
d = np.matrix(z).transpose()
lhs = A.transpose()*A
rhs = A.transpose()*d
p = np.linalg.inv(lhs)*rhs

# Parameters
p0 = p[0,0]
p1 = p[1,0]
p2 = p[2,0]
p3 = p[3,0]

p = [p0, p1, p2, p3]

# The parameters from the equation
a = p[3]
y0 = -p[2]/a/2
x0 = -p[1]/a/2
z0 = p[0] - a*x0**2 - a*y0**2

print('p0 =',p[0], 'p1 =',p[1], 'p2 =',p[2], 'p3 =',p[3])
print('\na =',a, 'x0 =',x0, 'y0 =',y0, 'z0 =',z0)

# So I hope I'm doing this correctly? And the value for the a parameter
# is one sigma?
di = np.sqrt(np.diag(np.linalg.inv(lhs)))

print('f = ', 1/(4*a), 'pm', (di[3]/a)*(1/(4*a)))