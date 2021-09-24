#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 21:32:19 2021

@author: lfairgrievepark12
"""

import numpy as np

# Using Prof. Sievers elegant choice of x, eps is machine prec.
x=42
eps=2**-52

# First we'll do f = e^x. For dx calc see pdf
dx = (eps/4)**(1/5)

# To make show the rough correctness of dx, we'll take a dx slightly above
# and below the optimal dx and demonstrate that the error is higher
dxu = 1.25*dx
dxd = 0.75*dx

# lets just redefine the exp function to save a little space - it'll be a lot
# of space for e^0.01x
def f1(x):
    return np.exp(x)


f1dtrue = np.exp(x) # True derivative of e^x for comparison

# Calculating for best dx and the worse dx above and below
f1dcalc = 4/3*((f1(x+dx)-f1(x-dx))/(2*dx)-(f1(x+2*dx)-f1(x-2*dx))/(16*dx))
f1dcalcu = 4/3*((f1(x+dxu)-f1(x-dxu))/(2*dxu)-(f1(x+2*dxu)-f1(x-2*dxu))/(16*dxu))
f1dcalcd = 4/3*((f1(x+dxd)-f1(x-dxd))/(2*dxd)-(f1(x+2*dxd)-f1(x-2*dxd))/(16*dxd))

print('for f(x) = e^(x)')
print('dx gives err', f1dcalc/f1dtrue-1, '\n1.25dx gives err', \
      f1dcalcu/f1dtrue-1, '\n0.75dx gives err', f1dcalcd/f1dtrue-1)
print('dx is error closest to zero\n')




# Repeat process for f = e^0.01x. Sorry, know it would have been neater
# to define a function to repeat the process but I already wrote this
dx = (eps/4e-10)**(1/5)
dxu = 1.25*dx
dxd = 0.75*dx

def f1(x):
    return np.exp(0.01*x)

f1dtrue = 0.01*np.exp(0.01*x)

f1dcalc = 4/3*((f1(x+dx)-f1(x-dx))/(2*dx)-(f1(x+2*dx)-f1(x-2*dx))/(16*dx))
f1dcalcu = 4/3*((f1(x+dxu)-f1(x-dxu))/(2*dxu)-(f1(x+2*dxu)-f1(x-2*dxu))/(16*dxu))
f1dcalcd = 4/3*((f1(x+dxd)-f1(x-dxd))/(2*dxd)-(f1(x+2*dxd)-f1(x-2*dxd))/(16*dxd))

print('for f(x) = e^(0.01x)')
print('dx gives err', f1dcalc/f1dtrue-1, '\n1.25dx gives err', \
      f1dcalcu/f1dtrue-1, '\n0.75dx gives err', f1dcalcd/f1dtrue-1)
print('dx is error closest to zero')