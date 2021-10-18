#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 21:31:40 2021

@author: lfairgrievepark12
"""

import numpy as np
from matplotlib import pyplot as plt

#solve y''=-y with RK4


def logistic(x,y):
    dydx=y/(1+x**2)
    return dydx


def f(x,y): 
    dydx=np.asarray([y[1],-y[0]])
    return dydx

def rk4_step(fun,x,y,h):
    k1=fun(x,y)*h
    k2=h*fun(x+h/2,y+k1/2)
    k3=h*fun(x+h/2,y+k2/2)
    k4=h*fun(x+h,y+k3)
    dy=(k1+2*k2+2*k3+k4)/6
    return y+dy

# Everything above this point was taken from Sievers rk4.py script because
# that's just the classic rk4 evaulation

# Here's our double step rk4 evaluation functon. For formulation see num 
# recipes section 17.2
def rk4_stepd(fun,x,y,h):
    y1 = rk4_step(fun,x,y,h)
    y2a = rk4_step(fun,x,y,h/2)
    y2b = rk4_step(fun,x+h/2,y2a,h/2)
    return (y2b+(y2b-y1)/15)


# First integrate using standard method, 200 steps
y0=1
x1=np.linspace(-20,20,200)
h=np.median(np.diff(x1))
y1=np.zeros(len(x1))
y1[0]=y0
for i in range(len(x1)-1):
    y1[i+1]=rk4_step(logistic,x1[i],y1[i],h)

# Now integrate with the modified rk4, only 66 steps because each evaluation
# does 3 rk4 evaluations
y0=1
x2=np.linspace(-20,20,66)
h=np.median(np.diff(x2))
y2=np.zeros(len(x2))
y2[0]=y0
for i in range(len(x2)-1):
    y2[i+1]=rk4_stepd(logistic,x2[i],y2[i],h)    

# Compare error to real functions
yreal1 = np.e**(np.arctan(x1)+np.arctan(20))
yreal2 = np.e**(np.arctan(x2)+np.arctan(20))
y1err = np.abs(yreal1-y1)
y2err = np.abs(yreal2-y2)

# Plot out function
plt.figure(1)
plt.title('rk4 evaluation of dy/dx=y/(1+x^2), y(-20)=1')
plt.plot(x1,y1,'k', label = 'True func')
plt.plot(x1,y1,'rs', markerfacecolor='none', label = 'Reg rk4')
plt.plot(x2,y2,'go', markerfacecolor='none', label = 'Mod rk4')
plt.legend()
plt.savefig('P1Fig1.png')

# Plot out error
plt.figure(2)
plt.title('rk4 error')
plt.semilogy(x1,y1err,'r', label = 'Reg rk4')
plt.semilogy(x2,y2err,'g', label = 'Mod rk4')
plt.legend()
plt.savefig('P1Fig2.png')


y1meanerr = np.mean(y1err)
y2meanerr = np.mean(y2err)

# Print mean error
print('Mean err for regular rk4 is', y1meanerr)
print('Mean err for modified rk4 is', y2meanerr)
print('The modified rk4 is more accurate')
