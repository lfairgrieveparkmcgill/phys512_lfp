#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 23:35:36 2021

@author: lfairgrievepark12
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# Lets define all our different time constants to make things easier
hr=60**2
day=hr*24
yr=day*365.25

# Define our function, halflives are taken from that slide in class
def fun(x,y,halflife=[4.468e9*yr,24.10*day,6.70*hr,245500*yr,75380*yr,1600*yr,3.8235*day,\
            3.1*60, 26.8*60, 19.9*60, 164.3e-3,22.3*yr,5015*yr,138.376*day]):
    dydx=np.zeros(len(halflife)+1)
    
    for i in range(len(halflife)+1):
        # Equation for starting substance (U238)
        if i==0:
            dydx[i] = -y[i]/halflife[i]
        # Equations for intermediary substances
        elif i>0 and i<len(halflife):
            dydx[i] = y[i-1]/halflife[i-1]-y[i]/halflife[i]
        # Equation for final, stable substance (Pb206)
        else:
            dydx[i] = y[i-1]/halflife[i-1]
    return dydx*np.log(2)

# Only starting with uranium
y0 = np.asarray([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
x0=0
x1=5e9*yr
# Solve that differential equation, log spacing so things are nicer
# Going to use Radau because it worked the best in his example
ans = integrate.solve_ivp(fun,[x0,x1],y0,t_eval=np.logspace(1e-3,17.15,100000),method='Radau')

#
plt.figure(1)
plt.semilogy(ans.t/yr,ans.y[-1]/ans.y[0])
plt.ylim([.001,1])
plt.xlim([0,10**17.15/yr])
plt.title('Pb206/U238 ratio')
plt.xlabel('Time (years)')
plt.savefig('P2Fig1.png')
# This result sense analytically. The ratio of U238/Pb206 starts out at an extremely
# high value as almost no uranium would have fully decayed down to lead. By one half
# life of U238 the ratio of U238/Pb206 is 1 as there are near equal amounts of the initial
# and final decay products

plt.figure(2)
plt.semilogy(ans.t/yr,ans.y[4]/ans.y[3])
plt.ylim([0.001,1])
plt.xlim([0,5e5])
plt.title('Th230/U234 ratio')
plt.xlabel('Time (years)')
plt.savefig('P2Fig2.png')

