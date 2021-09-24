#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 23:24:23 2021

@author: lfairgrievepark12
"""

import numpy as np


def lorentz(x):
    return 1/(1+x**2)

# First we'll define the upgraded recursion integrator
def integrate_adaptive(fun,x0,x1,tol,extra=None):
    # First execution of function proceeds as normal
    if extra==None:
        x=np.linspace(x0,x1,5)
        y=fun(x)
        # This is a counter to keep track of the number of f(x) evaluations
        integrate_adaptive.counter += 5
        dx=(x1-x0)/(len(x)-1)
        area1=2*dx*(y[0]+4*y[2]+y[4])/3 #coarse step
        area2=dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3 #finer step
        err=np.abs(area1-area2)
        if err<tol:
            return area2
        else:
            xmid=(x0+x1)/2
            # Will define the extra arguments for the next function call so that
            # we don't have to re-evaluate. After splitting it into left and right,
            # there will be three points in the next function call where we 
            # have already calculated the value, y[0] y[1] and y[2] for left
            # and y[2], y[3] and y[4] for right. Also dx is passed through as it
            # must be kept track of we no longer define a linear spaced x array
            # to evaluate
            left=integrate_adaptive(fun,x0,xmid,tol/2,extra=[y[0],y[1],y[2],dx])
            right=integrate_adaptive(fun,xmid,x1,tol/2,extra=[y[2],y[3],y[4],dx])
            return left+right
    
    # We will evaluate through this path every time after the first
    else:   
        # Generally y[0], y[2] and y[4] are now just extra[0], extra[1] and
        # extra[2] (which is function evaluated at min, middle and max value)
        # Just sub in those extras in their place
        x=np.array([x0+0.5*extra[3],x1-0.5*extra[3]])
        y=fun(x)
        # Only count 2 evaluations in this path as we only need to evaluate 2
        # points on this loop
        integrate_adaptive.counter += 2
        dx=extra[3]/2
        area1=2*dx*(extra[0]+4*extra[1]+extra[2])/3 #coarse step
        area2=dx*(extra[0]+4*y[0]+2*extra[1]+4*y[1]+extra[2])/3 #finer step
        err=np.abs(area1-area2)
        if err<tol:
            return area2
        else:
            xmid=(x0+x1)/2
            left=integrate_adaptive(fun,x0,xmid,tol/2,extra=[extra[0],y[0],extra[1],dx])
            right=integrate_adaptive(fun,xmid,x1,tol/2,extra=[extra[1],y[1],extra[2],dx])
            return left+right        

# Define points for lorentzian calc
x0=-100
x1=100


# We'll evaluate for a lorentizna between -100 and 100 and an exponential
# between 0 and 5. 
integrate_adaptive.counter = 0
ans =integrate_adaptive(lorentz,x0,x1,1e-7)
print('Improved recursive integrator:')
print('\nLorentz')
print('error in answer is',ans-(np.arctan(x1)-np.arctan(x0)))
print('f(x) calls = ',integrate_adaptive.counter)

integrate_adaptive.counter = 0
ans =integrate_adaptive(np.exp,0,5,1e-7)
print('\nexponential')
print('error in answer is',ans-(np.exp(5)-np.exp(0)))
print('f(x) calls = ',integrate_adaptive.counter)


# This is the function Prof coded in class
def integrate_adaptive(fun,x0,x1,tol):
    x=np.linspace(x0,x1,5)
    y=fun(x)
    integrate_adaptive.counter += 5
    dx=(x1-x0)/(len(x)-1)
    area1=2*dx*(y[0]+4*y[2]+y[4])/3 #coarse step
    area2=dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3 #finer step
    err=np.abs(area1-area2)
    if err<tol:
        return area2
    else:
        xmid=(x0+x1)/2
        left=integrate_adaptive(fun,x0,xmid,tol/2)
        right=integrate_adaptive(fun,xmid,x1,tol/2)
        return left+right


# Same as before, lets see how the old function compares
integrate_adaptive.counter = 0
ans =integrate_adaptive(lorentz,x0,x1,1e-7)
print('\n*********************************************')
print('\nOld recursive integrator:')
print('\nLorentz')
print('error in answer is',ans-(np.arctan(x1)-np.arctan(x0)))
print('f(x) calls = ',integrate_adaptive.counter)

integrate_adaptive.counter = 0
ans =integrate_adaptive(np.exp,0,5,1e-7)
print('\nexponential')
print('error in answer is',ans-(np.exp(5)-np.exp(0)))
print('f(x) calls = ',integrate_adaptive.counter)

# As expected, in both cases it takes our upgraded function 2/5 of the number 
# of f(x) calls as the original 