#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 22:27:06 2021

@author: lfairgrievepark12
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
dat=np.loadtxt('lakeshore.txt')

def lakeshore(V,data):
    # Need to reverse both arrays as cubic spline doesn't like it when
    # values aren't increasing
    temp = data[:,0][::-1]
    volt = data[:,1][::-1]
    
    # Just doing a cubic spline cause I have a feeling its the best 
    # (not because I thought it was easy to implement!)
    spln = interpolate.splrep(volt,temp)
    Tout = interpolate.splev(V,spln)

    # Bootstrapping error estimation. Most of this was taken from Rigels 
    # Sept 16th bootstrapping demo
    
    # Defining a random seed for consistent results. Not sure how to choose
    # a good number of samples/resamples so went random...
    rng = np.random.default_rng(1
    resamples = 10
    samples = 100
    error=[]
    
    for i in range(resamples):
        # Going to take 100 random samples 10 times, compare the true
        # values to the interpolated calculated variables and take
        # the average error to be the interpolation error
        indices = list(range(volt.size))
        tointerp = rng.choice(indices, size=samples, replace = False)
        tointerp.sort()
        check = [i for i in indices if not (i in tointerp)]
        newinterp = interpolate.splrep(volt[tointerp],temp[tointerp])
        interptemp = interpolate.splev(volt[check],newinterp)
        realtemp = temp[check]
        error.append(np.abs(interptemp-realtemp))
    err = np.mean(error)

    return (Tout, err)

# Works for arrays and for single values
(T, err) = lakeshore(np.array([1.2,1.3,1.1]),dat)
print('temps/temp are/is', T, 'error is', err)

(T, err) = lakeshore(1.2,dat)
print('temps/temp are/is', T, 'error is', err)




        