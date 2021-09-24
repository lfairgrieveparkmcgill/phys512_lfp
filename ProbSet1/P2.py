#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 22:47:30 2021

@author: lfairgrievepark12
"""

import numpy as np

# This function will return the 3rd derivative with 2nd order accuracy
# for use in determining error for derivative calculator. Coefficients
# were taken from https://en.wikipedia.org/wiki/Finite_difference_coefficient
def f3d(fun, x, dx):
     return (-0.5*fun(x-2*dx)+fun(x-dx)-fun(x+dx)+0.5*fun(x+2*dx))/dx**3
 
# Here's our function that actually takes the derivative
def ndiff(fun,x,full=False):

    eps=2**-52
    
    # Since we have a chicken and egg scenario, I sort of randomly took a
    # dx. I know eps**(1/2) is best but it was giving bad results for 3rd
    # derivative (and therefore bad error accuracy). This seems to give the
    # best for first and 3rd derivative
    dx = (eps)**(1/3)

    # Defining our derivative calculation
    fcalc = (fun(x+dx) - fun(x-dx))/(2*dx)
    
    #Error calc, see siever notes from lecture 1 
    err = fun(x)*eps/dx + f3d(fun,x,dx)*dx**2
    
    if full == True:
        return (fcalc, dx, err)
    
    else:
        return (fcalc)

# Proof of function, exp function allows for easy check of result
(fout, dx, err) = ndiff(np.exp, 5, full=True)
print(fout,dx,err)

(fout) = ndiff(np.exp, 5)
print(fout)