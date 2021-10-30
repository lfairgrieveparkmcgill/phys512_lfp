#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 10:12:36 2021

@author: lfairgrievepark12
"""

import numpy as np
import matplotlib.pyplot as plt

# Define our convolution function from John's notes
def conv(f,g):
    ft1 = np.fft.fft(f)
    ft2 = np.fft.fft(g)
    return np.real(np.fft.ifft(ft1*ft2))


def shifter(func,shift):
    # to shift, we convolve function with delta function where location of delta
    # is # array steps shifted
    f = func
    g = np.zeros(len(f))
    if shift >= 0:
        g[shift] = 1
    else:
        pass
    
    # convolve to shift function
    h = conv(f,g)
    return(h)
        
x = np.linspace(-10,10.01,101)
f = np.exp(-0.5*x**2)
h1 = shifter(f,50)

plt.plot(x,f,label = 'orig function')
plt.plot(x,h1,label='shifted by 1/2 array')
plt.legend()
plt.savefig('gauss_shift.png')

