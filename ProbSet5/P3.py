#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:50:03 2021

@author: lfairgrievepark12
"""

import numpy as np
import matplotlib.pyplot as plt

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
    
    # convolve to shift function
    h = conv(f,g)
    return(h)
    
def corr(f,g):
    ft1 = np.fft.fft(f)
    ft2 = np.fft.fft(g)
    return np.real(np.fft.fftshift(np.fft.ifft(ft1*np.conjugate(ft2))))

x = np.linspace(-10,10.01,100)
f = np.exp(-0.5*x**2)
h1 = corr(f,f)
h2 = corr(f,shifter(f,12))
h3 = corr(f,shifter(f,25))
h4 = corr(f,shifter(f,37))
h5 = corr(f,shifter(f,50))

plt.plot(h1, label = 'f corr f')
plt.plot(h2, label = 'f corr f shift 1/8')
plt.plot(h3, label = 'f corr f shift 1/4')
plt.plot(h4, label = 'f corr f shift 3/8')
plt.plot(h5, label = 'f corr f shift 1/2')
plt.legend()

plt.savefig('P3.png')