#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 18:56:31 2021

@author: lfairgrievepark12
"""
import numpy as np
import matplotlib.pyplot as plt


def conv_safe(f,g):
    # So this works the same way, except we pad each of the arrays with zeros
    # so they both have the length = sum of the length of the two arrays -1
    pad = abs(len(f)+len(g))
    f1 = np.pad(f,(0,pad-len(f)-1), 'constant')
    g1 = np.pad(g,(0,pad-len(g)-1), 'constant')

    ft1 = np.fft.fft(f1)
    ft2 = np.fft.fft(g1)

    return np.real(np.fft.ifft(ft1*ft2))

# We'll test it out on a gaussian defined over 100 points and a gaussian 
# defined over 200 points
x = np.linspace(-10,10,100)
x2 = np.linspace(-10,10,200)
f = np.exp(-0.5*x**2)
f2 =np.exp(-0.5*x2**2)
h1 = conv_safe(f,f2)


plt.plot(f/max(f),label='f1')
plt.plot(f2/max(f2),label='f2')
plt.plot(h1/max(h1),label='conv(f1,f2)')
plt.legend()
plt.savefig('P4.png')

