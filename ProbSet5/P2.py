#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 13:58:58 2021

@author: lfairgrievepark12
"""

import numpy as np
import matplotlib.pyplot as plt

def corr(f,g):
    ft1 = np.fft.fft(f)
    ft2 = np.fft.fft(g)
    return np.real(np.fft.fftshift(np.fft.ifft(ft1*np.conjugate(ft2))))

x = np.linspace(-10,10.01,100)
f = np.exp(-0.5*x**2)
h = corr(f,f)
plt.plot(h)
plt.savefig('P2.png')
