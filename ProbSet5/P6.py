#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 15:14:43 2021

@author: lfairgrievepark12
"""

import numpy as np
import matplotlib.pyplot as plt

x=1000
# Generate the random walk
rand = np.cumsum(np.random.randn(x))

# Generate k modes and clip k=0 cause it causes problems
k = np.fft.fftfreq(x)[1:]
k2 = k**-2
ps = abs(np.fft.fft(rand))**2

plt.semilogy(ps/np.mean((ps)), label='normalized rand walk PSD')
plt.semilogy(k2/np.mean(k2), label = 'normalized k^2')
plt.legend()
plt.savefig('P6.png')
            