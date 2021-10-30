#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 09:21:41 2021

@author: lfairgrievepark12
"""

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

kp=1.37
x=np.linspace(0,10,101)
f=np.sin(2*np.pi*kp*x)
fs=10

N=1024
nVals = np.arange(start = 0,stop = N)
fVals=np.arange(start = -N/2,stop = N/2)*fs/N
print(fVals)

fft = abs(np.fft.fftshift(np.fft.fft(f,N)))


def analfft(kt,kp,N):
    fft = np.zeros(len(kt),dtype=complex)
    for i in range(len(kt)):
        fft1 = (1-np.exp(-2*np.pi*1j*(kt[i]+kp)))/(1-np.exp(-2*np.pi*1j*(kt[i]+kp)/(N)))
        fft2 = (1-np.exp(-2*np.pi*1j*(kt[i]-kp)))/(1-np.exp(-2*np.pi*1j*(kt[i]-kp)/(N)))
        fft[i] = (fft1-fft2)/(2j)
    return abs(fft)  

analfft = analfft(fVals,kp,N)

plt.figure(1)
plt.plot(fVals,fft/max(fft),label='true_fft')
plt.plot(fVals,analfft/max(analfft),label='analytic_fft')

fwin = f*(0.5-0.5*np.cos(2*np.pi*x/N))
fftfwin = abs(np.fft.fftshift(np.fft.fft(fwin,N)))

plt.figure(2)
plt.plot(fVals,fft/max(fft),label='fft')
plt.plot(fVals,fftfwin/max(fftfwin),label='fft_window')
'''
fftwin = np.fft.fft((0.5-0.5*np.cos(2*np.pi*x/N)),N)
plt.figure(3)
plt.plot(fVals,np.(fftwin))
print(fftwin)
'''
