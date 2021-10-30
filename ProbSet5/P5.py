#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:30:05 2021

@author: lfairgrievepark12
"""
import numpy as np
import matplotlib.pyplot as plt

# Decide our non integer sin freq, N, define our sin wave and our k
kp=6.5
N=1000
x=np.linspace(0,1,N)
f=np.sin(2*np.pi*kp*x)
k=np.fft.fftfreq(len(x),x[1]-x[0])

# take the fft
fft = abs(np.fft.fft(f))

# analytic format of function, based on exponential form of sin
def analfft(kt,kp,N):
    fft = np.zeros(len(kt))
    for i in range(len(kt)):
        fft1 = (1-np.exp(-2*np.pi*1j*(kt[i]+kp)))/(1-np.exp(-2*np.pi*1j*(kt[i]+kp)/(N)))
        fft2 = (1-np.exp(-2*np.pi*1j*(kt[i]-kp)))/(1-np.exp(-2*np.pi*1j*(kt[i]-kp)/(N)))
        fft[i] = (fft1-fft2)/(2j)
    return abs(fft)
        
analfft = analfft(k,kp,N)

# They seem to match well
plt.figure(1)
plt.plot(k,fft,'-', label = 'true fft')
plt.plot(k,analfft,'--', label = 'analytic fft')
plt.xlim([0,10])
plt.legend()
plt.savefig('P5c.png')
analerr = np.mean(abs(fft-analfft))
print('analytic fft vs. true fft mean error is', analerr)

# define window function and calculate windowed fft
fwin = f*(0.5-0.5*np.cos(2*np.pi*x/N))
fftfwin = abs(np.fft.fft(fwin,N))

plt.figure(2)
plt.plot(k,fft/max(fft),'-', label = 'true fft')
plt.plot(k,fftfwin/max(fftfwin), label = 'window fft')
plt.xlim([0,10])
plt.legend()
plt.savefig('P5d.png')


# calculate fft of window function
fftwin = abs(np.fft.fft((0.5-0.5*np.cos(2*np.pi*x/N))))

plt.figure(3)
plt.plot(k,fftwin*N,'-')
plt.xlim([-10,10])
plt.title('fft of window function scaled to N')
plt.savefig('P5e.png')


