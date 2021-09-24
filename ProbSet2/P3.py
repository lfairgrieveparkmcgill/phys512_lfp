#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 22:37:52 2021

@author: lfairgrievepark12
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
x=np.linspace(0.5,1,100)
y=np.log2(x)

# Extracting and fitting 7th order chebyshev for log2(x) from 0.5 to 1 
# Order 7 was taken from trial and error that gives <1e-6 error
ccheb = np.polynomial.chebyshev.chebfit(x,y,7)
ypredcheb = np.polynomial.chebyshev.chebval(x,ccheb)

# Here's the fitted and "real" log2x
plt.figure(1)
plt.plot(x,y,'o',label='numpy log2(x)')
plt.plot(x,ypredcheb,'*',label='cheb approx. log2(x)')
plt.title('log base 2 of x')
plt.legend()

# Here's the error, looks below 1e-6 to me
plt.figure(2)
plt.plot(x,y-ypredcheb)
plt.title('log base 2 chebyshev error')


def mylogcheb(x):
    # So this function relies on some log identies
    # First of these is: loge(x) = log2(x)/log2(e)
    # Second is: given x = mantissa + 2**exp, log2(x) = log2(mantissa) + exp
    # As 0.5 <= mantissa <= 1, we can use our chebyshev trained on log2(x) for
    # 0.5 <= x <= 1 we can evaluate the log2 of any mantissa using chebval
    
    # First we split our x and e into mantissa and exponent
    x1,x2 = np.frexp(x)
    e1,e2 = np.frexp(np.e)
    
    # Then we calculate the log2 of x and e by using our log2 chevyshev
    # evaluator on the mantissa and adding the exponent
    log2x = np.polynomial.chebyshev.chebval(x1,ccheb) + x2
    log2e = np.polynomial.chebyshev.chebval(e1,ccheb) + e2
    
    # Then, as stated earlier loge(x) = log2(x)/log2(e)
    return log2x/log2e

# our chebyshev natural log evaluator performs quite similar to the built in 
# numpy log evaluator 
print('loge of e with mylogcheb is', mylogcheb(np.e))
print('loge of e with np.log is', np.log(np.e))
print('loge of 10 with mylogcheb is', mylogcheb(10))
print('loge of 10 with np.log is', np.log(10))

# Now we are going to compare to the performance of a legendre polynomial based
# log evaluator, same as before just using legendre instead of chebyshev
cleg = np.polynomial.legendre.legfit(x,y,7)
ypredleg = np.polynomial.legendre.legval(x,cleg)

chebmaxerr = max(abs(ypredcheb-y))
chebrmserr = mean_squared_error(ypredcheb, y, squared=False)
legmaxerr = max(abs(ypredleg-y))
legrmserr = mean_squared_error(ypredleg, y, squared=False)

# Performance is very similar, they provide tge same reult to within almost 8
# decimal places 
print('\nchebyshev log2(x), x= 0.5 to 1 max err is', chebmaxerr)
print('chebyshev log2(x), x= 0.5 to 1 rms err is', chebrmserr)
print('legendre log2(x), x= 0.5 to 1 max err is', legmaxerr)
print('legendre log2(x), x= 0.5 to 1 rms err is', legrmserr)

