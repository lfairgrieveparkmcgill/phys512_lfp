#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 22:42:05 2021

@author: lfairgrievepark12
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

# First we are going to do the cos fitting
x=np.linspace(-3.14/2,3.14/2,100)
dx=x[1]-x[0]
y=np.cos(x)

# cubic polynomial fitting modelled after lecture 2, cubic_interpolation.py code
# Going to ignore the first2 and last2 points for fitting so we don't get edge
# problems
xx=np.linspace(x[2],x[-3],100)
y_true=np.cos(xx)

yypoly=np.empty(len(xx))
for i in range(len(xx)):
    ind=(xx[i]-x[0])/dx
    ind=int(np.floor(ind))
    x_use=x[ind-1:ind+3]
    y_use=y[ind-1:ind+3]
    p=np.polyfit(x_use,y_use,3)
    yypoly[i]=np.polyval(p,xx[i])
        
# Spline interpolation 
spln=interpolate.splrep(x,y)
yyspline=interpolate.splev(xx,spln)

# Rational function interpolation  modelled on lecture 3 ratfit_exact.py code
# I modified it so like the polynomial fitting, it fits it in sections
# instead of just fitting the entire function with a rational function like
# the code shown in class
def rat_eval(p,q,x):
    top=0
    for i in range(len(p)):
        top=top+p[i]*x**i
    bot=1
    for i in range(len(q)):
        bot=bot+q[i]*x**(i+1)
    return top/bot

def rat_fit(x,y,n,m):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i
    pars=np.dot(np.linalg.inv(mat),y)
    p=pars[:n]
    q=pars[n:]
    return p,q


n=2
m=3
yyrat=np.empty(len(xx))
for i in range(len(xx)):
    ind=(xx[i]-x[0])/dx
    ind=int(np.floor(ind))
    x_use=x[ind-1:ind+3]
    y_use=y[ind-1:ind+3]
    p,q=rat_fit(x_use,y_use,n,m)
    yyrat[i]=rat_eval(p,q,xx[i])

print('for cos')
print('error in cubic poly is ',np.std(yypoly-y_true))
print('error in cubic spline is ',np.std(yyspline-y_true))
print('error in rat func is ',np.std(yyrat-y_true))

plt.figure(1)
plt.title('cos')
plt.plot(xx,y_true,'k')
plt.plot(xx,yypoly,'r')
plt.plot(xx,yyspline,'b')
plt.plot(xx,yyrat,'g')


##############################################################

# Sorry, should of defined a function instead of writing all this again but
# I was running out of time
x=np.linspace(-1,1,100)
dx=x[1]-x[0]
y=1/(1+x**2)

xx=np.linspace(x[2],x[-3],100)
y_true=1/(1+xx**2)

yypoly=np.empty(len(xx))
for i in range(len(xx)):
    ind=(xx[i]-x[0])/dx
    ind=int(np.floor(ind))
    x_use=x[ind-1:ind+3]
    y_use=y[ind-1:ind+3]
    p=np.polyfit(x_use,y_use,3)
    yypoly[i]=np.polyval(p,xx[i])
        

spln=interpolate.splrep(x,y)
yyspline=interpolate.splev(xx,spln)


n=2
m=3
yyrat=np.empty(len(xx))
for i in range(len(xx)):
    ind=(xx[i]-x[0])/dx
    ind=int(np.floor(ind))
    x_use=x[ind-1:ind+3]
    y_use=y[ind-1:ind+3]
    p,q=rat_fit(x_use,y_use,n,m)
    yyrat[i]=rat_eval(p,q,xx[i])
    

print('\nfor lorentzian')
print('error in cubic poly is ',np.std(yypoly-y_true))
print('error in cubic spline is ',np.std(yyspline-y_true))
print('error in rat func is ',np.std(yyrat-y_true))

plt.figure(2)
plt.title('lorentzian')
plt.plot(xx,y_true,'k')
plt.plot(xx,yypoly,'r')
plt.plot(xx,yyspline,'b')
plt.plot(xx,yyrat,'g')

# So this code is not going to produce the whacky lorentzian stuff for
# the rational function because it interpolates it in segments.
# This is because I didn't want to have to train my interpolation on
# m+n-1 points because that seems intentionally bad for reasonable m+n
# To recreate the whacky stuff, going to do the rat func interpolation 
# but not segmented

# Here's the rational function fitter redefined using pinv instead
def rat_fitpinv(x,y,n,m):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i
    pars=np.dot(np.linalg.pinv(mat),y)
    p=pars[:n]
    q=pars[n:]
    return p,q

# Okay fitting lorentzian with one rational function using regular linalg.inv
n=4
m=5
x=np.linspace(-1,1,n+m-1)
y=1/(1+x**2)
p,q=rat_fit(x,y,n,m)
xx=np.linspace(5*x[0],5*x[-1],1001)
y_true=1/(1+xx**2)
pred=rat_eval(p,q,xx)

plt.figure(3)
plt.title('rat fit lorentzian')
plt.plot(xx,y_true,'g*')
plt.plot(xx,pred,'r')

print('\nfor lorentzian, one rational func, linalg.inv')
print('error in rat func is ',np.std(pred-y_true))
print('p are',p)
print('q are',q)

# Now fitting lorentzian with one rational function but with linalg.pinv
p,q=rat_fitpinv(x,y,n,m)
pred=rat_eval(p,q,xx)
plt.plot(xx,pred,'b')
plt.ylim([-1,2])


print('\nfor lorentzian, one rational func, linalg.pinv')
print('error in rat func is ',np.std(pred-y_true))
print('p are',p)
print('q are',q)

# As you can see the error is huge (not in expectations when we use linalg.inv  
# and tiny when we use linalg.pinv. Because the lorentzian is itself a rational
# function is can be completely described by the rational function fitter.
# Additionally, there is not one unique rational function that will describe
# the lorentzian, but an infinite number of them (p and q produced by rat fit
# function are different from traditional formulation of lorentzian but are
# still correct).

# According to documentation pinv calculates the pseudo (Moore-Penrose) inverse
# From https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse, a use of
# the M-P inverse is "to find the minimum (Euclidean) norm solution to a system 
# of linear equations with multiple solutions." The lorentzian has multiple 
# rational function solutions, requiring we use the M-P inverse. If we don't
# crazy stuff happens and gets blown up when taking the regular inverse
# resulting in a very incorrect answer when using regular inv.

# Another way of thinking about this is since we expect to have some zero
# eigen values because there are exact rational function solutions. The regular
# inverse function doesn't work as you cannot take the inverse of a matrix with
# eigen values of zero