#python routine to help show how broken the C 
#standard library random number generator is.
#generate a bunch of random triples.  If plotted
#correctly, it becomes obvious they aren't 
#anywhere close to random.

import numpy as np
import ctypes
import numba as nbc
import time
from matplotlib import pyplot as plt



def get_rands_nb(vals):
    n=len(vals)
    for i in range(n):
        vals[i]=np.random.randint(0,int(2**31),1)
    return vals

def get_rands(n):
    vec=np.empty(n)
    get_rands_nb(vec)
    return vec

n=30000000
vec=get_rands(int(n*3))   
#vv=vec&(2**16-1)

vv=np.reshape(vec,[n,3])
vmax=np.max(vv,axis=1)

maxval=1e8
vv2=vv[vmax<maxval,:]

f=open('rand_points_python.txt','w')
for i in range(vv2.shape[0]):
    myline=repr(vv2[i,0])+' '+repr(vv2[i,1])+' '+ repr(vv2[i,2])+'\n'
    f.write(myline)
f.close()
print('done')

