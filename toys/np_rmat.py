#!/usr/bin/env python3

import numpy as np
import pandas as pd
import math
import gc
import sys
import matplotlib.pyplot as plt

def gen_rmat_edges(lgNv, Ne_per_v, p):
    # number of vertices
    Nv = 2**lgNv
    # number of edges
    Ne = Ne_per_v * Nv
    # probabilities
    a = p
    b = (1.0 - a)/ 3.0
    c = b
    d = b
    # init edge arrays
    ii = np.ones(Ne,dtype=np.int)
    jj = np.ones(Ne,dtype=np.int)
    # quantites to use in edge generation loop
    ab = a+b
    c_norm = c / (c + d)
    a_norm = a / (a + b)
    # generate edges
    for ib in range(1,lgNv):
        ii_bit = (np.random.rand(Ne) > ab)
        jj_bit = (np.random.rand(Ne) > (c_norm * ii_bit + a_norm * (~ ii_bit)))
        ii = ii + ((2**(ib-1)) * ii_bit)
        jj = jj + ((2**(ib-1)) * jj_bit)
    # generate permutation for new vertex numbers(names)
    ir = np.argsort(np.random.rand(Nv))
    # renumber(rename) vertices
    ii = ir[ii] # rename first vertex
    jj = ir[jj] # rename second vertex
    #
    # maybe: remove edges which are self-loops???
    # 
    # return pair of ndarrays
    return (ii,jj)


lgNv = 16
Ne_per_v = 4
p = 0.1
(ii,jj) = gen_rmat_edges(lgNv, Ne_per_v, p)

print("ii = ", (ii.size, ii))
print("ii(min,max) = ", (ii.min(), ii.max()))
print("jj = ", (jj.size, jj))
print("jj(min,max) = ", (jj.min(), jj.max()))

#plt.scatter(ii,jj)
#plt.show()

df = pd.DataFrame({"ii":ii, "jj":jj})

grps = df.groupby("ii")
cts = grps.agg("count")
#print(cts)
print("counts",(cts["jj"].min(),cts["jj"].max(), cts["jj"].mean()))
nBins = cts["jj"].max()
plt.hist(cts["jj"],bins=nBins)
plt.yscale('log')
plt.show()

nu = grps.agg("nunique")
#print(nu)
print("nunique",(nu["jj"].min(), nu["jj"].max(), nu["jj"].mean()))
nBins = nu["jj"].max()
plt.hist(nu["jj"],bins=nBins)
plt.yscale('log')
plt.show()
