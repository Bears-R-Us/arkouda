#!/usr/bin/env python3

import importlib
import numpy as np
import math
import gc
import sys

import arkouda as ak

ak.verbose = False
if len(sys.argv) > 1:
    ak.connect(server=sys.argv[1], port=sys.argv[2])
else:
    ak.connect()

a = ak.arange(0, 10, 1)
b = np.linspace(10, 20, 10)
c = ak.array(b)
d = a + c
e = d.to_ndarray()
    
a = ak.ones(10)
a[::2] = 0
print(a)

a = ak.ones(10)
b = ak.zeros(5)
a[1::2] = b
print(a)

a = ak.zeros(10) # float64
b = ak.arange(0,10,1) # int64
a[:] = b # cast b to float64
print(b,b.dtype)
print(a,a.dtype)

a = ak.randint(0,2,10,dtype=ak.bool)
a[1] = True
a[2] = True
print(a)

a = ak.ones(10,dtype=ak.int64)
iv = ak.arange(0,10,1)[::2]
a[iv] = 10
print(a)

a = ak.ones(10)
iv = ak.arange(0,10,1)[::2]
a[iv] = 10.0
print(a)

a = ak.ones(10,dtype=ak.bool)
iv = ak.arange(0,10,1)[::2]
a[iv] = False
print(a)

a = ak.ones(10,dtype=ak.bool)
iv = ak.arange(0,5,1)
b = ak.zeros(iv.size,dtype=ak.bool)
a[iv] = b
print(a)

a = ak.randint(10,20,10)
print(a)
iv = ak.randint(0,10,5)
print(iv)
b = ak.zeros(iv.size,dtype=ak.int64)
a[iv] = b
print(a)

ak.verbose = False
a = ak.randint(10,30,40)
vc = ak.value_counts(a)
print(vc[0].size,vc[0])
print(vc[1].size,vc[1])

ak.verbose = False

a = ak.arange(0,10,1)
b = a[a<5]
a = ak.linspace(0,9,10)
b = a[a<5]
print(b)

ak.verbose = True
ak.pdarrayIterThresh = 1000
a = ak.arange(0,10,1)
print(list(a))

ak.verbose = False
a = ak.randint(10,30,40)
u = ak.unique(a)
h = ak.histogram(a,bins=20)
print(a)
print(h.size,h)
print(u.size,u)

ak.verbose = False
a = ak.randint(10,30,50)
h = ak.histogram(a,bins=20)
print(a)
print(h)

ak.verbose = False
a = ak.randint(0,2,50,dtype=ak.bool)
print(a)
print(a.sum())

ak.verbose = False
a = ak.linspace(101,102,100)
h = ak.histogram(a,bins=50)
print(h)

ak.verbose = False
a = ak.arange(0,100,1)
h = ak.histogram(a,bins=10)
print(h)

ak.verbose = False
a = ak.arange(0,99,1)
b = a[::10] # take every tenth one
b = b[::-1] # reverse b
print(a)
print(b)
c = ak.in1d(a,b) # put out truth vector
print(c)
print(a[c]) # compress out false values

ak.verbose = False
a = np.ones(10,dtype=np.bool)
b = np.arange(0,10,1)

np.sum(a),np.cumsum(a),np.sum(b<4),b[b<4],b<5

ak.verbose = False
# currently... ak pdarray to np array
a = ak.linspace(0,9,10)
b = np.array(list(a))
print(a,a.dtype,b,b.dtype)

a = ak.arange(0,10,1)
b = np.array(list(a))
print(a,a.dtype,b,b.dtype)

a = ak.ones(10,dtype=ak.bool)
b = np.array(list(a))
print(a,a.dtype,b,b.dtype)

ak.verbose = False

b = np.linspace(1,10,10)
a = np.arange(1,11,1)
print(b/a)

ak.verbose = False

#a = np.ones(10000,dtype=np.int64)
a = np.linspace(0,99,100)
#a = np.arange(0,100,1)
print(a)

ak.verbose = False
print(a.__repr__())
print(a.__str__())

ak.verbose = False
print(a)
print(type(a), a.dtype, a.size, a.ndim, a.shape, a.itemsize)

ak.verbose = False
a = ak.arange(0,100,1)

ak.verbose = False
print(a)

ak.verbose = False

b = ak.linspace(0,99,100)
print(b.__repr__())

ak.verbose = False
b = ak.linspace(0,9,10)
a = ak.arange(0,10,1)
print(a.name, a.size, a.dtype, a)
print(b.name, b.size, b.dtype, b)
print(ak.info(a+b))


ak.verbose = False

c = ak.arange(0,10,1)
print(ak.info(c))
print(c.name, c.dtype, c.size, c.ndim, c.shape, c.itemsize)
print(c)

ak.verbose = False

print(5+c + 5)

c = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
c = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19.1])
print(c.__repr__(), c.dtype.__str__(), c.dtype.__repr__())

ak.verbose = False
a = np.ones(9)
b = np.arange(1,10,1)
print(a.dtype,b.dtype)
c = ak.ones(9)
d = ak.arange(1,10,1)
print(c.dtype,d.dtype)
y = a/b
z = c/d
print("truediv  \nnp out:",y,"\nak out:",z)
print(y[5],z[5],y[5] ==z[5])
y = a//b
z = c//d
print("floordiv \nnp out:",y,"\nak out:",z)
print(y[5],z[5],y[5] ==z[5])

ak.verbose = False

c = ak.arange(1,10,1)
c //= c
print(c)
c += c
print(c)
c *= c
print(c)

ak.verbose = False

a = np.ones(9,dtype=np.int64)
b = np.ones_like(a)
print(b)

ak.verbose = False

a = ak.ones(9,dtype=ak.int64)
b = ak.ones_like(a)
print(b)

ak.verbose = False

a = ak.arange(0,10,1)
b = np.arange(0,10,1)

print(a[5] == b[5])

ak.verbose = False

a = ak.arange(0,10,1)
b = np.arange(0,10,1)

a[5] = 10.2
print(a[5])

ak.verbose = False

a = ak.arange(0,10,1)
b = np.arange(0,10,1)
#print((a[:]),b[:])
#print(a[1:-1:2],b[1:-1:2])
#print(a[0:10:2],b[0:10:2])
print(a[4:20:-1],b[4:20:-1])
print(a[:1:-1],b[:1:-1])

ak.verbose = False
d = ak.arange(1,10,1)
#d.type.__class__,d.name,d.isnative,np.int64.__class__,bool
ak.info(d)
#dir(d)

ak.verbose = False

a = ak.ones(10,dtype=ak.bool)
print(a[1])

ak.verbose = False

a = ak.zeros(10,dtype=ak.bool)
print(a[1])

ak.verbose = False
a = ak.ones(10,dtype=ak.bool)
a[4] = False
a[1] = False
print(a)
print(a[::2])
print(a[1])
a = ak.ones(10,dtype=ak.int64)
a[4] = False
a[1] = False
print(a)
print(a[::2])
print(a[1])
a = ak.ones(10)
a[4] = False
a[1] = False
print(a)
print(a[::2])
print(a[1])

ak.verbose = False

a = ak.arange(0,10,1)
b = list(a)
print(b)

a = a<5
b = list(a)
print(b)

ak.verbose = False
a = ak.linspace(1,10,10)
print(ak.abs(a))
print(ak.log(a))
print(ak.exp(a))
a.fill(math.e)
print(ak.log(a))

type(bool),type(np.bool),type(ak.bool),type(True)

ak.verbose = False
a = ak.linspace(0,9,10)
print(a,ak.any(a),ak.all(a),ak.all(ak.ones(10,dtype=ak.float64)))
b = a<5
print(b,ak.any(b),ak.all(b),ak.all(ak.ones(10,dtype=ak.bool)))
c = ak.arange(0,10,1)
print(c,ak.any(c),ak.all(c),ak.all(ak.ones(10,dtype=ak.int64)))
print(a.any(),a.all(),b.any(),b.all())

ak.verbose = False
a = ak.linspace(0,9,10)
ak.sum(a)
b = np.linspace(0,9,10)
print(ak.sum(a) == np.sum(b),ak.sum(a),np.sum(b),a.sum(),b.sum())

ak.verbose = False
a = ak.linspace(1,10,10)
b = np.linspace(1,10,10)
print(ak.prod(a) == np.prod(b),ak.prod(a),np.prod(b),a.prod(),b.prod())

ak.verbose = False

a = np.arange(0,20,1)
b = a<10
print(b,np.sum(b),b.sum(),np.prod(b),b.prod(),np.cumsum(b),np.cumprod(b))
print()
b = a<5
print(b,np.sum(b),b.sum(),np.prod(b),b.prod(),np.cumsum(b),np.cumprod(b))
print()
a = ak.arange(0,20,1)
b = a<10
print(b,ak.sum(b),b.sum(),ak.prod(b),b.prod(),ak.cumsum(b),ak.cumprod(b))
b = a<5
print(b,ak.sum(b),b.sum(),ak.prod(b),b.prod(),ak.cumsum(b),ak.cumprod(b))

ak.verbose = False
a = ak.arange(0,10,1)
iv = a[::-1]
print(a,iv,a[iv])

ak.verbose = False
a = ak.arange(0,10,1)
iv = a[::-1]
print(a,iv,a[iv])

ak.verbose = False
a = ak.linspace(0,9,10)
iv = ak.arange(0,10,1)
iv = iv[::-1]
print(a,iv,a[iv])

ak.verbose = False
a = np.arange(0,10,1)
iv = a[::-1]
print(a,iv,a[iv])

ak.verbose = False
a = ak.arange(0,10,1)
b = a<20
print(a,b,a[b])

ak.verbose = False
a = ak.arange(0,10,1)
b = a<5
print(a,b,a[b])

ak.verbose = False
a = ak.arange(0,10,1)
b = a<0
print(a,b,a[b])

ak.verbose = False
a = ak.linspace(0,9,10)
b = a<5
print(a,b,a[b])

ak.verbose = False
N = 2**23 # 2**23 * 8 == 64MiB
A = ak.linspace(0,N-1,N)
B = ak.linspace(0,N-1,N)

C = A+B
print(ak.info(C),C)

# turn off verbose messages from arkouda package
ak.verbose = False
# set pdarrayIterThresh to 0 to only print the first 3 and last 3 of pdarray
ak.pdarrayIterThresh = 0
a = ak.linspace(0,9,10)
b = a<5
print(a)
print(b)
print(a[b])
print(a)

a = np.linspace(0,9,10)
b = a<5
print(a)
print(b)
print(a[b])
print(a)

ak.verbose = False
ak.pdarrayIterThresh = 0

a = ak.ones(10,ak.int64)
b = a | 0xff
print(a, b, a^b, b>>a, b<<1|1, 0xf & b, 0xaa ^ b, b ^ 0xff)
print(-a,~(~a))

a = ak.ones(10,dtype=ak.int64)
b = ~ak.zeros(10,dtype=ak.int64)
print(~a, -b)

a = np.ones(10,dtype=np.int64)
b = ~np.zeros(10,dtype=np.int64)
print(~a, -b)


ak.shutdown()

