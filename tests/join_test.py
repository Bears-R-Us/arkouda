#!/usr/bin/env python3                                                         

import importlib
import numpy as np
import math
import gc
import sys

import arkouda as ak

print(">>> Sanity checks on the arkouda_server")

ak.verbose = False
if len(sys.argv) > 1:
    ak.connect(server=sys.argv[1], port=sys.argv[2])
else:
    ak.connect()

N = 1000

a1 = ak.ones(N,dtype=np.int64)
a2 = ak.arange(0,N,1)
t1 = a1
t2 = a1 * 10
dt = 10

# should get N*N answers
I,J = ak.join_on_eq_with_dt(a1,a1,a1,a1,dt,"true_dt",result_limit=N*N)
print(I,J)
if (I.size == N*N) and (J.size == N*N):
    print("passed!")
else:
    print("failed!")

# should get N answers
I,J = ak.join_on_eq_with_dt(a2,a1,t1,t2,dt,"true_dt")
print(I,J)
if (I.size == N) and (J.size == N):
    print("passed!")
else:
    print("failed!")

# should get N answers
I,J = ak.join_on_eq_with_dt(a2,a1,t1,t2,dt,"abs_dt")
print(I,J)
if (I.size == N) and (J.size == N):
    print("passed!")
else:
    print("failed!")

# should get N answers
I,J = ak.join_on_eq_with_dt(a2,a1,t1,t2,dt,"pos_dt")
print(I,J)
if (I.size == N) and (J.size == N):
    print("passed!")
else:
    print("failed!")

# should get 0 answers
# N^2 matches but 0 within dt window
dt = 8
I,J = ak.join_on_eq_with_dt(a1,a1,t1,t1*10,dt,"abs_dt")
print(I,J)
if (I.size == 0) and (J.size == 0):
    print("passed!")
else:
    print("failed!")

# should get 0 answers
# N matches but 0 within dt window
dt = 8
I,J = ak.join_on_eq_with_dt(a2,a1,t1,t2,dt,"abs_dt")
print(I,J)
if (I.size == 0) and (J.size == 0):
    print("passed!")
else:
    print("failed!")

# should get 0 answers
# N matches but 0 within dt window
dt = 8
I,J = ak.join_on_eq_with_dt(a2,a1,t1,t2,dt,"pos_dt")
print(I,J)
if (I.size == 0) and (J.size == 0):
    print("passed!")
else:
    print("failed!")

