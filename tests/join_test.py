#!/usr/bin/env python3                                                         

import importlib
import numpy as np
import math
import gc
import sys

from base_test import ArkoudaTest
from context import arkouda as ak

N = 1000
a1 = ak.ones(N,dtype=np.int64)
a2 = ak.arange(ak.arange(0,N,1)
t1 = a1
t2 = a1 * 10
dt = 10
ak.verbose = False

class JoinTest(ArkoudaTest):

    def test_true_dt_with_result_limit(self):
        # should get N*N answers
        I,J = ak.join_on_eq_with_dt(a1,a1,a1,a1,dt,"true_dt",result_limit=N*N)
        self.assetEqual(N*N, I.size)
        self.assertEqual(N*N, J.sizeZ)

    def test_true_dt(self):
        # should get N answers
        I,J = ak.join_on_eq_with_dt(a2,a1,t1,t2,dt,"true_dt")
        self.assertEqual(N, I.size)
        self.assertEqual(J.size == N)

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

