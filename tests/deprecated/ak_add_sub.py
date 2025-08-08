#!/usr/bin/env python3

import gc
import importlib
import math
import sys

import numpy as np

import arkouda as ak

ak.verbose = False
if len(sys.argv) > 1:
    ak.connect(server=sys.argv[1], port=sys.argv[2])
else:
    ak.connect()

N = 10**9
a = ak.ones(N, dtype="int64")
b = ak.ones(N, dtype="int64")
print(a, b)

c = a + b
d = a - b
print(c)
print(d)

ak.shutdown()
