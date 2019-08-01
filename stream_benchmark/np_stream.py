#!/usr/bin/env python3                                                         

import numpy as np
import math
import gc
import sys
import time

N = 2*10**8
n = 6

print("array size = ",N)
print("times to time = ",n)

print("numpy stream")
a = np.ones(N)
b = np.ones(N)
alpha = 1.0

start = time.time()
for i in range(n):
    c = a+b*alpha
end = time.time()
ttime = (end - start) / n

print("secs = ",ttime)
bytes_per_sec = (c.size * c.itemsize * 3) / ttime
print("GiB/sec = ",bytes_per_sec/2**30)
