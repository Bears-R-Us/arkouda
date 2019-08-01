#!/usr/bin/env python3                                                         

import math
import gc
import sys
import time

import arkouda as ak

ak.v = False
if len(sys.argv) > 1:
    ak.connect(server=sys.argv[1], port=sys.argv[2])
else:
    ak.connect()

N = 2*10**8
n = 6

print("array size = ",N)
print("times to time = ",n)

print("arkouda stream")
a = ak.ones(N)
b = ak.ones(N)
alpha = 1.0

start = time.time()
for i in range(n):
    c = a+b*alpha
end = time.time()
ttime = (end - start) / n

print("secs = ",ttime)
bytes_per_sec = (c.size * c.itemsize * 3) / ttime
print("GiB/sec = ",bytes_per_sec/2**30)

ak.shutdown()

