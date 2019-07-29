#!/usr/bin/env python3                                                         

import importlib
import numpy as np
import math
import gc
import sys

import arkouda as ak

ak.v = False
if len(sys.argv) > 1:
    ak.connect(server=sys.argv[1], port=sys.argv[2])
else:
    ak.connect()

a = ak.array(np.arange(0,10,1))
b = ak.arange(0,10,1)
print(a,b)
c = a == b
print(type(c),c)
print(c.all())

ak.shutdown()
