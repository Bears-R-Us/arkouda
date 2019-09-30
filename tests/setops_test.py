#!/usr/bin/env python3

import arkouda as ak
import numpy as np
import pandas as pd

SIZE = 1000000

ak.connect()
a = ak.randint(0,100*SIZE,SIZE)
b = ak.randint(0,100*SIZE,SIZE)

set_union = ak.union1d(a,b)
passed = ak.all(ak.in1d(set_union,a) | ak.in1d(set_union,b))
print("union1d passed test: ",passed)

set_intersection = ak.intersect1d(a,b)
passed = ak.all(ak.in1d(set_intersection,a) & ak.in1d(set_intersection,b))
print("intersect1d passed test: ",passed)

set_difference = ak.setdiff1d(a,b)
passes = ak.all(ak.in1d(set_difference,a) & ak.in1d(set_difference,b,invert=True))
print("setdiff1d passed test: ",passed)

set_xor = ak.setxor1d(a,b)
passes = ak.all(ak.in1d(set_xor, set_intersection, invert=True))
print("setxor1d passed test: ",passed)

ak.disconnect()
