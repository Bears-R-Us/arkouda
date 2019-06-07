#!/usr/bin/env python3

import arkouda as ak
import numpy as np
import pandas as pd
from time import time

def measure_runtime(length, ncat, op, dtype, per_locale=True):
    keys = ak.randint(0, ncat, length)
    if dtype == 'int64':
        vals = ak.randint(0, length//ncat, length)
    elif dtype == 'bool':
        vals = ak.zeros(length, dtype='bool')
        for i in np.random.randint(0, length, ncat//2):
            vals[i] = True
    else:
        vals = ak.linspace(-1, 1, length)        
    print("Local groupby", end=' ')
    start = time()
    lg = ak.GroupBy(keys, per_locale)
    lgtime = time() - start
    print(lgtime)
    print("Local reduce", end=' ')
    start = time()
    lk, lv = lg.aggregate(vals, op)
    lrtime = time() - start
    print(lrtime)
    return lgtime, lrtime

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 5:
        print(f"Usage: {sys.argv[0]} <server> <port> <length> <num_categories> [op [dtype [global]]]")
    if len(sys.argv) < 6:
        op = 'sum'
    else:
        op = sys.argv[5]
    if len(sys.argv) < 7:
        dtype = 'float64'
    else:
        dtype = sys.argv[6]
    if len(sys.argv) < 8:
        per_locale = True
    else:
        per_locale = (sys.argv[7].lower() in ('0', 'False'))
    ak.connect(sys.argv[1], int(sys.argv[2]))
    measure_runtime(int(sys.argv[3]), int(sys.argv[4]), op, dtype, per_locale)
    sys.exit()
