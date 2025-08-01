from time import time

import numpy as np
import pandas as pd
from base_test import ArkoudaTest
from context import arkouda as ak


def compare_strategies(length, ncat, op, dtype):
    keys = ak.randint(0, ncat, length)
    if dtype == 'int64':
        vals = ak.randint(0, length//ncat, length)
    elif dtype == 'bool':
        vals = ak.zeros(length, dtype='bool')
        for i in np.random.randint(0, length, ncat//2):
            vals[i] = True
    else:
        vals = ak.linspace(-1, 1, length)        
    print("Global groupby", end=' ')                                        
    start = time()                                                
    gg = ak.GroupBy(keys, False)
    ggtime = time() - start
    print(ggtime)
    print("Global reduce", end=' ')
    start = time()
    gk, gv = gg.aggregate(vals, op)
    grtime = time() - start
    print(grtime)
    print("Local groupby", end=' ')
    start = time()
    lg = ak.GroupBy(keys, True)
    lgtime = time() - start
    print(lgtime)
    print("Local reduce", end=' ')
    start = time()
    lk, lv = lg.aggregate(vals, op)
    lrtime = time() - start
    print(lrtime)
    print(f"Keys match? {(gk == lk).all()}")
    print(f"Absolute diff of vals = {ak.abs(gv - lv).sum()}")
    return ggtime, grtime, lgtime, lrtime

class GroupByCompareStrategiesTest(ArkoudaTest):

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 5:
        print(f"Usage: {sys.argv[0]} <server> <port> <length> <num_categories> [op [dtype]]")
    if len(sys.argv) < 6:
        op = 'sum'
    else:
        op = sys.argv[5]
    if len(sys.argv) < 7:
        dtype = 'float64'
    else:
        dtype = sys.argv[6]
    ak.connect(sys.argv[1], int(sys.argv[2]))
    compare_strategies(int(sys.argv[3]), int(sys.argv[4]), op, dtype)
    sys.exit()
