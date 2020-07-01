import numpy as np
import pandas as pd
from context import arkouda as ak
from base_test import ArkoudaTest
import unittest

SIZE = 10
GROUPS = 3
verbose = True

OPS = frozenset(['mean'])

def groupby_to_arrays(df : pd.DataFrame, kname, vname, op):
    g = df.groupby(kname)[vname]
    agg = g.aggregate(op.replace('arg', 'idx'))
    keys = agg.index.values
    return keys, agg.values

def make_arrays():
    keys = np.random.randint(0, GROUPS, SIZE)
    f = np.empty(SIZE)#np.random.randn(SIZE) * 1000
    f.fill(5)

    for i in range(SIZE):
        if np.random.rand() < .7:
            f[i] = np.nan
    d = {'keys':keys, 'float64':f}

    return d

def compare_keys(pdkeys, akkeys, pdvals, akvals) -> int:
    '''
    Compares the numpy and arkouda arrays via the numpy.allclose method with the
    default relative and absolute tolerances, returning 0 if the arrays are similar
    element-wise within the tolerances, 1 if they are dissimilar.element
    
    :return: 0 (identical) or 1 (dissimilar)
    :rtype: int
    '''
    akkeys = akkeys.to_ndarray()

    if not np.allclose(pdkeys, akkeys):
        print("Different keys")
        return 1

    if not np.allclose(pdvals, akvals,rtol=1e-05):
        print(f"Different values (abs diff = {np.abs(pdvals - akvals).sum()})")
        return 1
    return 0


def run_test(verbose=True):
    d = make_arrays()
    df = pd.DataFrame(d)
    akdf = {k:ak.array(v) for k, v in d.items()}

    akg = ak.GroupBy(akdf['keys'])
    keyname = 'keys'

    tests = 0
    failures = 0
    not_impl = 0

    tests += 1
    pdkeys, pdvals = groupby_to_arrays(df, keyname, 'float64', 'count')
    akkeys, akvals = akg.count()
    akvals = akvals.to_ndarray()

    for op in OPS:
        tests += 1

        do_check = True
        try:
            pdkeys, pdvals = groupby_to_arrays(df, keyname, 'float64', op)
        except Exception as E:
            if verbose: print("Pandas does not implement")
            do_check = False
        try:
            akkeys, akvals = akg.aggregate(akdf['float64'], op, True)
            akvals = akvals.to_ndarray()
        except RuntimeError as E:
            if verbose: print("Arkouda error: ", E)
            not_impl += 1
            do_check = False
            continue
        if not do_check:
            continue

        print(akvals)
        print(pdvals)
        failures += compare_keys(pdkeys, akkeys, pdvals, akvals)

    return failures

class NanTest(ArkoudaTest):
    def test_nan(self):
        self.assertEqual(0, run_test())
