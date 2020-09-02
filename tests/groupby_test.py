import numpy as np
import pandas as pd
from context import arkouda as ak
from base_test import ArkoudaTest
import unittest

SIZE = 10000
GROUPS = 64
verbose = True

def groupby_to_arrays(df : pd.DataFrame, kname, vname, op, levels):
    g = df.groupby(kname)[vname]
    agg = g.aggregate(op.replace('arg', 'idx'))
    if op == 'prod':
        # There appears to be a bug in pandas where it sometimes
        # reports the product of a segment as NaN when it should be 0
        agg[agg.isna()] = 0
    if levels==1:
        keys = agg.index.values
    else:
        keys = tuple(zip(*(agg.index.values)))
    return keys, agg.values

def make_arrays():
    keys = np.random.randint(0, GROUPS, SIZE)
    keys2 = np.random.randint(0, GROUPS, SIZE)
    i = np.random.randint(0, SIZE//GROUPS, SIZE)
    f = np.random.randn(SIZE) # normally dist random numbers
    b = (i % 2) == 0
    d = {'keys':keys, 'keys2':keys2, 'int64':i, 'float64':f, 'bool':b}

    return d
  
def compare_keys(pdkeys, akkeys, levels, pdvals, akvals) -> int:
    '''
    Compares the numpy and arkouda arrays via the numpy.allclose method with the
    default relative and absolute tolerances, returning 0 if the arrays are similar
    element-wise within the tolerances, 1 if they are dissimilar.element
    
    :return: 0 (identical) or 1 (dissimilar)
    :rtype: int
    '''
    if levels == 1:
        akkeys = akkeys.to_ndarray()
        if not np.allclose(pdkeys, akkeys):
            print("Different keys")
            return 1
    else:
        for l in range(levels):
            if not np.allclose(pdkeys[l], akkeys[l].to_ndarray()):
                print("Different keys")
                return 1
    if not np.allclose(pdvals, akvals):
        print(f"Different values (abs diff = {np.abs(pdvals - akvals).sum()})")
        return 1
    return 0

def run_test(levels, verbose=False):
    '''
    The run_test method enables execution of ak.GroupBy and ak.GroupBy.Reductions
    on a randomized set of arrays on the specified number of levels. 
    
    Note: the current set of valid levels is {1,2}
    :return: 
    '''
    d = make_arrays()
    df = pd.DataFrame(d)
    akdf = {k:ak.array(v) for k, v in d.items()}

    if levels == 1:
        akg = ak.GroupBy(akdf['keys'])
        keyname = 'keys'
    elif levels == 2:
        akg = ak.GroupBy([akdf['keys'], akdf['keys2']])
        keyname = ['keys', 'keys2']
    tests = 0
    failures = 0
    not_impl = 0
    if verbose: print(f"Doing .count()")
    tests += 1
    pdkeys, pdvals = groupby_to_arrays(df, keyname, 'int64', 'count', levels)
    akkeys, akvals = akg.count()
    akvals = akvals.to_ndarray()
    failures += compare_keys(pdkeys, akkeys, levels, pdvals, akvals)
    for vname in ('int64', 'float64', 'bool'):
        for op in ak.GroupBy.Reductions:
            if verbose: print(f"\nDoing aggregate({vname}, {op})")
            tests += 1
            do_check = True
            try:
                pdkeys, pdvals = groupby_to_arrays(df, keyname, vname, op, levels)
            except Exception as E:
                if verbose: print("Pandas does not implement")
                do_check = False
            try:
                akkeys, akvals = akg.aggregate(akdf[vname], op)
                akvals = akvals.to_ndarray()
            except RuntimeError as E:
                if verbose: print("Arkouda error: ", E)
                not_impl += 1
                do_check = False
                continue
            if not do_check:
                continue
            if op.startswith('arg'):
                pdextrema = df[vname][pdvals]
                akextrema = akdf[vname][ak.array(akvals)].to_ndarray()
                if not np.allclose(pdextrema, akextrema):
                    print(f"Different argmin/argmax: Arkouda failed to find an extremum")
                    print("pd: ", pdextrema)
                    print("ak: ", akextrema)
                    failures += 1
            else:
                failures += compare_keys(pdkeys, akkeys, levels, pdvals, akvals)
    print(f"{tests - failures - not_impl} / {tests - not_impl} passed, {failures} errors, {not_impl} not implemented")
    return failures

'''
The GroupByTest class encapsulates specific calls to the run_test method within a Python unittest.TestCase object,
which enables integration into a pytest test harness.
'''
class GroupByTest(ArkoudaTest): 

    # https://github.com/mhmerrill/arkouda/issues/365
    #@unittest.skip
    def test_groupby_on_one_level(self):
        '''
        Executes run_test with levels=1 and asserts whether there are any errors
        
        :return: None
        :raise: AssertionError if there are any errors encountered in run_test with levels = 1
        '''
        self.assertEqual(0, run_test(1, verbose))
  
    def test_groupby_one_two_levels(self):
        '''
        Executes run_test with levels=1 and asserts whether there are any errors
        
        :return: None
        :raise: AssertionError if there are any errors encountered in run_test with levels = 2
        '''
        self.assertEqual(0, run_test(2, verbose))

