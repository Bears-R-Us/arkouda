import numpy as np
import pandas as pd
from context import arkouda as ak
from arkouda.dtypes import float64, int64
from base_test import ArkoudaTest
from arkouda.groupbyclass import GroupByReductionType

SIZE = 100
GROUPS = 8
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

    def setUp(self):
        ArkoudaTest.setUp(self)
        
        self.bvalues = ak.randint(0,1,10,dtype=bool)
        self.fvalues = ak.randint(0,1,10,dtype=float)
        self.ivalues = ak.array([4, 1, 3, 2, 2, 2, 5, 5, 2, 3])
        self.igb = ak.GroupBy(self.ivalues)

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

    def test_bitwise_aggregations(self):
        revs = ak.arange(self.igb.size) % 2
        self.assertTrue((self.igb.OR(revs)[1] == self.igb.max(revs)[1]).all())
        self.assertTrue((self.igb.AND(revs)[1] == self.igb.min(revs)[1]).all())
        self.assertTrue((self.igb.XOR(revs)[1] == (self.igb.sum(revs)[1] % 2)).all())
        
    def test_standalone_broadcast(self):
        segs = ak.arange(10)**2
        vals = ak.arange(10)
        size = 100
        check = ((2*vals + 1)*vals).sum()
        self.assertTrue(ak.broadcast(segs, vals, size=size).sum() == check)
        perm = ak.arange(99, -1, -1)
        bcast = ak.broadcast(segs, vals, permutation=perm)
        self.assertTrue(bcast.sum() == check)
        self.assertTrue((bcast[:-1] >= bcast[1:]).all())
        
    def test_broadcast_ints(self):
        keys,counts = self.igb.count()

        self.assertTrue((np.array([1,4,2,1,2]) == counts.to_ndarray()).all())
        self.assertTrue((np.array([1,2,3,4,5]) == keys.to_ndarray()).all())

        results = self.igb.broadcast(1*(counts > 2))
        self.assertTrue((np.array([0,1,1,1,1,0,0,0,0,0]),results.to_ndarray()))
        
        results = self.igb.broadcast(1*(counts == 2))
        self.assertTrue((np.array([0,0,0,0,0,1,1,0,1,1]),results.to_ndarray()))     
        
        results = self.igb.broadcast(1*(counts < 4))
        self.assertTrue((np.array([1,0,0,0,0,1,1,1,1,1]),results.to_ndarray()))  
        
    def test_broadcast_booleans(self):
        keys,counts = self.igb.count()

        self.assertTrue((np.array([1,4,2,1,2]) == counts.to_ndarray()).all())
        self.assertTrue((np.array([1,2,3,4,5]) == keys.to_ndarray()).all())

        results = self.igb.broadcast(counts > 2)
        self.assertTrue((np.array([0,1,1,1,1,0,0,0,0,0]),results.to_ndarray()))
        
        results = self.igb.broadcast(counts == 2)
        self.assertTrue((np.array([0,0,0,0,0,1,1,0,1,1]),results.to_ndarray()))     
        
        results = self.igb.broadcast(counts < 4)
        self.assertTrue((np.array([1,0,0,0,0,1,1,1,1,1]),results.to_ndarray()))    
        
    def test_count(self):   
        keys, counts = self.igb.count()
        
        self.assertTrue((np.array([1,2,3,4,5]) == keys.to_ndarray()).all())
        self.assertTrue((np.array([1,4,2,1,2]) == counts.to_ndarray()).all())
        
        
    def test_groupby_reduction_type(self):
        self.assertEqual('any', str(GroupByReductionType.ANY)) 
        self.assertEqual('all', str(GroupByReductionType.ALL))         
        self.assertEqual(GroupByReductionType.ANY, GroupByReductionType('any'))
        
        with self.assertRaises(ValueError):
            GroupByReductionType('an')
        
        self.assertIsInstance(ak.GROUPBY_REDUCTION_TYPES, frozenset)
        self.assertTrue('any' in ak.GROUPBY_REDUCTION_TYPES)
        
        
    def test_error_handling(self):
        d = make_arrays()
        akdf = {k:ak.array(v) for k, v in d.items()}        
        gb = ak.GroupBy([akdf['keys'], akdf['keys2']])
        
        with self.assertRaises(TypeError) as cm:
            ak.GroupBy(self.bvalues)  
        self.assertEqual('GroupBy only supports pdarrays with a dtype int64', 
                         cm.exception.args[0])    
        
        with self.assertRaises(TypeError) as cm:
            ak.GroupBy(self.fvalues)  
        self.assertEqual('GroupBy only supports pdarrays with a dtype int64', 
                         cm.exception.args[0])              

        with self.assertRaises(TypeError) as cm:
            gb.broadcast([])
        self.assertEqual('type of argument "values" must be arkouda.pdarrayclass.pdarray; got list instead', 
                         cm.exception.args[0])  
        
        with self.assertRaises(TypeError) as cm:
            self.igb.nunique(ak.randint(0,1,10,dtype=bool))
        self.assertEqual('the pdarray dtype must be int64', 
                         cm.exception.args[0])  

        with self.assertRaises(TypeError) as cm:
            self.igb.nunique(ak.randint(0,1,10,dtype=float64))
        self.assertEqual('the pdarray dtype must be int64', 
                         cm.exception.args[0])  
        
        with self.assertRaises(TypeError) as cm:
            self.igb.any(ak.randint(0,1,10,dtype=float64))
        self.assertEqual('any is only supported for pdarrays of dtype bool', 
                         cm.exception.args[0])  

        with self.assertRaises(TypeError) as cm:
            self.igb.any(ak.randint(0,1,10,dtype=int64))
        self.assertEqual('any is only supported for pdarrays of dtype bool', 
                         cm.exception.args[0])  
        
        with self.assertRaises(TypeError) as cm:
            self.igb.all(ak.randint(0,1,10,dtype=float64))
        self.assertEqual('all is only supported for pdarrays of dtype bool', 
                         cm.exception.args[0])  

        with self.assertRaises(TypeError) as cm:
            self.igb.all(ak.randint(0,1,10,dtype=int64))
        self.assertEqual('all is only supported for pdarrays of dtype bool', 
                         cm.exception.args[0])  
        
        with self.assertRaises(TypeError) as cm:
            self.igb.min(ak.randint(0,1,10,dtype=bool))
        self.assertEqual('min is only supported for pdarrays of dtype float64 and int64', 
                         cm.exception.args[0])  

        with self.assertRaises(TypeError) as cm:
            self.igb.max(ak.randint(0,1,10,dtype=bool))
        self.assertEqual('max is only supported for pdarrays of dtype float64 and int64', 
                         cm.exception.args[0])  
        
        with self.assertRaises(TypeError) as cm:
            self.igb.argmin(ak.randint(0,1,10,dtype=bool))
        self.assertEqual('argmin is only supported for pdarrays of dtype float64 and int64', 
                         cm.exception.args[0])  

        with self.assertRaises(TypeError) as cm:
            self.igb.argmax(ak.randint(0,1,10,dtype=bool))
        self.assertEqual('argmax is only supported for pdarrays of dtype float64 and int64', 
                         cm.exception.args[0])  

    def test_aggregate_strings(self):
        s = ak.array(['a', 'b', 'a', 'b', 'c'])
        i = ak.arange(s.size)
        grouping = ak.GroupBy(s)
        labels, values = grouping.nunique(i)

        expected = {'a': 2, 'b': 2, 'c': 1}
        actual = {label: value for (label, value) in zip(labels.to_ndarray(), values.to_ndarray())}

        self.assertDictEqual(expected, actual)
