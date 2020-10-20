from context import arkouda as ak
import numpy as np
import pandas as pd
from context import arkouda as ak
from base_test import ArkoudaTest

SIZE = 10
OPS = frozenset(['intersect1d', 'union1d', 'setxor1d', 'setdiff1d'])

def make_arrays():
    a = ak.randint(0, SIZE, SIZE)
    b = ak.randint(SIZE/2, 2*SIZE, SIZE)
    return a, b

def compare_results(akvals, npvals) -> int:
    '''
    Compares the numpy and arkouda arrays via the numpy.allclose method with the
    default relative and absolute tolerances, returning 0 if the arrays are similar
    element-wise within the tolerances, 1 if they are dissimilar.element
    
    :return: 0 (identical) or 1 (dissimilar)
    :rtype: int
    '''
    akvals = akvals.to_ndarray()
    
    if not np.array_equal(akvals,npvals):
        akvals = ak.array(akvals)
        npvals = ak.array(npvals)
        innp = npvals[ak.in1d(ak.array(npvals), ak.array(akvals), True)] # values in np array, but not ak array
        inak = akvals[ak.in1d(ak.array(akvals), ak.array(npvals), True)] # values in ak array, not not np array
        print(f"(values in np but not ak: {innp}) (values in ak but not np: {inak})")
        return 1
    return 0

def run_test(verbose=True):
    '''
    The run_test method enables execution of the set operations
    intersect1d, union1d, setxor1d, and setdiff1d
    on a randomized set of arrays. 
    :return: 
    '''
    aka, akb = make_arrays()

    tests = 0
    failures = 0
    not_impl = 0
    
    for op in OPS:
        tests += 1
        do_check = True
        try:
            fxn = getattr(ak, op)
            akres = fxn(aka,akb)
            fxn = getattr(np, op)
            npres = fxn(aka.to_ndarray(), akb.to_ndarray())
        except RuntimeError as E:
            if verbose: print("Arkouda error: ", E)
            not_impl += 1
            do_check = False
            continue
        if not do_check:
            continue
        failures += compare_results(akres, npres)
    
    return failures

class SetOpsTest(ArkoudaTest):
    def test_setops(self):
        '''
        Executes run_test and asserts whether there are any errors
        
        :return: None
        :raise: AssertionError if there are any errors encountered in run_test for set operations
        '''
        self.assertEqual(0, run_test())
        
    def test_error_handling(self):
        with self.assertRaises(RuntimeError) as cm:
            ak.concatenate([ak.ones(100),ak.array([True])])

        self.assertEqual('Error: concatenateMsg: Incompatible arguments: ' +
                         'Expected float64 dtype but got bool dtype', 
                         cm.exception.args[0])       
         
        with self.assertRaises(TypeError):
            ak.union1d([-1, 0, 1], [-2, 0, 2])