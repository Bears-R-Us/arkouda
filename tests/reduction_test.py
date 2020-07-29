from context import arkouda as ak
import numpy as np
from context import arkouda as ak
from base_test import ArkoudaTest

SIZE = 10
K = 5

def make_array():
    a = ak.randint(0, SIZE, SIZE)
    return a

def compare_results(akres, sortedres) -> int:
    '''
    Compares the numpy and arkouda arrays via the numpy.allclose method with the
    default relative and absolute tolerances, returning 0 if the arrays are similar
    element-wise within the tolerances, 1 if they are dissimilar.element
    
    :return: 0 (identical) or 1 (dissimilar)
    :rtype: int
    '''
    akres = akres.to_ndarray()
    
    if not np.array_equal(akres, sortedres):
        akres = ak.array(akres)
        sortedres = ak.array(sortedres)
        innp = sortedres[ak.in1d(ak.array(sortedres), ak.array(akres), True)] # values in np array, but not ak array
        inak = akres[ak.in1d(ak.array(akres), ak.array(sortedres), True)] # values in ak array, not not np array
        print(f"(values in np but not ak: {innp}) (values in ak but not np: {inak})")
        return 1
    return 0

def run_test(runMin=True, verbose=True):
    '''
    The run_test method runs execution of the mink reduction
    on a randomized array.
    :return: 
    '''
    aka = make_array()
    
    failures = 0
    try:
        if runMin:
            akres = ak.mink(aka, K)
            npres = np.sort(aka.to_ndarray())[:K] # first K elements from sorted array
        else:
            akres = ak.maxk(aka, K)
            npres = np.sort(aka.to_ndarray())[-K:] # last K elements from sorted array
    except RuntimeError as E:
        if verbose: print("Arkouda error: ", E)

    failures += compare_results(akres, npres)
    
    return failures

class MinKTest(ArkoudaTest):
    def test_mink(self):
        '''
        Executes run_test and asserts whether there are any errors
        
        :return: None
        :raise: AssertionError if there are any errors encountered in run_test for set operations
        '''
        self.assertEqual(0, run_test())

class MaxKTest(ArkoudaTest):
    def test_maxk(self):
        '''
        Executes run_test and asserts whether there are any errors
        
        :return: None
        :raise: AssertionError if there are any errors encountered in run_test for set operations
        '''
        self.assertEqual(0, run_test(runMin=False))
