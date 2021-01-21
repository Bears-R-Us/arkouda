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

def run_test(runMin=True, isInd=True, verbose=True):
    '''
    The run_test method runs execution of the mink reduction
    on a randomized array.
    :return: 
    '''
    aka = make_array()
    
    failures = 0
    try:
        if not isInd:
            if runMin:
                akres = ak.mink(aka, K)
                npres = np.sort(aka.to_ndarray())[:K] # first K elements from sorted array
            else:
                akres = ak.maxk(aka, K)
                npres = np.sort(aka.to_ndarray())[-K:] # last K elements from sorted array
        else:
            if runMin:
                akres = aka[ak.argmink(aka, K)]
                npres = np.sort(aka.to_ndarray())[:K] # first K elements from sorted array
            else:
                akres = aka[ak.argmaxk(aka, K)]
                npres = np.sort(aka.to_ndarray())[-K:] # last K elements from sorted array
    except RuntimeError as E:
        if verbose: print("Arkouda error: ", E)
        return 1

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
        
    def test_error_handling(self):
        testArray = ak.randint(0, 100, 100)

        with self.assertRaises(TypeError) as cm:
            ak.mink(list(range(0,10)), 1)
        self.assertEqual('type of argument "pda" must be arkouda.pdarrayclass.pdarray; got list instead', 
                         cm.exception.args[0])  

        with self.assertRaises(TypeError) as cm:
            ak.mink(testArray, '1')
        self.assertEqual('type of argument "k" must be one of (int, int64); got str instead', 
                         cm.exception.args[0])  
            
        with self.assertRaises(ValueError) as cm:
            ak.mink(testArray, -1)
        self.assertEqual("k must be 1 or greater", 
                         cm.exception.args[0])   
        
        with self.assertRaises(ValueError) as cm:
            ak.mink(ak.array([]), 1)
        self.assertEqual("must be a non-empty pdarray of type int or float", 
                         cm.exception.args[0])   

class MaxKTest(ArkoudaTest):
    def test_maxk(self):
        '''
        Executes run_test and asserts whether there are any errors
        
        :return: None
        :raise: AssertionError if there are any errors encountered in run_test for set operations
        '''
        self.assertEqual(0, run_test(runMin=False))
        
    def test_error_handling(self):
        testArray = ak.randint(0, 100, 100)

        with self.assertRaises(TypeError) as cm:
            ak.maxk(list(range(0,10)), 1)
        self.assertEqual('type of argument "pda" must be arkouda.pdarrayclass.pdarray; got list instead', 
                         cm.exception.args[0])  

        with self.assertRaises(TypeError) as cm:
            ak.maxk(testArray, '1')
        self.assertEqual('type of argument "k" must be one of (int, int64); got str instead', 
                         cm.exception.args[0])  
            
        with self.assertRaises(ValueError) as cm:
            ak.maxk(testArray, -1)
        self.assertEqual("k must be 1 or greater", 
                         cm.exception.args[0])   
        
        with self.assertRaises(ValueError) as cm:
            ak.maxk(ak.array([]), 1)
        self.assertEqual("must be a non-empty pdarray of type int or float", 
                         cm.exception.args[0])   

class ArgMinKTest(ArkoudaTest):
    def test_argmink(self):
        '''
        Executes run_test and asserts whether there are any errors
        
        :return: None
        :raise: AssertionError if there are any errors encountered in run_test for set operations
        '''
        self.assertEqual(0, run_test(isInd=True))
        
    def test_error_handling(self):
        testArray = ak.randint(0, 100, 100)

        with self.assertRaises(TypeError) as cm:
            ak.argmink(list(range(0,10)), 1)
        self.assertEqual('type of argument "pda" must be arkouda.pdarrayclass.pdarray; got list instead', 
                         cm.exception.args[0])  

        with self.assertRaises(TypeError) as cm:
            ak.argmink(testArray, '1')
        self.assertEqual('type of argument "k" must be one of (int, int64); got str instead', 
                         cm.exception.args[0])  
            
        with self.assertRaises(ValueError) as cm:
            ak.argmink(testArray, -1)
        self.assertEqual("k must be 1 or greater", 
                         cm.exception.args[0])   
        
        with self.assertRaises(ValueError) as cm:
            ak.argmink(ak.array([]), 1)
        self.assertEqual("must be a non-empty pdarray of type int or float", 
                         cm.exception.args[0])   

class ArgMaxKTest(ArkoudaTest):
    def test_argmaxk(self):
        '''
        Executes run_test and asserts whether there are any errors
        
        :return: None
        :raise: AssertionError if there are any errors encountered in run_test for set operations
        '''
        self.assertEqual(0, run_test(runMin=False, isInd=True))
    
    def test_error_handling(self):
        testArray = ak.randint(0, 100, 100)

        with self.assertRaises(TypeError) as cm:
            ak.argmaxk(list(range(0,10)), 1)
        self.assertEqual('type of argument "pda" must be arkouda.pdarrayclass.pdarray; got list instead', 
                         cm.exception.args[0])  
        
        with self.assertRaises(TypeError) as cm:
            ak.argmaxk(testArray, '1')
        self.assertEqual('type of argument "k" must be one of (int, int64); got str instead', 
                         cm.exception.args[0])  
            
        with self.assertRaises(ValueError) as cm:
            ak.argmaxk(testArray, -1)
        self.assertEqual("k must be 1 or greater", 
                         cm.exception.args[0])      
        
        with self.assertRaises(ValueError) as cm:
            ak.argmaxk(ak.array([]), 1)
        self.assertEqual("must be a non-empty pdarray of type int or float", 
                         cm.exception.args[0])           
