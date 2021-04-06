import numpy as np
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
        
    def testSetxor1d(self):
        pdaOne = ak.array([1, 2, 3, 2, 4])
        pdaTwo = ak.array([2, 3, 5, 7, 5])
        expected = ak.array([1, 4, 5, 7])
        
        self.assertTrue((expected == ak.setxor1d(pdaOne,pdaTwo)).all())
        
        with self.assertRaises(RuntimeError) as cm:
            ak.setxor1d(ak.array([-1.0, 0.0, 1.0]), ak.array([-2.0, 0.0, 2.0]))
        self.assertEqual('Error: unique: float64 not implemented', 
                         cm.exception.args[0])  
        
        with self.assertRaises(RuntimeError) as cm:
            ak.setxor1d(ak.array([True, False, True]), ak.array([True, True]))
        self.assertEqual('Error: unique: bool not implemented', 
                         cm.exception.args[0]) 
        with self.assertRaises(TypeError):
            ak.setxor1d([-1, 0, 1], [-2, 0, 2])     
        
    def testSetdiff1d(self):
        pdaOne = ak.array([1, 2, 3, 2, 4, 1])
        pdaTwo = ak.array([3, 4, 5, 6])
        expected = ak.array([1,2])
        
        self.assertTrue((expected == ak.setdiff1d(pdaOne,pdaTwo)).all())
        
        with self.assertRaises(RuntimeError) as cm:
            ak.setdiff1d(ak.array([-1.0, 0.0, 1.0]), ak.array([-2.0, 0.0, 2.0]))
        self.assertEqual('Error: unique: float64 not implemented', 
                         cm.exception.args[0])  
        
        with self.assertRaises(RuntimeError) as cm:
            ak.setdiff1d(ak.array([True, False, True]), ak.array([True, True]))
        self.assertEqual('Error: unique: bool not implemented', 
                         cm.exception.args[0]) 
        with self.assertRaises(TypeError):
            ak.setdiff1d([-1, 0, 1], [-2, 0, 2])     
        
    def testIntersectId(self):
        pdaOne = ak.array([1, 3, 4, 3])
        pdaTwo = ak.array([3, 1, 2, 1])
        expected = ak.array([1,3])
        self.assertTrue((expected == ak.intersect1d(pdaOne,pdaTwo)).all())
        
        with self.assertRaises(RuntimeError) as cm:
            ak.intersect1d(ak.array([-1.0, 0.0, 1.0]), ak.array([-2.0, 0.0, 2.0]))
        self.assertEqual('Error: unique: float64 not implemented', 
                         cm.exception.args[0])  
        
        with self.assertRaises(RuntimeError) as cm:
            ak.intersect1d(ak.array([True, False, True]), ak.array([True, True]))
        self.assertEqual('Error: unique: bool not implemented', 
                         cm.exception.args[0]) 
        with self.assertRaises(TypeError):
            ak.intersect1d([-1, 0, 1], [-2, 0, 2])     
        
    def testUnion1d(self):
        pdaOne = ak.array([-1, 0, 1])
        pdaTwo = ak.array([-2, 0, 2])
        expected = ak.array([-2, -1,  0,  1,  2])
        self.assertTrue((expected == ak.union1d(pdaOne,pdaTwo)).all())
        
        with self.assertRaises(RuntimeError) as cm:
            ak.union1d(ak.array([-1.0, 0.0, 1.0]), ak.array([-2.0, 0.0, 2.0]))
        self.assertEqual('Error: unique: float64 not implemented', 
                         cm.exception.args[0])  
        
        with self.assertRaises(RuntimeError) as cm:
            ak.union1d(ak.array([True, True, True]), ak.array([True,False,True]))
        self.assertEqual('Error: unique: bool not implemented', 
                         cm.exception.args[0])          
        with self.assertRaises(TypeError):
            ak.union1d([-1, 0, 1], [-2, 0, 2])     

    def testIn1d(self): 
        pdaOne = ak.array([-1, 0, 1, 3])
        pdaTwo = ak.array([-1, 2, 2, 3])
        self.assertTrue((ak.in1d(pdaOne, pdaTwo) == 
                         ak.array([True, False, False, True])).all())

        vals = [i % 3 for i in range(10)]
        valsTwo = [i % 2 for i in range(10)]
        stringsOne = ak.array(['String {}'.format(i) for i in vals])
        stringsTwo = ak.array(['String {}'.format(i) for i in valsTwo])

        answer = ak.array([x < 2 for x in vals])
        self.assertTrue((answer == ak.in1d(stringsOne,stringsTwo)).all())
