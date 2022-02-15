import numpy as np
import warnings
from itertools import product
from base_test import ArkoudaTest
from context import arkouda as ak
SIZE = 10
verbose = ArkoudaTest.verbose

def run_tests(verbose):
    # ignore numpy warnings like divide by 0
    np.seterr(all='ignore')
    global pdarrays
    pdarrays = {'int64': ak.arange(0, SIZE, 1),
                'uint64': ak.array(np.arange(0, SIZE, 1, dtype=np.uint64)),
                'float64': ak.linspace(0, 2, SIZE),
                'bool': (ak.arange(0, SIZE, 1) % 2) == 0}
    global ndarrays
    ndarrays = {'int64': np.arange(0, SIZE, 1),
                'uint64': np.arange(0, SIZE, 1, dtype=np.uint64),
                'float64': np.linspace(0, 2, SIZE),
                'bool': (np.arange(0, SIZE, 1) % 2) == 0}
    global scalars
    #scalars = {k: v[SIZE//2] for k, v in ndarrays.items()}
    scalars = {'int64': 5,
               'uint64': np.uint64(5),
               'float64': 3.14159,
               'bool': True}
    dtypes = pdarrays.keys()
    if verbose:
        print("Operators: ", ak.pdarray.BinOps)
        print("Dtypes: ", dtypes)
        print("pdarrays: ")
        for k, v in pdarrays.items():
            print(k, ": ", v)
        print("ndarrays: ")
        for k, v in ndarrays.items():
            print(k, ": ", v)
        print("scalars: ")
        for k, v in scalars.items():
            print(k, ": ", v)

    def do_op(lt, rt, ls, rs, isarkouda, oper):
        evalstr = ''
        if ls:
            evalstr += 'scalars["{}"]'.format(lt)
        else:
            evalstr += '{}["{}"]'.format(('ndarrays', 'pdarrays')[isarkouda], lt)
        evalstr += ' {} '.format(oper)
        if rs:
            evalstr += 'scalars["{}"]'.format(rt)
        else:
            evalstr += '{}["{}"]'.format(('ndarrays', 'pdarrays')[isarkouda], rt)
        #print(evalstr)
        res = eval(evalstr)
        return res

    results = {'neither_implement': [],   # (expression, ak_error)
               'arkouda_minus_numpy': [], # (expression, ak_result, error_on_exec?)
               'numpy_minus_arkouda': [], # (expression, ak_result, error_on_exec?)
               'both_implement': []}      # (expression, ak_result, error_on_exec?, dtype_mismatch?, value_mismatch?)
    tests = 0
    for ltype, rtype, op in product(dtypes, dtypes, ak.pdarray.BinOps):
        if op in ("<<<", ">>>"):
            continue
        for lscalar, rscalar in ((False, False), (False, True), (True, False)):
            tests += 1
            expression = "{}({}) {} {}({})".format(ltype, ('array', 'scalar')[lscalar], op, rtype, ('array', 'scalar')[rscalar])
            try:
                npres = do_op(ltype, rtype, lscalar, rscalar, False, op)
            except TypeError: # numpy doesn't implement operation
                try:
                    akres = do_op(ltype, rtype, lscalar, rscalar, True, op)
                except RuntimeError as e:
                    if 'not implemented' or 'unrecognized type' in str(e):  # neither numpy nor arkouda implement
                        results['neither_implement'].append((expression, str(e)))
                    else: # arkouda implements with error, np does not implement
                        results['arkouda_minus_numpy'].append((expression, str(e), True))
                    continue
                # arkouda implements but not numpy
                results['arkouda_minus_numpy'].append((expression, str(akres), False))
                continue
            try:
                akres = do_op(ltype, rtype, lscalar, rscalar, True, op)
            except RuntimeError as e:
                if 'not implemented' or 'unrecognized type' in str(e):  # numpy implements but not arkouda
                    results['numpy_minus_arkouda'].append((expression, str(e), True))
                else: # both implement, but arkouda errors
                    results['both_implement'].append((expression, str(e), True, False, False))
                continue
            # both numpy and arkouda execute without error
            try:
                akrestype = akres.dtype
            except Exception as e:
                warnings.warn("Cannot detect return dtype of ak result: {} (np result: {})".format(akres, npres))
                results['both_implement'].append((expression, str(akres), False, True, False))
                continue

            if akrestype != npres.dtype:
                restypes = "{}(np) vs. {}(ak)".format(npres.dtype, akrestype)
                #warnings.warn("dtype mismatch: {} = {}".format(expression, restypes))
                results['both_implement'].append((expression, restypes, False, True, False))
                continue
            try:
                akasnp = akres.to_ndarray()
            except Exception as e:
                warnings.warn("Could not convert to ndarray: {}".format(akres))
                results['both_implement'].append((expression, str(akres), True, False, False))
                continue
            if not np.allclose(akasnp, npres, equal_nan=True):
                res = "np: {}\nak: {}".format(npres, akasnp)
                # warnings.warn("result mismatch: {} =\n{}".format(expression, res))
                results['both_implement'].append((expression, res, False, False, True))
            # Finally, both numpy and arkouda agree on result
            results['both_implement'].append((expression, "", False, False, False))

    print("# ops not implemented by numpy or arkouda: {}".format(len(results['neither_implement'])))
    if verbose:
        for expression, err in results['neither_implement']:
            print(expression)
    print("# ops implemented by numpy but not arkouda: {}".format(len(results['numpy_minus_arkouda'])))
    if verbose:
        for expression, err, flag in results['numpy_minus_arkouda']:
            print(expression)
    print("# ops implemented by arkouda but not numpy: {}".format(len(results['arkouda_minus_numpy'])))
    if verbose:
        for expression, res, flag in results['arkouda_minus_numpy']:
            print(expression, " -> ", res)
    nboth = len(results['both_implement'])
    print("# ops implemented by both: {}".format(nboth))
    matches = 0
    execerrors = []
    dtypeerrors = []
    valueerrors = []
    for (expression, res, ex, dt, val) in results['both_implement']:
        matches += not any((ex, dt, val))
        if ex: execerrors.append((expression, res))
        if dt: dtypeerrors.append((expression, res))
        if val: valueerrors.append((expression, res))
    print("  Matching results:         {} / {}".format(matches, nboth))
    print("  Arkouda execution errors: {} / {}".format(len(execerrors), nboth))
    if verbose: print('\n'.join(map(': '.join, execerrors)))
    print("  Dtype mismatches:         {} / {}".format(len(dtypeerrors), nboth))
    if verbose: print('\n'.join(map(': '.join, dtypeerrors)))
    print("  Value mismatches:         {} / {}".format(len(valueerrors), nboth))
    if verbose: print('\n'.join(map(': '.join, valueerrors)))
    return matches == nboth

'''
Encapsulates test cases that invoke the run_tests method.
'''
class OperatorsTest(ArkoudaTest):

    def testPdArrayAddInt(self):
        aArray = ak.ones(100)
        addArray = aArray + 1
        self.assertIsInstance(addArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(2),addArray[0])

        addArray = 1 + aArray
        self.assertIsInstance(addArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(2), addArray[0])

    def testPdArrayAddNumpyInt(self):
        aArray = ak.ones(100)
        addArray = aArray + np.int64(1)
        self.assertIsInstance(addArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(2), addArray[0])

        addArray = np.int64(1) + aArray
        self.assertIsInstance(addArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(2), addArray[0])

    def testPdArraySubtractInt(self):
        aArray = ak.ones(100)
        subArray =  aArray - 2
        self.assertIsInstance(subArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(-1), subArray[0])

        subArray =  2 - aArray
        self.assertIsInstance(subArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(1), subArray[0])

    def testPdArraySubtractNumpyInt(self):
        aArray = ak.ones(100)
        subArray =  aArray - np.int64(2)
        self.assertIsInstance(subArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(-1), subArray[0])

        subArray =  np.int64(2) - aArray
        self.assertIsInstance(subArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(1), subArray[0])

    def testPdArrayMultInt(self):
        aArray = ak.ones(100)
        mArray =  aArray*5
        self.assertIsInstance(mArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(5), mArray[0])

        mArray =  5*aArray
        self.assertIsInstance(mArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(5), mArray[0])

    def testPdArrayMultNumpyInt(self):
        aArray = ak.ones(100)
        mArray =  aArray*np.int64(5)
        self.assertIsInstance(mArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(5), mArray[0])

        mArray =  np.int64(5)*aArray
        self.assertIsInstance(mArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(5), mArray[0])

    def testPdArrayDivideInt(self):
        aArray = ak.ones(100)
        dArray =  aArray*15/3
        self.assertIsInstance(dArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(5), dArray[0])

        dArray =  15*aArray/3
        self.assertIsInstance(dArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(5), dArray[0])

    def testPdArrayDivideNumpyInt(self):
        aArray = ak.ones(100)
        dArray =  aArray*np.int64(15)/3
        self.assertIsInstance(dArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(5), dArray[0])

        dArray =  np.int64(15)*aArray/3
        self.assertIsInstance(dArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(5), dArray[0])
        
    def testPdArrayConcatenation(self):
        onesOne = ak.randint(0, 100, 100)
        onesTwo = ak.randint(0, 100, 100)
        
        result = ak.concatenate([onesOne,onesTwo])
        self.assertEqual(200, len(result))
        self.assertEqual(np.int64,result.dtype)

    def testConcatenate(self):
        pdaOne = ak.arange(1,4)
        pdaTwo = ak.arange(4,7)  
        
        self.assertTrue((ak.array([1,2,3,4,5,6])
                              == ak.concatenate([pdaOne,pdaTwo])).all())
        self.assertTrue((ak.array([4,5,6,1,2,3])
                              == ak.concatenate([pdaTwo,pdaOne])).all())
        
        pdaOne = ak.linspace(start=1,stop=3,length=3)
        pdaTwo = ak.linspace(start=4,stop=6,length=3)        
        
        self.assertTrue((ak.array([1,2,3,4,5,6])
                              == ak.concatenate([pdaOne,pdaTwo])).all())
        self.assertTrue((ak.array([4,5,6,1,2,3])
                              == ak.concatenate([pdaTwo,pdaOne])).all())

        pdaOne = ak.array([True,False,True])
        pdaTwo = ak.array([False,True,True])
       
        self.assertTrue((ak.array([True, False, True, False, True, True]) == 
                ak.concatenate([pdaOne,pdaTwo])).all())
        
    def testAllOperators(self):
        run_tests(verbose)
        
    def testErrorHandling(self):
        # Test NotImplmentedError that prevents pddarray iteration       
        with self.assertRaises(NotImplementedError):
            iter(ak.ones(100))
            
        # Test NotImplmentedError that prevents Strings iteration       
        with self.assertRaises(NotImplementedError):
            iter(ak.array(['String {}'.format(i) for i in range(0,10)]))
        
        # Test ak,histogram against unsupported dtype
        #with self.assertRaises(ValueError) as cm:
        #    ak.histogram((ak.randint(0, 1, 100, dtype=ak.bool)))
            
        with self.assertRaises(RuntimeError) as cm:
            ak.concatenate([ak.array([True]),ak.array([True])]).is_sorted()
        
        with self.assertRaises(TypeError):
            ak.ones(100).any([0])
            
        with self.assertRaises(TypeError) as cm:
            ak.unique(list(range(0,10)))
        
        with self.assertRaises(RuntimeError) as cm:
            ak.concatenate([ak.ones(100),ak.array([True])])

    def test_str_repr(self):
        """
        Test 3 different types: int, float, bool with lengths under/over threshold
        Do this for both __str__() and __repr__()
        """
        ak.client.pdarrayIterThresh = 5
        # Test __str__()
        self.assertEqual("[1 2 3]", ak.array([1, 2, 3]).__str__())
        self.assertEqual("[1 2 3 ... 17 18 19]", ak.arange(1, 20).__str__())
        self.assertEqual("[1.100000e+00 2.300000e+00 5.000000e+00]", ak.array([1.1, 2.3, 5]).__str__())
        self.assertEqual("[0.000000e+00 5.263158e-01 1.052632e+00 ... 8.947368e+00 9.473684e+00 1.000000e+01]",
                         ak.linspace(0, 10, 20).__str__())
        self.assertEqual("[False False False]", ak.isnan(ak.array([1.1, 2.3, 5])).__str__())
        self.assertEqual("[False False False ... False False False]", ak.isnan(ak.linspace(0, 10, 20)).__str__())

        # Test __repr__()
        self.assertEqual("array([1 2 3])", ak.array([1, 2, 3]).__repr__())
        self.assertEqual("array([1 2 3 ... 17 18 19])", ak.arange(1, 20).__repr__())
        self.assertEqual("array([1.1000000000000001 2.2999999999999998 5])", ak.array([1.1, 2.3, 5]).__repr__())
        self.assertEqual("array([0 0.52631578947368418 1.0526315789473684 ... 8.9473684210526319 9.473684210526315 10])",
                         ak.linspace(0, 10, 20).__repr__())
        self.assertEqual("array([False False False])", ak.isnan(ak.array([1.1, 2.3, 5])).__repr__())
        self.assertEqual("array([False False False ... False False False])", ak.isnan(ak.linspace(0, 10, 20)).__repr__())
        ak.client.pdarrayIterThresh = ak.client.pdarrayIterThreshDefVal  # Don't forget to set this back for other tests.



if __name__ == '__main__':
    '''
    Enables invocation of operator tests outside of pytest test harness
    '''
    import sys
    if len(sys.argv) not in (3, 4):
        print(f"Usage: {sys.argv[0]} <server_name> <port> [<verbose>=(0|1)]")
    verbose = False
    if len(sys.argv) == 4 and sys.argv[3] == "1":
        verbose = True
    ak.connect(server=sys.argv[1], port=int(sys.argv[2]))
    success = run_tests(verbose)
    ak.disconnect()
    sys.exit((1, 0)[success])
