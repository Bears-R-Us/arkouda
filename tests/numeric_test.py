import numpy as np
from context import arkouda as ak
from arkouda.dtypes import npstr
from base_test import ArkoudaTest

"""
Encapsulates unit tests for the numeric module with the exception
of the where method, which is in the where_test module
"""
class NumericTest(ArkoudaTest):

    def testSeededRNG(self):
        N = 100
        seed = 8675309
        numericdtypes = [ak.int64, ak.float64, ak.bool]
        for dt in numericdtypes:
            # Make sure unseeded runs differ
            a = ak.randint(0, 2**32, N, dtype=dt)
            b = ak.randint(0, 2**32, N, dtype=dt)
            self.assertFalse((a == b).all())
            # Make sure seeded results are same
            a = ak.randint(0, 2**32, N, dtype=dt, seed=seed)
            b = ak.randint(0, 2**32, N, dtype=dt, seed=seed)
            self.assertTrue((a == b).all())
        # Uniform
        self.assertFalse((ak.uniform(N) == ak.uniform(N)).all())
        self.assertTrue((ak.uniform(N, seed=seed) == ak.uniform(N, seed=seed)).all())
        # Standard Normal
        self.assertFalse((ak.standard_normal(N) == ak.standard_normal(N)).all())
        self.assertTrue((ak.standard_normal(N, seed=seed) == ak.standard_normal(N, seed=seed)).all())
        # Strings (uniformly distributed length)
        self.assertFalse((ak.random_strings_uniform(1, 10, N) == ak.random_strings_uniform(1, 10, N)).all())
        self.assertTrue((ak.random_strings_uniform(1, 10, N, seed=seed) == ak.random_strings_uniform(1, 10, N, seed=seed)).all())
        # Strings (log-normally distributed length)
        self.assertFalse((ak.random_strings_lognormal(2, 1, N) == ak.random_strings_lognormal(2, 1, N)).all())
        self.assertTrue((ak.random_strings_lognormal(2, 1, N, seed=seed) == ak.random_strings_lognormal(2, 1, N, seed=seed)).all())
            
    def testCast(self):
        N = 100
        arrays = {ak.int64: ak.randint(-(2**48), 2**48, N),
                  ak.float64: ak.randint(0, 1, N, dtype=ak.float64),
                  ak.bool: ak.randint(0, 2, N, dtype=ak.bool)}
        roundtripable = set(((ak.bool, ak.bool),
                         (ak.int64, ak.int64),
                         (ak.int64, ak.float64),
                         (ak.int64, npstr),
                         (ak.float64, ak.float64),
                         (ak.float64, npstr),
                         (ak.uint8, ak.int64),
                         (ak.uint8, ak.float64),
                         (ak.uint8, npstr)))
        for t1, orig in arrays.items():
            for t2 in ak.DTypes:
                t2 = ak.dtype(t2)
                other = ak.cast(orig, t2)
                self.assertEqual(orig.size, other.size)
                if (t1, t2) in roundtripable:
                    roundtrip = ak.cast(other, t1)
                    self.assertTrue((orig == roundtrip).all(), f"{t1}: {orig[:5]}, {t2}: {roundtrip[:5]}")
                    
        self.assertTrue((ak.array([1, 2, 3, 4, 5]) == ak.cast(ak.linspace(1,5,5), dt=ak.int64)).all())
        self.assertEqual(ak.cast(ak.arange(0,5), dt=ak.float64).dtype, ak.float64)
        self.assertTrue((ak.array([False, True, True, True, True]) == ak.cast(ak.linspace(0,4,5), dt=ak.bool)).all())
    
    def testHistogram(self):
        pda = ak.randint(10,30,40)
        result = ak.histogram(pda, bins=20)  

        self.assertIsInstance(result, ak.pdarray)
        self.assertEqual(20, len(result))
        self.assertEqual(int, result.dtype)
        
        with self.assertRaises(TypeError) as cm:
            ak.histogram([range(0,10)], bins=1)
        self.assertEqual('type of argument "pda" must be arkouda.pdarrayclass.pdarray; got list instead', 
                        cm.exception.args[0])  
        
        with self.assertRaises(TypeError) as cm:
            ak.histogram(pda, bins='1')
        self.assertEqual('type of argument "bins" must be one of (int, int64); got str instead', 
                        cm.exception.args[0])  
        
        with self.assertRaises(TypeError) as cm:
            ak.histogram([range(0,10)], bins='1')
        self.assertEqual('type of argument "pda" must be arkouda.pdarrayclass.pdarray; got list instead', 
                        cm.exception.args[0])  
    
    def testLog(self):
        na = np.linspace(1,10,10)
        pda = ak.array(na)

        self.assertTrue((np.log(na) == ak.log(pda).to_ndarray()).all())
        with self.assertRaises(TypeError) as cm:
            ak.log([range(0,10)])
        self.assertEqual('type of argument "pda" must be arkouda.pdarrayclass.pdarray; got list instead', 
                        cm.exception.args[0])  
        
    def testExp(self):
        na = np.linspace(1,10,10)
        pda = ak.array(na)

        self.assertTrue((np.exp(na) == ak.exp(pda).to_ndarray()).all())
        with self.assertRaises(TypeError) as cm:
            ak.exp([range(0,10)])
        self.assertEqual('type of argument "pda" must be arkouda.pdarrayclass.pdarray; got list instead', 
                        cm.exception.args[0])  
        
    def testAbs(self):
        na = np.linspace(1,10,10)
        pda = ak.array(na)

        self.assertTrue((np.abs(na) == ak.abs(pda).to_ndarray()).all())
        self.assertTrue((ak.arange(5,1,-1) == ak.abs(ak.arange(-5,-1))).all())
        self.assertTrue((ak.array([5,4,3,2,1]) == ak.abs(ak.linspace(-5,-1,5))).all())
        
        with self.assertRaises(TypeError) as cm:
            ak.abs([range(0,10)])
        self.assertEqual('type of argument "pda" must be arkouda.pdarrayclass.pdarray; got list instead', 
                        cm.exception.args[0])  

    def testCumSum(self):
        na = np.linspace(1,10,10)
        pda = ak.array(na)

        self.assertTrue((np.cumsum(na) == ak.cumsum(pda).to_ndarray()).all())
        with self.assertRaises(TypeError) as cm:
            ak.cumsum([range(0,10)])
        self.assertEqual('type of argument "pda" must be arkouda.pdarrayclass.pdarray; got list instead', 
                        cm.exception.args[0])  
        
    def testCumProd(self):
        na = np.linspace(1,10,10)
        pda = ak.array(na)

        self.assertTrue((np.cumprod(na) == ak.cumprod(pda).to_ndarray()).all())
        with self.assertRaises(TypeError) as cm:
            ak.cumprod([range(0,10)])
        self.assertEqual('type of argument "pda" must be arkouda.pdarrayclass.pdarray; got list instead', 
                        cm.exception.args[0])  
        
    def testSin(self):
        na = np.linspace(1,10,10)
        pda = ak.array(na)
    
        self.assertTrue((np.sin(na) == ak.sin(pda).to_ndarray()).all())
        with self.assertRaises(TypeError) as cm:
            ak.cos([range(0,10)])
        self.assertEqual('type of argument "pda" must be arkouda.pdarrayclass.pdarray; got list instead', 
                        cm.exception.args[0])  
        
    def testCos(self):
        na = np.linspace(1,10,10)
        pda = ak.array(na)
  
        self.assertTrue((np.cos(na) == ak.cos(pda).to_ndarray()).all())    
        with self.assertRaises(TypeError) as cm:
            ak.cos([range(0,10)])
        self.assertEqual('type of argument "pda" must be arkouda.pdarrayclass.pdarray; got list instead', 
                        cm.exception.args[0])    
        
    def testValueCounts(self):
        pda = ak.ones(100, dtype=ak.int64)
        result = ak.value_counts(pda)

        self.assertEqual(ak.array([1]), result[0])
        self.assertEqual(ak.array([100]), result[1])
        
        pda = ak.linspace(1,10,10)
        with self.assertRaises(RuntimeError) as cm:
            ak.value_counts(pda) 
        self.assertEqual('Error: unique: float64 not implemented', 
                        cm.exception.args[0])    

        with self.assertRaises(TypeError) as cm:  
            ak.value_counts([0]) 
        self.assertEqual('type of argument "pda" must be arkouda.pdarrayclass.pdarray; got list instead', 
                        cm.exception.args[0])   
