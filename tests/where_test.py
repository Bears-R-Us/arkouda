import numpy as np
import warnings
from itertools import product
from context import arkouda as ak
from base_test import ArkoudaTest
warnings.simplefilter("always", UserWarning)
SIZE = 10

'''
Tests the Arkouda where functionality and compares results to the
analogous Numpy were results.
'''
class WhereTest(ArkoudaTest):
    
    def setUp(self): 
        ArkoudaTest.setUp(self)
        self.npA = {'int64': np.random.randint(0, 10, SIZE),
               'float64': np.random.randn(SIZE),
               'bool': np.random.randint(0, 2, SIZE, dtype='bool')}
        self.akA = {k: ak.array(v) for k, v in self.npA.items()}
        self.npB = {'int64': np.random.randint(10, 20, SIZE),
               'float64': np.random.randn(SIZE)+10,
               'bool': np.random.randint(0, 2, SIZE, dtype='bool')}
        self.akB = {k: ak.array(v) for k, v in self.npB.items()}
        self.npCond = np.random.randint(0, 2, SIZE, dtype='bool')
        self.akCond = ak.array(self.npCond)
        self.scA = {'int64': 42, 'float64': 2.71828, 'bool': True}
        self.scB = {'int64': -1, 'float64': 3.14159, 'bool': False}
        self.dtypes = set(self.npA.keys()) 

    def test_where_equivalence(self):
        failures = 0
        tests = 0
        for dtype in self.dtypes:
            for (self.ak1, self.ak2), (self.np1, self.np2) in zip(product((self.akA, self.scA), (self.akB, self.scB)),
                                              product((self.npA, self.scA), (self.npB, self.scB))):
                tests += 1
                akres = ak.where(self.akCond, self.ak1[dtype], self.ak2[dtype]).to_ndarray()
                npres = np.where(self.npCond, self.np1[dtype], self.np2[dtype])
                if not np.allclose(akres, npres, equal_nan=True):
                    self.assertWarning(warnings.warn("{} !=\n{}".format(akres, npres)))
                    failures += 1
        self.assertEqual(0,failures)
        
    def test_error_handling(self):
        
        with self.assertRaises(TypeError) as cm:
            ak.where([0], ak.linspace(1,10,10), ak.linspace(1,10,10))
        self.assertEqual('condition must be a pdarray, not list', 
                        cm.exception.args[0]) 
        
        
