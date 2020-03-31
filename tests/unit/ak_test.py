import arkouda as ak
import numpy as np
from unit.base_test import ArkoudaTest
from arkouda.pdarrayclass import pdarray

class AkTest(ArkoudaTest):
    
    def testOnes(self):
        ones = ak.ones(size=100)

        self.assertEquals(100, len(ones))
        self.assertTrue(isinstance(ones, pdarray))
        self.assertEqual(np.int64(1), ones[0])
        self.assertTrue(isinstance(ones[0], np.float64))
        
        ones = ak.ones(size=100, dtype=np.int64)
        
        self.assertEqual(np.float64(1), ones[0])
        self.assertTrue(isinstance(ones[0], np.int64))
        
        ones = ak.ones(size=100, dtype=np.bool)
        
        self.assertEqual(np.bool(1), ones[0])
        self.assertTrue(isinstance(ones[0], np.bool))
        self.assertTrue(ones[0])        
        
    def testZeros(self):
        ones = ak.zeros(size=100)

        self.assertEquals(100, len(ones))
        self.assertTrue(isinstance(ones, pdarray))
        self.assertEqual(np.int64(0), ones[0])
        self.assertTrue(isinstance(ones[0], np.float64))
        
        ones = ak.zeros(size=100, dtype=np.int64)
        
        self.assertEqual(np.float64(0), ones[0])
        self.assertTrue(isinstance(ones[0], np.int64))
        
        ones = ak.zeros(size=100, dtype=np.bool)
        
        self.assertEqual(np.bool(0), ones[0])
        self.assertTrue(isinstance(ones[0], np.bool))
        self.assertFalse(ones[0])    
        
        
    def testArange(self):
        a_range = ak.arange(0,5,1)
        
        self.assertTrue(isinstance(a_range, pdarray))
        self.assertEqual(5, len(a_range))
        self.assertEqual(0, a_range[0])
        self.assertEqual(4, a_range[4])
        
        a_range = ak.arange(2,10,2)
        
        self.assertTrue(isinstance(a_range, pdarray))
        self.assertEqual(4, len(a_range))
        self.assertEqual(2, a_range[0])
        self.assertEqual(8, a_range[3])
        