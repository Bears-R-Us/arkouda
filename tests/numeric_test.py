from context import arkouda as ak
from base_test import ArkoudaTest

"""
Encapsulates unit tests for the numeric module with the exception
of the where method, which is in the where_test module
"""
class NumericTest(ArkoudaTest):
    
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
        self.assertEqual('type of argument "bins" must be int; got str instead', 
                        cm.exception.args[0])  
        
        with self.assertRaises(TypeError) as cm:
            ak.histogram([range(0,10)], bins='1')
        self.assertEqual('type of argument "pda" must be arkouda.pdarrayclass.pdarray; got list instead', 
                        cm.exception.args[0])  
    
    def testLog(self):
        pda = ak.linspace(1,10,10)
        result = ak.log(pda) 

        self.assertIsInstance(result, ak.pdarray)
        self.assertEqual(10, len(result))
        self.assertEqual(float, result.dtype)
        
        with self.assertRaises(TypeError) as cm:
            ak.log([range(0,10)])
        self.assertEqual('type of argument "pda" must be arkouda.pdarrayclass.pdarray; got list instead', 
                        cm.exception.args[0])  
        
    def testExp(self):
        pda = ak.linspace(1,10,10)
        result = ak.exp(pda) 

        self.assertIsInstance(result, ak.pdarray)
        self.assertEqual(10, len(result))
        self.assertEqual(float, result.dtype)
        
        with self.assertRaises(TypeError) as cm:
            ak.exp([range(0,10)])
        self.assertEqual('type of argument "pda" must be arkouda.pdarrayclass.pdarray; got list instead', 
                        cm.exception.args[0])  
        
    def testAbs(self):
        pda = ak.linspace(1,10,10)
        result = ak.abs(pda) 

        self.assertIsInstance(result, ak.pdarray)
        self.assertEqual(10, len(result))
        self.assertEqual(float, result.dtype)
        
        with self.assertRaises(TypeError) as cm:
            ak.abs([range(0,10)])
        self.assertEqual('type of argument "pda" must be arkouda.pdarrayclass.pdarray; got list instead', 
                        cm.exception.args[0])  
        
        
    def testCumSum(self):
        pda = ak.linspace(1,10,10)
        result = ak.cumsum(pda) 

        self.assertIsInstance(result, ak.pdarray)
        self.assertEqual(10, len(result))
        self.assertEqual(float, result.dtype)
        
        with self.assertRaises(TypeError) as cm:
            ak.cumsum([range(0,10)])
        self.assertEqual('type of argument "pda" must be arkouda.pdarrayclass.pdarray; got list instead', 
                        cm.exception.args[0])  
        
    def testCumProd(self):
        pda = ak.linspace(1,10,10)
        result = ak.cumprod(pda) 

        self.assertIsInstance(result, ak.pdarray)
        self.assertEqual(10, len(result))
        self.assertEqual(float, result.dtype)
        
        with self.assertRaises(TypeError) as cm:
            ak.cumprod([range(0,10)])
        self.assertEqual('type of argument "pda" must be arkouda.pdarrayclass.pdarray; got list instead', 
                        cm.exception.args[0])  
        
    def testSin(self):
        pda = ak.linspace(1,10,10)
        result = ak.sin(pda) 

        self.assertIsInstance(result, ak.pdarray)
        self.assertEqual(10, len(result))
        self.assertEqual(float, result.dtype)
        
        with self.assertRaises(TypeError) as cm:
            ak.sin([range(0,10)])
        self.assertEqual('type of argument "pda" must be arkouda.pdarrayclass.pdarray; got list instead', 
                        cm.exception.args[0]) 
        
    def testCos(self):
        pda = ak.linspace(1,10,10)
        result = ak.cos(pda) 

        self.assertIsInstance(result, ak.pdarray)
        self.assertEqual(10, len(result))
        self.assertEqual(float, result.dtype)
        
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
        
            