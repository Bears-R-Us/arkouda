import numpy as np
import pandas as pd
import datetime as dt
from collections import deque
from base_test import ArkoudaTest
from context import arkouda as ak

'''
Encapsulates test cases for pdarray creation methods
'''
class PdarrayCreationTest(ArkoudaTest):
 
    def testArrayCreation(self):
        pda = ak.array(np.ones(100))
        self.assertIsInstance(pda, ak.pdarray)
        self.assertEqual(100, len(pda))
        self.assertEqual(float, pda.dtype)
        
        pda =  ak.array(list(range(0,100)))
        self.assertIsInstance(pda, ak.pdarray)
        self.assertEqual(100, len(pda))
        self.assertEqual(int, pda.dtype)        

        pda =  ak.array((range(5)))
        self.assertIsInstance(pda, ak.pdarray)
        self.assertEqual(5, len(pda))
        self.assertEqual(int, pda.dtype) 
        
        pda =  ak.array(deque(range(5)))
        self.assertIsInstance(pda, ak.pdarray)
        self.assertEqual(5, len(pda))
        self.assertEqual(int, pda.dtype)         

        with self.assertRaises(RuntimeError) as cm:          
            ak.array({range(0,100)})         
        self.assertEqual("Only rank-1 pdarrays or ndarrays supported", 
                         cm.exception.args[0])  
        
        with self.assertRaises(RuntimeError) as cm:          
            ak.array(np.array([[0,1],[0,1]]))         
        self.assertEqual("Only rank-1 pdarrays or ndarrays supported", 
                         cm.exception.args[0])  

        with self.assertRaises(RuntimeError) as cm:          
            ak.array('not an iterable')          
        self.assertEqual("Only rank-1 pdarrays or ndarrays supported", 
                         cm.exception.args[0]) 
        
        with self.assertRaises(TypeError) as cm:          
            ak.array(list(list(0)))          
        self.assertEqual("'int' object is not iterable", 
                         cm.exception.args[0])       

    def testRandint(self):
        testArray = ak.randint(0, 10, 5)
        self.assertIsInstance(testArray, ak.pdarray)
        self.assertEqual(5, len(testArray))
        self.assertEqual(ak.int64, testArray.dtype)
        self.assertEqual([5], testArray.shape)
        
        test_ndarray = testArray.to_ndarray()
        
        for value in test_ndarray:
            self.assertTrue(0 <= value <= 10)
                          
        test_array = ak.randint(0, 1, 3, dtype=ak.float64)
        self.assertEqual(ak.float64, test_array.dtype)
        
        test_array = ak.randint(0, 1, 5, dtype=ak.bool)
        self.assertEqual(ak.bool, test_array.dtype)
        
        test_ndarray = test_array.to_ndarray()
        
        for value in test_ndarray:
            self.assertTrue(value in [True,False])
           
        with self.assertRaises(TypeError):
            ak.randint(low=5)
            
        with self.assertRaises(TypeError):
            ak.randint(high=5)

        with self.assertRaises(TypeError):            
            ak.randint()

        with self.assertRaises(ValueError) as cm:
            ak.randint(low=0, high=1, size=-1, dtype=ak.float64)
        self.assertEqual("size must be > 0 and high > low", 
                         cm.exception.args[0])    
 
        with self.assertRaises(ValueError) as cm:
            ak.randint(low=1, high=0, size=1, dtype=ak.float64)  
        self.assertEqual("size must be > 0 and high > low", 
                         cm.exception.args[0])             

        with self.assertRaises(TypeError) as cm:              
            ak.randint(0,1,'1000')
        self.assertEqual("The size parameter must be an integer", 
                         cm.exception.args[0])    

        with self.assertRaises(TypeError) as cm:              
            ak.randint('0',1,1000)
        self.assertEqual("The low parameter must be an integer or float", 
                         cm.exception.args[0])     
        
        with self.assertRaises(TypeError) as cm:              
            ak.randint(0,'1',1000)
        self.assertEqual("The high parameter must be an integer or float", 
                         cm.exception.args[0])     
    
    def testUniform(self):
        testArray = ak.uniform(3)
        self.assertIsInstance(testArray, ak.pdarray)
        self.assertEqual(ak.float64, testArray.dtype)
        self.assertEqual([3], testArray.shape)

        with self.assertRaises(TypeError):
            ak.uniform(low=5)
    
        with self.assertRaises(TypeError) as cm:
            ak.uniform(low='0', high=5, size=100)
        self.assertEqual('type of argument "low" must be either float or int; got str instead', 
                         cm.exception.args[0])   
            
        with self.assertRaises(TypeError) as cm:
            ak.uniform(low=0, high='5', size=100)
        self.assertEqual('type of argument "high" must be either float or int; got str instead', 
                         cm.exception.args[0])   
        
        with self.assertRaises(TypeError) as cm:
            ak.uniform(low=0, high=5, size='100')
        self.assertEqual('type of argument "size" must be int; got str instead', 
                         cm.exception.args[0])  
 
    def testZeros(self):
        intZeros = ak.zeros(5, dtype=ak.int64)
        self.assertIsInstance(intZeros, ak.pdarray)
        self.assertEqual(ak.int64,intZeros.dtype)
        
        floatZeros = ak.zeros(5, dtype=ak.float64)
        self.assertEqual(ak.float64,floatZeros.dtype)
        
        boolZeros = ak.zeros(5, dtype=ak.bool)
        self.assertEqual(ak.bool,boolZeros.dtype)
        
        zeros  = ak.zeros('5')
        self.assertEqual(5, len(zeros))

        with self.assertRaises(TypeError):
            ak.zeros(5, dtype=ak.uint8)
            
        with self.assertRaises(TypeError):
            ak.zeros(5, dtype=str)        
            
    def testOnes(self):
        intOnes = ak.ones(5, dtype=ak.int64)
        self.assertIsInstance(intOnes, ak.pdarray)
        self.assertEqual(ak.int64,intOnes.dtype)
        
        floatOnes = ak.ones(5, dtype=ak.float64)
        self.assertEqual(ak.float64,floatOnes.dtype)
        
        boolOnes = ak.ones(5, dtype=ak.bool)
        self.assertEqual(ak.bool,boolOnes.dtype)

        ones = ak.ones('5')
        self.assertEqual(5, len(ones))
        
        with self.assertRaises(TypeError) as cm:
            ak.ones(5, dtype=ak.uint8)
        self.assertEqual('unsupported dtype uint8', 
                         cm.exception.args[0])  
                    
        with self.assertRaises(TypeError) as cm:
            ak.ones(5, dtype=str)
        self.assertEqual('unsupported dtype <U0', 
                         cm.exception.args[0])     
        
    def testOnesLike(self):      
        intOnes = ak.ones(5, dtype=ak.int64)
        intOnesLike = ak.ones_like(intOnes)

        self.assertIsInstance(intOnesLike, ak.pdarray)
        self.assertEqual(ak.int64,intOnesLike.dtype)
        
        floatOnes = ak.ones(5, dtype=ak.float64)
        floatOnesLike = ak.ones_like(floatOnes)
        
        self.assertEqual(ak.float64,floatOnesLike.dtype)
        
        boolOnes = ak.ones(5, dtype=ak.bool)
        boolOnesLike = ak.ones_like(boolOnes)
        
        self.assertEqual(ak.bool,boolOnesLike.dtype)        

    def testLinspace(self):
        pda = ak.linspace(0, 100, 1000)  
        self.assertEqual(1000, len(pda))
        self.assertEqual(float, pda.dtype)
        self.assertIsInstance(pda, ak.pdarray)
        
        pda = ak.linspace(start=5, stop=0, length=6)
        self.assertEqual(5.0000, pda[0])
        self.assertEqual(0.0000, pda[5])
        
        with self.assertRaises(TypeError) as cm:        
            ak.linspace(0,'100', 1000)
        self.assertEqual(("The stop parameter must be an int or a" +
                         " scalar that can be parsed to an int, but is a 'str'"), 
                         cm.exception.args[0])  
        
        with self.assertRaises(TypeError) as cm:        
            ak.linspace('0',100, 1000)
        self.assertEqual(("The start parameter must be an int or a" +
                         " scalar that can be parsed to an int, but is a 'str'"), 
                         cm.exception.args[0])  

        with self.assertRaises(TypeError) as cm:          
            ak.linspace(0,100,'1000')           
        self.assertEqual("The length parameter must be an int64", 
                         cm.exception.args[0])            

    def test_standard_normal(self):
        pda = ak.standard_normal(100)
        self.assertIsInstance(pda, ak.pdarray)
        self.assertEqual(100,len(pda))
        self.assertEqual(float,pda.dtype)

        with self.assertRaises(TypeError) as cm:          
            ak.standard_normal('100')          
        self.assertEqual('type of argument "size" must be int; got str instead', 
                         cm.exception.args[0]) 
   
        with self.assertRaises(TypeError) as cm:          
            ak.standard_normal(100.0)          
        self.assertEqual('type of argument "size" must be int; got float instead', 
                         cm.exception.args[0])   
    
        with self.assertRaises(ValueError) as cm:          
            ak.standard_normal(-1)          
        self.assertEqual("The size parameter must be > 0", 
                         cm.exception.args[0])  

    def test_random_strings_uniform(self):
        pda = ak.random_strings_uniform(minlen=1, maxlen=10, size=100)
        self.assertIsInstance(pda, ak.Strings)
        self.assertEqual(100, len(pda))
        self.assertEqual(str, pda.dtype)
        
        with self.assertRaises(ValueError) as cm:          
            ak.random_strings_uniform(maxlen=1,minlen=5, size=100)          
        self.assertEqual("Incompatible arguments: minlen < 0, maxlen < minlen, or size < 0", 
                         cm.exception.args[0])   
        
        with self.assertRaises(ValueError) as cm:          
            ak.random_strings_uniform(maxlen=5,minlen=1, size=-1)          
        self.assertEqual("Incompatible arguments: minlen < 0, maxlen < minlen, or size < 0", 
                         cm.exception.args[0])    
        
        with self.assertRaises(TypeError) as cm:          
            ak.random_strings_uniform(minlen='1', maxlen=5, size=10)          
        self.assertEqual('type of argument "minlen" must be int; got str instead', 
                         cm.exception.args[0])  
        
        with self.assertRaises(TypeError) as cm:          
            ak.random_strings_uniform(minlen=1, maxlen='5', size=10)          
        self.assertEqual('type of argument "maxlen" must be int; got str instead', 
                         cm.exception.args[0])     
        
        with self.assertRaises(TypeError) as cm:          
            ak.random_strings_uniform(minlen=1, maxlen=5, size='10')          
        self.assertEqual('type of argument "size" must be int; got str instead', 
                         cm.exception.args[0])              

    def test_random_strings_lognormal(self):
        pda = ak.random_strings_lognormal(2, 0.25, 100, characters='printable')
        self.assertIsInstance(pda,ak.Strings)
        self.assertEqual(100, len(pda))
        self.assertEqual(str, pda.dtype)
        
        with self.assertRaises(TypeError) as cm:          
            ak.random_strings_lognormal('2', 0.25, 100)          
        self.assertEqual('type of argument "logmean" must be one of (float, int); got str instead', 
                         cm.exception.args[0])   
                
        with self.assertRaises(TypeError) as cm:          
            ak.random_strings_lognormal(2, 0.25, '100')          
        self.assertEqual('type of argument "size" must be int; got str instead', 
                         cm.exception.args[0])       
        
        with self.assertRaises(TypeError) as cm:          
            ak.random_strings_lognormal(2, 0.25, 100, 1000000)          
        self.assertEqual('type of argument "characters" must be str; got int instead', 
                         cm.exception.args[0])         
    
    def testMulitdimensionalArrayCreation(self):
        with self.assertRaises(RuntimeError) as cm:
            ak.array([[0,0],[0,1],[1,1]])
            
        self.assertEqual('Only rank-1 pdarrays or ndarrays supported', 
                         cm.exception.args[0])
        
    def test_from_series(self):
        strings = ak.from_series(pd.Series(['a', 'b', 'c', 'd', 'e'], dtype="string"))
        
        self.assertIsInstance(strings, ak.Strings)
        self.assertEqual(5, len(strings))
        
        print(type(np.str))
        objects = ak.from_series(pd.Series(['a', 'b', 'c', 'd', 'e']), dtype=np.str)
        
        self.assertIsInstance(objects, ak.Strings)
        self.assertEqual(np.str, objects.dtype)

        p_array = ak.from_series(pd.Series(np.random.randint(0,10,10)))

        self.assertIsInstance(p_array,ak.pdarray)
        self.assertEqual(np.int64, p_array.dtype)
    
        p_i_objects_array = ak.from_series(pd.Series(np.random.randint(0,10,10), 
                                                   dtype='object'), dtype=np.int64)

        self.assertIsInstance(p_i_objects_array,ak.pdarray)
        self.assertEqual(np.int64, p_i_objects_array.dtype)
        
        p_array = ak.from_series(pd.Series(np.random.uniform(low=0.0,high=1.0,size=10)))

        self.assertIsInstance(p_array,ak.pdarray)
        self.assertEqual(np.float64, p_array.dtype)    
        
        p_f_objects_array = ak.from_series(pd.Series(np.random.uniform(low=0.0,high=1.0,size=10), 
                                           dtype='object'), dtype=np.float64)

        self.assertIsInstance(p_f_objects_array,ak.pdarray)
        self.assertEqual(np.float64, p_f_objects_array.dtype)          
        
        p_array = ak.from_series(pd.Series(np.random.choice([True, False],size=10)))

        self.assertIsInstance(p_array,ak.pdarray)
        self.assertEqual(bool, p_array.dtype)       
        
        p_b_objects_array = ak.from_series(pd.Series(np.random.choice([True, False],size=10), 
                                            dtype='object'), dtype=np.bool)

        self.assertIsInstance( p_b_objects_array,ak.pdarray)
        self.assertEqual(bool, p_b_objects_array.dtype)     

        p_array = ak.from_series(pd.Series([dt.datetime(2016,1,1,0,0,1)]))
        
        self.assertIsInstance(p_array,ak.pdarray)
        self.assertEqual(np.int64, p_array.dtype)   

        p_array = ak.from_series(pd.Series([np.datetime64('2018-01-01')]))
        
        self.assertIsInstance(p_array,ak.pdarray)
        self.assertEqual(np.int64, p_array.dtype)   
        
        p_array = ak.from_series(pd.Series(pd.to_datetime(['1/1/2018', 
                                    np.datetime64('2018-01-01'), dt.datetime(2018, 1, 1)])))
        
        self.assertIsInstance(p_array,ak.pdarray)
        self.assertEqual(np.int64, p_array.dtype)  
  
        with self.assertRaises(TypeError) as cm:          
            ak.from_series(np.ones(100))        
        self.assertEqual(('type of argument "series" must be pandas.core.series.Series; ' +
                         'got numpy.ndarray instead'), 
                         cm.exception.args[0])    

        with self.assertRaises(ValueError) as cm:          
            ak.from_series(pd.Series(np.random.randint(0,10,10), dtype=np.int8))        
        self.assertEqual(('dtype int8 is unsupported. Supported dtypes are bool, ' +
                          'float64, int64, string, and datetime64[ns]'), 
                         cm.exception.args[0])    
