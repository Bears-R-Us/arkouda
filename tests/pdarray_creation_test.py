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

    def test_arange(self):
        self.assertTrue((ak.array([0, 1, 2, 3, 4]) == ak.arange(0, 5, 1)).all())

        self.assertTrue((ak.array([5, 4, 3, 2, 1]) == ak.arange(5, 0, -1)).all())
        
        self.assertTrue((ak.array([-5, -6, -7, -8, -9]) == ak.arange(-5, -10, -1)).all())

        self.assertTrue((ak.array([0, 2, 4, 6, 8]) == ak.arange(0, 10, 2)).all())

    def test_randint(self):
        testArray = ak.randint(0, 10, 5)
        self.assertIsInstance(testArray, ak.pdarray)
        self.assertEqual(5, len(testArray))
        self.assertEqual(ak.int64, testArray.dtype)
        self.assertEqual([5], testArray.shape)
        
        testArray = ak.randint(np.int64(0), np.int64(10), np.int64(5))
        self.assertIsInstance(testArray, ak.pdarray)
        self.assertEqual(5, len(testArray))
        self.assertEqual(ak.int64, testArray.dtype)
        self.assertEqual([5], testArray.shape)
        
        testArray = ak.randint(np.float64(0), np.float64(10), np.int64(5))
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
        self.assertEqual('type of argument "size" must be one of (int, int64); got str instead', 
                         cm.exception.args[0])    

        with self.assertRaises(TypeError):              
            ak.randint('0',1,1000)
        
        with self.assertRaises(TypeError):              
            ak.randint(0,'1',1000)

    def test_randint_with_seed(self):
        values = ak.randint(1, 5, 10, seed=2)
        self.assertTrue((ak.array([4, 3, 1, 3, 4, 4, 2, 4, 3, 2]) == values).all())

        values = ak.randint(1, 5, 10, dtype=ak.float64, seed=2)
        self.assertTrue((ak.array([2.9160772326374946, 4.353429832157099, 4.5392023718621486, 
                                   4.4019932101126606, 3.3745324569952304, 1.1642002901528308, 
                                   4.4714086874555292, 3.7098921109084522, 4.5939589352472314, 
                                   4.0337935981006172]) == values).all())
    
        values = ak.randint(1, 5, 10, dtype=ak.bool, seed=2)
        self.assertTrue((ak.array([False, True, True, True, True, False, True, True, 
                                   True, True]) == values).all())
        
        values = ak.randint(1, 5, 10, dtype=bool, seed=2)
        self.assertTrue((ak.array([False, True, True, True, True, False, True, True, 
                                   True, True]) == values).all())

    def test_uniform(self):
        testArray = ak.uniform(3)
        self.assertIsInstance(testArray, ak.pdarray)
        self.assertEqual(ak.float64, testArray.dtype)
        self.assertEqual([3], testArray.shape)
        
        testArray = ak.uniform(np.int64(3))
        self.assertIsInstance(testArray, ak.pdarray)
        self.assertEqual(ak.float64, testArray.dtype)
        self.assertEqual([3], testArray.shape)


        uArray = ak.uniform(size=3,low=0,high=5,seed=0)
        self.assertTrue((ak.array([0.30013431967121934, 0.47383036230759112, 1.0441791878997098])
                        == uArray).all())
        
        uArray = ak.uniform(size=np.int64(3),low=np.int64(0),high=np.int64(5),seed=np.int64(0))
        self.assertTrue((ak.array([0.30013431967121934, 0.47383036230759112, 1.0441791878997098])
                        == uArray).all())
    
        with self.assertRaises(TypeError):
            ak.uniform(low='0', high=5, size=100)

        with self.assertRaises(TypeError):
            ak.uniform(low=0, high='5', size=100)

        with self.assertRaises(TypeError):
            ak.uniform(low=0, high=5, size='100')
 
    def test_zeros(self):
        intZeros = ak.zeros(5, dtype=ak.int64)
        self.assertIsInstance(intZeros, ak.pdarray)
        self.assertEqual(ak.int64,intZeros.dtype)

        floatZeros = ak.zeros(5, dtype=float)
        self.assertEqual(float,floatZeros.dtype)

        floatZeros = ak.zeros(5, dtype=ak.float64)
        self.assertEqual(ak.float64,floatZeros.dtype)

        boolZeros = ak.zeros(5, dtype=bool)
        self.assertEqual(bool,boolZeros.dtype)

        boolZeros = ak.zeros(5, dtype=ak.bool)
        self.assertEqual(ak.bool,boolZeros.dtype)
        
        zeros  = ak.zeros('5')
        self.assertEqual(5, len(zeros))

        with self.assertRaises(TypeError):
            ak.zeros(5, dtype=ak.uint8)
            
        with self.assertRaises(TypeError):
            ak.zeros(5, dtype=str)        
            
    def test_ones(self):   
        intOnes = ak.ones(5, dtype=int)
        self.assertIsInstance(intOnes, ak.pdarray)
        self.assertEqual(int,intOnes.dtype)
       
        intOnes = ak.ones(5, dtype=ak.int64)
        self.assertEqual(ak.int64,intOnes.dtype)

        floatOnes = ak.ones(5, dtype=float)
        self.assertEqual(float,floatOnes.dtype)
        
        floatOnes = ak.ones(5, dtype=ak.float64)
        self.assertEqual(ak.float64,floatOnes.dtype)

        boolOnes = ak.ones(5, dtype=bool)
        self.assertEqual(bool,boolOnes.dtype)
        
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
        
    def test_ones_like(self):      
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
        
    def test_eros_like(self):      
        intZeros = ak.zeros(5, dtype=ak.int64)
        intZerosLike = ak.zeros_like(intZeros)

        self.assertIsInstance(intZerosLike, ak.pdarray)
        self.assertEqual(ak.int64,intZerosLike.dtype)
        
        floatZeros = ak.ones(5, dtype=ak.float64)
        floatZerosLike = ak.ones_like(floatZeros)
        
        self.assertEqual(ak.float64,floatZerosLike.dtype)
        
        boolZeros = ak.ones(5, dtype=ak.bool)
        boolZerosLike = ak.ones_like(boolZeros)
        
        self.assertEqual(ak.bool,boolZerosLike.dtype)        

    def test_linspace(self):
        pda = ak.linspace(0, 100, 1000)  
        self.assertEqual(1000, len(pda))
        self.assertEqual(float, pda.dtype)
        self.assertIsInstance(pda, ak.pdarray)
        
        pda = ak.linspace(0.0, 100.0, 150)  
            
        pda = ak.linspace(start=5, stop=0, length=6)
        self.assertEqual(5.0000, pda[0])
        self.assertEqual(0.0000, pda[5])
        
        pda = ak.linspace(start=5.0, stop=0.0, length=6)
        self.assertEqual(5.0000, pda[0])
        self.assertEqual(0.0000, pda[5])
        
        pda = ak.linspace(start=np.float(5.0), stop=np.float(0.0), length=np.int64(6))
        self.assertEqual(5.0000, pda[0])
        self.assertEqual(0.0000, pda[5])
        
        with self.assertRaises(TypeError) as cm:        
            ak.linspace(0,'100', 1000)
        self.assertEqual(('both start and stop must be an int, np.int64, float, or np.float64'), 
                         cm.exception.args[0])  
        
        with self.assertRaises(TypeError) as cm:        
            ak.linspace('0',100, 1000)
        self.assertEqual(('both start and stop must be an int, np.int64, float, or np.float64'), 
                         cm.exception.args[0])  

        with self.assertRaises(TypeError) as cm:          
            ak.linspace(0,100,'1000')           
        self.assertEqual('type of argument "length" must be one of (int, int64); got str instead', 
                         cm.exception.args[0])            

    def test_standard_normal(self):
        pda = ak.standard_normal(100)
        self.assertIsInstance(pda, ak.pdarray)
        self.assertEqual(100,len(pda))
        self.assertEqual(float,pda.dtype)
        
        pda = ak.standard_normal(np.int64(100))
        self.assertIsInstance(pda, ak.pdarray)
        self.assertEqual(100,len(pda))
        self.assertEqual(float,pda.dtype)
        
        pda = ak.standard_normal(np.int64(100), np.int64(1))
        self.assertIsInstance(pda, ak.pdarray)
        self.assertEqual(100,len(pda))
        self.assertEqual(float,pda.dtype)
        
        npda = pda.to_ndarray()
        pda = ak.standard_normal(np.int64(100), np.int64(1))
        
        self.assertTrue((npda ==  pda.to_ndarray()).all())
        

        with self.assertRaises(TypeError) as cm:          
            ak.standard_normal('100')          
        self.assertEqual('type of argument "size" must be one of (int, int64); got str instead', 
                         cm.exception.args[0]) 
   
        with self.assertRaises(TypeError) as cm:          
            ak.standard_normal(100.0)          
        self.assertEqual('type of argument "size" must be one of (int, int64); got float instead', 
                         cm.exception.args[0])   
    
        with self.assertRaises(ValueError) as cm:          
            ak.standard_normal(-1)          
        self.assertEqual("The size parameter must be > 0", 
                         cm.exception.args[0])  

    def test_random_strings_uniform(self):
        pda = ak.random_strings_uniform(minlen=1, maxlen=5, size=100)
        nda = pda.to_ndarray()

        self.assertIsInstance(pda, ak.Strings)
        self.assertEqual(100, len(pda))
        self.assertEqual(str, pda.dtype)
        for string in nda:
            self.assertTrue(len(string) >= 1 and len(string) <= 5)
            self.assertTrue(string.isupper())
            
        pda = ak.random_strings_uniform(minlen=np.int64(1), maxlen=np.int64(5), size=np.int64(100))
        nda = pda.to_ndarray()

        self.assertIsInstance(pda, ak.Strings)
        self.assertEqual(100, len(pda))
        self.assertEqual(str, pda.dtype)
        for string in nda:
            self.assertTrue(len(string) >= 1 and len(string) <= 5)
            self.assertTrue(string.isupper())
        
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
        self.assertEqual('type of argument "minlen" must be one of (int, int64); got str instead', 
                         cm.exception.args[0])  
        
        with self.assertRaises(TypeError) as cm:          
            ak.random_strings_uniform(minlen=1, maxlen='5', size=10)          
        self.assertEqual('type of argument "maxlen" must be one of (int, int64); got str instead', 
                         cm.exception.args[0])     
        
        with self.assertRaises(TypeError) as cm:          
            ak.random_strings_uniform(minlen=1, maxlen=5, size='10')          
        self.assertEqual('type of argument "size" must be one of (int, int64); got str instead', 
                         cm.exception.args[0])              

    def test_random_strings_uniform_with_seed(self):
        pda = ak.random_strings_uniform(minlen=1, maxlen=5, seed=1, size=10)
 
        self.assertTrue((ak.array(['TVKJ', 'EWAB', 'CO', 'HFMD', 'U', 'MMGT', 
                        'N', 'WOQN', 'HZ', 'VSX']) == pda).all())
        
        pda = ak.random_strings_uniform(minlen=np.int64(1), maxlen=np.int64(5), seed=np.int64(1), 
                                        size=np.int64(10))
 
        self.assertTrue((ak.array(['TVKJ', 'EWAB', 'CO', 'HFMD', 'U', 'MMGT', 
                        'N', 'WOQN', 'HZ', 'VSX']) == pda).all())
        
        pda = ak.random_strings_uniform(minlen=1, maxlen=5, seed=1, size=10,
                                        characters='printable')
        self.assertTrue((ak.array(['+5"f', '-P]3', '4k', '~HFF', 'F', '`,IE', 
                        'Y', 'jkBa', '9(', '5oZ']) == pda).all())

    def test_random_strings_lognormal(self):
        pda = ak.random_strings_lognormal(2, 0.25, 100, characters='printable')
        self.assertIsInstance(pda,ak.Strings)
        self.assertEqual(100, len(pda))
        self.assertEqual(str, pda.dtype)
        
        pda = ak.random_strings_lognormal(np.int64(2), 0.25, np.int64(100), characters='printable')
        self.assertIsInstance(pda,ak.Strings)
        self.assertEqual(100, len(pda))
        self.assertEqual(str, pda.dtype)
        
        pda = ak.random_strings_lognormal(np.int64(2), np.float(0.25), np.int64(100), characters='printable')
        self.assertIsInstance(pda,ak.Strings)
        self.assertEqual(100, len(pda))
        self.assertEqual(str, pda.dtype)
        
        pda = ak.random_strings_lognormal(logmean=np.int64(2), logstd=0.25, size=np.int64(100), characters='printable', 
                                          seed=np.int64(0))
        self.assertIsInstance(pda,ak.Strings)
        self.assertEqual(100, len(pda))
        self.assertEqual(str, pda.dtype)
        
        pda = ak.random_strings_lognormal(logmean=np.float64(2), logstd=np.float64(0.25), size=np.int64(100), characters='printable', 
                                          seed=np.int64(0))
        self.assertIsInstance(pda,ak.Strings)
        self.assertEqual(100, len(pda))
        self.assertEqual(str, pda.dtype)
        
        pda = ak.random_strings_lognormal(np.float64(2), np.float64(0.25), np.int64(100), 
                                          characters='printable', seed=np.int64(0))
        self.assertIsInstance(pda,ak.Strings)
        self.assertEqual(100, len(pda))
        self.assertEqual(str, pda.dtype)
        
        with self.assertRaises(TypeError) as cm:          
            ak.random_strings_lognormal('2', 0.25, 100)          
        self.assertEqual('both logmean and logstd must be an int, np.int64, float, or np.float64', 
                         cm.exception.args[0])   
                
        with self.assertRaises(TypeError) as cm:          
            ak.random_strings_lognormal(2, 0.25, '100')          
        self.assertEqual('type of argument "size" must be one of (int, int64); got str instead', 
                         cm.exception.args[0])       
        
        with self.assertRaises(TypeError) as cm:          
            ak.random_strings_lognormal(2, 0.25, 100, 1000000)          
        self.assertEqual('type of argument "characters" must be str; got int instead', 
                         cm.exception.args[0])  
        
    def test_random_strings_lognormal_with_seed(self):
        pda = ak.random_strings_lognormal(2, 0.25, 10, seed=1)
        
        self.assertTrue((ak.array(['TVKJTE', 'ABOCORHFM', 'LUDMMGTB', 'KWOQNPHZ', 
                                   'VSXRRL', 'AKOZOEEWTB', 'GOSVGEJNOW', 'BFWSIO', 
                                   'MRIEJUSA', 'OLUKRJK'])
                        == pda).all())   
        
        pda = ak.random_strings_lognormal(np.int64(2), np.float64(0.25), np.int64(10), seed=1)
        
        self.assertTrue((ak.array(['TVKJTE', 'ABOCORHFM', 'LUDMMGTB', 'KWOQNPHZ', 
                                   'VSXRRL', 'AKOZOEEWTB', 'GOSVGEJNOW', 'BFWSIO', 
                                   'MRIEJUSA', 'OLUKRJK'])
                        == pda).all())          

        pda = ak.random_strings_lognormal(2, 0.25, 10, seed=1, characters='printable')

        self.assertTrue((ak.array(['+5"fp-', ']3Q4kC~HF', '=F=`,IE!', "DjkBa'9(", '5oZ1)=', 
                                   'T^.1@6aj";', '8b2$IX!Y7.', 'x|Y!eQ', '>1\\>2,on', '&#W":C3'])
                        == pda).all())   
        
        pda = ak.random_strings_lognormal(np.int64(2), np.float64(0.25), np.int64(10), seed=1, characters='printable')

        self.assertTrue((ak.array(['+5"fp-', ']3Q4kC~HF', '=F=`,IE!', "DjkBa'9(", '5oZ1)=', 
                                   'T^.1@6aj";', '8b2$IX!Y7.', 'x|Y!eQ', '>1\\>2,on', '&#W":C3'])
                        == pda).all())      
    
    def test_mulitdimensional_array_creation(self):
        with self.assertRaises(RuntimeError) as cm:
            ak.array([[0,0],[0,1],[1,1]])
            
        self.assertEqual('Only rank-1 pdarrays or ndarrays supported', 
                         cm.exception.args[0])
        
    def test_from_series(self):
        strings = ak.from_series(pd.Series(['a', 'b', 'c', 'd', 'e'], dtype="string"))
        
        self.assertIsInstance(strings, ak.Strings)
        self.assertEqual(5, len(strings))

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
                          'float64, int64, string, datetime64[ns], and timedelta64[ns]'), 
                         cm.exception.args[0])            
            
    def test_fill(self):
        ones = ak.ones(100)

        ones.fill(2)
        self.assertTrue((2 == ones.to_ndarray()).all())
        
        ones.fill(np.int64(2))  
        self.assertTrue((np.int64(2) == ones.to_ndarray()).all())     
        
        ones.fill(np.float(2))  
        self.assertTrue((float(2) == ones.to_ndarray()).all())  
        
        ones.fill(np.float64(2))  
        self.assertTrue((np.float64(2) == ones.to_ndarray()).all())  
