import numpy as np
from context import arkouda as ak
from arkouda import dtypes
from base_test import ArkoudaTest

'''
DtypesTest encapsulates arkouda dtypes module methods
'''
class DtypesTest(ArkoudaTest):

    def test_check_np_dtype(self):
      
        '''
        Tests dtypes.check_np_dtype method 
        
        :return: None
        :raise: AssertionError if 1.. test cases fail
        '''
        dtypes.check_np_dtype(np.dtype(np.bool))
        dtypes.check_np_dtype(np.dtype(np.int64))
        dtypes.check_np_dtype(np.dtype(np.float64))
        dtypes.check_np_dtype(np.dtype(np.uint8))
        dtypes.check_np_dtype(np.dtype(np.str))
        with self.assertRaises(TypeError):
            dtypes.check_np_dtype(np.dtype(np.int16))

    def test_translate_np_dtype(self):
        '''
        Tests dtypes.translate_np_dtype method
        
        :return: None
        :raise: AssertionError if 1.. test cases fail
        '''
        d_tuple = dtypes.translate_np_dtype(np.dtype(np.bool))
        self.assertEqual(1, d_tuple[1])
        self.assertEqual('bool', d_tuple[0])
        d_tuple = dtypes.translate_np_dtype(np.dtype(np.int64))
        self.assertEqual(8, d_tuple[1])
        self.assertEqual('int', d_tuple[0])  
        d_tuple = dtypes.translate_np_dtype(np.dtype(np.float64))
        self.assertEqual(8, d_tuple[1])
        self.assertEqual('float', d_tuple[0])        
        d_tuple = dtypes.translate_np_dtype(np.dtype(np.uint8))
        self.assertEqual(1, d_tuple[1])
        self.assertEqual('uint',d_tuple[0])    
        d_tuple = dtypes.translate_np_dtype(np.dtype(np.str))
        self.assertEqual(0, d_tuple[1])
        self.assertEqual('str', d_tuple[0])   
        with self.assertRaises(TypeError):
            dtypes.check_np_dtype(np.dtype(np.int16))
            
    def test_resolve_scalar_dtype(self):
        '''
        Tests dtypes.resolve_scalar_dtype method
        
        :return: None
        :raise: AssertionError if 1.. test cases fail
        '''
        self.assertEqual('bool', dtypes.resolve_scalar_dtype(True))
        self.assertEqual('int64', dtypes.resolve_scalar_dtype(1))
        self.assertEqual('float64', dtypes.resolve_scalar_dtype(float(0.0)))
        self.assertEqual('str', dtypes.resolve_scalar_dtype('test'))
        self.assertEqual('int64', dtypes.resolve_scalar_dtype(np.int64(1))) 
        self.assertEqual("<class 'list'>", dtypes.resolve_scalar_dtype([1]))
      