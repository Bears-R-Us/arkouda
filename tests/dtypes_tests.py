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
        with self.assertRaises(TypeError):
            dtypes.check_np_dtype('np.str')

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
        with self.assertRaises(TypeError):
            dtypes.translate_np_dtype('np.str')
            
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
        
    def test_pdarrays_datatypes(self):
        self.assertEqual(dtypes.dtype('float64'), ak.ones(10).dtype)
        self.assertEqual(dtypes.dtype('str'), 
                         ak.array(['string {}'.format(i) for i in range(0,10)]).dtype)

    def testIsSupportedInt(self):
        '''
        Tests for both True and False scenarios of the isSupportedInt method.
        '''
        self.assertTrue(dtypes.isSupportedInt(1))
        self.assertTrue(dtypes.isSupportedInt(np.int64(1)))
        self.assertTrue(dtypes.isSupportedInt(np.int64(1.0)))
        self.assertFalse(dtypes.isSupportedInt(1.0))
        self.assertFalse(dtypes.isSupportedInt('1'))
        self.assertFalse(dtypes.isSupportedInt('1.0'))
        
    def testIsSupportedFloat(self):
        '''
        Tests for both True and False scenarios of the isSupportedFloat method.
        '''
        self.assertTrue(dtypes.isSupportedFloat(1.0))
        self.assertTrue(dtypes.isSupportedFloat(float(1)))
        self.assertTrue(dtypes.isSupportedFloat(np.float64(1.0)))
        self.assertTrue(dtypes.isSupportedFloat(np.float64(1)))
        self.assertFalse(dtypes.isSupportedFloat(np.int64(1.0)))
        self.assertFalse(dtypes.isSupportedFloat(int(1.0)))
        self.assertFalse(dtypes.isSupportedFloat('1'))
        self.assertFalse(dtypes.isSupportedFloat('1.0'))