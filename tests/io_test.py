from base_test import ArkoudaTest
from util.test.generation import generate_alpha_string_array
from context import arkouda as ak
import numpy as np

'''
Tests writting Arkouda pdarrays to and from files
'''
class IOTest(ArkoudaTest):
 
    def testSaveAllLoadAllWithDict(self):
        int_tens_pdarray = ak.array(np.random.randint(-100,100,1000))
        int_hundreds_pdarray = ak.array(np.random.randint(-1000,1000,1000))
        float_pdarray = ak.array(np.random.default_rng().uniform(-100,100,1000))
 
        columns = {
          'int_tens_pdarray' : int_tens_pdarray,
          'int_hundreds_pdarray' : int_hundreds_pdarray,
          'float_pdarray' : float_pdarray
        }

        ak.save_all(columns=columns, path_prefix='/tmp/iotest_dict')
        retrieved_columns = ak.load_all('/tmp/iotest_dict')

        self.assertEqual(3, len(retrieved_columns))
        self.assertEqual(columns['int_tens_pdarray'].all(), retrieved_columns['int_tens_pdarray'].all())
        self.assertEqual(columns['int_hundreds_pdarray'].all(), retrieved_columns['int_hundreds_pdarray'].all())
        self.assertEqual(columns['float_pdarray'].all(), retrieved_columns['float_pdarray'].all())      
        self.assertEqual(3, len(ak.get_datasets('/tmp/iotest_dict_LOCALE0')))
        
    def testSaveAllLoadAllWithList(self):
        int_tens_pdarray = ak.array(np.random.randint(-100,100,1000))
        int_hundreds_pdarray = ak.array(np.random.randint(-1000,1000,1000))
        float_pdarray = ak.array(np.random.default_rng().uniform(-100,100,1000))
 
        columns = [
          int_tens_pdarray,
          int_hundreds_pdarray,
          float_pdarray
        ]

        names =  [
          'int_tens_pdarray',
          'int_hundreds_pdarray',
          'float_pdarray'
        ]

        ak.save_all(columns=columns, path_prefix='/tmp/iotest_list', names=names)
        retrieved_columns = ak.load_all('/tmp/iotest_list')

        self.assertEqual(3, len(retrieved_columns))
        self.assertEqual(columns[0].all(), retrieved_columns['int_tens_pdarray'].all())
        self.assertEqual(columns[1].all(), retrieved_columns['int_hundreds_pdarray'].all())
        self.assertEqual(columns[2].all(), retrieved_columns['float_pdarray'].all())      
        self.assertEqual(3, len(ak.get_datasets('/tmp/iotest_list_LOCALE0')))
