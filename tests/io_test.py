from base_test import ArkoudaTest
from util.test.generation import generate_alpha_string_array
from context import arkouda as ak
import numpy as np

'''
Tests writting Arkouda pdarrays to and from files
'''
class IOTest(ArkoudaTest):
 
    def testSaveAllWithDict(self):
        string_pdarray = ak.array(generate_alpha_string_array(array_length=1000, 
                                                       string_length=25, uppercase=True))
        int_tens_pdarray = ak.array(np.random.randint(-100,100,1000))
        int_hundreds_pdarray = ak.array(np.random.randint(-1000,1000,1000))
        float_pdarray = ak.array(np.random.default_rng().uniform(-100,100,1000))
 
        columns = {
          #'string_pdarray' : string_pdarray.bytes,
          #'string_pdarray_offsets' : string_pdarray.offsets,
          'int_tens_pdarray' : int_tens_pdarray,
          'int_hundreds_pdarray' : int_hundreds_pdarray,
          'float_pdarray' : float_pdarray
        }
        ak.save_all(columns=columns, path_prefix='/tmp/')
        
    
  