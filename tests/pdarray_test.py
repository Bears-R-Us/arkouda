import unittest
import arkouda as ak
import numpy as np
import subprocess, os
from unittest.case import skip

class PdArrayTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        try: 
            arkouda_path = os.environ['ARKOUDA_SERVER_PATH']
        except KeyError:
            raise EnvironmentError(('The ARKOUDA_SERVER_PATH env variable output from the ' + 
                                   '"which arkouda_server" command must be set'))
        
        PdArrayTest.ak_server = subprocess.Popen([arkouda_path, '--ServerPort=5566', '--quiet'])
    
    def setUp(self):
        ak.client.connect(port=5566)
      
    def testPdArrayAddInt(self):
        aArray = ak.ones(100)

        addArray = aArray + 1
        self.assertTrue(isinstance(addArray, ak.pdarray))
        self.assertEqual(np.float64(2), addArray[0])

        addArray = 1 + aArray
        self.assertTrue(isinstance(addArray, ak.pdarray))
        self.assertEqual(np.float64(2), addArray[0])

    def testPdArrayAddNumpyInt(self):
        aArray = ak.ones(100)

        addArray = aArray + np.int64(1)
        self.assertTrue(isinstance(addArray, ak.pdarray))
        self.assertEqual(np.float64(2), addArray[0])
        
        addArray = np.int64(1) + aArray
        self.assertTrue(isinstance(addArray, ak.pdarray))
        self.assertEqual(np.float64(2), addArray[0])
    
    def testPdArraySubtractInt(self):
        aArray = ak.ones(100)
        addArray =  aArray - 2
        self.assertTrue(isinstance(addArray, ak.pdarray))
        self.assertEqual(np.float64(-1), addArray[0])

        addArray =  2 - aArray
        self.assertTrue(isinstance(addArray, ak.pdarray))
        self.assertEqual(np.float64(1), addArray[0])
    
    def testPdArraySubtractNumpyInt(self):
        aArray = ak.ones(100)
        addArray =  aArray - np.int64(2)
        self.assertTrue(isinstance(addArray, ak.pdarray))
        self.assertEqual(np.float64(-1), addArray[0])

        addArray =  np.int64(2) - aArray
        self.assertTrue(isinstance(addArray, ak.pdarray))
        self.assertEqual(np.float64(1), addArray[0])
        
    def tearDown(self):
        ak.client.disconnect()
        
    @classmethod
    def tearDownClass(cls):
        PdArrayTest.ak_server.terminate()        