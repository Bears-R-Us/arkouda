from context import arkouda as ak 
from base_test import ArkoudaTest


'''
Encapsulates test cases that test sort functionality
'''
class SortTest(ArkoudaTest):

    def testSort(self):
        pda = ak.randint(0,100,100)
        spda = ak.sort(pda)
        maxIndex = spda.argmax()
        self.assertTrue(maxIndex > 0)
   
    def testErrorHandling(self):
        
        # Test RuntimeError from bool NotImplementedError
        akbools = ak.randint(0, 1, 1000, dtype=ak.bool)   
        bools = ak.randint(0, 1, 1000, dtype=bool) 
     
        with self.assertRaises(ValueError) as cm:
            ak.sort(akbools)
        self.assertEqual('ak.sort supports float64 or int64, not bool', 
                         cm.exception.args[0])
        
        with self.assertRaises(ValueError) as cm:
            ak.sort(bools)
        self.assertEqual('ak.sort supports float64 or int64, not bool', 
                         cm.exception.args[0])        
        
        # Test TypeError from sort attempt on non-pdarray
        with self.assertRaises(TypeError):
            ak.sort(list(range(0,10)))  
                
        # Test attempt to sort Strings object, which is unsupported
        with self.assertRaises(TypeError):
            ak.sort(ak.array(['String {}'.format(i) for i in range(0,10)]))
