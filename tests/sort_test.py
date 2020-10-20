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
        bools = ak.randint(0, 1, 1000, dtype=ak.bool)        
        with self.assertRaises(RuntimeError) as cm:
            ak.sort(bools)
        self.assertEqual('Error: sortMsg: bool not implemented', 
                         cm.exception.args[0])

        # Test TypeError from sort attempt on non-pdarray
        with self.assertRaises(TypeError):
            ak.sort(list(range(0,10)))  
            
        with self.assertRaises(TypeError):
            ak.local_argsort(list(range(0,10)))         
        
        with self.assertRaises(TypeError):
            ak.sort(ak.array(['String {}'.format(i) for i in range(0,10)]))
            
        with self.assertRaises(RuntimeError) as cm:
            ak.concatenate([ak.array([True]),ak.array([True])]).is_sorted()
        self.assertEqual('Error: reductionMsg: is_sorted bool not implemented', 
                         cm.exception.args[0])