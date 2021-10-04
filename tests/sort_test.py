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

    def testBitBoundaryHardcode(self):

        # test hardcoded 16/17-bit boundaries with and without negative values
        a = ak.array([1, -1, 32767]) # 16 bit
        b = ak.array([1,  0, 32768]) # 16 bit
        c = ak.array([1, -1, 32768]) # 17 bit
        assert ak.is_sorted(ak.sort(a))
        assert ak.is_sorted(ak.sort(b))
        assert ak.is_sorted(ak.sort(c))

        # test hardcoded 64-bit boundaries with and without negative values
        d = ak.array([1, -1, 2**63-1])
        e = ak.array([1,  0, 2**63-1])
        f = ak.array([1, -2**63, 2**63-1])
        assert ak.is_sorted(ak.sort(d))
        assert ak.is_sorted(ak.sort(e))
        assert ak.is_sorted(ak.sort(f))

    def testBitBoundary(self):

        # test 17-bit sort
        L = -2**15
        U = 2**16
        a = ak.randint(L, U, 100)
        assert ak.is_sorted(ak.sort(a))

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
