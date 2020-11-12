from base_test import ArkoudaTest
from context import arkouda
from arkouda.decorators import rangechecker


class DecoratorsTest(ArkoudaTest):
    
    def testRangechecker(self):
        
        @rangechecker(lowerField='low', lowerBound=0, upperField='high', upperBound=100)
        def testMethod(**args):
            pass
        
        with self.assertRaises(ValueError) as cm:
            testMethod(low=-1,high=100)
        self.assertEqual('the low value must be >= 0', 
                        cm.exception.args[0]) 
        
        with self.assertRaises(ValueError) as cm:
            testMethod(low=0,high=101)
        self.assertEqual('the high value must be <= 100', 
                        cm.exception.args[0]) 
        
        with self.assertRaises(ValueError) as cm:
            testMethod(low=5,high=1)
        self.assertEqual('the low value 5 must be less than the high value 1', 
                        cm.exception.args[0]) 