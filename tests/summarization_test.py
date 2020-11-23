import numpy as np
from context import arkouda as ak
from base_test import ArkoudaTest

"""
Encapsulates unit tests for the pdarrayclass module that provide
summarized values via reduction methods
"""
class SummarizationTest(ArkoudaTest):
    
    def setUp(self):
        ArkoudaTest.setUp(self)
        self.na = np.linspace(1,10,10)
        self.pda = ak.array(self.na)
    
    def testStd(self):
        self.assertEqual(self.na.std(), self.pda.std())      
        
    def testMin(self):
        self.assertEqual(self.na.min(), self.pda.min())   
    
    def testMax(self):
        self.assertEqual(self.na.max(), self.pda.max())   
        
    def testMean(self):
        self.assertEqual(self.na.mean(), self.pda.mean())   
        
    def testVar(self):
        self.assertEqual(self.na.var(), self.pda.var())   

    def testAny(self):
        self.assertEqual(self.na.any(), self.pda.any()) 
        
    def testAll(self):
        self.assertEqual(self.na.all(), self.pda.all()) 