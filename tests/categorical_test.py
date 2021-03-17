import numpy as np
from context import arkouda as ak
from base_test import ArkoudaTest


class CategoricalTest(ArkoudaTest):
    
    def _getCategorical(self, prefix : str='string', size : int=11) -> ak.Categorical:
        return ak.Categorical(ak.array(['{} {}'.format(prefix,i) for i in range(1,size)]))
    
    def _getRandomizedCategorical(self) -> ak.Categorical:
        return ak.Categorical(ak.array(['string', 'string1', 'non-string', 'non-string2', 
                                        'string', 'non-string', 'string3','non-string2', 
                                        'string', 'non-string']))
    
    def testBaseCategorical(self):
        cat = self._getCategorical()

        self.assertTrue((ak.array([7,5,9,8,2,1,4,0,3,6]) == cat.codes).all())
        self.assertTrue((ak.array([0,1,2,3,4,5,6,7,8,9]) == cat.segments).all())
        self.assertTrue((ak.array(['string 8', 'string 6', 'string 5', 'string 9', 
                                    'string 7', 'string 2', 'string 10', 'string 1', 
                                    'string 4', 'string 3']) == cat.categories).all())
        self.assertEqual(10,cat.size)
        self.assertEqual('category',cat.objtype)
        
        with self.assertRaises(ValueError) as cm:
            ak.Categorical(ak.arange(0,5,10))
        self.assertEqual('Categorical: inputs other than Strings not yet supported', 
                         cm.exception.args[0])        
        
    def testCategoricalFromCodesAndCategories(self):
        codes = ak.array([7,5,9,8,2,1,4,0,3,6])
        categories = ak.array(['string 8', 'string 6', 'string 5', 'string 9', 
                                    'string 7', 'string 2', 'string 10', 'string 1', 
                                    'string 4', 'string 3'])
        
        cat = ak.Categorical.from_codes(codes, categories)
        self.assertTrue((codes == cat.codes).all())
        self.assertTrue((categories == cat.categories).all())
        
    def testContains(self):
        cat = self._getCategorical()
        self.assertTrue(cat.contains('string').all())
        
    def testEndsWith(self):
        cat = self._getCategorical()
        self.assertTrue(cat.endswith('1').any())
        
    def testStartsWith(self):
        cat = self._getCategorical()
        self.assertTrue(cat.startswith('string').all())
        
    def testGroup(self):
        group = self._getRandomizedCategorical().group()
        self.assertTrue((ak.array([2,5,9,6,1,3,7,0,4,8]) == group).all())
        
    def testUnique(self):
        cat = self._getRandomizedCategorical()
        
        self.assertTrue((ak.Categorical(ak.array(['non-string', 'string3', 'string1', 
                                        'non-string2', 'string'])).to_ndarray() 
                                                  == cat.unique().to_ndarray()).all())  
        
    def testToNdarray(self):
        cat = self._getRandomizedCategorical()
        ndcat = np.array(['string', 'string1', 'non-string', 'non-string2', 
                            'string', 'non-string', 'string3','non-string2', 
                            'string', 'non-string'])
        self.assertTrue((cat.to_ndarray() == ndcat).all())
        
    def testEquality(self):
        cat = self._getCategorical()
        catDupe = self._getCategorical()
        catNonDupe = self._getRandomizedCategorical()
        
        self.assertTrue((cat == catDupe).all())
        self.assertTrue((cat != catNonDupe).all())
        
    def testBinop(self):
        cat = self._getCategorical()
        catDupe = self._getCategorical()
        catNonDupe = self._getRandomizedCategorical()

        
        self.assertTrue((cat._binop(catDupe,'==')).all())
        self.assertTrue((cat._binop(catNonDupe,'!=')).all())
        
        self.assertTrue((ak.array([True,True,True,True,True,True,True,
                                   True,True,True]) == cat._binop(catDupe,'==')).all())
        
        self.assertTrue((ak.array([False,False,False,False,False,False,
                                   False,False,False,False]) == cat._binop(catDupe,'!=')).all())

        self.assertTrue((ak.array([True,False,False,False,False,False,
                                   False,False,False,False]) == 
                                   cat._binop('string 1', '==')).all())
        self.assertTrue((ak.array([True,False,False,False,False,False,
                                   False,False,False,False]) == 
                                   cat._binop(np.str_('string 1'), '==')).all())
        
        self.assertTrue((ak.array([False,True,True,True,True,True,True,True,True,True]) ==
                   cat._binop('string 1', '!=')).all())
        self.assertTrue((ak.array([False,True,True,True,True,True,True,True,True,True]) ==
                   cat._binop(np.str_('string 1'), '!=')).all())
        
        with self.assertRaises(NotImplementedError):
            cat._binop('string 1', '===')
        
        with self.assertRaises(TypeError) as cm:
            cat._binop(1, '==')
        self.assertEqual(('type of argument "other" must be one of (Categorical, str, str_);' +
                          ' got int instead'), 
                         cm.exception.args[0])
            
    def testConcatenate(self):
        catOne = self._getCategorical('string',51)
        catTwo = self._getCategorical('string-two', 51)
        
        resultCat = catOne.concatenate([catTwo])
        self.assertEqual('category', resultCat.objtype)
        self.assertIsInstance(resultCat, ak.Categorical)
        self.assertEqual(100,resultCat.size)

        # Since Categorical.concatenate uses Categorical.from_codes method, confirm
        # that both permutation and segments are None
        self.assertFalse(resultCat.permutation)
        self.assertFalse(resultCat.segments)

        resultCat = ak.concatenate([catOne,catOne])
        self.assertEqual('category', resultCat.objtype)
        self.assertIsInstance(resultCat, ak.Categorical)
        self.assertEqual(100,resultCat.size)

        # Since Categorical.concatenate uses Categorical.from_codes method, confirm
        # that both permutation and segments are None
        self.assertFalse(resultCat.permutation)
        self.assertFalse(resultCat.segments)
        
        resultCat = ak.concatenate([catOne,catOne], ordered=False)
        self.assertEqual('category', resultCat.objtype)
        self.assertIsInstance(resultCat, ak.Categorical)
        self.assertEqual(100,resultCat.size)

        # Since Categorical.concatenate uses Categorical.from_codes method, confirm
        # that both permutation and segments are None
        self.assertFalse(resultCat.permutation)
        self.assertFalse(resultCat.segments)