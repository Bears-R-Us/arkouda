from context import arkouda as ak
from base_test import ArkoudaTest


class CategoricalTest(ArkoudaTest):
    
    
    def testBaseCategorical(self):
        strings = ak.array(['string {}'.format(i) for i in range(1,11)])
        cat = ak.Categorical(strings)

        self.assertTrue((ak.array([7,5,9,8,2,1,4,0,3,6]) == cat.codes).all())
        self.assertTrue((ak.array([0,1,2,3,4,5,6,7,8,9]) == cat.segments).all())
        self.assertTrue((ak.array(['string 8', 'string 6', 'string 5', 'string 9', 
                                    'string 7', 'string 2', 'string 10', 'string 1', 
                                    'string 4', 'string 3']) == cat.categories).all())
        self.assertEqual(10,cat.size)
        self.assertEqual('category',cat.objtype)
        
    def testCategoricalFromCodesAndCategories(self):
        codes = ak.array([7,5,9,8,2,1,4,0,3,6])
        categories = ak.array(['string 8', 'string 6', 'string 5', 'string 9', 
                                    'string 7', 'string 2', 'string 10', 'string 1', 
                                    'string 4', 'string 3'])
        
        cat = ak.Categorical.from_codes(codes, categories)
        self.assertTrue((codes == cat.codes).all())
        self.assertTrue((categories == cat.categories).all())
        
    def testContains(self):
        strings = ak.array(['string {}'.format(i) for i in range(1,11)])
        cat = ak.Categorical(strings)
        self.assertTrue(cat.contains('string').all())
        
    def testGroup(self):
        strings = ak.array(['string {}'.format(i) for i in range(1,11)])
        cat = ak.Categorical(strings)
        self.assertTrue((ak.array([7,5,4,8,6,1,9,0,3,2]) == cat.group()).all())