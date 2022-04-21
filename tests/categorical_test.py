import numpy as np
import glob
import shutil
import os
import tempfile
from arkouda import io_util
from context import arkouda as ak
from base_test import ArkoudaTest
from arkouda import pdarrayIO


class CategoricalTest(ArkoudaTest):

    @classmethod
    def setUpClass(cls):
        super(CategoricalTest, cls).setUpClass()
        CategoricalTest.cat_test_base_tmp = '{}/categorical_test'.format(os.getcwd())
        io_util.get_directory(CategoricalTest.cat_test_base_tmp )
    
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
                                    'string 4', 'string 3', 'N/A']) == cat.categories).all())
        self.assertEqual(10,cat.size)
        self.assertEqual('category',cat.objtype)
        
        with self.assertRaises(ValueError) as cm:
            ak.Categorical(ak.arange(0,5,10))
        
    def testCategoricalFromCodesAndCategories(self):
        codes = ak.array([7,5,9,8,2,1,4,0,3,6])
        categories = ak.unique(ak.array(['string 8', 'string 6', 'string 5', 'string 9', 
                                    'string 7', 'string 2', 'string 10', 'string 1', 
                                    'string 4', 'string 3', 'N/A']))
        
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

        c1 = ak.Categorical(ak.array(['a', 'b', 'c', 'a', 'b']))
        c2 = ak.Categorical(ak.array(['a', 'x', 'c', 'y', 'b']))
        res = (c1 == c2)
        ans = ak.array([True, False, True, False, True])
        self.assertTrue((res == ans).all())
        
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

    def testIn1d(self):
        vals = [i % 3 for i in range(10)]
        valsTwo = [i % 2 for i in range(10)]

        stringsOne = ak.array(['String {}'.format(i) for i in vals])
        stringsTwo = ak.array(['String {}'.format(i) for i in valsTwo])
        catOne = ak.Categorical(stringsOne)
        catTwo = ak.Categorical(stringsTwo)

        answer = ak.array([x < 2 for x in vals])

        self.assertTrue((answer == ak.in1d(catOne,catTwo)).all())
        self.assertTrue((answer == ak.in1d(catOne,stringsTwo)).all())

        with self.assertRaises(TypeError) as cm:
            ak.in1d(catOne, ak.randint(0,5,5))
       
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
        
        resultCat = ak.concatenate([catOne,catOne], ordered=False)
        self.assertEqual('category', resultCat.objtype)
        self.assertIsInstance(resultCat, ak.Categorical)
        self.assertEqual(100,resultCat.size)

        # Since Categorical.concatenate uses Categorical.from_codes method, confirm
        # that both permutation and segments are None
        self.assertFalse(resultCat.permutation)
        self.assertFalse(resultCat.segments)
        
        # Concatenate two Categoricals with different categories, and test result against original strings
        s1 = ak.array(['abc', 'de', 'abc', 'fghi', 'de'])
        s2 = ak.array(['jkl', 'mno', 'fghi', 'abc', 'fghi', 'mno'])
        c1 = ak.Categorical(s1)
        c2 = ak.Categorical(s2)
        # Ordered concatenation
        s12ord = ak.concatenate([s1, s2], ordered=True)
        c12ord = ak.concatenate([c1, c2], ordered=True)
        self.assertTrue((ak.Categorical(s12ord) == c12ord).all())
        # Unordered (but still deterministic) concatenation
        s12unord = ak.concatenate([s1, s2], ordered=False)
        c12unord = ak.concatenate([c1, c2], ordered=False)
        self.assertTrue((ak.Categorical(s12unord) == c12unord).all())

        # Tiny concatenation
        # Used to fail when length of array was less than numLocales
        # CI uses 2 locales, so try with length-1 arrays
        a = ak.Categorical(ak.array(['a']))
        b = ak.Categorical(ak.array(['b']))
        c = ak.concatenate((a, b), ordered=False)
        ans = ak.Categorical(ak.array(['a', 'b']))
        self.assertTrue((c == ans).all())
        
    def testSaveAndLoadCategorical(self):
        """
        Test to save categorical to hdf5 and read it back successfully
        """
        num_elems = 51  # _getCategorical starts counting at 1, so the size is really off by one
        cat = self._getCategorical(size=num_elems)
        with self.assertRaises(ValueError):  # Expect error for mode not being append or truncate
            cat.save("foo", dataset="bar", mode="not_allowed")

        with tempfile.TemporaryDirectory(dir=CategoricalTest.cat_test_base_tmp) as tmp_dirname:
            dset_name = "categorical_array"  # name of categorical array

            # Test the save functionality & confirm via h5py
            cat.save(f"{tmp_dirname}/cat-save-test", dataset=dset_name)

            import h5py
            f = h5py.File(tmp_dirname + "/cat-save-test_LOCALE0000", mode="r")
            keys = set(f.keys())
            if pdarrayIO.ARKOUDA_HDF5_FILE_METADATA_GROUP in keys:  # Ignore the metadata group if it exists
                keys.remove(pdarrayIO.ARKOUDA_HDF5_FILE_METADATA_GROUP)
            self.assertEqual(len(keys), 5, "Expected 5 keys")
            self.assertSetEqual(set(f"categorical_array.{k}" for k in cat._get_components_dict().keys()), keys)
            f.close()

            # Now try to read them back with load_all
            x = ak.load_all(path_prefix=f"{tmp_dirname}/cat-save-test")
            self.assertTrue(dset_name in x)
            cat_from_hdf = x[dset_name]

            expected_categories = [f"string {i}" for i in range(1, num_elems)] + ['N/A']

            # Note assertCountEqual asserts a and b have the same elements in the same amount regardless of order
            self.assertCountEqual(cat_from_hdf.categories.to_ndarray().tolist(), expected_categories)

            # Asserting the optional components and sizes are correct for both constructors should be sufficient
            self.assertTrue(cat_from_hdf.segments is not None)
            self.assertTrue(cat_from_hdf.permutation is not None)
            print(f"==> cat_from_hdf.size:{cat_from_hdf.size}")
            self.assertTrue(cat_from_hdf.size == num_elems-1)

    def test_unused_categories_logic(self):
        """
        Test that Categoricals built from_codes and from slices that have unused categories behave correctly
        """
        s = ak.array([str(i) for i in range(10)])
        s12 = s[1:3]
        cat = ak.Categorical(s)
        cat12 = cat[1:3]
        self.assertListEqual(ak.in1d(s, s12).to_ndarray().tolist(), ak.in1d(cat, cat12).to_ndarray().tolist())
        self.assertSetEqual(set(ak.unique(s12).to_ndarray().tolist()), set(ak.unique(cat12).to_ndarray().tolist()))

        cat_from_codes = ak.Categorical.from_codes(ak.array([1, 2]), s)
        self.assertListEqual(ak.in1d(s, s12).to_ndarray().tolist(), ak.in1d(cat, cat_from_codes).to_ndarray().tolist())
        self.assertSetEqual(set(ak.unique(s12).to_ndarray().tolist()), set(ak.unique(cat_from_codes).to_ndarray().tolist()))


    def testSaveAndLoadCategoricalMulti(self):
        """
        Test to build a pseudo dataframe with multiple categoricals, pdarrays, strings objects and successfully
        write/read it from HDF5
        """
        c1 = self._getCategorical(prefix="c1", size=51)
        c2 = self._getCategorical(prefix="c2", size=52)
        pda1 = ak.zeros(51)
        strings1 = ak.random_strings_uniform(9, 10, 52)

        with tempfile.TemporaryDirectory(dir=CategoricalTest.cat_test_base_tmp) as tmp_dirname:
            df = {
                "cat1": c1,
                "cat2": c2,
                "pda1": pda1,
                "strings1": strings1
            }
            ak.save_all(df, f"{tmp_dirname}/cat-save-test")
            x = ak.load_all(path_prefix=f"{tmp_dirname}/cat-save-test")
            self.assertTrue(len(x.items()) == 4)
            # Note assertCountEqual asserts a and b have the same elements in the same amount regardless of order
            self.assertCountEqual(x["cat1"].categories.to_ndarray().tolist(), c1.categories.to_ndarray().tolist())
            self.assertCountEqual(x["cat2"].categories.to_ndarray().tolist(), c2.categories.to_ndarray().tolist())
            self.assertCountEqual(x["pda1"].to_ndarray().tolist(), pda1.to_ndarray().tolist())
            self.assertCountEqual(x["strings1"].to_ndarray().tolist(), strings1.to_ndarray().tolist())

    def testNA(self):
        s = ak.array(['A', 'B', 'C', 'B', 'C'])
        # NAval present in categories
        c = ak.Categorical(s, NAvalue='C')
        self.assertListEqual(c.isna().to_ndarray().tolist(), [False, False, True, False, True])
        self.assertTrue(c.NAvalue == 'C')
        # Test that NAval survives registration
        c.register('my_categorical')
        c2 = ak.Categorical.attach('my_categorical')
        self.assertTrue(c2.NAvalue == 'C')

        # default NAval not present in categories
        c = ak.Categorical(s)
        self.assertTrue(not c.isna().any())
        self.assertTrue(c.NAvalue == 'N/A')

    def testStandardizeCategories(self):
        c1 = ak.Categorical(ak.array(['A', 'B', 'C']))
        c2 = ak.Categorical(ak.array(['B', 'C', 'D']))
        c3, c4 = ak.Categorical.standardize_categories([c1, c2])
        self.assertTrue((c3.categories == c4.categories).all())
        self.assertTrue(not c3.isna().any())
        self.assertTrue(not c4.isna().any())
        self.assertTrue(c3.categories.size == c1.categories.size+1)
        self.assertTrue(c4.categories.size == c2.categories.size+1)

    def testLookup(self):
        keys = ak.array([1, 2, 3])
        values = ak.Categorical(ak.array(['A', 'B', 'C']))
        args = ak.array([3, 2, 1, 0])
        ret = ak.lookup(keys, values, args)
        expected = ['C', 'B', 'A', 'N/A']
        self.assertListEqual(ret.to_ndarray().tolist(), expected)
        
    def tearDown(self):
        super(CategoricalTest, self).tearDown()
        for f in glob.glob('{}/*'.format(CategoricalTest.cat_test_base_tmp)):
            os.remove(f)

    @classmethod
    def tearDownClass(cls):
        super(CategoricalTest, cls).tearDownClass()
        shutil.rmtree(CategoricalTest.cat_test_base_tmp)
