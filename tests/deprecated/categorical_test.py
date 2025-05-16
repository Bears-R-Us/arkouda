import glob
import os
import shutil
import tempfile

from arkouda.pandas import io, io_util
import numpy as np
import pytest
from base_test import ArkoudaTest
from context import arkouda as ak

from arkouda import Strings
from arkouda.pandas.categorical import Categorical


class CategoricalTest(ArkoudaTest):
    @classmethod
    def setUpClass(cls):
        super(CategoricalTest, cls).setUpClass()
        CategoricalTest.cat_test_base_tmp = "{}/categorical_test".format(os.getcwd())
        io_util.get_directory(CategoricalTest.cat_test_base_tmp)

    def _getCategorical(self, prefix: str = "string", size: int = 11) -> ak.Categorical:
        return ak.Categorical(ak.array(["{} {}".format(prefix, i) for i in range(1, size)]))

    def _getRandomizedCategorical(self) -> ak.Categorical:
        return ak.Categorical(
            ak.array(
                [
                    "string",
                    "string1",
                    "non-string",
                    "non-string2",
                    "string",
                    "non-string",
                    "string3",
                    "non-string2",
                    "string",
                    "non-string",
                ]
            )
        )

    def testBaseCategorical(self):
        cat = self._getCategorical()

        self.assertListEqual(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            cat.codes.to_list(),
        )
        self.assertListEqual(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            cat.segments.to_list(),
        )
        self.assertListEqual(
            [
                "string 1",
                "string 2",
                "string 3",
                "string 4",
                "string 5",
                "string 6",
                "string 7",
                "string 8",
                "string 9",
                "string 10",
                "N/A",
            ],
            cat.categories.to_list(),
        )
        self.assertEqual(10, cat.size)
        self.assertEqual("Categorical", cat.objType)

        with self.assertRaises(ValueError):
            ak.Categorical(ak.arange(0, 5, 10))

    def testCategoricalFromCodesAndCategories(self):
        codes = ak.array([7, 5, 9, 8, 2, 1, 4, 0, 3, 6])
        categories = ak.unique(
            ak.array(
                [
                    "string 8",
                    "string 6",
                    "string 5",
                    "string 9",
                    "string 7",
                    "string 2",
                    "string 10",
                    "string 1",
                    "string 4",
                    "string 3",
                    "N/A",
                ]
            )
        )

        cat = ak.Categorical.from_codes(codes, categories)
        self.assertListEqual(codes.to_list(), cat.codes.to_list())
        self.assertListEqual(categories.to_list(), cat.categories.to_list())

    def test_inferred_type(self):
        cat = self._getCategorical()
        self.assertEqual(cat.inferred_type, "categorical")

    def test_equals(self):
        c = Categorical(ak.array(["a", "b", "c"]))
        c_cpy = Categorical(ak.array(["a", "b", "c"]))
        self.assertTrue(ak.sum((c == c_cpy) != ak.array([True, True, True])) == 0)
        self.assertTrue(ak.sum((c != c_cpy) != ak.array([False, False, False])) == 0)
        assert c.equals(c_cpy)

        c2 = Categorical(ak.array(["a", "x", "c"]))
        self.assertTrue(ak.sum((c == c2) != ak.array([True, False, True])) == 0)
        self.assertTrue(ak.sum((c != c2) != ak.array([False, True, False])) == 0)
        assert not c.equals(c2)

        c3 = ak.array(["a", "b", "c", "d"])
        assert not c.equals(c3)

    def test_substring_search(self):
        cat = ak.Categorical(ak.array([f"{i} string {i}" for i in range(10)]))
        self.assertTrue(cat.contains("tri").all())
        self.assertTrue(cat.endswith("ing 1").any())
        self.assertTrue(cat.startswith("1 str").any())

        self.assertTrue(cat.contains("\\w", regex=True).all())
        self.assertTrue(cat.endswith("ing \\d", regex=True).all())
        self.assertTrue(cat.startswith("\\d str", regex=True).all())

    def testGroup(self):
        group = self._getRandomizedCategorical().group()
        self.assertListEqual([0, 4, 8, 1, 6, 2, 5, 9, 3, 7], group.to_list())

    def testUnique(self):
        cat = self._getRandomizedCategorical()

        self.assertListEqual(
            ak.Categorical(
                ak.array(["string", "string1", "string3", "non-string", "non-string2"])
            ).to_list(),
            cat.unique().to_list(),
        )

    def testToNdarray(self):
        cat = self._getRandomizedCategorical()
        ndcat = np.array(
            [
                "string",
                "string1",
                "non-string",
                "non-string2",
                "string",
                "non-string",
                "string3",
                "non-string2",
                "string",
                "non-string",
            ]
        )
        self.assertListEqual(cat.to_list(), ndcat.tolist())

    def testToStrings(self):
        cat = self._getRandomizedCategorical()
        cat_list = [
            "string",
            "string1",
            "non-string",
            "non-string2",
            "string",
            "non-string",
            "string3",
            "non-string2",
            "string",
            "non-string",
        ]

        self.assertListEqual(cat.to_strings().to_list(), cat_list)
        self.assertTrue(isinstance(cat.to_strings(), Strings))

    def testEquality(self):
        cat = self._getCategorical()
        catDupe = self._getCategorical()
        catNonDupe = self._getRandomizedCategorical()

        self.assertListEqual(cat.to_list(), catDupe.to_list())
        self.assertFalse((cat == catNonDupe).any())

        c1 = ak.Categorical(ak.array(["a", "b", "c", "a", "b"]))
        c2 = ak.Categorical(ak.array(["a", "x", "c", "y", "b"]))
        res = c1 == c2
        self.assertListEqual(res.to_list(), [True, False, True, False, True])

    def testBinop(self):
        cat = self._getCategorical()
        catDupe = self._getCategorical()
        catNonDupe = self._getRandomizedCategorical()

        self.assertTrue(cat._binop(catDupe, "==").all())
        self.assertTrue(cat._binop(catNonDupe, "!=").all())

        self.assertListEqual(
            ak.ones(10, dtype=bool).to_list(),
            cat._binop(catDupe, "==").to_list(),
        )

        self.assertListEqual(
            ak.zeros(10, dtype=bool).to_list(),
            cat._binop(catDupe, "!=").to_list(),
        )

        self.assertListEqual(
            (ak.arange(10) == 0).to_list(),
            cat._binop("string 1", "==").to_list(),
        )

        self.assertListEqual(
            (ak.arange(10) == 0).to_list(),
            cat._binop(np.str_("string 1"), "==").to_list(),
        )

        self.assertListEqual(
            (ak.arange(10) != 0).to_list(),
            cat._binop("string 1", "!=").to_list(),
        )

        self.assertListEqual(
            (ak.arange(10) != 0).to_list(),
            cat._binop(np.str_("string 1"), "!=").to_list(),
        )

        with self.assertRaises(NotImplementedError):
            cat._binop("string 1", "===")

        with self.assertRaises(TypeError):
            cat._binop(1, "==")

    def testIn1d(self):
        vals = [i % 3 for i in range(10)]
        valsTwo = [i % 2 for i in range(10)]

        stringsOne = ak.array(["String {}".format(i) for i in vals])
        stringsTwo = ak.array(["String {}".format(i) for i in valsTwo])
        catOne = ak.Categorical(stringsOne)
        catTwo = ak.Categorical(stringsTwo)

        answer = [x < 2 for x in vals]

        self.assertListEqual(answer, ak.in1d(catOne, catTwo).to_list())
        self.assertListEqual(answer, ak.in1d(catOne, stringsTwo).to_list())

        with self.assertRaises(TypeError):
            ak.in1d(catOne, ak.randint(0, 5, 5))

    def test_where(self):
        revs = ak.arange(10) % 2 == 0
        cat1 = ak.Categorical(ak.array([f"str {i}" for i in range(10)]))

        # str in categories, cat first
        str_in_cat = "str 1"
        ans = ak.where(revs, cat1, str_in_cat)
        self.assertListEqual(cat1[revs].to_list(), ans[revs].to_list())
        for s in ans[~revs].to_list():
            self.assertEqual(s, str_in_cat)

        # str in categories, str first
        ans = ak.where(revs, str_in_cat, cat1)
        self.assertListEqual(cat1[~revs].to_list(), ans[~revs].to_list())
        for s in ans[revs].to_list():
            self.assertEqual(s, str_in_cat)

        # str not in categories, cat first
        str_not_in_cat = "str 122222"
        ans = ak.where(revs, cat1, str_not_in_cat)
        self.assertListEqual(cat1[revs].to_list(), ans[revs].to_list())
        for s in ans[~revs].to_list():
            self.assertEqual(s, str_not_in_cat)

        # str not in categories, str first
        ans = ak.where(revs, str_not_in_cat, cat1)
        self.assertListEqual(cat1[~revs].to_list(), ans[~revs].to_list())
        for s in ans[revs].to_list():
            self.assertEqual(s, str_not_in_cat)

        # 2 categorical, same categories
        cat2 = ak.Categorical(ak.array([f"str {i}" for i in range(9, -1, -1)]))
        ans = ak.where(revs, cat1, cat2)
        self.assertListEqual(cat1[revs].to_list(), ans[revs].to_list())
        self.assertListEqual(cat2[~revs].to_list(), ans[~revs].to_list())

        # 2 categorical, different categories
        cat2 = ak.Categorical(ak.array([f"str {i*2}" for i in range(10)]))
        ans = ak.where(revs, cat1, cat2)
        self.assertListEqual(cat1[revs].to_list(), ans[revs].to_list())
        self.assertListEqual(cat2[~revs].to_list(), ans[~revs].to_list())

    def testConcatenate(self):
        catOne = self._getCategorical("string", 51)
        catTwo = self._getCategorical("string-two", 51)

        resultCat = catOne.concatenate([catTwo])
        self.assertEqual("Categorical", resultCat.objType)
        self.assertIsInstance(resultCat, ak.Categorical)
        self.assertEqual(100, resultCat.size)

        # Since Categorical.concatenate uses Categorical.from_codes method, confirm
        # that both permutation and segments are None
        self.assertFalse(resultCat.permutation)
        self.assertFalse(resultCat.segments)

        resultCat = ak.concatenate([catOne, catOne], ordered=False)
        self.assertEqual("Categorical", resultCat.objType)
        self.assertIsInstance(resultCat, ak.Categorical)
        self.assertEqual(100, resultCat.size)

        # Since Categorical.concatenate uses Categorical.from_codes method, confirm
        # that both permutation and segments are None
        self.assertFalse(resultCat.permutation)
        self.assertFalse(resultCat.segments)

        # Concatenate two Categoricals with different categories,
        # and test result against original strings
        s1 = ak.array(["abc", "de", "abc", "fghi", "de"])
        s2 = ak.array(["jkl", "mno", "fghi", "abc", "fghi", "mno"])
        c1 = ak.Categorical(s1)
        c2 = ak.Categorical(s2)
        # Ordered concatenation
        s12ord = ak.concatenate([s1, s2], ordered=True)
        c12ord = ak.concatenate([c1, c2], ordered=True)
        self.assertListEqual(ak.Categorical(s12ord).to_list(), c12ord.to_list())
        # Unordered (but still deterministic) concatenation
        s12unord = ak.concatenate([s1, s2], ordered=False)
        c12unord = ak.concatenate([c1, c2], ordered=False)
        self.assertListEqual(ak.Categorical(s12unord).to_list(), c12unord.to_list())

        # Tiny concatenation
        # Used to fail when length of array was less than numLocales
        # CI uses 2 locales, so try with length-1 arrays
        a = ak.Categorical(ak.array(["a"]))
        b = ak.Categorical(ak.array(["b"]))
        c = ak.concatenate((a, b), ordered=False)
        ans = ak.Categorical(ak.array(["a", "b"]))
        self.assertListEqual(c.to_list(), ans.to_list())

    def testSaveAndLoadCategorical(self):
        """
        Test to save categorical to hdf5 and read it back successfully
        """
        num_elems = 51  # _getCategorical starts counting at 1, so the size is really off by one
        cat = self._getCategorical(size=num_elems)
        with self.assertRaises(ValueError):
            # Expect error for mode not being append or truncate
            cat.to_hdf("foo", dataset="bar", mode="not_allowed")

        with tempfile.TemporaryDirectory(dir=CategoricalTest.cat_test_base_tmp) as tmp_dirname:
            dset_name = "categorical_array"  # name of categorical array

            # Test the save functionality & confirm via h5py
            cat.to_hdf(f"{tmp_dirname}/cat-save-test", dataset=dset_name)

            import h5py

            f = h5py.File(tmp_dirname + "/cat-save-test_LOCALE0000", mode="r")
            keys = list(f.keys())
            if io.ARKOUDA_HDF5_FILE_METADATA_GROUP in keys:  # Ignore the metadata group if it exists
                keys.remove(io.ARKOUDA_HDF5_FILE_METADATA_GROUP)
            self.assertEqual(len(keys), 1, f"Expected 1 key, {dset_name}")
            self.assertListEqual([dset_name], keys)
            d = f[dset_name]
            f.close()

            # Now try to read them back with load_all
            x = ak.load_all(path_prefix=f"{tmp_dirname}/cat-save-test")
            self.assertTrue(dset_name in x)
            cat_from_hdf = x[dset_name]

            expected_categories = [f"string {i}" for i in range(1, num_elems)] + ["N/A"]

            # Note assertCountEqual asserts a and b have the same elements
            # in the same amount regardless of order
            self.assertCountEqual(cat_from_hdf.categories.to_list(), expected_categories)

            # Asserting the optional components and sizes are correct
            # for both constructors should be sufficient
            self.assertTrue(cat_from_hdf.segments is not None)
            self.assertTrue(cat_from_hdf.permutation is not None)
            print(f"==> cat_from_hdf.size:{cat_from_hdf.size}")
            self.assertEqual(cat_from_hdf.size, num_elems - 1)

    def test_hdf_update(self):
        num_elems = 51  # _getCategorical starts counting at 1, so the size is really off by one
        cat = self._getCategorical(size=num_elems)
        with tempfile.TemporaryDirectory(dir=CategoricalTest.cat_test_base_tmp) as tmp_dirname:
            dset_name = "categorical_array"  # name of categorical array
            cat.to_hdf(f"{tmp_dirname}/cat-save-test", dataset=dset_name)

            dset_name2 = "to_replace"
            cat.to_hdf(f"{tmp_dirname}/cat-save-test", dataset=dset_name2, mode="append")

            dset_name3 = "cat_array2"
            cat.to_hdf(f"{tmp_dirname}/cat-save-test", dataset=dset_name3, mode="append")

            replace_cat = self._getCategorical(size=23)
            replace_cat.update_hdf(f"{tmp_dirname}/cat-save-test", dataset=dset_name2)

            data = ak.read_hdf(f"{tmp_dirname}/cat-save-test_*")
            self.assertTrue(dset_name in data)
            self.assertTrue(dset_name2 in data)
            self.assertTrue(dset_name3 in data)

            d = data[dset_name2]
            self.assertListEqual(d.codes.to_list(), replace_cat.codes.to_list())
            self.assertListEqual(d.permutation.to_list(), replace_cat.permutation.to_list())
            self.assertListEqual(d.segments.to_list(), replace_cat.segments.to_list())
            self.assertListEqual(d._akNAcode.to_list(), replace_cat._akNAcode.to_list())
            self.assertListEqual(d.categories.to_list(), replace_cat.categories.to_list())

    def test_unused_categories_logic(self):
        """
        Test that Categoricals built from_codes and from slices
        that have unused categories behave correctly
        """
        s = ak.array([str(i) for i in range(10)])
        s12 = s[1:3]
        cat = ak.Categorical(s)
        cat12 = cat[1:3]
        self.assertListEqual(ak.in1d(s, s12).to_list(), ak.in1d(cat, cat12).to_list())
        self.assertSetEqual(set(ak.unique(s12).to_list()), set(ak.unique(cat12).to_list()))

        cat_from_codes = ak.Categorical.from_codes(ak.array([1, 2]), s)
        self.assertListEqual(ak.in1d(s, s12).to_list(), ak.in1d(cat, cat_from_codes).to_list())
        self.assertSetEqual(
            set(ak.unique(s12).to_list()),
            set(ak.unique(cat_from_codes).to_list()),
        )

    def testSaveAndLoadCategoricalMulti(self):
        """
        Test to build a pseudo dataframe with multiple
        categoricals, pdarrays, strings objects and successfully
        write/read it from HDF5
        """
        c1 = self._getCategorical(prefix="c1", size=51)
        c2 = self._getCategorical(prefix="c2", size=52)
        pda1 = ak.zeros(51)
        strings1 = ak.random_strings_uniform(9, 10, 52)

        with tempfile.TemporaryDirectory(dir=CategoricalTest.cat_test_base_tmp) as tmp_dirname:
            df = {"cat1": c1, "cat2": c2, "pda1": pda1, "strings1": strings1}
            ak.to_hdf(df, f"{tmp_dirname}/cat-save-test")
            x = ak.load_all(path_prefix=f"{tmp_dirname}/cat-save-test")
            self.assertEqual(len(x.items()), 4)
            # Note assertCountEqual asserts a and b have the same
            # elements in the same amount regardless of order
            self.assertCountEqual(x["cat1"].categories.to_list(), c1.categories.to_list())
            self.assertCountEqual(x["cat2"].categories.to_list(), c2.categories.to_list())
            self.assertCountEqual(x["pda1"].to_list(), pda1.to_list())
            self.assertCountEqual(x["strings1"].to_list(), strings1.to_list())

    def testNA(self):
        s = ak.array(["A", "B", "C", "B", "C"])
        # NAval present in categories
        c = ak.Categorical(s, NAvalue="C")
        self.assertListEqual(c.isna().to_list(), [False, False, True, False, True])
        self.assertEqual(c.NAvalue, "C")
        # Test that NAval survives registration
        c.register("my_categorical")
        c2 = ak.attach("my_categorical")
        self.assertEqual(c2.NAvalue, "C")

        c.unregister()

        # default NAval not present in categories
        c = ak.Categorical(s)
        self.assertTrue(not c.isna().any())
        self.assertEqual(c.NAvalue, "N/A")

    def testStandardizeCategories(self):
        c1 = ak.Categorical(ak.array(["A", "B", "C"]))
        c2 = ak.Categorical(ak.array(["B", "C", "D"]))
        c3, c4 = ak.Categorical.standardize_categories([c1, c2])
        self.assertListEqual(c3.categories.to_list(), c4.categories.to_list())
        self.assertTrue(not c3.isna().any())
        self.assertTrue(not c4.isna().any())
        self.assertEqual(c3.categories.size, c1.categories.size + 1)
        self.assertEqual(c4.categories.size, c2.categories.size + 1)

    def testLookup(self):
        keys = ak.array([1, 2, 3])
        values = ak.Categorical(ak.array(["A", "B", "C"]))
        args = ak.array([3, 2, 1, 0])
        ret = ak.lookup(keys, values, args)
        self.assertListEqual(ret.to_list(), ["C", "B", "A", "N/A"])

    def test_deletion(self):
        cat = ak.Categorical(ak.array(["a", "b", "c"]))
        # validate registration with server
        self.assertTrue(len(ak.list_symbol_table()) > 0)

        # set to none and validate no entries in symbol table
        cat = None
        self.assertEqual(len(ak.list_symbol_table()), 0)

    def test_sort(self):
        rand_cats = ak.random_strings_uniform(1, 16, 10)
        rand_codes = ak.randint(0, rand_cats.size, 100)
        cat = ak.Categorical.from_codes(codes=rand_codes, categories=rand_cats)

        self.assertEqual(sorted(cat.to_list()), cat.sort_values().to_list())

    def tearDown(self):
        super(CategoricalTest, self).tearDown()
        for f in glob.glob("{}/*".format(CategoricalTest.cat_test_base_tmp)):
            os.remove(f)

    @classmethod
    def tearDownClass(cls):
        super(CategoricalTest, cls).tearDownClass()
        shutil.rmtree(CategoricalTest.cat_test_base_tmp)
