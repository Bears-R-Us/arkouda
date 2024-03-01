import glob
import os
import tempfile

import pandas as pd
from base_test import ArkoudaTest
from context import arkouda as ak

from arkouda import io_util
from arkouda.dtypes import dtype
from arkouda.pdarrayclass import pdarray


class IndexTest(ArkoudaTest):
    @classmethod
    def setUpClass(cls):
        super(IndexTest, cls).setUpClass()
        IndexTest.ind_test_base_tmp = "{}/ind_io_test".format(os.getcwd())
        io_util.get_directory(IndexTest.ind_test_base_tmp)

    def test_index_creation(self):
        idx = ak.Index(ak.arange(5))

        self.assertIsInstance(idx, ak.Index)
        self.assertEqual(idx.size, 5)
        self.assertListEqual(idx.to_list(), [i for i in range(5)])

    def test_index_creation_lists(self):
        i = ak.Index([1, 2, 3])
        self.assertIsInstance(i.values, pdarray)

        i2 = ak.Index([1, 2, 3], allow_list=True)
        self.assertIsInstance(i2.values, list)
        self.assertEqual(i2.dtype, dtype("int64"))

        i3 = ak.Index(["a", "b", "c"], allow_list=True)
        self.assertIsInstance(i3.values, list)
        self.assertEqual(i3.dtype, dtype("<U"))

        with self.assertRaises(ValueError):
            ak.Index([1, 2, 3], allow_list=True, max_list_size=2)

    def test_multiindex_creation(self):
        # test list generation
        idx = ak.MultiIndex([ak.arange(5), ak.arange(5)])
        self.assertIsInstance(idx, ak.MultiIndex)
        self.assertEqual(idx.levels, 2)
        self.assertEqual(idx.size, 5)

        # test tuple generation
        idx = ak.MultiIndex((ak.arange(5), ak.arange(5)))
        self.assertIsInstance(idx, ak.MultiIndex)
        self.assertEqual(idx.levels, 2)
        self.assertEqual(idx.size, 5)

        with self.assertRaises(TypeError):
            idx = ak.MultiIndex(ak.arange(5))

        with self.assertRaises(ValueError):
            idx = ak.MultiIndex([ak.arange(5), ak.arange(3)])

    def test_is_unique(self):
        i = ak.Index(ak.array([0, 1, 2]))
        self.assertTrue(i.is_unique)

        i = ak.Index(ak.array([0, 1, 1]))
        self.assertFalse(i.is_unique)

        i = ak.Index([0, 1, 2], allow_list=True)
        self.assertTrue(i.is_unique)

        i = ak.Index([0, 1, 1], allow_list=True)
        self.assertFalse(i.is_unique)

    def test_factory(self):
        idx = ak.Index.factory(ak.arange(5))
        self.assertIsInstance(idx, ak.Index)

        idx = ak.Index.factory([ak.arange(5), ak.arange(5)])
        self.assertIsInstance(idx, ak.MultiIndex)

    def test_argsort(self):
        idx = ak.Index.factory(ak.arange(5))
        i = idx.argsort(False)
        self.assertListEqual(i.to_list(), [4, 3, 2, 1, 0])

        idx = ak.Index(ak.array([1, 0, 4, 2, 5, 3]))
        i = idx.argsort()
        # values should be the indexes in the array of idx
        self.assertListEqual(i.to_list(), [1, 0, 3, 5, 2, 4])

        i = ak.Index([1, 2, 3])
        self.assertListEqual(i.argsort(ascending=True).to_list(), [0, 1, 2])
        self.assertListEqual(i.argsort(ascending=False).to_list(), [2, 1, 0])

        i2 = ak.Index([1, 2, 3], allow_list=True)
        self.assertListEqual(i2.argsort(ascending=True), [0, 1, 2])
        self.assertListEqual(i2.argsort(ascending=False), [2, 1, 0])

        i3 = ak.Index(["a", "b", "c"], allow_list=True)
        self.assertListEqual(i3.argsort(ascending=True), [0, 1, 2])
        self.assertListEqual(i3.argsort(ascending=False), [2, 1, 0])

        i4 = ak.Index(ak.array(["a", "b", "c"]))
        self.assertListEqual(i4.argsort(ascending=True).to_list(), [0, 1, 2])
        self.assertListEqual(i4.argsort(ascending=False).to_list(), [2, 1, 0])

    def test_concat(self):
        idx_1 = ak.Index.factory(ak.arange(5))
        idx_2 = ak.Index(ak.array([2, 4, 1, 3, 0]))

        idx_full = idx_1.concat(idx_2)
        self.assertListEqual(idx_full.to_list(), [0, 1, 2, 3, 4, 2, 4, 1, 3, 0])

        i = ak.Index([1, 2, 3], allow_list=True)
        i2 = ak.Index(["a", "b", "c"], allow_list=True)
        self.assertListEqual(i.concat(i2).to_list(), ["1", "2", "3", "a", "b", "c"])

    def test_lookup(self):
        idx = ak.Index.factory(ak.arange(5))
        lk = idx.lookup(ak.array([0, 4]))
        self.assertListEqual(lk.to_list(), [True, False, False, False, True])

    def test_multi_argsort(self):
        idx = ak.Index.factory([ak.arange(5), ak.arange(5)])
        s = idx.argsort(False)
        self.assertListEqual(s.to_list(), [4, 3, 2, 1, 0])

        s = idx.argsort()
        self.assertListEqual(s.to_list(), [i for i in range(5)])

    def test_multi_concat(self):
        idx = ak.Index.factory([ak.arange(5), ak.arange(5)])
        idx_2 = ak.Index.factory(ak.array([0.1, 1.1, 2.2, 3.3, 4.4]))
        with self.assertRaises(TypeError):
            idx.concat(idx_2)

        idx_2 = ak.Index.factory([ak.arange(5), ak.arange(5)])
        idx_full = idx.concat(idx_2)
        self.assertListEqual(
            idx_full.to_pandas().tolist(),
            [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
        )

    def test_multi_lookup(self):
        idx = ak.Index.factory([ak.arange(5), ak.arange(5)])

        lk = ak.array([0, 3, 2])

        result = idx.lookup([lk, lk])
        self.assertListEqual(result.to_list(), [True, False, True, True, False])

    def test_save(self):
        locale_count = ak.get_config()["numLocales"]
        with tempfile.TemporaryDirectory(dir=IndexTest.ind_test_base_tmp) as tmp_dirname:
            idx = ak.Index(ak.arange(5))
            idx.to_hdf(f"{tmp_dirname}/idx_file.h5")
            self.assertEqual(len(glob.glob(f"{tmp_dirname}/idx_file_*.h5")), locale_count)

    def test_to_pandas(self):
        i = ak.Index([1, 2, 3])
        self.assertTrue(i.to_pandas().equals(pd.Index([1, 2, 3])))

        i2 = ak.Index([1, 2, 3], allow_list=True)
        self.assertTrue(i2.to_pandas().equals(pd.Index([1, 2, 3])))

        i3 = ak.Index(["a", "b", "c"], allow_list=True)
        self.assertTrue(i3.to_pandas().equals(pd.Index(["a", "b", "c"])))

        i4 = ak.Index(ak.array(["a", "b", "c"]))
        self.assertTrue(i4.to_pandas().equals(pd.Index(["a", "b", "c"])))

    def test_to_ndarray(self):
        from numpy import array as ndarray
        from numpy import array_equal

        i = ak.Index([1, 2, 3])
        self.assertTrue(array_equal(i.to_ndarray(), ndarray([1, 2, 3])))

        i2 = ak.Index([1, 2, 3], allow_list=True)
        self.assertTrue(array_equal(i2.to_ndarray(), ndarray([1, 2, 3])))

        i3 = ak.Index(["a", "b", "c"], allow_list=True)
        self.assertTrue(array_equal(i3.to_ndarray(), ndarray(["a", "b", "c"])))

        i4 = ak.Index(ak.array(["a", "b", "c"]))
        self.assertTrue(array_equal(i4.to_ndarray(), ndarray(["a", "b", "c"])))

    def test_to_list(self):
        i = ak.Index([1, 2, 3])
        self.assertListEqual(i.to_list(), [1, 2, 3])

        i2 = ak.Index([1, 2, 3], allow_list=True)
        self.assertListEqual(i2.to_list(), [1, 2, 3])

        i3 = ak.Index(["a", "b", "c"], allow_list=True)
        self.assertListEqual(i3.to_list(), ["a", "b", "c"])

        i4 = ak.Index(ak.array(["a", "b", "c"]))
        self.assertListEqual(i4.to_list(), ["a", "b", "c"])

    def test_register_list_values(self):
        i = ak.Index([1, 2, 3], allow_list=True)
        with self.assertRaises(TypeError):
            i.register("test")
