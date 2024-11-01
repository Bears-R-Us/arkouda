import glob
import os
import tempfile

import pandas as pd
from base_test import ArkoudaTest
from context import arkouda as ak
from numpy import dtype as npdtype

from arkouda import io_util
from arkouda.numpy.dtypes import dtype
from arkouda.index import Index
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
        self.assertEqual(idx.nlevels, 2)
        self.assertEqual(idx.size, 5)

        # test tuple generation
        idx = ak.MultiIndex((ak.arange(5), ak.arange(5)))
        self.assertIsInstance(idx, ak.MultiIndex)
        self.assertEqual(idx.nlevels, 2)
        self.assertEqual(idx.size, 5)

        with self.assertRaises(TypeError):
            idx = ak.MultiIndex(ak.arange(5))

        with self.assertRaises(ValueError):
            idx = ak.MultiIndex([ak.arange(5), ak.arange(3)])

    def test_name_names(self):
        i = ak.Index([1, 2, 3], name="test")
        self.assertEqual(i.name, "test")
        self.assertListEqual(i.names, ["test"])

        size = 10
        m = ak.MultiIndex([ak.arange(size), ak.arange(size) * -1], names=["test", "test2"])
        self.assertListEqual(m.names, ["test", "test2"])

    def test_nlevels(self):
        i = ak.Index([1, 2, 3], name="test")
        assert i.nlevels == 1

        size = 10
        m = ak.MultiIndex([ak.arange(size), ak.arange(size) * -1])
        assert m.nlevels == 2

    def test_ndim(self):
        i = ak.Index([1, 2, 3], name="test")
        assert i.ndim == 1

        size = 10
        m = ak.MultiIndex([ak.arange(size), ak.arange(size) * -1])
        assert m.ndim == 1

    def test_dtypes(self):
        size = 10
        i = ak.Index(ak.arange(size, dtype="float64"))
        assert i.dtype == dtype("float64")

        m = ak.MultiIndex([ak.arange(size), ak.arange(size) * -1])
        assert m.dtype == npdtype("O")

    def test_inferred_type(self):
        i = ak.Index([1, 2, 3])
        self.assertEqual(i.inferred_type, "integer")

        i2 = ak.Index([1.0, 2, 3])
        self.assertEqual(i2.inferred_type, "floating")

        i3 = ak.Index(["a", "b", "c"], allow_list=True)
        self.assertEqual(i3.inferred_type, "string")

        from arkouda.categorical import Categorical

        i4 = ak.Index(Categorical(ak.array(["a", "b", "c"])))
        self.assertEqual(i4.inferred_type, "categorical")

        size = 10
        m = ak.MultiIndex([ak.arange(size), ak.arange(size) * -1], names=["test", "test2"])
        self.assertEqual(m.inferred_type, "mixed")

    def assert_equal(self, pda1, pda2):
        from arkouda import sum as aksum

        assert pda1.size == pda2.size
        assert aksum(pda1 != pda2) == 0

    def test_get_item(self):
        i = ak.Index([1, 2, 3])
        self.assertEqual(i[2], 3)
        self.assertTrue(i[[0, 1]].equals(Index([1, 2])))

        i2 = ak.Index([1, 2, 3], allow_list=True)
        self.assertEqual(i2[2], 3)
        self.assertTrue(i2[[0, 1]].equals(Index([1, 2], allow_list=True)))

        i3 = ak.Index(["a", "b", "c"], allow_list=True)
        self.assertEqual(i3[2], "c")
        self.assertTrue(i3[[0, 1]].equals(Index(["a", "b"], allow_list=True)))

    def test_eq(self):
        i = ak.Index([1, 2, 3])
        i_cpy = ak.Index([1, 2, 3])
        self.assert_equal(i == i_cpy, ak.array([True, True, True]))
        self.assert_equal(i != i_cpy, ak.array([False, False, False]))
        self.assertTrue(i.equals(i_cpy))

        i2 = ak.Index([1, 2, 3], allow_list=True)
        i2_cpy = ak.Index([1, 2, 3], allow_list=True)
        self.assert_equal(i2 == i2_cpy, ak.array([True, True, True]))
        self.assert_equal(i2 != i2_cpy, ak.array([False, False, False]))
        self.assertTrue(i2.equals(i2_cpy))

        self.assert_equal(i == i2, ak.array([True, True, True]))
        self.assert_equal(i != i2, ak.array([False, False, False]))
        self.assertTrue(i.equals(i2))

        i3 = ak.Index(["a", "b", "c"], allow_list=True)
        i3_cpy = ak.Index(["a", "b", "c"], allow_list=True)
        self.assert_equal(i3 == i3_cpy, ak.array([True, True, True]))
        self.assert_equal(i3 != i3_cpy, ak.array([False, False, False]))
        self.assertTrue(i3.equals(i3_cpy))

        i4 = ak.Index(["a", "b", "c"], allow_list=False)
        i4_cpy = ak.Index(["a", "b", "c"], allow_list=False)
        self.assert_equal(i4 == i4_cpy, ak.array([True, True, True]))
        self.assert_equal(i4 != i4_cpy, ak.array([False, False, False]))
        self.assertTrue(i3.equals(i3_cpy))

        i5 = ak.Index(["a", "x", "c"], allow_list=True)
        self.assert_equal(i3 == i5, ak.array([True, False, True]))
        self.assert_equal(i3 != i5, ak.array([False, True, False]))
        self.assertFalse(i3.equals(i5))

        i6 = ak.Index(["a", "b", "c", "d"], allow_list=True)
        self.assertFalse(i5.equals(i6))

        with self.assertRaises(ValueError):
            i5 == i6

        with self.assertRaises(ValueError):
            i5 != i6

        with self.assertRaises(TypeError):
            i.equals("string")

    def test_multiindex_equals(self):
        size = 10
        arrays = [ak.array([1, 1, 2, 2]), ak.array(["red", "blue", "red", "blue"])]
        m = ak.MultiIndex(arrays, names=["numbers", "colors"])
        self.assertTrue(m.equals(m))

        arrays2 = [ak.array([1, 1, 2, 2]), ak.array(["red", "blue", "red", "blue"])]
        m2 = ak.MultiIndex(arrays2, names=["numbers2", "colors2"])
        self.assertTrue(m.equals(m2))

        arrays3 = [
            ak.array([1, 1, 2, 2]),
            ak.array(["red", "blue", "red", "blue"]),
            ak.array([1, 1, 2, 2]),
        ]
        m3 = ak.MultiIndex(arrays3, names=["numbers", "colors2", "numbers3"])
        self.assertFalse(m.equals(m3))

        arrays4 = [ak.array([1, 1, 2, 2]), ak.array(["red", "blue", "red", "green"])]
        m4 = ak.MultiIndex(arrays4, names=["numbers2", "colors2"])
        self.assertFalse(m.equals(m4))

        m5 = ak.MultiIndex([ak.arange(size)])
        i = ak.Index(ak.arange(size))
        self.assertFalse(m5.equals(i))
        self.assertFalse(i.equals(m5))

    def test_equal_levels(self):
        m = ak.MultiIndex(
            [ak.arange(3), ak.arange(3) * -1, ak.array(["a", 'b","c', "d"])],
            names=["col1", "col2", "col3"],
        )
        m2 = ak.MultiIndex(
            [ak.arange(3), ak.arange(3) * -1, ak.array(["a", 'b","c', "d"])],
            names=["A", "B", "C"],
        )

        self.assertTrue(m.equal_levels(m2))

        m3 = ak.MultiIndex(
            [ak.arange(3), ak.arange(3) * -1, ak.array(["a", 'b","c', "d"]), 2 * ak.arange(3)],
            names=["col1", "col2", "col3"],
        )

        self.assertFalse(m.equal_levels(m3))

        m4 = ak.MultiIndex(
            [ak.arange(3), ak.arange(3) * 2, ak.array(["a", 'b","c', "d"])],
            names=["col1", "col2", "col3"],
        )

        self.assertFalse(m.equal_levels(m4))

    def test_get_level_values(self):
        m = ak.MultiIndex(
            [ak.arange(3), ak.arange(3) * -1, ak.array(["a", 'b","c', "d"])],
            names=["col1", "col2", "col3"],
        )

        i1 = Index(ak.arange(3), name="col1")
        self.assert_equal(m.get_level_values(0), i1)
        self.assert_equal(m.get_level_values("col1"), i1)

        i2 = Index(ak.arange(3) * -1, name="col2")
        self.assert_equal(m.get_level_values(1), i2)
        self.assert_equal(m.get_level_values("col2"), i2)

        i3 = Index(ak.array(["a", 'b","c', "d"]), name="col3")
        self.assert_equal(m.get_level_values(2), i3)
        self.assert_equal(m.get_level_values("col3"), i3)

        with self.assertRaises(ValueError):
            m.get_level_values("col4")

        #   Test when names=None
        m2 = ak.MultiIndex(
            [ak.arange(3), ak.arange(3) * -1, ak.array(["a", 'b","c', "d"])],
        )
        i4 = Index(ak.arange(3))
        self.assert_equal(m2.get_level_values(0), i4)

        with self.assertRaises(RuntimeError):
            m2.get_level_values("col")

        with self.assertRaises(ValueError):
            m2.get_level_values(m2.nlevels)

        with self.assertRaises(ValueError):
            m2.get_level_values(-1 * m2.nlevels)

    def test_memory_usage(self):
        from arkouda.numpy.dtypes import BigInt
        from arkouda.index import Index, MultiIndex

        idx = Index(ak.cast(ak.array([1, 2, 3]), dt="bigint"))
        self.assertEqual(idx.memory_usage(), 3 * BigInt.itemsize)

        n = 2000
        idx = Index(ak.cast(ak.arange(n), dt="int64"))
        self.assertEqual(
            idx.memory_usage(unit="GB"), n * ak.dtypes.int64.itemsize / (1024 * 1024 * 1024)
        )
        self.assertEqual(idx.memory_usage(unit="MB"), n * ak.dtypes.int64.itemsize / (1024 * 1024))
        self.assertEqual(idx.memory_usage(unit="KB"), n * ak.dtypes.int64.itemsize / (1024))
        self.assertEqual(idx.memory_usage(unit="B"), n * ak.dtypes.int64.itemsize)

        midx = MultiIndex([ak.cast(ak.arange(n), dt="int64"), ak.cast(ak.arange(n), dt="int64")])
        self.assertEqual(
            midx.memory_usage(unit="GB"), 2 * n * ak.dtypes.int64.itemsize / (1024 * 1024 * 1024)
        )
        self.assertEqual(midx.memory_usage(unit="MB"), 2 * n * ak.dtypes.int64.itemsize / (1024 * 1024))
        self.assertEqual(midx.memory_usage(unit="KB"), 2 * n * ak.dtypes.int64.itemsize / (1024))
        self.assertEqual(midx.memory_usage(unit="B"), 2 * n * ak.dtypes.int64.itemsize)

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

    def test_map(self):
        idx = ak.Index(ak.array([2, 3, 2, 3, 4]))

        result = idx.map({4: 25.0, 2: 30.0, 1: 7.0, 3: 5.0})
        self.assertListEqual(result.values.to_list(), [30.0, 5.0, 30.0, 5.0, 25.0])

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
