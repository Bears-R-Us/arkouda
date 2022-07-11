import glob
import os
from shutil import rmtree

from base_test import ArkoudaTest
from context import arkouda as ak


class IndexTest(ArkoudaTest):
    def test_index_creation(self):
        idx = ak.Index(ak.arange(5))

        self.assertIsInstance(idx, ak.Index)
        self.assertEqual(idx.size, 5)
        self.assertListEqual(idx.to_ndarray().tolist(), [i for i in range(5)])

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

    def test_factory(self):
        idx = ak.Index.factory(ak.arange(5))
        self.assertIsInstance(idx, ak.Index)

        idx = ak.Index.factory([ak.arange(5), ak.arange(5)])
        self.assertIsInstance(idx, ak.MultiIndex)

    def test_argsort(self):
        idx = ak.Index.factory(ak.arange(5))
        i = idx.argsort(False)
        self.assertListEqual(i.to_ndarray().tolist(), [4, 3, 2, 1, 0])

        idx = ak.Index(ak.array([1, 0, 4, 2, 5, 3]))
        i = idx.argsort()
        # values should be the indexes in the array of idx
        self.assertListEqual(i.to_ndarray().tolist(), [1, 0, 3, 5, 2, 4])

    def test_concat(self):
        idx_1 = ak.Index.factory(ak.arange(5))

        idx_2 = ak.Index(ak.array([2, 4, 1, 3, 0]))

        idx_full = idx_1.concat(idx_2)
        self.assertListEqual(idx_full.to_ndarray().tolist(), [0, 1, 2, 3, 4, 2, 4, 1, 3, 0])

    def test_lookup(self):
        idx = ak.Index.factory(ak.arange(5))
        lk = idx.lookup(ak.array([0, 4]))
        self.assertListEqual(lk.to_ndarray().tolist(), [True, False, False, False, True])

    def test_multi_argsort(self):
        idx = ak.Index.factory([ak.arange(5), ak.arange(5)])
        s = idx.argsort(False)
        self.assertListEqual(s.to_ndarray().tolist(), [4, 3, 2, 1, 0])

        s = idx.argsort()
        self.assertListEqual(s.to_ndarray().tolist(), [i for i in range(5)])

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
        self.assertListEqual(result.to_ndarray().tolist(), [True, False, True, True, False])

    def test_save(self):
        locale_count = ak.get_config()["numLocales"]
        d = f"{os.getcwd()}/save_index_test"
        # make directory to save to so pandas read works
        os.mkdir(d)

        idx = ak.Index(ak.arange(5))
        idx.save(f"{d}/idx_file.h5")
        self.assertTrue(len(glob.glob(f"{d}/idx_file_*.h5")) == locale_count)

        # clean up test files
        rmtree(d)
