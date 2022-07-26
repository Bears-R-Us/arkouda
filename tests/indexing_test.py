import numpy as np
from base_test import ArkoudaTest
from context import arkouda as ak

SIZE = 100


class IndexingTest(ArkoudaTest):
    def setUp(self):
        ArkoudaTest.setUp(self)

        self.ikeys = ak.arange(SIZE)
        self.ukeys = ak.arange(SIZE, dtype=ak.uint64)
        self.i = ak.randint(0, SIZE, SIZE)
        self.u = ak.cast(self.i, ak.uint64)
        self.f = ak.array(np.random.randn(SIZE))  # normally dist random numbers
        self.b = (self.i % 2) == 0
        self.s = ak.cast(self.i, str)
        self.array_dict = {"int64": self.i, "uint64": self.u, "float64": self.f, "bool": self.b}

    def test_pdarray_uint_indexing(self):
        # for every pda in array_dict test indexing with uint array and uint scalar
        for pda in self.array_dict.values():
            self.assertEqual(pda[np.uint(2)], pda[2])
            self.assertListEqual(pda[self.ukeys].to_list(), pda[self.ikeys].to_list())

    def test_strings_uint_indexing(self):
        # test Strings array indexing with uint array and uint scalar
        self.assertEqual(self.s[np.uint(2)], self.s[2])
        self.assertListEqual(self.s[self.ukeys].to_list(), self.s[self.ikeys].to_list())

    def test_uint_bool_indexing(self):
        # test uint array with bool indexing
        self.assertListEqual(self.u[self.b].to_list(), self.i[self.b].to_list())

    def test_set_uint(self):
        # for every pda in array_dict test __setitem__ indexing with uint array and uint scalar
        for t, pda in self.array_dict.items():
            # set [int] = val with uint key and value
            pda[np.uint(2)] = np.uint(5)
            self.assertEqual(pda[np.uint(2)], pda[2])

            # set [slice] = scalar/pdarray
            pda[:10] = np.uint(-2)
            self.assertListEqual(pda[self.ukeys].to_list(), pda[self.ikeys].to_list())
            pda[:10] = ak.cast(ak.arange(10), t)
            self.assertListEqual(pda[self.ukeys].to_list(), pda[self.ikeys].to_list())

            # set [pdarray] = scalar/pdarray with uint key pdarray
            pda[ak.arange(10, dtype=ak.uint64)] = np.uint(3)
            self.assertListEqual(pda[self.ukeys].to_list(), pda[self.ikeys].to_list())
            pda[ak.arange(10, dtype=ak.uint64)] = ak.cast(ak.arange(10), t)
            self.assertListEqual(pda[self.ukeys].to_list(), pda[self.ikeys].to_list())

    def test_indexing_with_uint(self):
        # verify reproducer from #1210 no longer fails
        a = ak.arange(10) * 2
        b = ak.cast(ak.array([3, 0, 8]), ak.uint64)
        a[b]
