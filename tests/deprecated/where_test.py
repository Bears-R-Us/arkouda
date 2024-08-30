import warnings
from itertools import product

import numpy as np
from base_test import ArkoudaTest
from context import arkouda as ak

warnings.simplefilter("always", UserWarning)
SIZE = 10

"""
Tests the Arkouda where functionality and compares with analogous Numpy results.
"""


class WhereTest(ArkoudaTest):
    def setUp(self):
        ArkoudaTest.setUp(self)
        self.npA = {
            "int64": np.random.randint(0, 10, SIZE),
            "float64": np.random.randn(SIZE),
            "bool": np.random.randint(0, 2, SIZE, dtype="bool"),
        }
        self.akA = {k: ak.array(v) for k, v in self.npA.items()}
        self.npB = {
            "int64": np.random.randint(10, 20, SIZE),
            "float64": np.random.randn(SIZE) + 10,
            "bool": np.random.randint(0, 2, SIZE, dtype="bool"),
        }
        self.akB = {k: ak.array(v) for k, v in self.npB.items()}
        self.npCond = np.random.randint(0, 2, SIZE, dtype="bool")
        self.akCond = ak.array(self.npCond)
        self.scA = {"int64": 42, "float64": 2.71828, "bool": True}
        self.scB = {"int64": -1, "float64": 3.14159, "bool": False}
        self.dtypes = set(self.npA.keys())

    def test_where_equivalence(self):
        failures = 0
        tests = 0
        for dtype in self.dtypes:
            for (self.ak1, self.ak2), (self.np1, self.np2) in zip(
                product((self.akA, self.scA), (self.akB, self.scB)),
                product((self.npA, self.scA), (self.npB, self.scB)),
            ):
                tests += 1
                akres = ak.where(self.akCond, self.ak1[dtype], self.ak2[dtype]).to_ndarray()
                npres = np.where(self.npCond, self.np1[dtype], self.np2[dtype])
                if not np.allclose(akres, npres, equal_nan=True):
                    self.assertWarning(warnings.warn("{} !=\n{}".format(akres, npres)))
                    failures += 1
        self.assertEqual(0, failures)

    def test_error_handling(self):
        with self.assertRaises(TypeError):
            ak.where([0], ak.linspace(1, 10, 10), ak.linspace(1, 10, 10))

        with self.assertRaises(TypeError):
            ak.where(ak.linspace(1, 10, 10), [0], ak.linspace(1, 10, 10))

        with self.assertRaises(TypeError):
            ak.where(ak.linspace(1, 10, 10), ak.linspace(1, 10, 10), [0])

    def test_less_than_where_clause(self):
        n1 = np.arange(1, 10)
        n2 = np.ones(9, dtype=np.int64)
        a1 = ak.array(n1)
        a2 = ak.array(n2)

        cond = n1 < 5
        result = np.where(cond, n1, n2)
        self.assertListEqual([1, 2, 3, 4, 1, 1, 1, 1, 1], result.tolist())

        cond = a1 < 5
        result = ak.where(cond, a1, a2)
        self.assertListEqual([1, 2, 3, 4, 1, 1, 1, 1, 1], result.to_list())

    def test_greater_than_where_clause(self):
        n1 = np.arange(1, 10)
        n2 = np.ones(9, dtype=np.int64)
        a1 = ak.array(n1)
        a2 = ak.array(n2)

        cond = n1 > 5
        result = np.where(cond, n1, n2)
        self.assertListEqual([1, 1, 1, 1, 1, 6, 7, 8, 9], result.tolist())

        cond = a1 > 5
        result = ak.where(cond, a1, a2)
        self.assertListEqual([1, 1, 1, 1, 1, 6, 7, 8, 9], result.to_list())

    def test_greater_than_where_clause_with_scalars(self):
        n1 = np.arange(1, 10)
        a1 = ak.array(n1)

        condN = n1 > 5
        result = np.where(condN, n1, 1)
        self.assertListEqual([1, 1, 1, 1, 1, 6, 7, 8, 9], result.tolist())

        condA = a1 > 5
        result = ak.where(condA, a1, 1)
        self.assertListEqual([1, 1, 1, 1, 1, 6, 7, 8, 9], result.to_list())

        result = np.where(condN, 1, n1)
        self.assertListEqual([1, 2, 3, 4, 5, 1, 1, 1, 1], result.tolist())

        result = ak.where(condA, 1, a1)
        self.assertListEqual([1, 2, 3, 4, 5, 1, 1, 1, 1], result.to_list())

    def test_not_equal_where_clause(self):
        n1 = np.arange(1, 10)
        n2 = np.ones(9, dtype=np.int64)
        a1 = ak.array(n1)
        a2 = ak.array(n2)

        cond = n1 != 5
        result = np.where(cond, n1, n2)
        self.assertListEqual([1, 2, 3, 4, 1, 6, 7, 8, 9], result.tolist())

        cond = a1 != 5
        result = ak.where(cond, a1, a2)
        self.assertListEqual([1, 2, 3, 4, 1, 6, 7, 8, 9], result.to_list())

    def test_equals_where_clause(self):
        n1 = np.arange(1, 10)
        n2 = np.ones(9, dtype=np.int64)
        a1 = ak.array(n1)
        a2 = ak.array(n2)

        cond = n1 == 5
        result = np.where(cond, n1, n2)
        self.assertListEqual([1, 1, 1, 1, 5, 1, 1, 1, 1], result.tolist())

        cond = a1 == 5
        result = ak.where(cond, a1, a2)
        self.assertListEqual([1, 1, 1, 1, 5, 1, 1, 1, 1], result.to_list())

    def test_where_filter(self):
        n1 = np.arange(1, 10)
        a1 = ak.array(n1)
        n2 = np.arange(6, 10)
        a2 = ak.array(n2)

        self.assertListEqual(n2.tolist(), n1[n1 > 5].tolist())
        self.assertListEqual(a2.to_list(), a1[a1 > 5].to_list())

    def test_multiple_where_clauses(self):
        n1 = np.arange(1, 10)
        n2 = np.ones(9, dtype=np.int64)
        a1 = ak.array(n1)
        a2 = ak.array(n2)

        cond = n1 > 2, n1 < 8
        result = np.where(cond, n1, n2)
        self.assertListEqual(
            np.array([[1, 1, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 1, 1]]).tolist(),
            result.tolist(),
        )
        # Arkouda does not support multiple where clauses
        cond = a1 > 5, a1 < 8
        with self.assertRaises(TypeError):
            ak.where(cond, a1, a2)

    def test_dtypes(self):
        cond = (ak.arange(10) % 2) == 0
        for dt in (ak.int64, ak.uint64, ak.float64, ak.bool_):
            a = ak.ones(10, dtype=dt)
            b = ak.ones(10, dtype=dt)
            self.assertListEqual(ak.where(cond, a, b).to_list(), a.to_list())
            self.assertListEqual(ak.where(cond, 1, b).to_list(), a.to_list())
            self.assertListEqual(ak.where(cond, a, 1).to_list(), a.to_list())
