import warnings
from itertools import product

import numpy as np
import pytest

import arkouda as ak

warnings.simplefilter("always", UserWarning)


class TestWhere:
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_where(self, size):
        npA = {
            "int64": np.random.randint(0, 10, size),
            "float64": np.random.randn(size),
            "bool": np.random.randint(0, 2, size, dtype="bool"),
        }
        akA = {k: ak.array(v) for k, v in npA.items()}
        npB = {
            "int64": np.random.randint(10, 20, size),
            "float64": np.random.randn(size) + 10,
            "bool": np.random.randint(0, 2, size, dtype="bool"),
        }
        akB = {k: ak.array(v) for k, v in npB.items()}
        npCond = np.random.randint(0, 2, size, dtype="bool")
        akCond = ak.array(npCond)
        scA = {"int64": 42, "float64": 2.71828, "bool": True}
        scB = {"int64": -1, "float64": 3.14159, "bool": False}
        dtypes = set(npA.keys())

        for dtype in dtypes:
            for (ak1, ak2), (np1, np2) in zip(
                product((akA, scA), (akB, scB)),
                product((npA, scA), (npB, scB)),
            ):
                akres = ak.where(akCond, ak1[dtype], ak2[dtype]).to_ndarray()
                npres = np.where(npCond, np1[dtype], np2[dtype])
                assert np.allclose(akres, npres, equal_nan=True)

    def test_error_handling(self):
        with pytest.raises(TypeError):
            ak.where([0], ak.linspace(1, 10, 10), ak.linspace(1, 10, 10))

        with pytest.raises(TypeError):
            ak.where(ak.linspace(1, 10, 10), [0], ak.linspace(1, 10, 10))

        with pytest.raises(TypeError):
            ak.where(ak.linspace(1, 10, 10), ak.linspace(1, 10, 10), [0])

    def test_less_than_where_clause(self):
        n1 = np.arange(1, 10)
        n2 = np.ones(9, dtype=np.int64)
        a1 = ak.array(n1)
        a2 = ak.array(n2)

        cond = n1 < 5
        np_result = np.where(cond, n1, n2)

        cond = a1 < 5
        result = ak.where(cond, a1, a2)
        assert np_result.tolist() == result.to_list()

    def test_greater_than_where_clause(self):
        n1 = np.arange(1, 10)
        n2 = np.ones(9, dtype=np.int64)
        a1 = ak.array(n1)
        a2 = ak.array(n2)

        cond = n1 > 5
        np_result = np.where(cond, n1, n2)

        cond = a1 > 5
        result = ak.where(cond, a1, a2)
        assert np_result.tolist() == result.to_list()

    def test_greater_than_where_clause_with_scalars(self):
        n1 = np.arange(1, 10)
        a1 = ak.array(n1)

        condN = n1 > 5
        np_result = np.where(condN, n1, 1)

        condA = a1 > 5
        result = ak.where(condA, a1, 1)
        assert np_result.tolist() == result.to_list()

        np_result = np.where(condN, 1, n1)

        result = ak.where(condA, 1, a1)
        assert np_result.tolist() == result.to_list()

    def test_not_equal_where_clause(self):
        n1 = np.arange(1, 10)
        n2 = np.ones(9, dtype=np.int64)
        a1 = ak.array(n1)
        a2 = ak.array(n2)

        cond = n1 != 5
        np_result = np.where(cond, n1, n2)

        cond = a1 != 5
        result = ak.where(cond, a1, a2)
        assert np_result.tolist() == result.to_list()

    def test_equals_where_clause(self):
        n1 = np.arange(1, 10)
        n2 = np.ones(9, dtype=np.int64)
        a1 = ak.array(n1)
        a2 = ak.array(n2)

        cond = n1 == 5
        np_result = np.where(cond, n1, n2)

        cond = a1 == 5
        result = ak.where(cond, a1, a2)
        assert np_result.tolist() == result.to_list()

    def test_where_filter(self):
        n1 = np.arange(1, 10)
        a1 = ak.array(n1)
        n2 = np.arange(6, 10)
        a2 = ak.array(n2)

        assert n2.tolist() == n1[n1 > 5].tolist()
        assert a2.to_list() == a1[a1 > 5].to_list()
        assert n1[n1 > 5].tolist() == a1[a1 > 5].to_list()

    def test_multiple_where_clauses(self):
        n1 = np.arange(1, 10)
        n2 = np.ones(9, dtype=np.int64)
        a1 = ak.array(n1)
        a2 = ak.array(n2)

        cond = n1 > 2, n1 < 8
        result = np.where(cond, n1, n2)
        assert (
            np.array([[1, 1, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 1, 1]]).tolist()
            == result.tolist()
        )
        # Arkouda does not support multiple where clauses
        cond = a1 > 5, a1 < 8
        with pytest.raises(TypeError):
            ak.where(cond, a1, a2)

    def test_dtypes(self):
        cond = (ak.arange(10) % 2) == 0
        for dt in (ak.int64, ak.uint64, ak.float64, ak.bool_):
            a = ak.ones(10, dtype=dt)
            b = ak.ones(10, dtype=dt)
            assert ak.where(cond, a, b).to_list() == a.to_list()
            assert ak.where(cond, 1, b).to_list() == a.to_list()
            assert ak.where(cond, a, 1).to_list() == a.to_list()
