import warnings

import numpy as np
import pytest

import arkouda as ak

warnings.simplefilter("always", UserWarning)


# TODO: Parametrize test_where
class TestWhere:
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_where(self, size):
        npA = {
            "uint64": np.random.randint(0, 10, size),
            "int64": np.random.randint(0, 10, size),
            "float64": np.random.randn(size),
            "bool": np.random.randint(0, 2, size, dtype="bool"),
        }
        akA = {k: ak.array(v) for k, v in npA.items()}
        npB = {
            "uint64": np.random.randint(0, 10, size),
            "int64": np.random.randint(10, 20, size),
            "float64": np.random.randn(size) + 10,
            "bool": np.random.randint(0, 2, size, dtype="bool"),
        }
        akB = {k: ak.array(v) for k, v in npB.items()}
        npCond = np.random.randint(0, 2, size, dtype="bool")
        akCond = ak.array(npCond)
        dtypes = set(npA.keys())

        for dtype1, dtype2 in zip(dtypes, dtypes):
            akres = ak.where(akCond, akA[dtype1], akB[dtype2]).to_ndarray()
            npres = np.where(npCond, npA[dtype1], npB[dtype2])
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
        assert np_result.tolist() == result.tolist()

    def test_greater_than_where_clause(self):
        n1 = np.arange(1, 10)
        n2 = np.ones(9, dtype=np.int64)
        a1 = ak.array(n1)
        a2 = ak.array(n2)

        cond = n1 > 5
        np_result = np.where(cond, n1, n2)

        cond = a1 > 5
        result = ak.where(cond, a1, a2)
        assert np_result.tolist() == result.tolist()

    def test_greater_than_where_clause_with_scalars(self):
        n1 = np.arange(1, 10)
        a1 = ak.array(n1)

        condN = n1 > 5
        np_result = np.where(condN, n1, 1)

        condA = a1 > 5
        result = ak.where(condA, a1, 1)
        assert np_result.tolist() == result.tolist()

        np_result = np.where(condN, 1, n1)

        result = ak.where(condA, 1, a1)
        assert np_result.tolist() == result.tolist()

    def test_not_equal_where_clause(self):
        n1 = np.arange(1, 10)
        n2 = np.ones(9, dtype=np.int64)
        a1 = ak.array(n1)
        a2 = ak.array(n2)

        cond = n1 != 5
        np_result = np.where(cond, n1, n2)

        cond = a1 != 5
        result = ak.where(cond, a1, a2)
        assert np_result.tolist() == result.tolist()

    def test_equals_where_clause(self):
        n1 = np.arange(1, 10)
        n2 = np.ones(9, dtype=np.int64)
        a1 = ak.array(n1)
        a2 = ak.array(n2)

        cond = n1 == 5
        np_result = np.where(cond, n1, n2)

        cond = a1 == 5
        result = ak.where(cond, a1, a2)
        assert np_result.tolist() == result.tolist()

    def test_where_filter(self):
        n1 = np.arange(1, 10)
        a1 = ak.array(n1)
        n2 = np.arange(6, 10)
        a2 = ak.array(n2)

        assert n2.tolist() == n1[n1 > 5].tolist()
        assert a2.tolist() == a1[a1 > 5].tolist()
        assert n1[n1 > 5].tolist() == a1[a1 > 5].tolist()

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
            assert ak.where(cond, a, b).tolist() == a.tolist()
            assert ak.where(cond, 1, b).tolist() == a.tolist()
            assert ak.where(cond, a, 1).tolist() == a.tolist()
