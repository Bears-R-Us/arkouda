import numpy as np
import pytest

import arkouda as ak
from arkouda.testing import assert_arkouda_array_equivalent


class TestManipulationFunctions:

    @pytest.mark.skip_if_rank_not_compiled([2])
    def test_vstack(self):
        # These tests will be implemented soon, hopefully.
        """
        a = [ak.random.randint(0, 10, 25) for _ in range(4)]
        n = [x.to_ndarray() for x in a]

        n_vstack = np.vstack(n)
        a_vstack = ak.vstack(a)

        assert n_vstack.tolist() == a_vstack.to_list()
        """

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, ak.bigint])
    def test_delete(self, size, dtype):
        dim_size = size
        if dtype == ak.bigint:
            a = ak.arange(2**200, 2**200 + dim_size)
            n = a.to_ndarray()
        else:
            a = ak.randint(0, 100, dim_size, dtype=dtype)
            n = a.to_ndarray()
        a_size = a.size

        ind = int(ak.randint(-a_size, a_size)[0])
        n_delete = np.delete(n, ind)
        a_delete = ak.delete(a, ind)
        assert_arkouda_array_equivalent(n_delete, a_delete)

        delete_inds = ak.randint(-a_size, a_size, max(1, dim_size // 3))
        n_delete = np.delete(n, delete_inds.to_list())
        a_delete = ak.delete(a, delete_inds.to_list())
        assert_arkouda_array_equivalent(n_delete, a_delete)

        delete_bounds = sorted(ak.randint(-a_size, a_size, 2).to_list())
        low, high = delete_bounds
        n_delete = np.delete(n, slice(low, high))
        a_delete = ak.delete(a, slice(low, high))
        assert_arkouda_array_equivalent(n_delete, a_delete)

        delete_bools = ak.randint(0, 2, a_size, dtype=bool)
        n_delete = np.delete(n, delete_bools.to_ndarray())
        a_delete = ak.delete(a, delete_bools)
        assert_arkouda_array_equivalent(n_delete, a_delete)

        for i in range(1):

            ind = int(ak.randint(-dim_size, dim_size)[0])
            n_delete = np.delete(n, ind, axis=i)
            a_delete = ak.delete(a, ind, axis=i)
            assert_arkouda_array_equivalent(n_delete, a_delete)

            delete_inds = ak.randint(-dim_size, dim_size, max(1, dim_size // 3))
            n_delete = np.delete(n, delete_inds.to_ndarray(), axis=i)
            a_delete = ak.delete(a, delete_inds, axis=i)
            assert_arkouda_array_equivalent(n_delete, a_delete)

            n_delete = np.delete(n, delete_inds.to_list(), axis=i)
            a_delete = ak.delete(a, delete_inds.to_list(), axis=i)
            assert_arkouda_array_equivalent(n_delete, a_delete)

            delete_bounds = sorted(ak.randint(-dim_size, dim_size, 2).to_list())
            low, high = delete_bounds
            n_delete = np.delete(n, slice(low, high), axis=i)
            a_delete = ak.delete(a, slice(low, high), axis=i)
            assert_arkouda_array_equivalent(n_delete, a_delete)

            delete_bools = ak.randint(0, 2, dim_size, dtype=bool)
            n_delete = np.delete(n, delete_bools.to_ndarray(), axis=i)
            a_delete = ak.delete(a, delete_bools, axis=i)
            assert_arkouda_array_equivalent(n_delete, a_delete)

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, ak.bigint])
    def test_delete_dim_2(self, size, dtype):
        dim_size = int(size**0.5) + 1
        if dtype == ak.bigint:
            a = ak.arange(2**200, 2**200 + dim_size**2).reshape((dim_size, dim_size))
            n = a.to_ndarray()
        else:
            a = ak.randint(0, 100, (dim_size, dim_size), dtype=dtype)
            n = a.to_ndarray()
        a_size = a.size

        ind = int(ak.randint(-a_size, a_size)[0])
        n_delete = np.delete(n, ind)
        a_delete = ak.delete(a, ind)
        assert_arkouda_array_equivalent(n_delete, a_delete)

        delete_inds = ak.randint(-a_size, a_size, max(1, a_size // 3))
        n_delete = np.delete(n, delete_inds.to_list())
        a_delete = ak.delete(a, delete_inds.to_list())
        assert_arkouda_array_equivalent(n_delete, a_delete)

        delete_bounds = sorted(ak.randint(-a_size, a_size, 2).to_list())
        low, high = delete_bounds
        n_delete = np.delete(n, slice(low, high))
        a_delete = ak.delete(a, slice(low, high))
        assert_arkouda_array_equivalent(n_delete, a_delete)

        delete_bools = ak.randint(0, 2, a_size, dtype=bool)
        n_delete = np.delete(n, delete_bools.to_ndarray())
        a_delete = ak.delete(a, delete_bools)
        assert_arkouda_array_equivalent(n_delete, a_delete)

        for i in range(2):

            ind = int(ak.randint(-dim_size, dim_size)[0])
            n_delete = np.delete(n, ind, axis=i)
            a_delete = ak.delete(a, ind, axis=i)
            assert_arkouda_array_equivalent(n_delete, a_delete)

            delete_inds = ak.randint(-dim_size, dim_size, max(1, dim_size // 3))
            n_delete = np.delete(n, delete_inds.to_ndarray(), axis=i)
            a_delete = ak.delete(a, delete_inds, axis=i)
            assert_arkouda_array_equivalent(n_delete, a_delete)

            n_delete = np.delete(n, delete_inds.to_list(), axis=i)
            a_delete = ak.delete(a, delete_inds.to_list(), axis=i)
            assert_arkouda_array_equivalent(n_delete, a_delete)

            delete_bounds = sorted(ak.randint(-dim_size, dim_size, 2).to_list())
            low, high = delete_bounds
            n_delete = np.delete(n, slice(low, high), axis=i)
            a_delete = ak.delete(a, slice(low, high), axis=i)
            assert_arkouda_array_equivalent(n_delete, a_delete)

            delete_bools = ak.randint(0, 2, dim_size, dtype=bool)
            n_delete = np.delete(n, delete_bools.to_ndarray(), axis=i)
            a_delete = ak.delete(a, delete_bools, axis=i)
            assert_arkouda_array_equivalent(n_delete, a_delete)

    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, ak.bigint])
    def test_delete_dim_3(self, size, dtype):
        dim_size = int(size ** (1 / 3)) + 1
        if dtype == ak.bigint:
            a = ak.arange(2**200, 2**200 + dim_size**3).reshape((dim_size, dim_size, dim_size))
            n = a.to_ndarray()
        else:
            a = ak.randint(0, 100, (dim_size, dim_size, dim_size), dtype=dtype)
            n = a.to_ndarray()
        a_size = a.size

        ind = int(ak.randint(-a_size, a_size)[0])
        n_delete = np.delete(n, ind)
        a_delete = ak.delete(a, ind)
        assert_arkouda_array_equivalent(n_delete, a_delete)

        delete_inds = ak.randint(-a_size, a_size, max(1, a_size // 3))
        n_delete = np.delete(n, delete_inds.to_list())
        a_delete = ak.delete(a, delete_inds.to_list())
        assert_arkouda_array_equivalent(n_delete, a_delete)

        delete_bounds = sorted(ak.randint(-a_size, a_size, 2).to_list())
        low, high = delete_bounds
        n_delete = np.delete(n, slice(low, high))
        a_delete = ak.delete(a, slice(low, high))
        assert_arkouda_array_equivalent(n_delete, a_delete)

        delete_bools = ak.randint(0, 2, a_size, dtype=bool)
        n_delete = np.delete(n, delete_bools.to_ndarray())
        a_delete = ak.delete(a, delete_bools)
        assert_arkouda_array_equivalent(n_delete, a_delete)

        for i in range(3):

            ind = int(ak.randint(-dim_size, dim_size)[0])
            n_delete = np.delete(n, ind, axis=i)
            a_delete = ak.delete(a, ind, axis=i)
            assert_arkouda_array_equivalent(n_delete, a_delete)

            delete_inds = ak.randint(-dim_size, dim_size, max(1, dim_size // 3))
            n_delete = np.delete(n, delete_inds.to_ndarray(), axis=i)
            a_delete = ak.delete(a, delete_inds, axis=i)
            assert_arkouda_array_equivalent(n_delete, a_delete)

            n_delete = np.delete(n, delete_inds.to_list(), axis=i)
            a_delete = ak.delete(a, delete_inds.to_list(), axis=i)
            assert_arkouda_array_equivalent(n_delete, a_delete)

            delete_bounds = sorted(ak.randint(-dim_size, dim_size, 2).to_list())
            low, high = delete_bounds
            n_delete = np.delete(n, slice(low, high), axis=i)
            a_delete = ak.delete(a, slice(low, high), axis=i)
            assert_arkouda_array_equivalent(n_delete, a_delete)

            delete_bools = ak.randint(0, 2, dim_size, dtype=bool)
            n_delete = np.delete(n, delete_bools.to_ndarray(), axis=i)
            a_delete = ak.delete(a, delete_bools, axis=i)
            assert_arkouda_array_equivalent(n_delete, a_delete)
