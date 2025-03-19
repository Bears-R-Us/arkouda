import numpy as np
import pytest

import arkouda as ak
from arkouda.client import get_array_ranks
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
    @pytest.mark.parametrize("num_dims", [1, 2, 3])
    def test_delete(self, size, dtype, num_dims):

        if num_dims in get_array_ranks():

            test_dims = [None] + list(range(num_dims))

            dim_size = int(size ** (1 / num_dims)) + 1
            if dtype == ak.bigint:
                a = ak.arange(2**200, 2**200 + dim_size**num_dims)
                a = a.reshape((dim_size,) * num_dims)
                n = a.to_ndarray()
            else:
                a = ak.randint(0, 100, dim_size**num_dims, dtype=dtype)
                a = a.reshape((dim_size,) * num_dims)
                n = a.to_ndarray()
            a_size = a.size

            for i in test_dims:

                axis_dim_size = a_size if i is None else a.shape[i]

                ind = int(ak.randint(-axis_dim_size, axis_dim_size)[0])
                n_delete = np.delete(n, ind, axis=i)
                a_delete = ak.delete(a, ind, axis=i)
                assert_arkouda_array_equivalent(n_delete, a_delete)

                delete_inds = ak.randint(-axis_dim_size, axis_dim_size, max(1, axis_dim_size // 3))
                n_delete = np.delete(n, delete_inds.to_ndarray(), axis=i)
                a_delete = ak.delete(a, delete_inds, axis=i)
                assert_arkouda_array_equivalent(n_delete, a_delete)

                n_delete = np.delete(n, delete_inds.to_list(), axis=i)
                a_delete = ak.delete(a, delete_inds.to_list(), axis=i)
                assert_arkouda_array_equivalent(n_delete, a_delete)

                delete_bounds = sorted(ak.randint(-axis_dim_size, axis_dim_size, 2).to_list())
                if delete_bounds[0] < 0 <= delete_bounds[1] < delete_bounds[0] + a_size:
                    delete_bounds = delete_bounds[::-1]
                low, high = delete_bounds
                n_delete = np.delete(n, slice(low, high), axis=i)
                a_delete = ak.delete(a, slice(low, high), axis=i)
                assert_arkouda_array_equivalent(n_delete, a_delete)

                delete_bools = ak.randint(0, 2, axis_dim_size, dtype=bool)
                n_delete = np.delete(n, delete_bools.to_ndarray(), axis=i)
                a_delete = ak.delete(a, delete_bools, axis=i)
                assert_arkouda_array_equivalent(n_delete, a_delete)

        else:
            pytest.skip(f"Sever not compiled for rank {num_dims}")
