import numpy as np
import pytest

import arkouda as ak
from arkouda.testing import assert_arkouda_array_equivalent


class TestManipulationFunctions:

    @pytest.mark.skip_if_rank_not_compiled([2])
    def test_vstack(self):
        a = [ak.random.randint(0, 10, 25) for _ in range(4)]
        n = [x.to_ndarray() for x in a]

        n_vstack = np.vstack(n)
        a_vstack = ak.vstack(a)

        assert_arkouda_array_equivalent(n_vstack, a_vstack)

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64])
    def test_vstack_with_dtypes(self, size, dtype):
        a = [ak.arange(i * size, (i + 1) * size, dtype=dtype) for i in range(4)]
        n = [x.to_ndarray() for x in a]

        n_vstack = np.vstack(n)
        a_vstack = ak.vstack(a)

        assert_arkouda_array_equivalent(n_vstack, a_vstack)

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64])
    @pytest.mark.parametrize("shapes", [[(3,), (3,)], [(2, 3), (3,)], [(3, 3), (4, 3)]])
    def test_vstack2D_with_shapes(self, dtype, shapes):
        shape1, shape2 = shapes

        shape1_prod = 1
        for i in shape1:
            shape1_prod = shape1_prod * i

        shape2_prod = 1
        for i in shape2:
            shape2_prod = shape2_prod * i

        ak_a = ak.arange(shape1_prod, dtype=dtype).reshape(shape1)
        ak_b = ak.arange(shape1_prod, (shape1_prod + shape2_prod), dtype=dtype).reshape(shape2)
        ak_vstack = ak.vstack((ak_a, ak_b))

        np_a = np.arange(shape1_prod, dtype=dtype).reshape(shape1)
        np_b = np.arange(shape1_prod, (shape1_prod + shape2_prod), dtype=dtype).reshape(shape2)
        np_vstack = np.vstack((np_a, np_b))

        assert_arkouda_array_equivalent(np_vstack, ak_vstack)

    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64])
    @pytest.mark.parametrize(
        "shapes", [[(1, 2, 3), (1, 2, 3)], [(1, 2, 3), (2, 2, 3)], [(2, 2, 2), (4, 2, 2)]]
    )
    def test_vstack3D_with_shapes(self, dtype, shapes):
        shape1, shape2 = shapes

        shape1_prod = 1
        for i in shape1:
            shape1_prod = shape1_prod * i

        shape2_prod = 1
        for i in shape2:
            shape2_prod = shape2_prod * i

        ak_a = ak.arange(shape1_prod, dtype=dtype).reshape(shape1)
        ak_b = ak.arange(shape1_prod, (shape1_prod + shape2_prod), dtype=dtype).reshape(shape2)
        ak_vstack = ak.vstack((ak_a, ak_b))

        np_a = np.arange(shape1_prod, dtype=dtype).reshape(shape1)
        np_b = np.arange(shape1_prod, (shape1_prod + shape2_prod), dtype=dtype).reshape(shape2)
        np_vstack = np.vstack((np_a, np_b))

        assert_arkouda_array_equivalent(np_vstack, ak_vstack)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64])
    def test_hstack(self, size, dtype):
        a = [ak.arange(i * size, (i + 1) * size, dtype=dtype) for i in range(4)]
        n = [x.to_ndarray() for x in a]

        n_hstack = np.hstack(n)
        a_hstack = ak.hstack(a)

        assert_arkouda_array_equivalent(n_hstack, a_hstack)

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64])
    @pytest.mark.parametrize("shapes", [[(3, 1), (3, 1)], [(2, 3), (2, 4)], [(3, 5), (3, 2)]])
    def test_hstack2D_with_shapes(self, dtype, shapes):
        shape1, shape2 = shapes

        shape1_prod = 1
        for i in shape1:
            shape1_prod = shape1_prod * i

        shape2_prod = 1
        for i in shape2:
            shape2_prod = shape2_prod * i

        ak_a = ak.arange(shape1_prod, dtype=dtype).reshape(shape1)
        ak_b = ak.arange(shape1_prod, (shape1_prod + shape2_prod), dtype=dtype).reshape(shape2)
        ak_hstack = ak.hstack((ak_a, ak_b))

        np_a = np.arange(shape1_prod, dtype=dtype).reshape(shape1)
        np_b = np.arange(shape1_prod, (shape1_prod + shape2_prod), dtype=dtype).reshape(shape2)
        np_hstack = np.hstack((np_a, np_b))

        assert_arkouda_array_equivalent(np_hstack, ak_hstack)

    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64])
    @pytest.mark.parametrize(
        "shapes", [[(1, 2, 3), (1, 2, 3)], [(1, 2, 3), (1, 1, 3)], [(2, 2, 2), (2, 4, 2)]]
    )
    def test_hstack3D_with_shapes(self, dtype, shapes):
        shape1, shape2 = shapes

        shape1_prod = 1
        for i in shape1:
            shape1_prod = shape1_prod * i

        shape2_prod = 1
        for i in shape2:
            shape2_prod = shape2_prod * i

        ak_a = ak.arange(shape1_prod, dtype=dtype).reshape(shape1)
        ak_b = ak.arange(shape1_prod, (shape1_prod + shape2_prod), dtype=dtype).reshape(shape2)
        ak_hstack = ak.hstack((ak_a, ak_b))

        np_a = np.arange(shape1_prod, dtype=dtype).reshape(shape1)
        np_b = np.arange(shape1_prod, (shape1_prod + shape2_prod), dtype=dtype).reshape(shape2)
        np_hstack = np.hstack((np_a, np_b))

        assert_arkouda_array_equivalent(np_hstack, ak_hstack)

    @pytest.mark.skip_if_rank_not_compiled([2])
    def test_delete(self):
        # This test fails, but wasn't included in pytest.ini before
        """
        a = ak.randint(0, 100, (10, 10))
        n = a.to_ndarray()

        n_delete = np.delete(n, 5, axis=1)
        a_delete = ak.delete(a, 5, axis=1)

        assert n_delete.tolist() == a_delete.to_list()

        n_delete = np.delete(n, np.array([1, 3, 5]), axis=0)
        a_delete = ak.delete(a, ak.array([1, 3, 5]), axis=0)

        assert n_delete.tolist() == a_delete.to_list()

        n_delete = np.delete(n, np.array([1, 3, 5]))
        a_delete = ak.delete(a, ak.array([1, 3, 5]))

        assert n_delete.tolist() == a_delete.to_list()

        n_delete = np.delete(n, slice(3, 5), axis=1)
        a_delete = ak.delete(a, slice(3, 5), axis=1)

        assert n_delete.tolist() == a_delete.to_list()
        """
