import numpy as np
import pytest

import arkouda as ak
from arkouda.client import get_array_ranks
from arkouda.numpy import pdarraymanipulation
from arkouda.testing import assert_arkouda_array_equivalent


class TestManipulationFunctions:
    @pytest.mark.skip_if_rank_not_compiled([1, 2, 3])
    def test_pdarraymanipulation_docstrings(self):
        import doctest

        result = doctest.testmod(
            pdarraymanipulation,
            optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.skip_if_rank_not_compiled([2])
    def test_vstack(self):
        a = [ak.random.randint(0, 10, 25) for _ in range(4)]
        n = [x.to_ndarray() for x in a]

        n_vstack = np.vstack(n)
        a_vstack = ak.vstack(a)

        assert_arkouda_array_equivalent(n_vstack, a_vstack)

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, ak.bigint])
    def test_vstack_with_dtypes(self, size, dtype):
        if dtype == ak.bigint:
            a = [ak.arange(2**200 + i * size, 2**200 + (i + 1) * size, dtype=dtype) for i in range(4)]
            n = [np.arange(2**200 + i * size, 2**200 + (i + 1) * size) for i in range(4)]
        else:
            a = [ak.arange(i * size, (i + 1) * size, dtype=dtype) for i in range(4)]
            n = [x.to_ndarray() for x in a]
        n_vstack = np.vstack(n)
        a_vstack = ak.vstack(a)

        assert_arkouda_array_equivalent(n_vstack, a_vstack)

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, ak.bigint])
    @pytest.mark.parametrize("shapes", [[(3,), (3,)], [(2, 3), (3,)], [(3, 3), (4, 3)]])
    def test_vstack2D_with_shapes(self, dtype, shapes):
        shape1, shape2 = shapes

        shape1_prod = 1
        for i in shape1:
            shape1_prod = shape1_prod * i

        shape2_prod = 1
        for i in shape2:
            shape2_prod = shape2_prod * i

        if dtype == ak.bigint:
            ak_a = ak.arange(2**200, 2**200 + shape1_prod, dtype=dtype).reshape(shape1)
            ak_b = ak.arange(
                2**200 + shape1_prod, 2**200 + (shape1_prod + shape2_prod), dtype=dtype
            ).reshape(shape2)
            ak_vstack = ak.vstack((ak_a, ak_b))

            np_a = np.arange(2**200, 2**200 + shape1_prod).reshape(shape1)
            np_b = np.arange(2**200 + shape1_prod, 2**200 + (shape1_prod + shape2_prod)).reshape(shape2)
            np_vstack = np.vstack((np_a, np_b))
        else:
            ak_a = ak.arange(shape1_prod, dtype=dtype).reshape(shape1)
            ak_b = ak.arange(shape1_prod, (shape1_prod + shape2_prod), dtype=dtype).reshape(shape2)
            ak_vstack = ak.vstack((ak_a, ak_b))

            np_a = np.arange(shape1_prod, dtype=dtype).reshape(shape1)
            np_b = np.arange(shape1_prod, (shape1_prod + shape2_prod), dtype=dtype).reshape(shape2)
            np_vstack = np.vstack((np_a, np_b))

        assert_arkouda_array_equivalent(np_vstack, ak_vstack)

    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, ak.bigint])
    @pytest.mark.parametrize(
        "shapes",
        [[(1, 2, 3), (1, 2, 3)], [(1, 2, 3), (2, 2, 3)], [(2, 2, 2), (4, 2, 2)]],
    )
    def test_vstack3D_with_shapes(self, dtype, shapes):
        shape1, shape2 = shapes

        shape1_prod = 1
        for i in shape1:
            shape1_prod = shape1_prod * i

        shape2_prod = 1
        for i in shape2:
            shape2_prod = shape2_prod * i

        if dtype == ak.bigint:
            ak_a = ak.arange(2**200, 2**200 + shape1_prod, dtype=dtype).reshape(shape1)
            ak_b = ak.arange(
                2**200 + shape1_prod, 2**200 + (shape1_prod + shape2_prod), dtype=dtype
            ).reshape(shape2)
            ak_vstack = ak.vstack((ak_a, ak_b))

            np_a = np.arange(2**200, 2**200 + shape1_prod).reshape(shape1)
            np_b = np.arange(2**200 + shape1_prod, 2**200 + (shape1_prod + shape2_prod)).reshape(shape2)
            np_vstack = np.vstack((np_a, np_b))
        else:
            ak_a = ak.arange(shape1_prod, dtype=dtype).reshape(shape1)
            ak_b = ak.arange(shape1_prod, (shape1_prod + shape2_prod), dtype=dtype).reshape(shape2)
            ak_vstack = ak.vstack((ak_a, ak_b))

            np_a = np.arange(shape1_prod, dtype=dtype).reshape(shape1)
            np_b = np.arange(shape1_prod, (shape1_prod + shape2_prod), dtype=dtype).reshape(shape2)
            np_vstack = np.vstack((np_a, np_b))

        assert_arkouda_array_equivalent(np_vstack, ak_vstack)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, ak.bigint])
    def test_hstack(self, size, dtype):
        if dtype == ak.bigint:
            a = [ak.arange(2**200 + i * size, 2**200 + (i + 1) * size, dtype=dtype) for i in range(4)]
            n = [np.arange(2**200 + i * size, 2**200 + (i + 1) * size) for i in range(4)]
        else:
            a = [ak.arange(i * size, (i + 1) * size, dtype=dtype) for i in range(4)]
            n = [x.to_ndarray() for x in a]

        n_hstack = np.hstack(n)
        a_hstack = ak.hstack(a)

        assert_arkouda_array_equivalent(n_hstack, a_hstack)

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, ak.bigint])
    @pytest.mark.parametrize("shapes", [[(3, 1), (3, 1)], [(2, 3), (2, 4)], [(3, 5), (3, 2)]])
    def test_hstack2D_with_shapes(self, dtype, shapes):
        shape1, shape2 = shapes

        shape1_prod = 1
        for i in shape1:
            shape1_prod = shape1_prod * i

        shape2_prod = 1
        for i in shape2:
            shape2_prod = shape2_prod * i

        if dtype == ak.bigint:
            ak_a = ak.arange(2**200, 2**200 + shape1_prod, dtype=dtype).reshape(shape1)
            ak_b = ak.arange(
                2**200 + shape1_prod, 2**200 + (shape1_prod + shape2_prod), dtype=dtype
            ).reshape(shape2)
            ak_hstack = ak.hstack((ak_a, ak_b))

            np_a = np.arange(2**200, 2**200 + shape1_prod).reshape(shape1)
            np_b = np.arange(2**200 + shape1_prod, 2**200 + (shape1_prod + shape2_prod)).reshape(shape2)
            np_hstack = np.hstack((np_a, np_b))
        else:
            ak_a = ak.arange(shape1_prod, dtype=dtype).reshape(shape1)
            ak_b = ak.arange(shape1_prod, (shape1_prod + shape2_prod), dtype=dtype).reshape(shape2)
            ak_hstack = ak.hstack((ak_a, ak_b))

            np_a = np.arange(shape1_prod, dtype=dtype).reshape(shape1)
            np_b = np.arange(shape1_prod, (shape1_prod + shape2_prod), dtype=dtype).reshape(shape2)
            np_hstack = np.hstack((np_a, np_b))

        assert_arkouda_array_equivalent(np_hstack, ak_hstack)

    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, ak.bigint])
    @pytest.mark.parametrize(
        "shapes",
        [[(1, 2, 3), (1, 2, 3)], [(1, 2, 3), (1, 1, 3)], [(2, 2, 2), (2, 4, 2)]],
    )
    def test_hstack3D_with_shapes(self, dtype, shapes):
        shape1, shape2 = shapes

        shape1_prod = 1
        for i in shape1:
            shape1_prod = shape1_prod * i

        shape2_prod = 1
        for i in shape2:
            shape2_prod = shape2_prod * i

        if dtype == ak.bigint:
            ak_a = ak.arange(2**200, 2**200 + shape1_prod, dtype=dtype).reshape(shape1)
            ak_b = ak.arange(
                2**200 + shape1_prod, 2**200 + (shape1_prod + shape2_prod), dtype=dtype
            ).reshape(shape2)
            ak_hstack = ak.hstack((ak_a, ak_b))

            np_a = np.arange(2**200, 2**200 + shape1_prod).reshape(shape1)
            np_b = np.arange(2**200 + shape1_prod, 2**200 + (shape1_prod + shape2_prod)).reshape(shape2)
            np_hstack = np.hstack((np_a, np_b))
        else:
            ak_a = ak.arange(shape1_prod, dtype=dtype).reshape(shape1)
            ak_b = ak.arange(shape1_prod, (shape1_prod + shape2_prod), dtype=dtype).reshape(shape2)
            ak_hstack = ak.hstack((ak_a, ak_b))

            np_a = np.arange(shape1_prod, dtype=dtype).reshape(shape1)
            np_b = np.arange(shape1_prod, (shape1_prod + shape2_prod), dtype=dtype).reshape(shape2)
            np_hstack = np.hstack((np_a, np_b))

        assert_arkouda_array_equivalent(np_hstack, ak_hstack)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, ak.bigint])
    def test_append(self, size, dtype):
        if dtype == ak.bigint:
            a = [ak.arange(2**200 + i * size, 2**200 + (i + 1) * size, dtype=dtype) for i in range(2)]
            n = [np.arange(2**200 + i * size, 2**200 + (i + 1) * size) for i in range(2)]
        else:
            a = [ak.arange(i * size, (i + 1) * size, dtype=dtype) for i in range(2)]
            n = [x.to_ndarray() for x in a]

        n_append = np.append(*n)
        a_append = ak.append(*a)

        assert_arkouda_array_equivalent(n_append, a_append)

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, ak.bigint])
    @pytest.mark.parametrize("shapes", [[(3, 1), (3, 1)], [(2, 3), (2, 4)], [(3, 5), (3, 2)]])
    def test_append2D_with_shapes(self, dtype, shapes):
        shape1, shape2 = shapes

        shape1_prod = 1
        for i in shape1:
            shape1_prod = shape1_prod * i

        shape2_prod = 1
        for i in shape2:
            shape2_prod = shape2_prod * i

        for axis in [None, 0, 1]:
            if axis is not None and shape1_prod // shape1[axis] == shape2_prod // shape2[axis]:
                if dtype == ak.bigint:
                    ak_a = ak.arange(2**200, 2**200 + shape1_prod, dtype=dtype).reshape(shape1)
                    ak_b = ak.arange(
                        2**200 + shape1_prod, 2**200 + (shape1_prod + shape2_prod), dtype=dtype
                    ).reshape(shape2)
                    ak_append = ak.append(ak_a, ak_b, axis=axis)

                    np_a = np.arange(2**200, 2**200 + shape1_prod).reshape(shape1)
                    np_b = np.arange(2**200 + shape1_prod, 2**200 + (shape1_prod + shape2_prod)).reshape(
                        shape2
                    )
                    np_append = np.append(np_a, np_b, axis=axis)
                else:
                    ak_a = ak.arange(shape1_prod, dtype=dtype).reshape(shape1)
                    ak_b = ak.arange(shape1_prod, (shape1_prod + shape2_prod), dtype=dtype).reshape(
                        shape2
                    )
                    ak_append = ak.append(ak_a, ak_b, axis=axis)

                    np_a = np.arange(shape1_prod, dtype=dtype).reshape(shape1)
                    np_b = np.arange(shape1_prod, (shape1_prod + shape2_prod), dtype=dtype).reshape(
                        shape2
                    )
                    np_append = np.append(np_a, np_b, axis=axis)

                assert_arkouda_array_equivalent(np_append, ak_append)

    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, ak.bigint])
    @pytest.mark.parametrize(
        "shapes",
        [[(1, 2, 3), (1, 2, 3)], [(1, 2, 3), (1, 1, 3)], [(2, 2, 2), (2, 4, 2)]],
    )
    def test_append3D_with_shapes(self, dtype, shapes):
        shape1, shape2 = shapes

        shape1_prod = 1
        for i in shape1:
            shape1_prod = shape1_prod * i

        shape2_prod = 1
        for i in shape2:
            shape2_prod = shape2_prod * i

        for axis in [None, 0, 1, 2]:
            if axis is not None and shape1_prod // shape1[axis] == shape2_prod // shape2[axis]:
                if dtype == ak.bigint:
                    ak_a = ak.arange(2**200, 2**200 + shape1_prod, dtype=dtype).reshape(shape1)
                    ak_b = ak.arange(
                        2**200 + shape1_prod, 2**200 + (shape1_prod + shape2_prod), dtype=dtype
                    ).reshape(shape2)
                    ak_append = ak.append(ak_a, ak_b, axis=axis)

                    np_a = np.arange(2**200, 2**200 + shape1_prod).reshape(shape1)
                    np_b = np.arange(2**200 + shape1_prod, 2**200 + (shape1_prod + shape2_prod)).reshape(
                        shape2
                    )
                    np_append = np.append(np_a, np_b, axis=axis)
                else:
                    ak_a = ak.arange(shape1_prod, dtype=dtype).reshape(shape1)
                    ak_b = ak.arange(shape1_prod, (shape1_prod + shape2_prod), dtype=dtype).reshape(
                        shape2
                    )
                    ak_append = ak.append(ak_a, ak_b, axis=axis)

                    np_a = np.arange(shape1_prod, dtype=dtype).reshape(shape1)
                    np_b = np.arange(shape1_prod, (shape1_prod + shape2_prod), dtype=dtype).reshape(
                        shape2
                    )
                    np_append = np.append(np_a, np_b, axis=axis)

                assert_arkouda_array_equivalent(np_append, ak_append)

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

                n_delete = np.delete(n, delete_inds.tolist(), axis=i)
                a_delete = ak.delete(a, delete_inds.tolist(), axis=i)
                assert_arkouda_array_equivalent(n_delete, a_delete)

                delete_bounds = sorted(ak.randint(-axis_dim_size, axis_dim_size, 2).tolist())
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
            pytest.skip(f"Server not compiled for rank {num_dims}")
