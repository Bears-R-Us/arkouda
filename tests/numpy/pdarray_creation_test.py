import math
import statistics

from collections import deque

import numpy as np
import pandas as pd
import pytest

import arkouda as ak

from arkouda.numpy import newaxis, pdarraycreation
from arkouda.numpy.util import _generate_test_shape, _infer_shape_from_size
from arkouda.testing import assert_almost_equivalent, assert_arkouda_array_equal, assert_equivalent
from arkouda.testing import assert_equal as ak_assert_equal


INT_SCALARS = list(ak.numpy.dtypes.int_scalars.__args__)
NUMERIC_SCALARS = list(ak.numpy.dtypes.numeric_scalars.__args__)

DTYPES = [
    bool,
    float,
    ak.float64,
    int,
    ak.int64,
    str,
    ak.str_,
    ak.uint64,
    ak.uint8,
    ak.bigint,
]

#   multi_dim_ranks is used in multi_dim_testing


def multi_dim_ranks():
    ranks = ak.client.get_array_ranks()[:]
    ranks.remove(1)
    return ranks


class TestPdarrayCreation:
    @pytest.mark.skip_if_rank_not_compiled([1, 2, 3])
    def test_pdarraycreation_docstrings(self):
        import doctest

        result = doctest.testmod(
            pdarraycreation, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_array_creation(self, dtype, subtests):
        fixed_size = 100
        input_cases = [
            ("ak.ones(int)", ak.ones(fixed_size, int)),
            ("np.ones", np.ones(fixed_size)),
            ("list(range)", list(range(fixed_size))),
            ("tuple(range)", tuple(range(fixed_size))),
            ("deque(range)", deque(range(fixed_size))),
            ("list[str]", [f"{i}" for i in range(fixed_size)]),
        ]

        for name, input_data in input_cases:
            with subtests.test(source=name, dtype=dtype):
                pda = ak.array(input_data, dtype=dtype)

                expected_type = ak.Strings if "str" in str(dtype) else ak.pdarray

                assert isinstance(pda, expected_type), f"{name}: Type mismatch"
                assert len(pda) == fixed_size, f"{name}: Length mismatch"
                assert pda.dtype == dtype, f"{name}: Dtype mismatch ({pda.dtype} != {dtype})"

    # TODO: combine the many instances of 1D and multi-dim tests into one function each.
    #      e.g., there will just be test_array_creation, which will handle both 1D and
    #      multi-dim tests

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, bool, ak.bool_])
    def test_array_creation_multi_dim(self, size, dtype, subtests):
        # Tests using "range"-based inputs are omitted as they are 1D-only.
        for rank in multi_dim_ranks():
            shape, local_size = _generate_test_shape(rank, size)
            input_sources = [
                ("ak.ones", ak.ones(shape, int)),
                ("np.ones", np.ones(shape)),
            ]
            for name, input_data in input_sources:
                with subtests.test(rank=rank, dtype=dtype, source=name):
                    pda = ak.array(input_data, dtype=dtype)

                    assert isinstance(pda, ak.pdarray), f"{name}: Unexpected type {type(pda)}"
                    assert pda.shape == shape, f"{name}: Shape mismatch {pda.shape} != {shape}"
                    assert pda.dtype == dtype, f"{name}: Dtype mismatch {pda.dtype} != {dtype}"

                    expected = np.ones(shape, dtype=np.dtype(dtype))
                    assert_equivalent(pda, expected)

    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, bool, ak.bool_])
    def test_array_creation_error(self, dtype, subtests):
        rank = ak.client.get_max_array_rank() + 1
        assert rank > ak.client.get_max_array_rank(), "Test rank must exceed supported max"
        shape, _ = _generate_test_shape(rank, 2**rank)

        # Attempt to create an array with rank > max supported; should raise ValueError
        with subtests.test(dtype=dtype, shape=shape):
            with pytest.raises(ValueError):
                ak.array(np.ones(shape), dtype=dtype)

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_large_array_creation(self, size, subtests):
        """
        Test large-array creation using various Arkouda constructors.

        Ensures correct length and type.
        """
        test_cases = [
            ("ak.ones", lambda: ak.ones(size, int)),
            ("ak.array(ak.ones)", lambda: ak.array(ak.ones(size, int))),
            ("ak.zeros", lambda: ak.zeros(size)),
            ("ak.ones (default dtype)", lambda: ak.ones(size)),
            ("ak.full (str)", lambda: ak.full(size, "test")),
            ("ak.zeros_like", lambda: ak.zeros_like(ak.ones(size))),
            ("ak.ones_like", lambda: ak.ones_like(ak.zeros(size))),
            ("ak.full_like", lambda: ak.full_like(ak.zeros(size), 9)),
            ("ak.arange", lambda: ak.arange(size)),
            ("ak.linspace", lambda: ak.linspace(0, size, size)),
            ("ak.randint", lambda: ak.randint(0, size, size)),
            ("ak.uniform", lambda: ak.uniform(size, 0, 100)),
            ("ak.standard_normal", lambda: ak.standard_normal(size)),
            (
                "ak.random_strings_uniform",
                lambda: ak.random_strings_uniform(3, 30, size),
            ),
            (
                "ak.random_strings_lognormal",
                lambda: ak.random_strings_lognormal(2, 0.25, size),
            ),
            (
                "ak.from_series",
                lambda: ak.from_series(pd.Series(ak.arange(size).to_ndarray())),
            ),
            (
                "ak.bigint_from_uint_arrays",
                lambda: ak.bigint_from_uint_arrays([ak.ones(size, dtype=ak.uint64)]),
            ),
        ]

        for name, constructor in test_cases:
            with subtests.test(source=name):
                pda = constructor()
                expected_type = ak.Strings if pda.dtype == str else ak.pdarray
                assert isinstance(pda, expected_type), f"{name}: Type mismatch"
                assert len(pda) == size, f"{name}: Size mismatch: {len(pda)} != {size}"

    @pytest.mark.parametrize("dtype", [ak.int64, ak.float64, ak.bool_, ak.bigint])
    def test_array_copy(self, dtype):
        a = ak.arange(100, dtype=dtype)

        b = ak.array(a, copy=True)
        assert a is not b
        ak_assert_equal(a, b)

        c = ak.array(a, copy=False)
        assert a is c
        ak_assert_equal(a, c)

    from functools import partial

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_large_array_creation_multi_dim(self, size, subtests):
        """
        Test multi-dimensional array creation for supported Arkouda constructors.
        Excludes 1D-only functions like linspace, arange, and random strings.
        """
        for rank in multi_dim_ranks():
            shape, local_size = _generate_test_shape(rank, size)

            # Bind loop variables at definition time to satisfy Ruff B023
            test_cases = [
                ("ak.ones", lambda s=shape: ak.ones(s, int)),
                ("ak.array(ak.ones)", lambda s=shape: ak.array(ak.ones(s, int))),
                ("ak.zeros", lambda s=shape: ak.zeros(s)),
                ("ak.ones (default)", lambda s=shape: ak.ones(s)),
                ("ak.full", lambda s=shape: ak.full(s, 9)),  # only numeric for multi-dim
                ("ak.zeros_like", lambda s=shape: ak.zeros_like(ak.ones(s))),
                ("ak.ones_like", lambda s=shape: ak.ones_like(ak.zeros(s))),
                ("ak.full_like", lambda s=shape: ak.full_like(ak.zeros(s), 9)),
                ("ak.randint", lambda s=shape, n=local_size: ak.randint(0, n, s)),
            ]

            for name, constructor in test_cases:
                with subtests.test(func=name, rank=rank, shape=shape):
                    pda = constructor()
                    assert isinstance(pda, ak.pdarray), f"{name}: Expected ak.pdarray, got {type(pda)}"
                    assert pda.shape == shape, f"{name}: Shape mismatch: {pda.shape} != {shape}"
                    assert len(pda) == local_size, f"{name}: Length mismatch: {len(pda)} != {local_size}"

    @pytest.mark.parametrize(
        "input_data",
        [
            pytest.param({range(0, 10)}, id="set-of-range"),
            pytest.param("not an iterable", id="string"),
            pytest.param(0, id="non-iterable-int"),
        ],
    )
    def test_array_creation_misc(self, input_data, subtests):
        """Ensure that ak.array() rejects unsupported inputs with a TypeError."""
        with subtests.test(input_data=input_data):
            with pytest.raises(TypeError):
                ak.array(input_data)

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize(
        "rows,cols",
        [
            (5, 5),
            (10, 3),
            (3, 10),
        ],
    )
    def test_array_creation_transpose_bug_reproducer(self, rows, cols, subtests):
        """
        Reproducer for transpose bug: ensure ak.transpose(ak.array(nda))
        matches ak.array(np.transpose(nda)) for various 2D shapes.
        """
        np.random.seed(pytest.seed)
        nda = np.random.randint(1, 10, (rows, cols))

        ak_arr = ak.array(nda)
        ak_t = ak.transpose(ak_arr)
        np_t = np.transpose(nda)
        ak_np_t = ak.array(np_t)

        with subtests.test(shape=(rows, cols)):
            assert_arkouda_array_equal(ak_t, ak_np_t, check_dtype=True)

    def test_infer_shape_multi_dim(self, subtests):
        for rank in multi_dim_ranks():
            with subtests.test(rank=rank):
                proposed_shape = (2,) * rank
                proposed_size = 2**rank

                shape, ndim, full_size = _infer_shape_from_size(proposed_shape)

                assert ndim == rank, f"ndim mismatch for rank={rank}: expected {rank}, got {ndim}"
                assert full_size == proposed_size, (
                    f"full_size mismatch for shape={proposed_shape}: "
                    f"expected {proposed_size}, got {full_size}"
                )
                assert shape == proposed_shape, (
                    f"shape mismatch for rank={rank}: expected {proposed_shape}, got {shape}"
                )

    def test_infer_shape_from_scalar(self):
        shape, ndim, full_size = _infer_shape_from_size(7)
        assert ndim == 1
        assert full_size == 7
        assert shape == 7

    def test_infer_shape_from_tuple(self):
        input_shape = (3, 5, 2)
        expected_size = 3 * 5 * 2
        shape, ndim, full_size = _infer_shape_from_size(input_shape)
        assert ndim == 3
        assert full_size == expected_size
        assert shape == input_shape

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_generate_shape_from_rank(self, size):
        for rank in multi_dim_ranks():
            local_shape, local_size = _generate_test_shape(rank, size)
            assert len(local_shape) == rank
            assert all(item > 0 for item in local_shape)
            assert local_size <= size

    def test_bigint_creation(self):
        bi = 2**200

        pda_from_str = ak.array([f"{i}" for i in range(bi, bi + 10)], dtype=ak.bigint)
        pda_from_int = ak.array([i for i in range(bi, bi + 10)])
        cast_from_segstr = ak.cast(ak.array([f"{i}" for i in range(bi, bi + 10)]), ak.bigint)
        for pda in [pda_from_str, pda_from_int, cast_from_segstr]:
            assert isinstance(pda, ak.pdarray)
            assert 10 == len(pda)
            assert ak.bigint == pda.dtype
            assert pda[-1] == bi + 10 - 1

        # test array and arange infer dtype
        assert ak.array([bi, bi + 1, bi + 2, bi + 3, bi + 4]).tolist() == ak.arange(bi, bi + 5).tolist()

        # test that max_bits being set results in a mod
        assert ak.arange(bi, bi + 5, max_bits=200).tolist() == ak.arange(5).tolist()

        # test ak.bigint_from_uint_arrays
        # top bits are all 1 which should be 2**64
        top_bits = ak.ones(5, ak.uint64)
        bot_bits = ak.arange(5, dtype=ak.uint64)
        two_arrays = ak.bigint_from_uint_arrays([top_bits, bot_bits])
        assert ak.bigint == two_arrays.dtype
        assert two_arrays.tolist() == [2**64 + i for i in range(5)]
        # top bits should represent 2**128
        mid_bits = ak.zeros(5, ak.uint64)
        three_arrays = ak.bigint_from_uint_arrays([top_bits, mid_bits, bot_bits])
        assert three_arrays.tolist() == [2**128 + i for i in range(5)]

        # test round_trip of ak.bigint_to/from_uint_arrays
        t = ak.arange(bi - 1, bi + 9)
        t_dup = ak.bigint_from_uint_arrays(t.bigint_to_uint_arrays())
        assert t.tolist() == t_dup.tolist()
        assert t_dup.max_bits == -1

        # test setting max_bits after creation still mods
        t_dup.max_bits = 200
        assert t_dup.tolist() == [bi - 1, 0, 1, 2, 3, 4, 5, 6, 7, 8]

        # test slice_bits along 64 bit boundaries matches return from bigint_to_uint_arrays
        for i, uint_bits in enumerate(t.bigint_to_uint_arrays()):
            slice_bits = t.slice_bits(64 * (4 - (i + 1)), 64 * (4 - i) - 1)
            assert uint_bits.tolist() == slice_bits.tolist()

    @pytest.mark.skip_if_max_rank_less_than(2)
    def test_bigint_creation_multi_dim(self):
        # Strings does not have a reshape method, so those tests are skipped

        bi = 2**200

        for rank in multi_dim_ranks():
            size = 2**rank
            shape, local_size = _generate_test_shape(rank, size)

            pda_from_str = ak.array([f"{i}" for i in range(bi, bi + size)], dtype=ak.bigint).reshape(
                shape
            )
            pda_from_int = ak.array([i for i in range(bi, bi + size)]).reshape(shape)
            for pda in [pda_from_str, pda_from_int]:
                assert isinstance(pda, ak.pdarray)
                assert size == len(pda)
                assert ak.bigint == pda.dtype
            np_arr = np.array([bi + i for i in range(size)]).reshape(shape)
            ak_arr = ak.array([bi + i for i in range(size)]).reshape(shape)
            assert_arkouda_array_equal(ak_arr, ak.array(np_arr))

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_newaxis(self, size):
        a = ak.arange(size)
        b = a[:, newaxis]
        c = a[newaxis, :]
        assert_arkouda_array_equal(a, b[:, 0])
        assert_arkouda_array_equal(a, c[0, :])

    def test_arange(self):
        assert np.arange(0, 10, 1).tolist() == ak.arange(0, 10, 1).tolist()
        assert np.arange(10, 0, -1).tolist() == ak.arange(10, 0, -1).tolist()
        assert np.arange(-5, -10, -1).tolist() == ak.arange(-5, -10, -1).tolist()
        assert np.arange(0, 10, 2).tolist() == ak.arange(0, 10, 2).tolist()

    @pytest.mark.parametrize("dtype", ak.intTypes)
    def test_arange_dtype(self, dtype):
        # test dtype works with optional start/step
        stop = ak.arange(100, dtype=dtype)
        assert np.arange(100, dtype=dtype).tolist() == stop.tolist()
        assert dtype == stop.dtype

        start_stop = ak.arange(100, 105, dtype=dtype)
        assert np.arange(100, 105, dtype=dtype).tolist() == start_stop.tolist()
        assert dtype == start_stop.dtype

        start_stop_step = ak.arange(100, 105, 2, dtype=dtype)
        assert np.arange(100, 105, 2, dtype=dtype).tolist() == start_stop_step.tolist()
        assert dtype == start_stop_step.dtype

        # also test for start/stop/step that cause empty ranges
        start_stop_step = ak.arange(100, 10, 2, dtype=dtype)
        assert np.arange(100, 10, 2, dtype=dtype).tolist() == start_stop_step.tolist()
        assert dtype == start_stop_step.dtype

        start_stop_step = ak.arange(10, 15, -2, dtype=dtype)
        assert np.arange(10, 15, -2, dtype=dtype).tolist() == start_stop_step.tolist()
        assert dtype == start_stop_step.dtype

        start_stop_step = ak.arange(10, 10, -2, dtype=dtype)
        assert np.arange(10, 10, 2, dtype=dtype).tolist() == start_stop_step.tolist()
        assert dtype == start_stop_step.dtype

    def test_arange_misc(self):
        # test uint64 handles negatives correctly
        np_arange_uint = np.arange(2**64 - 5, 2**64 - 10, -1, dtype=np.uint64)
        ak_arange_uint = ak.arange(-5, -10, -1, dtype=ak.uint64)
        # np_arange_uint = array([18446744073709551611, 18446744073709551610, 18446744073709551609,
        #        18446744073709551608, 18446744073709551607], dtype=uint64)
        assert np_arange_uint.tolist() == ak_arange_uint.tolist()
        assert ak.uint64 == ak_arange_uint.dtype

        uint_start_stop = ak.arange(2**63 + 3, 2**63 + 7)
        ans = ak.arange(3, 7, dtype=ak.uint64) + 2**63
        assert ans.tolist() == uint_start_stop.tolist()
        assert ak.uint64 == uint_start_stop.dtype

        # test correct conversion to float64
        np_arange_float = np.arange(-5, -10, -1, dtype=np.float64)
        ak_arange_float = ak.arange(-5, -10, -1, dtype=ak.float64)
        # array([-5., -6., -7., -8., -9.])
        assert np_arange_float.tolist() == ak_arange_float.tolist()
        assert ak.float64 == ak_arange_float.dtype

        # test correct conversion to bool
        expected_bool = [False, True, True, True, True]
        ak_arange_bool = ak.arange(0, 10, 2, dtype=ak.bool_)
        assert expected_bool == ak_arange_bool.tolist()
        assert ak.bool_ == ak_arange_bool.dtype

        # test int_scalars covers uint8, uint16, uint32
        uint_array = ak.arange(np.uint8(1), np.uint16(1000), np.uint32(1))
        int_array = ak.arange(1, 1000, 1)
        assert (uint_array == int_array).all()

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [ak.int64, ak.uint64, ak.float64])
    @pytest.mark.parametrize("start", [0, 2, 5])
    @pytest.mark.parametrize("step", [1, 3])
    def test_compare_arange(self, size, dtype, start, step):
        # create np version
        nArange = np.arange(start, size, step, dtype=dtype)
        # create ak version
        aArange = ak.arange(start, size, step, dtype=dtype)
        assert np.allclose(nArange, aArange.to_ndarray())

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("array_type", [ak.int64, ak.float64, bool])
    def test_randint_array_dtype(self, size, array_type):
        high = size if array_type != bool else 2
        test_array = ak.randint(0, high, size, array_type)
        assert isinstance(test_array, ak.pdarray)
        assert size == len(test_array)
        assert array_type == test_array.dtype
        assert (size,) == test_array.shape
        assert ((0 <= test_array) & (test_array <= size)).all()

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("array_type", [ak.int64, ak.float64, bool])
    def test_randint_array_dtype_multi_dim(self, size, array_type):
        for rank in multi_dim_ranks():
            shape, local_size = _generate_test_shape(rank, size)
            high = size if array_type != bool else 2
            test_array = ak.randint(0, high, shape, array_type)
            assert isinstance(test_array, ak.pdarray)
            assert local_size == len(test_array)
            assert array_type == test_array.dtype
            assert shape == test_array.shape
            assert ((0 <= test_array) & (test_array <= size)).all()

    # (The above function tests randint with various ARRAY dtypes; the function below
    #  tests with various dtypes for the other parameters passed to randint)

    @pytest.mark.parametrize("dtype", NUMERIC_SCALARS)
    def test_randint_num_dtype(self, dtype):
        for test_array in ak.randint(dtype(0), 100, 1000), ak.randint(0, dtype(100), 1000):
            assert isinstance(test_array, ak.pdarray)
            assert 1000 == len(test_array)
            assert ak.int64 == test_array.dtype
            assert (1000,) == test_array.shape
            assert ((0 <= test_array) & (test_array <= 100)).all()

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("dtype", NUMERIC_SCALARS)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_randint_num_dtype_multi_dim(self, size, dtype):
        for rank in multi_dim_ranks():
            shape, local_size = _generate_test_shape(rank, size)
            for test_array in ak.randint(dtype(0), 100, shape), ak.randint(0, dtype(100), shape):
                assert isinstance(test_array, ak.pdarray)
                assert local_size == len(test_array)
                assert ak.int64 == test_array.dtype
                assert shape == test_array.shape
                assert ((0 <= test_array) & (test_array <= 100)).all()

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_randint_misc(self, size):
        # Test that int_scalars covers uint8, uint16, uint32
        ak.randint(low=np.uint8(1), high=np.uint16(100), size=np.uint32(100))

        # test resolution of modulus overflow - issue #1174
        assert (ak.randint(-(2**63), 2**63 - 1, size) != ak.full(size, -(2**63))).any()

        with pytest.raises(TypeError):
            ak.randint(low=5)

        with pytest.raises(TypeError):
            ak.randint(high=5)

        with pytest.raises(TypeError):
            ak.randint()

        with pytest.raises(ValueError):
            ak.randint(low=0, high=1, size=-1, dtype=ak.float64)

        with pytest.raises(ValueError):
            ak.randint(low=1, high=0, size=1, dtype=ak.float64)

        with pytest.raises(TypeError):
            ak.randint(0, 1, "1000")

        with pytest.raises(TypeError):
            ak.randint("0", 1, 1000)

        with pytest.raises(TypeError):
            ak.randint(0, "1", 1000)

    #   The tests below retain the non pytest.seed because they assert specific values.

    def test_randint_with_seed(self):
        values = ak.randint(1, 5, 10, seed=2)

        assert [4, 3, 1, 3, 2, 4, 4, 2, 3, 4] == values.tolist()

        values = ak.randint(1, 5, 10, dtype=ak.float64, seed=2)

        ans = [
            2.9160772326374946,
            4.353429832157099,
            4.5392023718621486,
            4.4019932101126606,
            3.3745324569952304,
            1.1642002901528308,
            4.4714086874555292,
            3.7098921109084522,
            4.5939589352472314,
            4.0337935981006172,
        ]

        assert ans == values.tolist()

        bools = [False, True, True, True, True, False, True, True, True, True]
        values = ak.randint(0, 2, 10, dtype=ak.bool_, seed=2)
        assert values.tolist() == bools

        values = ak.randint(0, 2, 10, dtype=bool, seed=2)
        assert values.tolist() == bools

        # Test that int_scalars covers uint8, uint16, uint32
        uint_arr = ak.randint(np.uint8(1), np.uint32(5), np.uint16(10), seed=np.uint8(2))
        int_arr = ak.randint(1, 5, 10, seed=2)
        assert (uint_arr == int_arr).all()

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_uniform(self, size):
        test_array = ak.uniform(size)
        assert isinstance(test_array, ak.pdarray)
        assert ak.float64 == test_array.dtype
        assert (size,) == test_array.shape

        u_array = ak.uniform(size=3, low=0, high=5, seed=0)
        assert [
            0.30013431967121934,
            0.47383036230759112,
            1.0441791878997098,
        ] == u_array.tolist()

        u_array = ak.uniform(size=np.int64(3), low=np.int64(0), high=np.int64(5), seed=np.int64(0))
        assert [
            0.30013431967121934,
            0.47383036230759112,
            1.0441791878997098,
        ] == u_array.tolist()

        with pytest.raises(TypeError):
            ak.uniform(low="0", high=5, size=size)

        with pytest.raises(TypeError):
            ak.uniform(low=0, high="5", size=size)

        with pytest.raises(TypeError):
            ak.uniform(low=0, high=5, size="100")

        # Test that int_scalars covers uint8, uint16, uint32
        uint_arr = ak.uniform(low=np.uint8(0), high=np.uint16(5), size=np.uint32(100), seed=np.uint8(1))
        int_arr = ak.uniform(low=0, high=5, size=100, seed=1)
        assert (uint_arr == int_arr).all()

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [ak.int64, float, ak.float64, bool, ak.bool_, ak.bigint])
    def test_zeros_dtype(self, size, dtype):
        zeros = ak.zeros(size, dtype)
        assert isinstance(zeros, ak.pdarray)
        assert dtype == zeros.dtype
        assert (0 == zeros).all()

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [ak.int64, float, ak.float64, bool, ak.bool_, ak.bigint])
    def test_zeros_dtype_multi_dim(self, size, dtype):
        for rank in multi_dim_ranks():
            shape, local_size = _generate_test_shape(rank, size)
            zeros = ak.zeros(shape, dtype)
            assert isinstance(zeros, ak.pdarray)
            assert dtype == zeros.dtype
            assert zeros.shape == shape
            assert (0 == zeros).all()

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, float, ak.float64, bool, ak.bool_])
    def test_zeros_match_numpy(self, size, dtype):
        for rank in multi_dim_ranks():
            shape, local_size = _generate_test_shape(rank, size)
            assert_equivalent(ak.zeros(shape, dtype=dtype), np.zeros(shape, dtype=dtype))

    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, bool, ak.bool_])
    def test_zeros_error(self, dtype):
        rank = ak.client.get_max_array_rank() + 1
        shape, local_size = _generate_test_shape(rank, 2**rank)
        with pytest.raises(ValueError):
            ak.zeros(shape, dtype)

    def test_zeros_misc(self):
        zeros = ak.ones("5")
        assert 5 == len(zeros)

        with pytest.raises(TypeError):
            ak.zeros(5, dtype=ak.uint8)

        with pytest.raises(TypeError):
            ak.zeros(5, dtype=str)

        # Test that int_scalars covers uint8, uint16, uint32, str
        int_arr = ak.zeros(5)
        for arg in np.uint8(5), np.uint16(5), np.uint32(5), str(5):
            assert (int_arr == ak.zeros(arg, dtype=ak.int64)).all()

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, bool, ak.bool_])
    def test_zeros_like(self, size, dtype):
        ran_arr = ak.array(ak.arange(size, dtype=dtype))
        zeros_like_arr = ak.zeros_like(ran_arr)
        assert isinstance(zeros_like_arr, ak.pdarray)
        assert dtype == zeros_like_arr.dtype
        assert (zeros_like_arr == 0).all()
        assert zeros_like_arr.size == ran_arr.size

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, bool, ak.bool_])
    def test_zeros_like_multi_dim(self, size, dtype):
        for rank in multi_dim_ranks():
            shape, local_size = _generate_test_shape(rank, size)
            ran_arr = ak.array(ak.arange(local_size, dtype=dtype)).reshape(shape)
            zeros_like_arr = ak.zeros_like(ran_arr)
            assert isinstance(zeros_like_arr, ak.pdarray)
            assert dtype == zeros_like_arr.dtype
            assert (zeros_like_arr == 0).all()
            assert zeros_like_arr.size == ran_arr.size

    @pytest.mark.parametrize("dtype", [int, ak.int64, float, ak.float64, bool, ak.bool_, ak.bigint])
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_ones_dtype(self, size, dtype):
        ones = ak.ones(size, dtype)
        assert isinstance(ones, ak.pdarray)
        assert dtype == ones.dtype
        assert (1 == ones).all()

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [ak.int64, float, ak.float64, bool, ak.bool_, ak.bigint])
    def test_ones_dtype_multi_dim(self, size, dtype):
        for rank in multi_dim_ranks():
            shape, local_size = _generate_test_shape(rank, size)
            ones = ak.ones(shape, dtype)
            assert isinstance(ones, ak.pdarray)
            assert dtype == ones.dtype
            assert ones.shape == shape
            assert (1 == ones).all()

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, float, ak.float64, bool, ak.bool_])
    def test_ones_match_numpy(self, size, dtype):
        for rank in multi_dim_ranks():
            shape, local_size = _generate_test_shape(rank, size)
            assert_equivalent(ak.ones(shape, dtype=dtype), np.ones(shape, dtype=dtype))

    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, bool, ak.bool_])
    def test_ones_error(self, dtype):
        rank = ak.client.get_max_array_rank() + 1
        shape, local_size = _generate_test_shape(rank, 2**rank)
        with pytest.raises(ValueError):
            ak.ones(shape, dtype)

    def test_ones_misc(self):
        ones = ak.ones("5")
        assert 5 == len(ones)

        with pytest.raises(TypeError):
            ak.ones(5, dtype=ak.uint8)

        # Test that int_scalars covers uint8, uint16, uint32
        int_arr = ak.ones(5)
        for arg in np.uint8(5), np.uint16(5), np.uint32(5):
            assert (int_arr == ak.ones(arg, dtype=ak.int64)).all()

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [ak.int64, ak.float64, ak.bool_, ak.bigint])
    def test_ones_like(self, size, dtype):
        ran_arr = ak.array(ak.arange(size, dtype=dtype))
        ones_like_arr = ak.ones_like(ran_arr)
        assert isinstance(ones_like_arr, ak.pdarray)
        assert dtype == ones_like_arr.dtype
        assert (1 == ones_like_arr).all()
        assert ones_like_arr.size == ran_arr.size

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, bool, ak.bool_])
    def test_ones_like_multi_dim(self, size, dtype):
        for rank in multi_dim_ranks():
            shape, local_size = _generate_test_shape(rank, size)
            ran_arr = ak.array(ak.arange(local_size, dtype=dtype)).reshape(shape)
            ones_like_arr = ak.ones_like(ran_arr)
            assert isinstance(ones_like_arr, ak.pdarray)
            assert dtype == ones_like_arr.dtype
            assert (ones_like_arr == 1).all()
            assert ones_like_arr.size == ran_arr.size

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, bool, ak.bool_])
    def test_full_dtype(self, size, dtype):
        type_full = ak.full(size, 1, dtype)
        assert isinstance(type_full, ak.pdarray)
        assert dtype == type_full.dtype
        assert (1 == type_full).all()

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [ak.int64, float, ak.float64, bool, ak.bool_, ak.bigint])
    def test_full_dtype_multi_dim(self, size, dtype):
        for rank in multi_dim_ranks():
            shape, local_size = _generate_test_shape(rank, size)
            type_full = ak.full(shape, 1, dtype)
            assert isinstance(type_full, ak.pdarray)
            assert dtype == type_full.dtype
            assert type_full.shape == shape
            assert (1 == type_full).all()

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, float, ak.float64, bool, ak.bool_])
    def test_full_match_numpy(self, size, dtype):
        for rank in ak.client.get_array_ranks():
            if rank == 1:
                continue
            shape, local_size = _generate_test_shape(rank, size)
            assert_equivalent(
                ak.full(shape, fill_value=2, dtype=dtype),
                np.full(shape, fill_value=2, dtype=dtype),
            )

    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, bool, ak.bool_])
    def test_full_error(self, dtype):
        rank = ak.client.get_max_array_rank() + 1
        shape, local_size = _generate_test_shape(rank, 2**rank)
        with pytest.raises(ValueError):
            ak.full(shape, 1, dtype)

    def test_full_misc(self):
        for arg in -1, False:
            bool_full = ak.full(5, arg, dtype=bool)
            assert bool == bool_full.dtype
            assert bool_full.all() if arg else not bool_full.any()

        string_len_full = ak.full("5", 5)
        assert 5 == len(string_len_full)

        strings_full = ak.full(5, "test")
        assert isinstance(strings_full, ak.Strings)
        assert 5 == len(strings_full)
        assert strings_full.tolist() == ["test"] * 5

        with pytest.raises(TypeError):
            ak.full(5, 1, dtype=ak.uint8)

        # Test that int_scalars covers uint8, uint16, uint32
        int_arr = ak.full(5, 5)
        for args in [
            (np.uint8(5), np.uint16(5)),
            (np.uint16(5), np.uint32(5)),
            (np.uint32(5), np.uint8(5)),
        ]:
            assert (int_arr == ak.full(*args, dtype=int)).all()

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, bool, ak.bool_])
    def test_full_like(self, size, dtype):
        ran_arr = ak.full(size, 5, dtype)
        full_like_arr = ak.full_like(ran_arr, 1)
        assert isinstance(full_like_arr, ak.pdarray)
        assert dtype == full_like_arr.dtype
        assert (full_like_arr == 1).all()
        assert full_like_arr.size == ran_arr.size

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, bool, ak.bool_])
    def test_full_like_multi_dim(self, size, dtype):
        for rank in multi_dim_ranks():
            shape, local_size = _generate_test_shape(rank, size)
            ran_arr = ak.full(shape, 5, dtype)
            full_like_arr = ak.full_like(ran_arr, 1)
            assert isinstance(full_like_arr, ak.pdarray)
            assert dtype == full_like_arr.dtype
            assert (full_like_arr == 1).all()
            assert full_like_arr.size == ran_arr.size

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_linspace_1D(self, size):
        pda = ak.linspace(0, 100, size)
        nda = np.linspace(0, 100, size)
        assert size == len(pda)
        assert float == pda.dtype
        assert isinstance(pda, ak.pdarray)
        assert_almost_equivalent(pda, nda)

        pda = ak.linspace(start=5, stop=0, num=6)
        nda = np.linspace(start=5, stop=0, num=6)
        assert 5.0000 == pda[0]
        assert 0.0000 == pda[5]
        assert_almost_equivalent(pda, nda)

        pda = ak.linspace(start=5, stop=0, num=6, endpoint=False)
        nda = np.linspace(5, 0, 6, endpoint=False)
        assert 5.0000 == pda[0]
        assert 0.0000 != pda[5]
        assert_almost_equivalent(pda, nda)

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_linspace_2D(self, size):
        pedge = ak.array([4, 5])
        pda = ak.linspace(0, pedge, size)
        nda = np.linspace(0, pedge.to_ndarray(), size)
        assert 2 * size == pda.size
        assert float == pda.dtype
        assert isinstance(pda, ak.pdarray)
        assert_almost_equivalent(pda, nda)

        pda = ak.linspace(pedge, 10, size)
        nda = np.linspace(pedge.to_ndarray(), 10, size)
        assert 2 * size == pda.size
        assert float == pda.dtype
        assert isinstance(pda, ak.pdarray)
        assert_almost_equivalent(pda, nda)

    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_linspace_3D(self, size):
        # without having to broadcast shapes

        p_lo = ak.array([4, 5])
        p_hi = ak.array([7, 20])
        pda = ak.linspace(p_lo, p_hi, size)
        nda = np.linspace(p_lo.to_ndarray(), p_hi.to_ndarray(), size)
        assert 2 * size == pda.size
        assert float == pda.dtype
        assert isinstance(pda, ak.pdarray)
        assert_almost_equivalent(pda, nda)

        # with broadcasting start and stop to common shape

        p_hi = ak.array([[7, 8], [9, 10]])
        pda = ak.linspace(p_lo, p_hi, size)
        nda = np.linspace(p_lo.to_ndarray(), p_hi.to_ndarray(), size)
        assert 4 * size == pda.size
        assert float == pda.dtype
        assert isinstance(pda, ak.pdarray)
        assert_almost_equivalent(pda, nda)

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_logspace_1D(self, size):
        pda = ak.logspace(0, 10, size, endpoint=True, base=2.0)
        nda = np.logspace(0, 10, size, endpoint=True, base=2.0)
        assert size == len(pda)
        assert float == pda.dtype
        assert isinstance(pda, ak.pdarray)
        assert_almost_equivalent(pda, nda)

        pda = ak.logspace(start=5, stop=0, num=6, endpoint=True, base=2)
        nda = np.logspace(start=5, stop=0, num=6, endpoint=True, base=2)
        assert math.isclose(32.0000, pda[0])
        assert math.isclose(1.0000, pda[5])
        assert_almost_equivalent(pda, nda)

        pda = ak.logspace(start=5, stop=0, num=6, endpoint=False, base=2)
        nda = np.logspace(start=5, stop=0, num=6, endpoint=False, base=2)
        assert math.isclose(32.0000, pda[0])
        assert not math.isclose(1.0000, pda[5])
        assert_almost_equivalent(pda, nda)

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_logspace_2D(self, size):
        pedge = ak.array([4, 5])
        pda = ak.logspace(0, pedge, size, endpoint=True, base=3.0)
        nda = np.logspace(0, pedge.to_ndarray(), size, endpoint=True, base=3.0)
        assert 2 * size == pda.size
        assert float == pda.dtype
        assert isinstance(pda, ak.pdarray)
        assert_almost_equivalent(pda, nda)

        pda = ak.logspace(pedge, 10, size, endpoint=True, base=3.0)
        nda = np.logspace(pedge.to_ndarray(), 10, size, endpoint=True, base=3.0)
        assert 2 * size == pda.size
        assert float == pda.dtype
        assert isinstance(pda, ak.pdarray)
        assert_almost_equivalent(pda, nda)

    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_logspace_3D(self, size):
        # without having to broadcast shapes

        p_lo = ak.array([4, 5])
        p_hi = ak.array([7, 20])
        pda = ak.logspace(p_lo, p_hi, size, endpoint=True, base=1.7)
        nda = np.logspace(p_lo.to_ndarray(), p_hi.to_ndarray(), size, endpoint=True, base=1.7)
        assert 2 * size == pda.size
        assert float == pda.dtype
        assert isinstance(pda, ak.pdarray)
        assert_almost_equivalent(pda, nda)

        # with broadcasting start and stop to common shape

        p_hi = ak.array([[7, 8], [9, 10]])
        pda = ak.logspace(p_lo, p_hi, size, endpoint=True, base=2.1)
        nda = np.logspace(p_lo.to_ndarray(), p_hi.to_ndarray(), size, endpoint=True, base=2.1)
        assert 4 * size == pda.size
        assert float == pda.dtype
        assert isinstance(pda, ak.pdarray)
        assert_almost_equivalent(pda, nda)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", INT_SCALARS)
    def test_standard_normal(self, size, dtype):
        pda = ak.standard_normal(100)
        assert isinstance(pda, ak.pdarray)
        assert 100 == len(pda)
        assert float == pda.dtype

        pda = ak.standard_normal(dtype(100))
        assert isinstance(pda, ak.pdarray)
        assert 100 == len(pda)
        assert float == pda.dtype

        pda = ak.standard_normal(dtype(100), dtype(1))
        assert isinstance(pda, ak.pdarray)
        assert 100 == len(pda)
        assert float == pda.dtype

        npda = pda.to_ndarray()
        pda = ak.standard_normal(dtype(100), dtype(1))
        assert npda.tolist() == pda.tolist()

    def test_standard_normal_errors(self):
        with pytest.raises(TypeError):
            ak.standard_normal("100")

        with pytest.raises(TypeError):
            ak.standard_normal(100.0)

        with pytest.raises(ValueError):
            ak.standard_normal(-1)

        # Test that int_scalars covers uint8, uint16, uint32
        int_arr = ak.standard_normal(5, seed=1)
        for args in [
            (np.uint8(5), np.uint16(1)),
            (np.uint16(5), np.uint32(1)),
            (np.uint32(5), np.uint8(1)),
        ]:
            assert (int_arr == ak.standard_normal(*args)).all()

    @pytest.mark.parametrize("dtype", INT_SCALARS)
    def test_random_strings_uniform(self, dtype):
        pda = ak.random_strings_uniform(minlen=dtype(1), maxlen=dtype(5), size=dtype(100))
        assert isinstance(pda, ak.Strings)
        assert 100 == len(pda)
        assert str == pda.dtype

        assert ((1 <= pda.get_lengths()) & (pda.get_lengths() <= 5)).all()
        assert (pda.isupper()).all()

    def test_random_strings_uniform_errors(self):
        with pytest.raises(ValueError):
            ak.random_strings_uniform(maxlen=1, minlen=5, size=10)

        with pytest.raises(ValueError):
            ak.random_strings_uniform(maxlen=5, minlen=1, size=-1)

        with pytest.raises(ValueError):
            ak.random_strings_uniform(maxlen=5, minlen=5, size=10)

        with pytest.raises(TypeError):
            ak.random_strings_uniform(minlen="1", maxlen=5, size=10)

        with pytest.raises(TypeError):
            ak.random_strings_uniform(minlen=1, maxlen="5", size=10)

        with pytest.raises(TypeError):
            ak.random_strings_uniform(minlen=1, maxlen=5, size="10")

        # Test that int_scalars covers uint8, uint16, uint32
        np_arr = ak.random_strings_uniform(
            minlen=np.uint8(1),
            maxlen=np.uint32(5),
            seed=np.uint16(1),
            size=np.uint8(10),
            characters="printable",
        )

        int_arr = ak.random_strings_uniform(
            minlen=(1),
            maxlen=(5),
            seed=(1),
            size=(10),
            characters="printable",
        )

        assert (np_arr == int_arr).all()

    def test_random_strings_uniform_with_seed(self):
        pda = ak.random_strings_uniform(minlen=1, maxlen=5, seed=1, size=10)
        assert [
            "VW",
            "JEXI",
            "EBBX",
            "HG",
            "S",
            "WOVK",
            "U",
            "WL",
            "JCSD",
            "DSN",
        ] == pda.tolist()

        pda = ak.random_strings_uniform(minlen=1, maxlen=5, seed=1, size=10, characters="printable")
        assert [
            "eL",
            "6<OD",
            "o-GO",
            " l",
            "m",
            "PV y",
            "f",
            "}.",
            "b3Yc",
            "Kw,",
        ] == pda.tolist()

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("num_dtype", NUMERIC_SCALARS)
    def test_random_strings_lognormal(self, size, num_dtype):
        pda = ak.random_strings_lognormal(
            logmean=num_dtype(2),
            logstd=num_dtype(1),
            size=np.int64(size),
            characters="printable",
        )
        assert isinstance(pda, ak.Strings)
        assert size == len(pda)
        assert str == pda.dtype

    def test_random_strings_lognormal_errors(self):
        with pytest.raises(TypeError):
            ak.random_strings_lognormal("2", 0.25, 100)

        with pytest.raises(TypeError):
            ak.random_strings_lognormal(2, 0.25, "100")

        with pytest.raises(TypeError):
            ak.random_strings_lognormal(2, 0.25, 100, 1000000)

        # Test that int_scalars covers uint8, uint16, uint32
        ak.random_strings_lognormal(np.uint8(2), 0.25, np.uint16(100))

    def test_random_strings_lognormal_with_seed(self):
        randoms = [
            "VWHJEX",
            "BEBBXJHGM",
            "RWOVKBUR",
            "LNJCSDXD",
            "NKEDQC",
            "GIBAFPAVWF",
            "IJIFHGDHKA",
            "VUDYRA",
            "QHQETTEZ",
            "DJBPWJV",
        ]

        pda = ak.random_strings_lognormal(2, 0.25, 10, seed=1)
        assert randoms == pda.tolist()

        pda = ak.random_strings_lognormal(float(2), np.float64(0.25), np.int64(10), seed=1)
        assert randoms == pda.tolist()

        printable_randoms = [
            "eL96<O",
            ")o-GOe lR",
            ")PV yHf(",
            "._b3Yc&K",
            ",7Wjef",
            "R{lQs_g]5T",
            "E[2dk\\2a9J",
            "I*VknZ",
            "0!u~e$Lm",
            "9Q{TtHq",
        ]

        pda = ak.random_strings_lognormal(2, 0.25, 10, seed=1, characters="printable")
        assert printable_randoms == pda.tolist()

        pda = ak.random_strings_lognormal(
            np.int64(2), np.float64(0.25), np.int64(10), seed=1, characters="printable"
        )
        assert printable_randoms == pda.tolist()

    @pytest.mark.parametrize("dtype", NUMERIC_SCALARS)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_fill(self, size, dtype):
        ones = ak.ones(size)
        ones.fill(dtype(2))
        assert (dtype(2) == ones).all()

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("dtype", NUMERIC_SCALARS)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_fill_multi_dim(self, size, dtype):
        for rank in multi_dim_ranks():
            shape, local_size = _generate_test_shape(rank, size)
            ones = ak.ones(shape)
            ones.fill(dtype(2))
            assert (dtype(2) == ones).all()

    def test_endian(self):
        a = np.random.randint(1, 100, 100)
        aka = ak.array(a)
        npa = aka.to_ndarray()
        assert np.allclose(a, npa)

        a = a.view(a.dtype.newbyteorder("<"))
        aka = ak.array(a)
        npa = aka.to_ndarray()
        assert np.allclose(a, npa)

        a = a.view(a.dtype.newbyteorder(">"))
        aka = ak.array(a)
        npa = aka.to_ndarray()
        assert np.allclose(a, npa)

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_endian_multi_dim(self, size):
        for rank in multi_dim_ranks():
            shape, local_size = _generate_test_shape(rank, size)
            a = np.random.randint(1, 100, shape)
            aka = ak.array(a)
            npa = aka.to_ndarray()
            assert np.allclose(a, npa)

            a = a.view(a.dtype.newbyteorder("S"))
            aka = ak.array(a)
            npa = aka.to_ndarray()
            assert np.allclose(a, npa)

            a = a.view(a.dtype.newbyteorder("S"))
            aka = ak.array(a)
            npa = aka.to_ndarray()
            assert np.allclose(a, npa)

    def test_clobber(self):
        n_arrs = 10

        arrs = [np.random.randint(1, 100, 100) for _ in range(n_arrs)]
        ak_arrs = [ak.array(arr) for arr in arrs]
        np_arrs = [arr.to_ndarray() for arr in ak_arrs]
        for a, npa in zip(arrs, np_arrs):
            assert np.allclose(a, npa)

        arrs = [np.full(100, i) for i in range(n_arrs)]
        ak_arrs = [ak.array(arr) for arr in arrs]
        np_arrs = [arr.to_ndarray() for arr in ak_arrs]

        for a, npa, i in zip(arrs, np_arrs, range(n_arrs)):
            assert np.all(a == i)
            assert np.all(npa == i)

            a += 1
            assert np.all(a == i + 1)
            assert np.all(npa == i)

            npa += 1
            assert np.all(a == i + 1)
            assert np.all(npa == i + 1)

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_clobber_multi_dim(self, size):
        for rank in multi_dim_ranks():
            shape, local_size = _generate_test_shape(rank, size)
            n_arrs = 10

            arrs = [np.random.randint(1, 100, shape) for _ in range(n_arrs)]
            ak_arrs = [ak.array(arr) for arr in arrs]
            np_arrs = [arr.to_ndarray() for arr in ak_arrs]
            for a, npa in zip(arrs, np_arrs):
                assert np.allclose(a, npa)

            arrs = [np.full(shape, i) for i in range(n_arrs)]
            ak_arrs = [ak.array(arr) for arr in arrs]
            np_arrs = [arr.to_ndarray() for arr in ak_arrs]

            for a, npa, i in zip(arrs, np_arrs, range(n_arrs)):
                assert np.all(a == i)
                assert np.all(npa == i)

                a += 1
                assert np.all(a == i + 1)
                assert np.all(npa == i)

                npa += 1
                assert np.all(a == i + 1)
                assert np.all(npa == i + 1)

    def test_uint_greediness(self):
        # default to uint when all supportedInt and any value > 2**63
        # to avoid loss of precision see (#1297)
        for greedy_list in (
            [2**63, 6, 2**63 - 1, 2**63 + 1],
            [2**64 - 1, 0, 2**64 - 1],
        ):
            greedy_pda = ak.array(greedy_list)
            assert greedy_pda.dtype == ak.uint64
            assert greedy_list == greedy_pda.tolist()

    @pytest.mark.parametrize("size", pytest.prob_size)
    def randint_randomness(self, size):
        # THIS TEST DOES NOT RUN, see Issue #1672
        # To run rename to `test_randint_randomness`
        min_val = 0
        max_val = 2**32
        passed = 0
        trials = 20

        for x in range(trials):
            l_int = ak.randint(min_val, max_val, size)
            l_median = statistics.median(l_int.to_ndarray())

            runs, n1, n2 = 0, 0, 0

            # Checking for start of new run
            for i in range(len(l_int)):
                # no. of runs
                if (l_int[i] >= l_median > l_int[i - 1]) or (l_int[i] < l_median <= l_int[i - 1]):
                    runs += 1

                # no. of positive values
                if (l_int[i]) >= l_median:
                    n1 += 1
                # no. of negative values
                else:
                    n2 += 1

            runs_exp = ((2 * n1 * n2) / (n1 + n2)) + 1
            stan_dev = math.sqrt(
                (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
            )

            if abs((runs - runs_exp) / stan_dev) < 1.9:
                passed += 1

        assert passed >= trials * 0.8

    def test_inferred_type(self):
        a = ak.array([1, 2, 3])
        assert a.inferred_type == "integer"

        a2 = ak.array([1.0, 2, 3])
        assert a2.inferred_type == "floating"

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_inferred_type_multi_dim(self, size):
        for rank in multi_dim_ranks():
            shape, local_size = _generate_test_shape(rank, size)
            a = ak.arange(local_size).reshape(shape)
            assert a.inferred_type == "integer"

            a2 = ak.full(shape, 2.0)
            assert a2.inferred_type == "floating"

    def test_to_ndarray(self):
        ones = ak.ones(10)
        n_ones = ones.to_ndarray()
        new_ones = ak.array(n_ones)
        assert ones.tolist() == new_ones.tolist()

        empty_ones = ak.ones(0)
        n_empty_ones = empty_ones.to_ndarray()
        new_empty_ones = ak.array(n_empty_ones)
        assert empty_ones.tolist() == new_empty_ones.tolist()

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_to_ndarray_multi_dim(self, size):
        for rank in multi_dim_ranks():
            shape, local_size = _generate_test_shape(rank, size)
            ones = ak.ones(shape)
            n_ones = ones.to_ndarray()
            new_ones = ak.array(n_ones)
            assert ones.tolist() == new_ones.tolist()

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_to_ndarray_multi_dim_bigint(self, size):
        for rank in multi_dim_ranks():
            shape, local_size = _generate_test_shape(rank, size)
            ones = ak.ones(shape, ak.bigint)
            n_ones = ones.to_ndarray()
            assert_arkouda_array_equal(ones, ak.array(n_ones))

    def test_np_bigint_zeros_conversion(self):
        a = ak.array(np.array([2**200] * 100) * 0)
        assert a.size == 100
        assert ak.all(a == 0)
        assert ak.all(a == ak.array([0] * 100, dtype=ak.bigint))

    @pytest.mark.skip_if_max_rank_less_than(2)
    def test_np_bigint_zeros_conversion_multidim(self):
        a = ak.array((np.array([2**200] * 100) * 0).reshape(5, 20))
        assert a.size == 100
        assert ak.all(a == 0)
        assert ak.all(a == ak.array([0] * 100, dtype=ak.bigint).reshape(5, 20))

    def test_bigint_large_negative_values(self):
        a = ak.array([-(2**200)])
        b = np.array([2**200])
        val = 2**200
        c = [f"{-val}"]
        ak_c = ak.array(c, dtype=ak.bigint)
        assert_equivalent(a, -b)
        assert_equivalent(a, ak_c)

    def test_range_conversion(self):
        a = ak.array(range(0, 10))
        b = ak.array(list(range(10)))
        assert_equivalent(a, b)

    def test_should_be_uint(self):
        a = ak.array([2**63])
        assert a.dtype == ak.uint64

    def test_should_not_be_uint(self):
        a = ak.array([-1, 2**63])
        b = np.array([-1, 2**63])
        assert_equivalent(a, b)
        assert a.dtype == ak.float64
