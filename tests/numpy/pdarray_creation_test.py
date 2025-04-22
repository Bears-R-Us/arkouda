import datetime as dt
import math
import statistics
from collections import deque

import numpy as np
import pandas as pd
import pytest

import arkouda as ak
from arkouda.numpy import newaxis, pdarraycreation
from arkouda.numpy.util import _generate_test_shape, _infer_shape_from_size
from arkouda.testing import assert_arkouda_array_equal, assert_equivalent

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
        Test large-array creation using various Arkouda constructors. Ensures correct length and type.
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

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_large_array_creation_multi_dim(self, size, subtests):
        """
        Test multi-dimensional array creation for supported Arkouda constructors.
        Excludes 1D-only functions like linspace, arange, and random strings.
        """
        for rank in multi_dim_ranks():
            shape, local_size = _generate_test_shape(rank, size)
            test_cases = [
                ("ak.ones", lambda: ak.ones(shape, int)),
                ("ak.array(ak.ones)", lambda: ak.array(ak.ones(shape, int))),
                ("ak.zeros", lambda: ak.zeros(shape)),
                ("ak.ones (default)", lambda: ak.ones(shape)),
                ("ak.full", lambda: ak.full(shape, 9)),  # only numeric for multi-dim
                ("ak.zeros_like", lambda: ak.zeros_like(ak.ones(shape))),
                ("ak.ones_like", lambda: ak.ones_like(ak.zeros(shape))),
                ("ak.full_like", lambda: ak.full_like(ak.zeros(shape), 9)),
                ("ak.randint", lambda: ak.randint(0, local_size, shape)),
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
        """
        Ensure that ak.array() rejects unsupported inputs with a TypeError.
        """
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
        np.random.seed(0)
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
    def test_generate_shape_from_rank(self, size, subtests):
        """
        Verify that `_generate_test_shape(rank, size)` returns:
          1. A tuple of length `rank`.
          2. Every dimension > 0.
          3. A total element count ≤ `size`.
        """
        for rank in multi_dim_ranks():
            with subtests.test(rank=rank):
                shape, total = _generate_test_shape(rank, size)

                # 1. Correct number of dimensions
                assert len(shape) == rank, f"Rank {rank}: Expected shape length {rank}, got {len(shape)}"

                # 2. All dimensions positive
                assert all(dim > 0 for dim in shape), f"Rank {rank}: Expected all dims > 0, got {shape}"

                # 3. Total size within limit
                assert total <= size, f"Rank {rank}: Total elements {total} exceed size {size}"

    bi = 2**200

    @pytest.mark.parametrize(
        "factory,name",
        [
            (
                lambda bi: ak.array([f"{i}" for i in range(bi, bi + 10)], dtype=ak.bigint),
                "from_str",
            ),
            (lambda bi: ak.array([i for i in range(bi, bi + 10)]), "from_int"),
            (
                lambda bi: ak.cast(ak.array([f"{i}" for i in range(bi, bi + 10)]), ak.bigint),
                "cast",
            ),
        ],
    )
    def test_bigint_array_construction(self, factory, name, subtests):
        """Construction from strings, ints, and cast must yield a bigint pdarray of length 10."""
        with subtests.test(case=name):
            pda = factory(self.bi)
            assert isinstance(pda, ak.pdarray)
            assert pda.dtype == ak.bigint
            assert len(pda) == 10
            assert pda[-1] == self.bi + 9

    def test_arange_infers_bigint(self):
        """ak.arange(start, …) should infer bigint if start is ≥ 2**64."""
        expected = [self.bi + i for i in range(5)]
        assert ak.array(expected).to_list() == ak.arange(self.bi, self.bi + 5).to_list()

    def test_arange_max_bits_wraps(self):
        """Specifying max_bits should wrap values modulo 2**max_bits."""
        wrapped = ak.arange(self.bi, self.bi + 5, max_bits=200).to_list()
        assert wrapped == ak.arange(5).to_list()

    def test_bigint_from_uint_arrays(self):
        """Reassemble bigints from 2‑ and 3‑segment uint64 arrays correctly."""
        top = ak.ones(5, ak.uint64)
        bot = ak.arange(5, dtype=ak.uint64)

        two = ak.bigint_from_uint_arrays([top, bot])
        assert two.dtype == ak.bigint
        assert two.to_list() == [2**64 + i for i in range(5)]

        mid = ak.zeros(5, ak.uint64)
        three = ak.bigint_from_uint_arrays([top, mid, bot])
        assert three.to_list() == [2**128 + i for i in range(5)]

    def test_bigint_roundtrip_and_max_bits_property(self, subtests):
        """bigint_to_uint_arrays → bigint_from_uint_arrays roundtrips, and max_bits setter wraps."""
        t = ak.arange(self.bi - 1, self.bi + 9)
        uint_segs = t.bigint_to_uint_arrays()

        # Round‑trip
        t_dup = ak.bigint_from_uint_arrays(uint_segs)
        assert t.to_list() == t_dup.to_list()
        assert t_dup.max_bits == -1  # no wrapping by default

        # Setting max_bits should wrap
        t_dup.max_bits = 200
        expected = [self.bi - 1] + list(range(9))
        assert t_dup.to_list() == expected

    def test_slice_bits_matches_uint_segments(self):
        """slice_bits should reproduce each 64‑bit segment returned by bigint_to_uint_arrays."""
        t = ak.arange(self.bi - 1, self.bi + 9)
        for i, seg in enumerate(t.bigint_to_uint_arrays()):
            low = 64 * (4 - (i + 1))
            high = 64 * (4 - i) - 1
            slice_seg = t.slice_bits(low, high)
            assert seg.to_list() == slice_seg.to_list()

    @pytest.mark.skip_if_max_rank_less_than(2)
    def test_bigint_creation_multi_dim(self, subtests):
        """
        Test multi-dimensional bigint array creation from strings and ints,
        and verify equivalence to NumPy-based creation.
        """
        bi = 2**200

        for rank in multi_dim_ranks():
            size = 2**rank
            shape, _ = _generate_test_shape(rank, size)

            # Construction from strings and ints
            cases = [
                (
                    "from_str",
                    lambda: ak.array([f"{i}" for i in range(bi, bi + size)], dtype=ak.bigint),
                ),
                ("from_int", lambda: ak.array([i for i in range(bi, bi + size)])),
            ]
            for name, constructor in cases:
                with subtests.test(rank=rank, source=name):
                    pda = constructor().reshape(shape)
                    assert isinstance(pda, ak.pdarray), f"{name}: Expected ak.pdarray, got {type(pda)}"
                    assert len(pda) == size, f"{name}: Length mismatch {len(pda)} != {size}"
                    assert pda.dtype == ak.bigint, f"{name}: Dtype mismatch {pda.dtype} != bigint"

            # Equivalence to NumPy-backed creation
        with subtests.test(rank=rank, source="numpy_equiv"):
            np_arr = np.array([bi + i for i in range(size)], dtype=object).reshape(shape)
            ak_arr = ak.array([bi + i for i in range(size)]).reshape(shape)
            assert_arkouda_array_equal(
                ak_arr,
                ak.array(np_arr),
                err_msg=f"numpy_equiv: Mismatch at rank={rank}",
            )

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_newaxis(self, size):
        a = ak.arange(size)
        b = a[:, newaxis]
        c = a[newaxis, :]
        assert_arkouda_array_equal(a, b[:, 0])
        assert_arkouda_array_equal(a, c[0, :])

    @pytest.mark.parametrize(
        "start, stop, step",
        [
            pytest.param(0, 10, 1, id="ascending"),
            pytest.param(10, 0, -1, id="descending"),
            pytest.param(-5, -10, -1, id="neg_to_neg"),
            pytest.param(0, 10, 2, id="step2"),
        ],
    )
    def test_arange_matches_numpy(self, start, stop, step):
        """ak.arange should exactly match numpy.arange for various (start,stop,step)."""
        expected = np.arange(start, stop, step).tolist()
        result = ak.arange(start, stop, step).to_list()
        assert result == expected, f"ak.arange({start},{stop},{step}) → {result} != {expected}"
        assert len(result) == len(expected)

    def test_arange_default_step(self):
        """When only stop is given, ak.arange(stop) should equal list(range(stop))."""
        assert ak.arange(5).to_list() == list(range(5))

    @pytest.mark.parametrize("step", [0, 0.0])
    def test_arange_zero_step_raises(self, step):
        """ak.arange must raise ValueError if step is zero."""
        with pytest.raises(ZeroDivisionError):
            ak.arange(0, 10, step)

    @pytest.mark.parametrize("dtype", ak.intTypes)
    def test_arange_dtype(self, dtype):
        # test dtype works with optional start/step
        stop = ak.arange(100, dtype=dtype)
        assert np.arange(100, dtype=dtype).tolist() == stop.to_list()
        assert dtype == stop.dtype

        start_stop = ak.arange(100, 105, dtype=dtype)
        assert np.arange(100, 105, dtype=dtype).tolist() == start_stop.to_list()
        assert dtype == start_stop.dtype

        start_stop_step = ak.arange(100, 105, 2, dtype=dtype)
        assert np.arange(100, 105, 2, dtype=dtype).tolist() == start_stop_step.to_list()
        assert dtype == start_stop_step.dtype

        # also test for start/stop/step that cause empty ranges
        start_stop_step = ak.arange(100, 10, 2, dtype=dtype)
        assert np.arange(100, 10, 2, dtype=dtype).tolist() == start_stop_step.to_list()
        assert dtype == start_stop_step.dtype

        start_stop_step = ak.arange(10, 15, -2, dtype=dtype)
        assert np.arange(10, 15, -2, dtype=dtype).tolist() == start_stop_step.to_list()
        assert dtype == start_stop_step.dtype

        start_stop_step = ak.arange(10, 10, -2, dtype=dtype)
        assert np.arange(10, 10, 2, dtype=dtype).tolist() == start_stop_step.to_list()
        assert dtype == start_stop_step.dtype

    def test_arange_misc(self):
        # test uint64 handles negatives correctly
        np_arange_uint = np.arange(2**64 - 5, 2**64 - 10, -1, dtype=np.uint64)
        ak_arange_uint = ak.arange(-5, -10, -1, dtype=ak.uint64)
        # np_arange_uint = array([18446744073709551611, 18446744073709551610, 18446744073709551609,
        #        18446744073709551608, 18446744073709551607], dtype=uint64)
        assert np_arange_uint.tolist() == ak_arange_uint.to_list()
        assert ak.uint64 == ak_arange_uint.dtype

        uint_start_stop = ak.arange(2**63 + 3, 2**63 + 7)
        ans = ak.arange(3, 7, dtype=ak.uint64) + 2**63
        assert ans.to_list() == uint_start_stop.to_list()
        assert ak.uint64 == uint_start_stop.dtype

        # test correct conversion to float64
        np_arange_float = np.arange(-5, -10, -1, dtype=np.float64)
        ak_arange_float = ak.arange(-5, -10, -1, dtype=ak.float64)
        # array([-5., -6., -7., -8., -9.])
        assert np_arange_float.tolist() == ak_arange_float.to_list()
        assert ak.float64 == ak_arange_float.dtype

        # test correct conversion to bool
        expected_bool = [False, True, True, True, True]
        ak_arange_bool = ak.arange(0, 10, 2, dtype=ak.bool_)
        assert expected_bool == ak_arange_bool.to_list()
        assert ak.bool_ == ak_arange_bool.dtype

        # test int_scalars covers uint8, uint16, uint32
        uint_array = ak.arange(np.uint8(1), np.uint16(1000), np.uint32(1))
        int_array = ak.arange(1, 1000, 1)
        assert (uint_array == int_array).all()

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [ak.float64, ak.uint64, ak.float64])
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
        test_array = ak.randint(0, size, size, array_type)
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
            test_array = ak.randint(0, size, shape, array_type)
            assert isinstance(test_array, ak.pdarray)
            assert size == len(test_array)
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

    def test_randint_with_seed(self):
        values = ak.randint(1, 5, 10, seed=2)

        assert [4, 3, 1, 3, 2, 4, 4, 2, 3, 4] == values.to_list()

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

        assert ans == values.to_list()

        bools = [False, True, True, True, True, False, True, True, True, True]
        values = ak.randint(1, 5, 10, dtype=ak.bool_, seed=2)
        assert values.to_list() == bools

        values = ak.randint(1, 5, 10, dtype=bool, seed=2)
        assert values.to_list() == bools

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
        ] == u_array.to_list()

        u_array = ak.uniform(size=np.int64(3), low=np.int64(0), high=np.int64(5), seed=np.int64(0))
        assert [
            0.30013431967121934,
            0.47383036230759112,
            1.0441791878997098,
        ] == u_array.to_list()

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

        with pytest.raises(TypeError):
            ak.ones(5, dtype=str)

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
    @pytest.mark.parametrize("fill_value", [0, 1, 2, True, False, 3.14])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, float, ak.float64, bool, ak.bool_])
    def test_full_match_numpy(self, size, dtype, fill_value, subtests):
        for rank in pytest.compiled_ranks:
            if rank == 1:
                continue
            with subtests.test(rank=rank, dtype=dtype, fill_value=fill_value):
                shape, _ = _generate_test_shape(rank, size)

                ak_arr = ak.full(shape, fill_value=fill_value, dtype=dtype)
                np_arr = np.full(shape, fill_value=fill_value, dtype=np.dtype(dtype))

                assert ak_arr.shape == np_arr.shape
                assert str(ak_arr.dtype) == str(np_arr.dtype)

                assert_equivalent(
                    ak_arr,
                    np_arr,
                    err_msg=f"Failed for rank={rank}, dtype={dtype}, fill_value={fill_value}",
                )

    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, bool, ak.bool_])
    def test_full_error(self, dtype, subtests):
        """
        Attempt to create a multi‐dimensional "full" array with rank > max supported.
        Should always raise ValueError for “rank too large.”
        """
        max_rank = ak.client.get_max_array_rank()
        test_rank = max_rank + 1
        # sanity check
        assert test_rank > max_rank, f"Test rank ({test_rank}) must exceed supported max ({max_rank})"

        shape, local_size = _generate_test_shape(test_rank, 2**test_rank)

        for fill_value in (0, 9):  # try a couple of numeric fill_values
            with subtests.test(dtype=dtype, fill_value=fill_value, rank=test_rank):
                with pytest.raises(ValueError):
                    # use keyword for clarity:
                    ak.full(shape, fill_value=fill_value, dtype=dtype)

    @pytest.mark.parametrize(
        "arg, expected_all",
        [
            (-1, np.True_),
            (np.False_, np.False_),
        ],
    )
    def test_full_bool(self, arg, expected_all, subtests):
        """
        Bool‐dtype branches: negative ints → True, False → all False.
        """
        with subtests.test(arg=arg):
            pda = ak.full(5, arg, dtype=ak.bool_)
            assert pda.dtype == ak.bool_, f"dtype mismatch: got {pda.dtype}"
            # `.all()` only valid when dtype is bool
            assert pda.all() is expected_all, f"bool contents wrong for arg={arg}"

    @pytest.mark.parametrize(
        "shape,fill,dtype,expected_len",
        [
            (5, 5, None, 5),  # shape=int, no dtype→Strings?
            (5, "test", ak.Strings, 5),  # shape=int, fill=string→Strings
            ("5", 5, ak.Strings, 5),  # shape=str→cast to int
        ],
    )
    def test_full_string_shapes(self, shape, fill, dtype, expected_len, subtests):
        """
        String‐based full: shape may be int or str; fill must be str or int.
        """
        with subtests.test(shape=shape, fill=fill):
            pda = ak.full(shape, fill, dtype=dtype) if dtype else ak.full(shape, fill)
            # if we passed a string fill or explicit Strings dtype, it should be Strings
            expect_type = ak.Strings if (isinstance(fill, str) or dtype is ak.Strings) else ak.pdarray
            assert isinstance(pda, expect_type), f"expected {expect_type}, got {type(pda)}"
            assert len(pda) == expected_len, f"length wrong: {len(pda)} != {expected_len}"
            if expect_type is ak.Strings:
                assert pda.to_list() == [str(fill)] * expected_len

    @pytest.mark.parametrize(
        "shape, fill, bad_dtype",
        [
            pytest.param(5, 1, ak.uint8, id="invalid_uint8_dtype"),
        ],
    )
    @pytest.mark.parametrize("style", ["positional", "keyword"])
    def test_full_type_errors(self, shape, fill, bad_dtype, subtests, style):
        """
        Ensure ak.full(shape, fill, dtype=bad_dtype) raises TypeError
        when the dtype/fill combination is unsupported.
        """

        def call():
            if style == "positional":
                return ak.full(shape, fill, dtype=bad_dtype)
            else:
                return ak.full(shape=shape, fill_value=fill, dtype=bad_dtype)

        with pytest.raises(TypeError):
            call()

        # Test that int_scalars covers uint8, uint16, uint32
        int_arr = ak.full(5, 5)
        for args in [
            (np.uint8(5), np.uint16(5)),
            (np.uint16(5), np.uint32(5)),
            (np.uint32(5), np.uint8(5)),
        ]:
            assert (int_arr == ak.full(*args, dtype=int)).all()

    @pytest.mark.parametrize(
        "shape_scalar,fill_scalar",
        [
            pytest.param(np.uint8(5), np.uint16(5), id="uint8_shape_uint16_fill"),
            pytest.param(np.uint16(5), np.uint32(5), id="uint16_shape_uint32_fill"),
            pytest.param(np.uint32(5), np.uint8(5), id="uint32_shape_uint8_fill"),
        ],
    )
    def test_full_int_scalar_roundtrip(self, shape_scalar, fill_scalar, subtests):
        """
        Using any uint8/16/32 scalar for shape and fill with dtype=int
        must produce the same int-typed array as ak.full(5, 5).
        """
        # Sanity check default array
        #   TODO:  Uncomment when #4312 is resolved
        # default_arr = ak.full(5, 5)
        # assert default_arr.dtype is int, f"default dtype is {default_arr.dtype}, expected int"
        # assert len(default_arr) == 5, f"default length is {len(default_arr)}, expected 5"
        for call_style in ("positional", "keyword"):
            with subtests.test(call_style=call_style):
                if call_style == "positional":
                    arr = ak.full(shape_scalar, fill_scalar, dtype=int)
                else:
                    arr = ak.full(shape_scalar, fill_value=fill_scalar, dtype=int)

                # Check dtype and length
                assert arr.dtype == ak.int64, f"dtype {arr.dtype} is not int"
                assert len(arr) == 5, f"length {len(arr)} != 5"
                #   TODO:  Uncomment when #4312 is resolved
                # assert_arkouda_array_equal(default_arr, arr)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize(
        "dtype, np_dtype",
        [
            (int, np.int64),
            (float, np.float64),
            (bool, np.bool_),
            (ak.int64, np.int64),
            (ak.uint64, np.uint64),
            (ak.float64, np.float64),
            (ak.bool_, np.bool_),
        ],
    )
    def test_full_like(self, size, dtype, np_dtype, subtests):
        """
        full_like: given an array of a certain dtype, produce a new array
        of the same shape and dtype, filled with the specified value.
        """
        # Create a “random” source array via full
        src = ak.full(size, 5, dtype=dtype)

        for fill_value in [0, 1, 3.3, True]:
            with subtests.test(dtype=dtype, fill_value=fill_value):
                result = ak.full_like(src, fill_value)

                # 1) Type & shape
                assert isinstance(result, ak.pdarray), f"Expected pdarray, got {type(result)}"
                assert result.dtype == dtype, f"Dtype mismatch: got {result.dtype}, expected {dtype}"
                assert result.size == src.size, f"Size mismatch: {result.size} != {src.size}"
                assert result.shape == src.shape, f"Shape mismatch: {result.shape} != {src.shape}"

                # 2) Content check via elementwise equality
                if dtype is bool:
                    # for bool, ensure truthiness matches
                    expected = np.full(src.size, fill_value, dtype=np_dtype)
                    assert (result.to_ndarray() == expected).all(), (
                        f"Boolean content mismatch for fill={fill_value}"
                    )
                else:
                    # compare against numpy.full_like
                    expected_np = np.full_like(src.to_ndarray(), fill_value, dtype=np_dtype)
                    assert_equivalent(result, expected_np)

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize(
        "dtype, np_dtype",
        [
            (int, np.int64),
            (ak.int64, np.int64),
            (ak.uint64, np.uint64),
            (float, np.float64),
            (ak.float64, np.float64),
            (bool, np.bool_),
            (ak.bool_, np.bool_),
        ],
    )
    def test_full_like_multi_dim(self, size, dtype, np_dtype, subtests):
        """
        For each supported multi-dimensional rank ≥2, ensure that
        ak.full_like(src, fill) produces the same shape, dtype, size,
        and contents as numpy.full_like(src_nd, fill, np_dtype).
        """
        for rank in multi_dim_ranks():
            shape, _ = _generate_test_shape(rank, size)
            # create a source array of known values/type
            src = ak.full(shape, 5, dtype=dtype)

            for fill_value in [0, 1, 7.7, True]:
                with subtests.test(rank=rank, dtype=dtype, fill=fill_value):
                    result = ak.full_like(src, fill_value)

                    # 1) Type & metadata checks
                    assert isinstance(result, ak.pdarray), (
                        f"Expected pdarray, got {type(result)} at rank={rank}"
                    )
                    assert result.shape == shape, (
                        f"Shape mismatch: {result.shape} != {shape} for rank={rank}"
                    )
                    assert result.size == src.size, f"Size mismatch: {result.size} != {src.size}"
                    assert result.dtype == dtype, f"Dtype mismatch: {result.dtype} != {dtype}"

                    # 2) Content check via NumPy alignment
                    np_src = src.to_ndarray()
                    expected = np.full_like(np_src, fill_value, dtype=np_dtype)
                    ak_arr = result.to_ndarray()
                    # elementwise equality
                    assert_equivalent(ak_arr, expected)

    @pytest.mark.parametrize(
        "start, stop, length",
        [
            pytest.param(0, 100, 1000, id="ints_positional"),
            pytest.param(5, 0, 6, id="ints_reverse"),
            pytest.param(5.0, 0.0, 6, id="floats"),
            pytest.param(5, 0, np.int64(6), id="np_int64_length"),
        ],
    )
    def test_linspace_valid(self, start, stop, length, subtests):
        """
        ak.linspace(start, stop, length) must match numpy.linspace(start, stop, length_int).
        """
        length_int = int(length)
        with subtests.test(start=start, stop=stop, length=length):
            pda = ak.linspace(start, stop, length)
            np_arr = np.linspace(start, stop, length_int)

            assert isinstance(pda, ak.pdarray), f"wrong type {type(pda)}"
            assert len(pda) == length_int, f"length {len(pda)} != {length_int}"
            assert pda.dtype == float, f"dtype {pda.dtype} is not float"
            assert (pda.to_ndarray() == np_arr).all(), "content mismatch"

    @pytest.mark.parametrize(
        "case_id, args, expected_err",
        [
            ("stop_str", (0, "100", 1000), TypeError),
            ("start_str", ("0", 100, 1000), TypeError),
            ("length_str", (0, 100, "1000"), TypeError),
            # ("length_zero", (0, 100, 0), ValueError),
            # ("length_negative", (0, 100, -5), ValueError),
        ],
    )
    def test_linspace_errors(self, case_id, args, expected_err, subtests):
        """
        Passing invalid types or non-positive length to linspace should raise.
        """
        with subtests.test(case=case_id):
            with pytest.raises(expected_err):
                ak.linspace(*args)

    @pytest.mark.parametrize(
        "start, stop",
        [
            pytest.param(0, 50, id="int_to_int"),
            pytest.param(0.5, 101, id="float_to_int"),
            pytest.param(2, 50, id="int_to_int_small"),
        ],
    )
    @pytest.mark.parametrize("length", pytest.prob_size)
    def test_linspace_numpy_alignment(self, length, start, stop, subtests):
        """
        ak.linspace(start, stop, length) should produce the same sequence
        (within floating-point tolerance) as np.linspace(start, stop, length).
        """
        with subtests.test(start=start, stop=stop, length=length):
            # produce both arrays
            np_arr = np.linspace(start, stop, length)
            ak_arr = ak.linspace(start, stop, length)

            # 1) Type & metadata
            assert isinstance(ak_arr, ak.pdarray), f"Expected pdarray, got {type(ak_arr)}"
            assert ak_arr.dtype == float, f"Expected dtype float, got {ak_arr.dtype}"
            assert len(ak_arr) == length, f"Expected length {length}, got {len(ak_arr)}"

            # 2) Value check
            # Use allclose to allow for any tiny FP differences
            assert np.allclose(ak_arr.to_ndarray(), np_arr, rtol=1e-7, atol=0), (
                f"Values differ for start={start}, stop={stop}, length={length}"
            )

    @pytest.mark.parametrize(
        "shape_scalar",
        [
            pytest.param(100, id="int_shape"),
            pytest.param(np.int64(100), id="np_int64_shape"),
            pytest.param(np.uint16(50), id="np_uint16_shape"),
        ],
    )
    def test_standard_normal_shape_dtype(self, shape_scalar, subtests):
        """
        Verify that ak.standard_normal called with various integer‐scalar shapes
        returns a float pdarray of the correct length.
        """
        expected_len = int(shape_scalar)
        for call_style in ("positional", "keyword"):
            with subtests.test(shape=shape_scalar, style=call_style):
                if call_style == "positional":
                    arr = ak.standard_normal(shape_scalar)
                else:
                    arr = ak.standard_normal(size=shape_scalar)

                assert isinstance(arr, ak.pdarray), f"{call_style}: Expected pdarray, got {type(arr)}"
                assert arr.dtype == float, f"{call_style}: Expected dtype float, got {arr.dtype}"
                assert len(arr) == expected_len, (
                    f"{call_style}: Expected length {expected_len}, got {len(arr)}"
                )

    @pytest.mark.parametrize(
        "shape_scalar, seed_scalar",
        [
            pytest.param(100, 42, id="int_shape_int_seed"),
            pytest.param(np.uint8(10), np.uint16(7), id="np_scalars"),
        ],
    )
    def test_standard_normal_reproducible(self, shape_scalar, seed_scalar, subtests):
        """
        When a seed is provided, ak.standard_normal must be bit‐for‐bit
        reproducible across calls, for both pos. and kw. arg styles.
        """
        for call_style in ("positional", "keyword"):
            with subtests.test(seed=seed_scalar, style=call_style):
                if call_style == "positional":
                    a1 = ak.standard_normal(shape_scalar, seed_scalar)
                    a2 = ak.standard_normal(shape_scalar, seed_scalar)
                else:
                    a1 = ak.standard_normal(size=shape_scalar, seed=seed_scalar)
                    a2 = ak.standard_normal(size=shape_scalar, seed=seed_scalar)

                assert isinstance(a1, ak.pdarray)
                # same object shape & dtype
                assert a1.dtype == float
                assert len(a1) == len(a2)
                # reproducibility
                assert (a1 == a2).all(), "Outputs differ with same seed"

    @pytest.mark.parametrize(
        "size, seed, expected_err",
        [
            pytest.param("100", None, TypeError, id="size_str"),
            pytest.param(100.0, None, TypeError, id="size_float"),
            pytest.param(-5, None, ValueError, id="size_negative"),
            pytest.param(5, "1", TypeError, id="seed_str"),
            pytest.param(5, 1.5, TypeError, id="seed_float"),
        ],
    )
    def test_standard_normal_errors(self, size, seed, expected_err, subtests):
        """
        Passing invalid types or values to ak.standard_normal should raise
        TypeError for non-ints and ValueError for negative sizes.
        """
        for call_style in ("positional", "keyword"):
            with subtests.test(size=size, seed=seed, style=call_style):

                def call():
                    if seed is None:
                        return ak.standard_normal(size)
                    elif call_style == "positional":
                        return ak.standard_normal(size, seed)
                    else:
                        return ak.standard_normal(size=size, seed=seed)

                with pytest.raises(expected_err):
                    call()

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
        ] == pda.to_list()

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
        ] == pda.to_list()

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
        assert randoms == pda.to_list()

        pda = ak.random_strings_lognormal(float(2), np.float64(0.25), np.int64(10), seed=1)
        assert randoms == pda.to_list()

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
        assert printable_randoms == pda.to_list()

        pda = ak.random_strings_lognormal(
            np.int64(2), np.float64(0.25), np.int64(10), seed=1, characters="printable"
        )
        assert printable_randoms == pda.to_list()

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [bool, np.float64, np.int64, str])
    def test_from_series_dtypes(self, size, dtype):
        p_array = ak.from_series(pd.Series(np.random.randint(0, 10, size)), dtype)
        assert isinstance(p_array, ak.pdarray if dtype is not str else ak.Strings)
        assert dtype == p_array.dtype

        p_objects_array = ak.from_series(
            pd.Series(np.random.randint(0, 10, size), dtype="object"), dtype=dtype
        )
        assert isinstance(p_objects_array, ak.pdarray if dtype is not str else ak.Strings)
        assert dtype == p_objects_array.dtype

    def test_from_series_misc(self):
        p_array = ak.from_series(pd.Series(["a", "b", "c", "d", "e"]))
        assert isinstance(p_array, ak.Strings)
        assert str == p_array.dtype

        p_array = ak.from_series(pd.Series(np.random.choice([True, False], size=10)))

        assert isinstance(p_array, ak.pdarray)
        assert bool == p_array.dtype

        p_array = ak.from_series(pd.Series([dt.datetime(2016, 1, 1, 0, 0, 1)]))

        assert isinstance(p_array, ak.pdarray)
        assert np.int64 == p_array.dtype

        p_array = ak.from_series(pd.Series([np.datetime64("2018-01-01")]))

        assert isinstance(p_array, ak.pdarray)
        assert np.int64 == p_array.dtype

        p_array = ak.from_series(
            pd.Series(pd.to_datetime(["1/1/2018", np.datetime64("2018-01-01"), dt.datetime(2018, 1, 1)]))
        )

        assert isinstance(p_array, ak.pdarray)
        assert np.int64 == p_array.dtype

        with pytest.raises(TypeError):
            ak.from_series(np.ones(10))

        with pytest.raises(ValueError):
            ak.from_series(pd.Series(np.random.randint(0, 10, 10), dtype=np.int8))

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
            assert greedy_list == greedy_pda.to_list()

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
        assert ones.to_list() == new_ones.to_list()

        empty_ones = ak.ones(0)
        n_empty_ones = empty_ones.to_ndarray()
        new_empty_ones = ak.array(n_empty_ones)
        assert empty_ones.to_list() == new_empty_ones.to_list()

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_to_ndarray_multi_dim(self, size):
        for rank in multi_dim_ranks():
            shape, local_size = _generate_test_shape(rank, size)
            ones = ak.ones(shape)
            n_ones = ones.to_ndarray()
            new_ones = ak.array(n_ones)
            assert ones.to_list() == new_ones.to_list()

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_to_ndarray_multi_dim_bigint(self, size):
        for rank in multi_dim_ranks():
            shape, local_size = _generate_test_shape(rank, size)
            ones = ak.ones(shape, ak.bigint)
            n_ones = ones.to_ndarray()
            assert_arkouda_array_equal(ones, ak.array(n_ones))
