import datetime as dt
import math
import statistics
from collections import deque

import numpy as np
import pandas as pd
import pytest

import arkouda as ak
from arkouda.testing import assert_arkouda_array_equal, assert_equivalent

INT_SCALARS = list(ak.dtypes.int_scalars.__args__)
NUMERIC_SCALARS = list(ak.dtypes.numeric_scalars.__args__)

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


class TestPdarrayCreation:
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_array_creation(self, dtype):
        fixed_size = 100
        for pda in [
            ak.array(ak.ones(fixed_size, int), dtype),
            ak.array(np.ones(fixed_size), dtype),
            ak.array(list(range(fixed_size)), dtype=dtype),
            ak.array((range(fixed_size)), dtype),
            ak.array(deque(range(fixed_size)), dtype),
            ak.array([f"{i}" for i in range(fixed_size)], dtype=dtype),
        ]:
            assert isinstance(pda, ak.pdarray if ak.dtype(dtype) != "str_" else ak.Strings)
            assert len(pda) == fixed_size
            assert dtype == pda.dtype

    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, bool, ak.bool_])
    def test_array_creation_multi_dim(self, size, dtype):
        shape = (2, 2, size)
        for pda in [
            ak.array(ak.ones(shape, int), dtype),
            ak.array(np.ones(shape), dtype),
        ]:
            assert isinstance(pda, ak.pdarray)
            assert pda.shape == shape
            assert dtype == pda.dtype

    @pytest.mark.skip_if_max_rank_greater_than(3)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, bool, ak.bool_])
    def test_array_creation_error(self, dtype):
        shape = (2, 2, 2, 2)
        with pytest.raises(ValueError):
            ak.array(np.ones(shape), dtype)

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_large_array_creation(self, size):
        # Using pytest.prob_size in various other tests can be problematic; this
        # test is here simply to verify the ability of the various functions to
        # create large pdarrays, while the function-specific tests below are testing
        # the core functionality of each function
        for pda in [
            ak.ones(size, int),
            ak.array(ak.ones(size, int)),
            ak.zeros(size),
            ak.ones(size),
            ak.full(size, "test"),
            ak.zeros_like(ak.ones(size)),
            ak.ones_like(ak.zeros(size)),
            ak.full_like(ak.zeros(size), 9),
            ak.arange(size),
            ak.linspace(0, size, size),
            ak.randint(0, size, size),
            ak.uniform(size, 0, 100),
            ak.standard_normal(size),
            ak.random_strings_uniform(3, 30, size),
            ak.random_strings_lognormal(2, 0.25, size),
            ak.from_series(pd.Series(ak.arange(size).to_ndarray())),
            ak.bigint_from_uint_arrays([ak.ones(size, dtype=ak.uint64)]),
        ]:
            assert isinstance(pda, ak.pdarray if pda.dtype != str else ak.Strings)
            assert len(pda) == size

    @pytest.mark.skip_if_rank_not_compiled([2])
    def test_array_creation_misc(self):
        av = ak.array(np.array([[0, 1], [0, 1]]))
        assert isinstance(av, ak.pdarray)

        with pytest.raises(TypeError):
            ak.array({range(0, 10)})

        with pytest.raises(TypeError):
            ak.array("not an iterable")

        with pytest.raises(TypeError):
            ak.array(list(list(0)))

    @pytest.mark.skip_if_rank_not_compiled([2])
    def test_array_creation_transpose_bug_reproducer(self):

        import numpy as np

        rows = 5
        cols = 5
        nda = np.random.randint(1, 10, (rows, cols))

        assert_arkouda_array_equal(ak.transpose(ak.array(nda)), ak.array(np.transpose(nda)))

    def test_infer_shape_from_size(self):
        from arkouda.util import _infer_shape_from_size

        a = np.array([[0, 1], [0, 1]])
        shape, ndim, full_size = _infer_shape_from_size(a.shape)
        assert ndim == 2
        assert full_size == 4
        assert shape == (2, 2)

        shape, ndim, full_size = _infer_shape_from_size(7)
        assert ndim == 1
        assert full_size == 7
        assert shape == (7)

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
        assert (
            ak.array([bi, bi + 1, bi + 2, bi + 3, bi + 4]).to_list() == ak.arange(bi, bi + 5).to_list()
        )

        # test that max_bits being set results in a mod
        assert ak.arange(bi, bi + 5, max_bits=200).to_list() == ak.arange(5).to_list()

        # test ak.bigint_from_uint_arrays
        # top bits are all 1 which should be 2**64
        top_bits = ak.ones(5, ak.uint64)
        bot_bits = ak.arange(5, dtype=ak.uint64)
        two_arrays = ak.bigint_from_uint_arrays([top_bits, bot_bits])
        assert ak.bigint == two_arrays.dtype
        assert two_arrays.to_list() == [2**64 + i for i in range(5)]
        # top bits should represent 2**128
        mid_bits = ak.zeros(5, ak.uint64)
        three_arrays = ak.bigint_from_uint_arrays([top_bits, mid_bits, bot_bits])
        assert three_arrays.to_list() == [2**128 + i for i in range(5)]

        # test round_trip of ak.bigint_to/from_uint_arrays
        t = ak.arange(bi - 1, bi + 9)
        t_dup = ak.bigint_from_uint_arrays(t.bigint_to_uint_arrays())
        assert t.to_list() == t_dup.to_list()
        assert t_dup.max_bits == -1

        # test setting max_bits after creation still mods
        t_dup.max_bits = 200
        assert t_dup.to_list() == [bi - 1, 0, 1, 2, 3, 4, 5, 6, 7, 8]

        # test slice_bits along 64 bit boundaries matches return from bigint_to_uint_arrays
        for i, uint_bits in enumerate(t.bigint_to_uint_arrays()):
            slice_bits = t.slice_bits(64 * (4 - (i + 1)), 64 * (4 - i) - 1)
            assert uint_bits.to_list() == slice_bits.to_list()

    def test_arange(self):
        assert np.arange(0, 10, 1).tolist() == ak.arange(0, 10, 1).to_list()
        assert np.arange(10, 0, -1).tolist() == ak.arange(10, 0, -1).to_list()
        assert np.arange(-5, -10, -1).tolist() == ak.arange(-5, -10, -1).to_list()
        assert np.arange(0, 10, 2).tolist() == ak.arange(0, 10, 2).to_list()

    @pytest.mark.parametrize("dtype", ak.intTypes)
    def test_arange_dtype(self, dtype):
        # test dtype works with optional start/stride
        stop = ak.arange(100, dtype=dtype)
        assert np.arange(100, dtype=dtype).tolist() == stop.to_list()
        assert dtype == stop.dtype

        start_stop = ak.arange(100, 105, dtype=dtype)
        assert np.arange(100, 105, dtype=dtype).tolist() == start_stop.to_list()
        assert dtype == start_stop.dtype

        start_stop_stride = ak.arange(100, 105, 2, dtype=dtype)
        assert np.arange(100, 105, 2, dtype=dtype).tolist() == start_stop_stride.to_list()
        assert dtype == start_stop_stride.dtype

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
    @pytest.mark.parametrize("stride", [1, 3])
    def test_compare_arange(self, size, dtype, start, stride):
        # create np version
        nArange = np.arange(start, size, stride, dtype=dtype)
        # create ak version
        aArange = ak.arange(start, size, stride, dtype=dtype)
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

    # (The above function tests randint with various ARRAY dtypes; the function below
    #  tests with various dtypes for the other parameters passed to randint)
    @pytest.mark.parametrize("dtype", NUMERIC_SCALARS)
    def test_randint_num_dtype(self, dtype):
        for test_array in ak.randint(dtype(0), 100, 1000), ak.randint(0, dtype(100), 1000):
            assert isinstance(test_array, ak.pdarray)
            assert 1000 == len(test_array)
            assert ak.int64 == test_array.dtype
            assert (1000,) == test_array.shape
            assert ((0 <= test_array) & (test_array <= 1000)).all()

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
        assert [0.30013431967121934, 0.47383036230759112, 1.0441791878997098] == u_array.to_list()

        u_array = ak.uniform(size=np.int64(3), low=np.int64(0), high=np.int64(5), seed=np.int64(0))
        assert [0.30013431967121934, 0.47383036230759112, 1.0441791878997098] == u_array.to_list()

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

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("dtype", [int, ak.int64, float, ak.float64, bool, ak.bool_])
    @pytest.mark.parametrize("shape", [0, 2, (2, 3)])
    def test_ones_match_numpy(self, shape, dtype):
        assert_equivalent(ak.zeros(shape, dtype=dtype), np.zeros(shape, dtype=dtype))

    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [ak.int64, float, ak.float64, bool, ak.bool_, ak.bigint])
    def test_zeros_dtype_mult_dim(self, size, dtype):
        shape = (2, 2, size)
        zeros = ak.zeros(shape, dtype)
        assert isinstance(zeros, ak.pdarray)
        assert dtype == zeros.dtype
        assert zeros.shape == shape
        assert (0 == zeros).all()

    @pytest.mark.skip_if_max_rank_greater_than(3)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, bool, ak.bool_])
    def test_zeros_error(self, dtype):
        shape = (2, 2, 2, 2)
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

    @pytest.mark.parametrize("dtype", [int, ak.int64, float, ak.float64, bool, ak.bool_, ak.bigint])
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_ones_dtype(self, size, dtype):
        ones = ak.ones(size, dtype)
        assert isinstance(ones, ak.pdarray)
        assert dtype == ones.dtype
        assert (1 == ones).all()

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("dtype", [int, ak.int64, float, ak.float64, bool, ak.bool_])
    @pytest.mark.parametrize("shape", [0, 2, (2, 3)])
    def test_ones_match_numpy(self, shape, dtype):
        assert_equivalent(ak.ones(shape, dtype=dtype), np.ones(shape, dtype=dtype))

    @pytest.mark.parametrize("dtype", [int, ak.int64, float, ak.float64, bool, ak.bool_, ak.bigint])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.skip_if_rank_not_compiled([3])
    def test_ones_dtype_multi_dim(self, size, dtype):
        shape = (2, 2, size)
        ones = ak.ones(shape, dtype)
        assert isinstance(ones, ak.pdarray)
        assert ones.shape == shape
        assert dtype == ones.dtype
        assert (1 == ones).all()

    @pytest.mark.skip_if_max_rank_greater_than(3)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, bool, ak.bool_])
    def test_ones_error(self, dtype):
        shape = (2, 2, 2, 2)
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

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, bool, ak.bool_])
    def test_full_dtype(self, size, dtype):
        type_full = ak.full(size, 1, dtype)
        assert isinstance(type_full, ak.pdarray)
        assert dtype == type_full.dtype
        assert (1 == type_full).all()

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("dtype", [int, ak.int64, float, ak.float64, bool, ak.bool_])
    @pytest.mark.parametrize("shape", [0, 2, (2, 3)])
    def test_full_match_numpy(self, shape, dtype):
        assert_equivalent(
            ak.full(shape, fill_value=2, dtype=dtype), np.full(shape, fill_value=2, dtype=dtype)
        )

    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, bool, ak.bool_])
    def test_full_dtype_multi_dim(self, size, dtype):
        shape = (2, 2, size)
        type_full = ak.full(shape, 1, dtype)
        assert isinstance(type_full, ak.pdarray)
        assert dtype == type_full.dtype
        assert type_full.shape == shape
        assert (1 == type_full).all()

    @pytest.mark.skip_if_max_rank_greater_than(3)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, bool, ak.bool_])
    def test_full_error(self, dtype):
        shape = (2, 2, 2, 2)
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
        assert strings_full.to_list() == ["test"] * 5

        with pytest.raises(TypeError):
            ak.full(5, 1, dtype=ak.uint8)

        with pytest.raises(TypeError):
            ak.full(5, 8, dtype=str)

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

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [int, ak.int64, ak.uint64, float, ak.float64, bool, ak.bool_])
    def test_zeros_like(self, size, dtype):
        ran_arr = ak.array(ak.arange(size, dtype=dtype))
        zeros_like_arr = ak.zeros_like(ran_arr)
        assert isinstance(zeros_like_arr, ak.pdarray)
        assert dtype == zeros_like_arr.dtype
        assert (zeros_like_arr == 0).all()
        assert zeros_like_arr.size == ran_arr.size

    def test_linspace(self):
        pda = ak.linspace(0, 100, 1000)
        assert 1000 == len(pda)
        assert float == pda.dtype
        assert isinstance(pda, ak.pdarray)
        assert (pda.to_ndarray() == np.linspace(0, 100, 1000)).all()

        pda = ak.linspace(start=5, stop=0, length=6)
        assert 5.0000 == pda[0]
        assert 0.0000 == pda[5]
        assert (pda.to_ndarray() == np.linspace(5, 0, 6)).all()

        pda = ak.linspace(start=5.0, stop=0.0, length=6)
        assert 5.0000 == pda[0]
        assert 0.0000 == pda[5]
        assert (pda.to_ndarray() == np.linspace(5.0, 0.0, 6)).all()

        pda = ak.linspace(start=float(5.0), stop=float(0.0), length=np.int64(6))
        assert 5.0000 == pda[0]
        assert 0.0000 == pda[5]
        assert (pda.to_ndarray() == np.linspace(float(5.0), float(0.0), np.int64(6))).all()

        with pytest.raises(TypeError):
            ak.linspace(0, "100", 1000)

        with pytest.raises(TypeError):
            ak.linspace("0", 100, 1000)

        with pytest.raises(TypeError):
            ak.linspace(0, 100, "1000")

        # Test that int_scalars covers uint8, uint16, uint32
        int_arr = ak.linspace(0, 100, (1000 % 256))
        for args in [
            (np.uint8(0), np.uint16(100), np.uint32(1000 % 256)),
            (np.uint32(0), np.uint8(100), np.uint16(1000 % 256)),
            (np.uint16(0), np.uint32(100), np.uint8(1000 % 256)),
        ]:
            assert (int_arr == ak.linspace(*args)).all()

    @pytest.mark.parametrize("start", [0, 0.5, 2])
    @pytest.mark.parametrize("stop", [50, 101])
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_compare_linspace(self, size, start, stop):
        # create np version
        a = np.linspace(start, stop, size)
        # create ak version
        b = ak.linspace(start, stop, size)
        assert np.allclose(a, b.to_ndarray())

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
        assert npda.tolist() == pda.to_list()

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
        assert ["VW", "JEXI", "EBBX", "HG", "S", "WOVK", "U", "WL", "JCSD", "DSN"] == pda.to_list()

        pda = ak.random_strings_uniform(minlen=1, maxlen=5, seed=1, size=10, characters="printable")
        assert ["eL", "6<OD", "o-GO", " l", "m", "PV y", "f", "}.", "b3Yc", "Kw,"] == pda.to_list()

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("num_dtype", NUMERIC_SCALARS)
    def test_random_strings_lognormal(self, size, num_dtype):
        pda = ak.random_strings_lognormal(
            logmean=num_dtype(2), logstd=num_dtype(1), size=np.int64(size), characters="printable"
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

    @pytest.mark.skip_if_rank_not_compiled([2])
    def test_mulitdimensional_array_creation(self):
        a = ak.array([[0, 0], [0, 1], [1, 1]])
        assert isinstance(a, ak.pdarray)
        assert a.shape == (3, 2)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [bool, np.float64, np.int64, str])
    def test_from_series_dtypes(self, size, dtype):
        p_array = ak.from_series(pd.Series(np.random.randint(0, 10, size)), dtype)
        assert isinstance(p_array, ak.pdarray if dtype != str else ak.Strings)
        assert dtype == p_array.dtype

        p_objects_array = ak.from_series(
            pd.Series(np.random.randint(0, 10, size), dtype="object"), dtype=dtype
        )
        assert isinstance(p_objects_array, ak.pdarray if dtype != str else ak.Strings)
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

        # Test that int_scalars covers uint8, uint16, uint32
        ones.fill(np.uint8(2))
        ones.fill(np.uint16(2))
        ones.fill(np.uint32(2))

    def test_endian(self):
        a = np.random.randint(1, 100, 100)
        aka = ak.array(a)
        npa = aka.to_ndarray()
        assert np.allclose(a, npa)

        a = a.newbyteorder().byteswap()
        aka = ak.array(a)
        npa = aka.to_ndarray()
        assert np.allclose(a, npa)

        a = a.newbyteorder().byteswap()
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

    def test_uint_greediness(self):
        # default to uint when all supportedInt and any value > 2**63
        # to avoid loss of precision see (#1297)
        for greedy_list in ([2**63, 6, 2**63 - 1, 2**63 + 1], [2**64 - 1, 0, 2**64 - 1]):
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

    def testTo_ndarray(self):
        ones = ak.ones(10)
        n_ones = ones.to_ndarray()
        new_ones = ak.array(n_ones)
        assert ones.to_list() == new_ones.to_list()

        empty_ones = ak.ones(0)
        n_empty_ones = empty_ones.to_ndarray()
        new_empty_ones = ak.array(n_empty_ones)
        assert empty_ones.to_list() == new_empty_ones.to_list()
