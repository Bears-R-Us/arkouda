import numpy as np
import pytest

import arkouda as ak


NUM_TYPES = [ak.int64, ak.uint64, ak.float64, ak.bool_, ak.bigint]


def key_arrays(size):
    ikeys = ak.arange(size)
    ukeys = ak.arange(size, dtype=ak.uint64)
    return ikeys, ukeys


def value_array(dtype, size):
    if dtype in [ak.int64, ak.float64]:
        return ak.randint(-size, size, size, dtype=dtype)
    elif dtype is ak.uint64:
        return ak.randint(0, size, size, dtype=dtype)
    elif dtype is ak.bool_:
        return (ak.randint(0, size, size) % 2) == 0
    elif dtype is ak.bigint:
        return ak.arange(2**200, 2**200 + size, dtype=ak.bigint)
    elif dtype is ak.str_:
        return ak.random_strings_uniform(1, 16, size=size)
    return None


def value_scalar(dtype, size):
    if dtype in [ak.int64, ak.float64]:
        return ak.randint(-size, 0, 1, dtype=dtype)
    elif dtype is ak.uint64:
        return ak.randint(2**63, 2**64, 1, dtype=dtype)
    elif dtype is ak.bool_:
        return (ak.randint(0, 2, 1) % 2) == 0
    elif dtype is ak.bigint:
        return ak.randint(0, size, 1, dtype=ak.uint64) + 2**200
    elif dtype is ak.str_:
        return ak.random_strings_uniform(1, 16, size=1)
    return None


class TestIndexing:
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUM_TYPES + [ak.str_])
    def test_pdarray_uint_indexing(self, prob_size, dtype):
        ikeys, ukeys = key_arrays(prob_size)
        pda = value_array(dtype, prob_size)
        assert pda[np.uint(2)] == pda[2]
        assert pda[ukeys].tolist() == pda[ikeys].tolist()

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_bool_indexing(self, prob_size):
        u = value_array(ak.uint64, prob_size)
        b = value_array(ak.bool_, prob_size)
        assert u[b].tolist() == ak.cast(u, ak.int64)[b].tolist()
        assert u[b].tolist() == ak.cast(u, ak.bigint)[b].tolist()

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUM_TYPES)
    def test_set_uint(self, prob_size, dtype):
        test_size = prob_size // 2
        ikeys, ukeys = key_arrays(prob_size)
        pda = value_array(dtype, prob_size)

        # set [int] = val with uint key and value
        pda[np.uint(2)] = np.uint(5)
        assert pda[np.uint(2)] == pda[2]

        # set [slice] = scalar/pdarray
        pda[:test_size] = -2
        assert pda[ukeys].tolist() == pda[ikeys].tolist()
        pda[:test_size] = ak.cast(ak.arange(test_size), dtype)
        assert pda[ukeys].tolist() == pda[ikeys].tolist()

        # set [int] = val with uint key and value
        val = value_scalar(dtype, prob_size)[0]
        pda[np.uint(2)] = val
        assert np.allclose(pda[2], val)

        # set [slice] = scalar/pdarray
        pda[:prob_size] = val
        assert pda[:prob_size].tolist() == ak.full(prob_size, val, dtype=dtype).tolist()
        pda_value_array = value_array(dtype, prob_size)
        pda[:prob_size] = pda_value_array
        assert pda[:prob_size].tolist() == pda_value_array.tolist()

        # set [pdarray] = scalar/pdarray with uint key pdarray
        pda[ak.arange(prob_size, dtype=ak.uint64)] = val
        assert pda[:prob_size].tolist() == ak.full(prob_size, val, dtype=dtype).tolist()
        pda_value_array = value_array(dtype, prob_size)
        pda[ak.arange(prob_size)] = pda_value_array
        assert pda[:prob_size].tolist() == pda_value_array.tolist()

    def test_indexing_with_uint(self):
        # verify reproducer from #1210 no longer fails
        a = ak.arange(10) * 2
        b = ak.cast(ak.array([3, 0, 8]), ak.uint64)
        a[b]

    def test_bigint_indexing_preserves_max_bits(self):
        max_bits = 64
        a = ak.arange(2**200 - 1, 2**200 + 11, max_bits=max_bits)
        assert max_bits == a[ak.arange(10)].max_bits
        assert max_bits == a[:].max_bits

    def test_handling_bigint_max_bits(self):
        a = ak.arange(2**200 - 1, 2**200 + 11, max_bits=3)
        a[:] = ak.arange(2**200 - 1, 2**200 + 11)
        assert [7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2] == a.tolist()

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_compare_get_slice(self, size):
        # create np version
        a = np.arange(size)
        a = a[::2]
        # create ak version
        b = ak.arange(size)
        b = b[::2]
        assert np.allclose(a, b.to_ndarray())

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_compare_set_slice_value(self, size):
        # create np version
        a = np.ones(size)
        a[::2] = -1
        # create ak version
        b = ak.ones(size)
        b[::2] = -1
        assert np.allclose(a, b.to_ndarray())

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_compare_set_slice(self, size):
        # create np version
        a = np.ones(size)
        a[::2] = a[::2] * -1
        # create ak version
        b = ak.ones(size)
        b[::2] = b[::2] * -1
        assert np.allclose(a, b.to_ndarray())

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_compare_get_bool_iv(self, size):
        # create np version
        a = np.arange(size)
        a = a[a < size // 2]
        # create ak version
        b = ak.arange(size)
        b = b[b < size // 2]
        assert np.allclose(a, b.to_ndarray())

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_compare_set_bool_iv_value(self, size):
        # create np version
        a = np.arange(size)
        a[a < size // 2] = -1
        # create ak version
        b = ak.arange(size)
        b[b < size // 2] = -1
        assert np.allclose(a, b.to_ndarray())

    @pytest.mark.parametrize("size", pytest.prob_size)
    def check_set_bool_iv(self, size):
        # create np version
        a = np.arange(size)
        a[a < size // 2] = a[: size // 2] * -1
        # create ak version
        b = ak.arange(size)
        b[b < size // 2] = b[: size // 2] * -1
        assert np.allclose(a, b.to_ndarray())

    @pytest.mark.parametrize("size", pytest.prob_size)
    def check_get_integer_iv(self, size):
        # create np version
        a = np.arange(size)
        iv = np.arange(size // 2)
        a = a[iv]
        # create ak version
        b = ak.arange(size)
        iv = ak.arange(size // 2)
        b = b[iv]
        assert np.allclose(a, b.to_ndarray())

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_compare_set_integer_iv_val(self, size):
        # create np version
        a = np.arange(size)
        iv = np.arange(size // 2)
        a[iv] = -1
        # create ak version
        b = ak.arange(size)
        iv = ak.arange(size // 2)
        b[iv] = -1
        assert np.allclose(a, b.to_ndarray())

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_compare_set_integer_iv(self, size):
        # create np version
        a = np.arange(size)
        iv = np.arange(size // 2)
        a[iv] = iv * 10
        # create ak version
        b = ak.arange(size)
        iv = ak.arange(size // 2)
        b[iv] = iv * 10
        assert np.allclose(a, b.to_ndarray())

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_compare_get_integer_idx(self, size):
        # create np version
        a = np.arange(size)
        v1 = a[size // 2]
        # create ak version
        b = ak.arange(size)
        v2 = b[size // 2]
        assert v1 == v2
        assert a[-1] == b[-1]

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_compare_set_integer_idx(self, size):
        # create np version
        a = np.arange(size)
        a[size // 2] = -1
        a[-1] = -1
        v1 = a[size // 2]
        # create ak version
        b = ak.arange(size)
        b[size // 2] = -1
        b[-1] = -1
        v2 = b[size // 2]
        assert v1 == v2
        assert a[-1] == b[-1]
