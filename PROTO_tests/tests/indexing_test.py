import numpy as np
import pytest

import arkouda as ak

NUM_TYPES = [ak.int64, ak.uint64, ak.float64, ak.bool, ak.bigint]


def key_arrays(size):
    ikeys = ak.arange(size)
    ukeys = ak.arange(size, dtype=ak.uint64)
    return ikeys, ukeys


def value_array(dtype, size):
    if dtype in [ak.int64, ak.float64]:
        return ak.randint(-size, size, size, dtype=dtype)
    elif dtype is ak.uint64:
        return ak.randint(0, size, size, dtype=dtype)
    elif dtype is ak.bool:
        return (ak.randint(0, size, size) % 2) == 0
    elif dtype is ak.bigint:
        return ak.randint(0, size, size, dtype=ak.uint64) + 2**200
    elif dtype is ak.str_:
        return ak.random_strings_uniform(1, 16, size=size)
    return None


def value_scalar(dtype, size):
    if dtype in [ak.int64, ak.float64]:
        return ak.randint(-size, 0, 1, dtype=dtype)
    elif dtype is ak.uint64:
        return ak.randint(2**63, 2**64, 1, dtype=dtype)
    elif dtype is ak.bool:
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
        assert pda[ukeys].to_list() == pda[ikeys].to_list()

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_bool_indexing(self, prob_size):
        u = value_array(ak.uint64, prob_size)
        b = value_array(ak.bool, prob_size)
        assert u[b].to_list() == ak.cast(u, ak.int64)[b].to_list()
        assert u[b].to_list() == ak.cast(u, ak.bigint)[b].to_list()

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
        assert pda[ukeys].to_list() == pda[ikeys].to_list()
        pda[:test_size] = ak.cast(ak.arange(test_size), dtype)
        assert pda[ukeys].to_list() == pda[ikeys].to_list()

        # set [int] = val with uint key and value
        val = value_scalar(dtype, prob_size)[0]
        pda[np.uint(2)] = val
        assert pda[2] == val

        # set [slice] = scalar/pdarray
        pda[:prob_size] = val
        assert pda[:prob_size].to_list() == ak.full(prob_size, val, dtype=dtype).to_list()
        pda_value_array = value_array(dtype, prob_size)
        pda[:prob_size] = pda_value_array
        assert pda[:prob_size].to_list() == pda_value_array.to_list()

        # set [pdarray] = scalar/pdarray with uint key pdarray
        pda[ak.arange(prob_size, dtype=ak.uint64)] = val
        assert pda[:prob_size].to_list() == ak.full(prob_size, val, dtype=dtype).to_list()
        pda_value_array = value_array(dtype, prob_size)
        pda[ak.arange(prob_size)] = pda_value_array
        assert pda[:prob_size].to_list() == pda_value_array.to_list()

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
        assert [7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2] == a.to_list()
