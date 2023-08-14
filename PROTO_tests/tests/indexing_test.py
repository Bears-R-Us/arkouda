import numpy as np
import pytest

import arkouda as ak

NUM_TYPES = [ak.int64, ak.uint64, ak.float64, ak.bool, ak.bigint]


def key_arrays(size):
    ikeys = ak.arange(size)
    ukeys = ak.arange(size, dtype=ak.uint64)
    return ikeys, ukeys


def value_array(dtype, size):
    if dtype is ak.int64:
        return ak.randint(-size, size, size)
    elif dtype is ak.uint64:
        return ak.randint(0, size, size, dtype=dtype)
    elif dtype is ak.float64:
        return ak.randint(-size, size, size, dtype=dtype)
    elif dtype is ak.bool:
        return (ak.randint(0, size, size) % 2) == 0
    elif dtype is ak.bigint:
        return ak.randint(0, size, size, dtype=ak.uint64) + 2**200
    elif dtype is ak.str_:
        return ak.random_strings_uniform(1, 16, size=size)
    return None


class TestIndexing:
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", NUM_TYPES + [ak.str_])
    def test_pdarray_uint_indexing(self, prob_size, dtype):
        # for every pda in array_dict test indexing with uint array and uint scalar
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
    @pytest.mark.parametrize("num_types", NUM_TYPES)
    def test_set_uint(self, prob_size, num_types):
        test_size = prob_size // 2
        ikeys, ukeys = key_arrays(prob_size)
        pda = value_array(num_types, prob_size)

        # set [int] = val with uint key and value
        pda[np.uint(2)] = np.uint(5)
        assert pda[np.uint(2)] == pda[2]

        # set [slice] = scalar/pdarray
        pda[:test_size] = -2
        assert pda[ukeys].to_list() == pda[ikeys].to_list()
        pda[:test_size] = ak.cast(ak.arange(test_size), num_types)
        assert pda[ukeys].to_list() == pda[ikeys].to_list()

        # set [pdarray] = scalar/pdarray with uint key pdarray
        pda[ak.arange(test_size, dtype=ak.uint64)] = np.uint(3)
        assert pda[ukeys].to_list() == pda[ikeys].to_list()
        pda[ak.arange(test_size, dtype=ak.uint64)] = ak.cast(ak.arange(test_size), num_types)
        assert pda[ukeys].to_list() == pda[ikeys].to_list()

        if num_types == ak.bigint:
            # bigint specific set [int] = val with uint key and value
            pda[np.uint(2)] = 2**200
            assert pda[2] == 2**200

            # bigint specific set [slice] = scalar/pdarray
            pda[:prob_size] = 2**200
            assert pda[:prob_size].to_list() == ak.full(prob_size, 2**200, ak.bigint).to_list()
            pda[:prob_size] = ak.arange(prob_size, dtype=ak.bigint)
            assert pda[:prob_size].to_list() == ak.arange(prob_size, dtype=ak.uint64).to_list()

            # bigint specific set [pdarray] = scalar/pdarray with uint key pdarray
            pda[ak.arange(prob_size, dtype=ak.uint64)] = 2**200
            assert pda[:prob_size].to_list() == ak.full(prob_size, 2**200, ak.bigint).to_list()
            pda[ak.arange(prob_size)] = ak.arange(prob_size, dtype=ak.bigint)
            assert pda[:prob_size].to_list() == ak.arange(prob_size, dtype=ak.uint64).to_list()

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
