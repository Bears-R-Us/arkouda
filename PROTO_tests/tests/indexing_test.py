import numpy as np
import pytest

import arkouda as ak

NUM_TYPES = [ak.int64, ak.uint64, ak.float64, ak.bool, ak.bigint]
BIGINT_AND_UINT_TYPES = [ak.bigint, ak.uint64]


def key_arrays(size):
    ikeys = ak.arange(size)
    ukeys = ak.arange(size, dtype=ak.uint64)
    return ikeys, ukeys


def value_array(num_types, size):
    array_dict = {
        ak.int64: (i := ak.randint(0, size, size)),
        ak.uint64: (u := ak.cast(i, ak.uint64)),
        ak.float64: ak.array(np.random.randn(size)),
        ak.bool: (i % 2) == 0,
        ak.bigint: ak.cast(u, ak.bigint),
        ak.str_: ak.cast(i, str),
    }
    return array_dict[num_types]


class TestIndexing:
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("num_types", NUM_TYPES)
    def test_pdarray_uint_indexing(self, prob_size, num_types):
        # for every pda in array_dict test indexing with uint array and uint scalar
        ikeys, ukeys = key_arrays(prob_size)
        pda = value_array(num_types, prob_size)
        assert pda[np.uint(2)] == pda[2]
        assert pda[ukeys].to_list() == pda[ikeys].to_list()

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_strings_uint_indexing(self, prob_size):
        # test Strings array indexing with uint array and uint scalar
        ikeys, ukeys = key_arrays(prob_size)
        pda = value_array(ak.str_, prob_size)
        assert pda[np.uint(2)] == pda[2]
        assert pda[ukeys].to_list() == pda[ikeys].to_list()

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("dtypes", BIGINT_AND_UINT_TYPES)
    def test_bool_indexing(self, prob_size, dtypes):
        pda_test = value_array(dtypes, prob_size)
        pda_uint = ak.cast(pda_test, ak.uint64)
        pda_bool = value_array(ak.bool, prob_size)
        assert pda_uint[pda_bool].to_list() == pda_test[pda_bool].to_list()

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
