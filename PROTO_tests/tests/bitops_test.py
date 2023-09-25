import pytest

import arkouda as ak


class TestBitOps:
    @classmethod
    def setup_class(cls):
        cls.int_arr = ak.arange(10)
        cls.uint_arr = ak.cast(cls.int_arr, ak.uint64)
        cls.bigint_arr = ak.cast(cls.int_arr, ak.bigint)
        cls.bigint_arr.max_bits = 64
        cls.edge_cases = ak.array([-(2**63), -1, 2**63 - 1])
        cls.edge_cases_uint = ak.cast(ak.array([-(2**63), -1, 2**63 - 1]), ak.uint64)
        cls.edge_cases_bigint = ak.cast(cls.edge_cases_uint, ak.bigint)
        cls.edge_cases_bigint.max_bits = 64

    def test_popcount(self):
        for arr in self.int_arr, self.uint_arr, self.bigint_arr:
            assert arr.popcount().to_list() == [0, 1, 1, 2, 1, 2, 2, 3, 1, 2]

        for arr in self.edge_cases, self.edge_cases_uint, self.edge_cases_bigint:
            assert ak.popcount(arr).to_list() == [1, 64, 63]

    def test_parity(self):
        for arr in self.int_arr, self.uint_arr, self.bigint_arr:
            assert arr.parity().to_list() == [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]

        for arr in self.edge_cases, self.edge_cases_uint, self.edge_cases_bigint:
            assert ak.parity(arr).to_list() == [1, 0, 1]

    def test_clz(self):
        for arr in self.int_arr, self.uint_arr, self.bigint_arr:
            assert arr.clz().to_list() == [64, 63, 62, 62, 61, 61, 61, 61, 60, 60]

        for arr in self.edge_cases, self.edge_cases_uint, self.edge_cases_bigint:
            assert ak.clz(arr).to_list() == [0, 0, 1]

    def test_ctz(self):
        for arr in self.int_arr, self.uint_arr, self.bigint_arr:
            assert arr.ctz().to_list() == [0, 0, 1, 0, 2, 0, 1, 0, 3, 0]

        for arr in self.edge_cases, self.edge_cases_uint, self.edge_cases_bigint:
            assert ak.ctz(arr).to_list() == [63, 0, 0]

    def test_bigint_bitops(self):
        # compare against int pdarray with variety of max_bits, should be the same except for clz
        i = ak.arange(10)
        bi = ak.arange(10, dtype=ak.bigint)

        pop_ans = ak.popcount(i)
        par_ans = ak.parity(i)
        base_clz = ak.clz(i)
        ctz_ans = ak.ctz(i)

        for max_bits in [10, 64, 201, 256]:
            bi.max_bits = max_bits
            # base_clz plus the difference between max_bits and the number bits used to store the bigint
            clz_ans = base_clz + (max_bits - 64)

            assert pop_ans.to_list() == bi.popcount().to_list()
            assert par_ans.to_list() == bi.parity().to_list()
            assert clz_ans.to_list() == bi.clz().to_list()
            assert ctz_ans.to_list() == bi.ctz().to_list()

        # set one more bit (the 201st)
        bi += 2**200
        # every elem has one more set bit, so popcount increases by one and parity swaps/is XORed by 1
        pop_ans += 1
        par_ans = par_ans ^ 1
        # clz_ans will be max_bits - 201 for all indices since that's the
        # first nonzero bit everywhere now
        # ctz_ans is unchanged other than first position which previously had no set bits
        ctz_ans[0] = 200
        for max_bits in [201, 256]:
            bi.max_bits = max_bits
            clz_ans = ak.full_like(bi, max_bits - 201)

            assert pop_ans.to_list() == bi.popcount().to_list()
            assert par_ans.to_list() == bi.parity().to_list()
            assert clz_ans.to_list() == bi.clz().to_list()
            assert ctz_ans.to_list() == bi.ctz().to_list()

        # test with lots of trailing zeros
        bi = ak.bigint_from_uint_arrays(
            [
                ak.arange(10, dtype=ak.uint64),
                ak.zeros(10, dtype=ak.uint64),
                ak.zeros(10, dtype=ak.uint64),
            ]
        )

        # popcount and parity just look at number of bits, so this is equivalent to arange(10)
        pop_ans = ak.popcount(i)
        par_ans = ak.parity(i)
        # ctz will include 2 new 64 bit arrays of zeros, but ctz(0) is still 0
        ctz_ans = ak.ctz(i) + 128
        ctz_ans[0] = 0

        for max_bits in [138, 192, 201, 256]:
            bi.max_bits = max_bits
            # base_clz plus the amount that max_bits exceeds the bits used to store the bigint
            clz_ans = base_clz + (max_bits - 192)
            # except for the first position doesn't have any set bits,
            # so we want the 128 bits after accounted
            clz_ans[0] += 128

            assert pop_ans.to_list() == bi.popcount().to_list()
            assert par_ans.to_list() == bi.parity().to_list()
            assert clz_ans.to_list() == bi.clz().to_list()
            assert ctz_ans.to_list() == bi.ctz().to_list()

        # test edge cases
        edge_case = ak.cast(ak.array([-(2**63), -1, 2**63 - 1]), ak.uint64)
        bi = ak.bigint_from_uint_arrays([edge_case, edge_case, edge_case])

        pop_ans = ak.popcount(edge_case) * 3
        # parity is the same as edge cases,
        # because anything XORed with itself becomes zero so (x ^ x ^ x) = x
        par_ans = ak.parity(edge_case)
        base_clz = ak.clz(edge_case)
        ctz_ans = ak.ctz(edge_case)

        for max_bits in [192, 201, 256]:
            bi.max_bits = max_bits
            # base_clz plus the amount that max_bits exceeds the bits used to store the bigint
            clz_ans = base_clz + (max_bits - 192)

            assert pop_ans.to_list() == bi.popcount().to_list()
            assert par_ans.to_list() == bi.parity().to_list()
            assert clz_ans.to_list() == bi.clz().to_list()
            assert ctz_ans.to_list() == bi.ctz().to_list()

    @pytest.mark.parametrize("dtype", [bool, str, float])
    def test_dtypes_errors(self, dtype):
        arr = ak.cast(ak.array(range(10)), dtype)

        with pytest.raises(TypeError):
            ak.popcount(arr)

        if dtype is not str:
            with pytest.raises(TypeError):
                arr.popcount()

    def test_rotl(self):
        # vector <<< scalar
        rotated = self.int_arr.rotl(5)
        shifted = self.int_arr << 5
        # No wraparound, so these should be equal
        assert rotated.to_list() == shifted.to_list()

        assert ak.rotl(self.edge_cases, 1).to_list() == [1, -1, -2]

        # vector <<< vector
        rotated = self.int_arr.rotl(self.int_arr)
        shifted = self.int_arr << self.int_arr
        # No wraparound, so these should be equal
        assert rotated.to_list() == shifted.to_list()

        assert ak.rotl(self.edge_cases, ak.array([1, 1, 1])).to_list() == [1, -1, -2]

        # scalar <<< vector
        rotated = ak.rotl(-(2**63), self.int_arr)
        assert rotated.to_list() == [-(2**63), 1, 2, 4, 8, 16, 32, 64, 128, 256]

    def test_rotr(self):
        # vector <<< scalar
        rotated = (1024 * self.int_arr).rotr(5)
        shifted = (1024 * self.int_arr) >> 5
        # No wraparound, so these should be equal
        assert rotated.to_list() == shifted.to_list()

        assert ak.rotr(self.edge_cases, 1).to_list() == [2**62, -1, -(2**62) - 1]

        # vector <<< vector
        rotated = (1024 * self.int_arr).rotr(self.int_arr)
        shifted = (1024 * self.int_arr) >> self.int_arr
        # No wraparound, so these should be equal
        assert rotated.to_list() == shifted.to_list()

        r = ak.rotr(self.edge_cases, ak.array([1, 1, 1]))
        assert r.to_list() == [2**62, -1, -(2**62) - 1]

        # scalar <<< vector
        rotated = ak.rotr(1, self.int_arr)
        ans = [1, -(2**63), 2**62, 2**61, 2**60, 2**59, 2**58, 2**57, 2**56, 2**55]
        assert rotated.to_list() == ans
