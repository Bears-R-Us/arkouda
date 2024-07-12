from base_test import ArkoudaTest
from context import arkouda as ak
import numpy as np


class BitOpsTest(ArkoudaTest):
    def setUp(self):
        ArkoudaTest.setUp(self)
        self.a = ak.arange(10)
        self.b = ak.cast(self.a, ak.uint64)
        self.bi = ak.cast(self.a, ak.bigint)
        self.bi.max_bits = 64
        self.edgeCases = ak.array([-(2**63), -1, 2**63 - 1])
        self.edgeCasesUint = ak.cast(ak.array([-(2**63), -1, 2**63 - 1]), ak.uint64)
        self.edgeCasesBigint = ak.cast(self.edgeCasesUint, ak.bigint)
        self.edgeCasesBigint.max_bits = 64

    def test_popcount(self):
        # Method invocation
        # Toy input
        ans = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2]
        self.assertListEqual(self.a.popcount().to_list(), ans)

        # Test uint case
        ans = np.array(ans, ak.uint64)
        self.assertListEqual(self.b.popcount().to_list(), ans.tolist())

        # Test bigint case
        self.assertListEqual(self.bi.popcount().to_list(), ans.tolist())

        # Function invocation
        # Edge case input
        ans = [1, 64, 63]
        self.assertListEqual(ak.popcount(self.edgeCases).to_list(), ans)

        # Test uint case
        ans = np.array(ans, ak.uint64)
        self.assertListEqual(ak.popcount(self.edgeCasesUint).to_list(), ans.tolist())

        # Test bigint case
        self.assertListEqual(ak.popcount(self.edgeCasesBigint).to_list(), ans.tolist())

    def test_parity(self):
        ans = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
        self.assertListEqual(self.a.parity().to_list(), ans)

        # Test uint case
        ans = np.array(ans, ak.uint64)
        self.assertListEqual(self.b.parity().to_list(), ans.tolist())

        # Test bigint case
        self.assertListEqual(self.bi.parity().to_list(), ans.tolist())

        ans = [1, 0, 1]
        self.assertListEqual(ak.parity(self.edgeCases).to_list(), ans)

        # Test uint case
        ans = np.array(ans, ak.uint64)
        self.assertListEqual(ak.parity(self.edgeCasesUint).to_list(), ans.tolist())

        # Test bigint case
        self.assertListEqual(ak.parity(self.edgeCasesBigint).to_list(), ans.tolist())

    def test_clz(self):
        ans = [64, 63, 62, 62, 61, 61, 61, 61, 60, 60]
        self.assertListEqual(self.a.clz().to_list(), ans)

        # Test uint case
        ans = np.array(ans, ak.uint64)
        self.assertListEqual(self.b.clz().to_list(), ans.tolist())

        # Test bigint case
        self.assertListEqual(self.bi.clz().to_list(), ans.tolist())

        ans = [0, 0, 1]
        self.assertListEqual(ak.clz(self.edgeCases).to_list(), ans)

        # Test uint case
        ans = np.array(ans, ak.uint64)
        self.assertListEqual(ak.clz(self.edgeCasesUint).to_list(), ans.tolist())

        # Test bigint case
        self.assertListEqual(ak.clz(self.edgeCasesBigint).to_list(), ans.tolist())

    def test_ctz(self):
        ans = [0, 0, 1, 0, 2, 0, 1, 0, 3, 0]
        self.assertListEqual(self.a.ctz().to_list(), ans)

        # Test uint case
        ans = np.array(ans, ak.uint64)
        self.assertListEqual(self.b.ctz().to_list(), ans.tolist())

        # Test bigint case
        self.assertListEqual(self.bi.ctz().to_list(), ans.tolist())

        ans = [63, 0, 0]
        self.assertListEqual(ak.ctz(self.edgeCases).to_list(), ans)

        # Test uint case
        ans = np.array(ans, ak.uint64)
        self.assertListEqual(ak.ctz(self.edgeCasesUint).to_list(), ans.tolist())

        # Test bigint case
        self.assertListEqual(ak.ctz(self.edgeCasesBigint).to_list(), ans.tolist())

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

            self.assertListEqual(pop_ans.to_list(), bi.popcount().to_list())
            self.assertListEqual(par_ans.to_list(), bi.parity().to_list())
            self.assertListEqual(clz_ans.to_list(), bi.clz().to_list())
            self.assertListEqual(ctz_ans.to_list(), bi.ctz().to_list())

        # set one more bit (the 201st)
        bi += 2**200
        # every elem has one more set bit, so popcount increases by one and parity swaps/is XORed by 1
        pop_ans += 1
        par_ans = par_ans ^ 1
        # clz_ans will be max_bits - 201 for all indices since that's the first nonzero bit everywhere now
        # ctz_ans is unchanged other than first position which previously had no set bits
        ctz_ans[0] = 200
        for max_bits in [201, 256]:
            bi.max_bits = max_bits
            clz_ans = ak.full_like(bi, max_bits - 201)

            self.assertListEqual(pop_ans.to_list(), bi.popcount().to_list())
            self.assertListEqual(par_ans.to_list(), bi.parity().to_list())
            self.assertListEqual(clz_ans.to_list(), bi.clz().to_list())
            self.assertListEqual(ctz_ans.to_list(), bi.ctz().to_list())

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
            # except for the first position doesn't have any set bits, so we want the 128 bits after accounted
            clz_ans[0] += 128

            self.assertListEqual(pop_ans.to_list(), bi.popcount().to_list())
            self.assertListEqual(par_ans.to_list(), bi.parity().to_list())
            self.assertListEqual(clz_ans.to_list(), bi.clz().to_list())
            self.assertListEqual(ctz_ans.to_list(), bi.ctz().to_list())

        # test edge cases
        edge_case = ak.cast(ak.array([-(2**63), -1, 2**63 - 1]), ak.uint64)
        bi = ak.bigint_from_uint_arrays([edge_case, edge_case, edge_case])

        pop_ans = ak.popcount(edge_case) * 3
        # parity is the same as edge cases, because anything XORed with itself becomes zero so (x ^ x ^ x) = x
        par_ans = ak.parity(edge_case)
        base_clz = ak.clz(edge_case)
        ctz_ans = ak.ctz(edge_case)

        for max_bits in [192, 201, 256]:
            bi.max_bits = max_bits
            # base_clz plus the amount that max_bits exceeds the bits used to store the bigint
            clz_ans = base_clz + (max_bits - 192)

            self.assertListEqual(pop_ans.to_list(), bi.popcount().to_list())
            self.assertListEqual(par_ans.to_list(), bi.parity().to_list())
            self.assertListEqual(clz_ans.to_list(), bi.clz().to_list())
            self.assertListEqual(ctz_ans.to_list(), bi.ctz().to_list())

    def test_dtypes(self):
        f = ak.zeros(10, dtype=ak.float64)
        with self.assertRaises(TypeError):
            f.popcount()

        with self.assertRaises(TypeError):
            ak.popcount(f)

    def test_rotl(self):
        # vector <<< scalar
        rotated = self.a.rotl(5)
        shifted = self.a << 5
        # No wraparound, so these should be equal
        self.assertListEqual(rotated.to_list(), shifted.to_list())

        r = ak.rotl(self.edgeCases, 1)
        self.assertListEqual(r.to_list(), [1, -1, -2])

        # vector <<< vector
        rotated = self.a.rotl(self.a)
        shifted = self.a << self.a
        # No wraparound, so these should be equal
        self.assertListEqual(rotated.to_list(), shifted.to_list())

        r = ak.rotl(self.edgeCases, ak.array([1, 1, 1]))
        self.assertListEqual(r.to_list(), [1, -1, -2])

        # scalar <<< vector
        rotated = ak.rotl(-(2**63), self.a)
        ans = [-(2**63), 1, 2, 4, 8, 16, 32, 64, 128, 256]
        self.assertListEqual(rotated.to_list(), ans)

    def test_rotr(self):
        # vector <<< scalar
        rotated = (1024 * self.a).rotr(5)
        shifted = (1024 * self.a) >> 5
        # No wraparound, so these should be equal
        self.assertListEqual(rotated.to_list(), shifted.to_list())

        r = ak.rotr(self.edgeCases, 1)
        self.assertListEqual(r.to_list(), [2**62, -1, -(2**62) - 1])

        # vector <<< vector
        rotated = (1024 * self.a).rotr(self.a)
        shifted = (1024 * self.a) >> self.a
        # No wraparound, so these should be equal
        self.assertListEqual(rotated.to_list(), shifted.to_list())

        r = ak.rotr(self.edgeCases, ak.array([1, 1, 1]))
        self.assertListEqual(r.to_list(), [2**62, -1, -(2**62) - 1])

        # scalar <<< vector
        rotated = ak.rotr(1, self.a)
        ans = [1, -(2**63), 2**62, 2**61, 2**60, 2**59, 2**58, 2**57, 2**56, 2**55]
        self.assertListEqual(rotated.to_list(), ans)
