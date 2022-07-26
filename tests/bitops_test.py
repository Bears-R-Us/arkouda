from base_test import ArkoudaTest
from context import arkouda as ak
import numpy as np


class BitOpsTest(ArkoudaTest):
    def setUp(self):
        ArkoudaTest.setUp(self)
        self.a = ak.arange(10)
        self.b = ak.cast(self.a, ak.uint64)
        self.edgeCases = ak.array([-(2**63), -1, 2**63 - 1])
        self.edgeCasesUint = ak.cast(ak.array([-(2**63), -1, 2**63 - 1]), ak.uint64)

    def test_popcount(self):
        # Method invocation
        # Toy input
        ans = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2]
        self.assertListEqual(self.a.popcount().to_list(), ans)

        # Test uint case
        ans = np.array(ans, ak.uint64)
        self.assertListEqual(self.b.popcount().to_list(), ans.tolist())

        # Function invocation
        # Edge case input
        ans = [1, 64, 63]
        self.assertListEqual(ak.popcount(self.edgeCases).to_list(), ans)

        # Test uint case
        ans = np.array(ans, ak.uint64)
        self.assertListEqual(ak.popcount(self.edgeCasesUint).to_list(), ans.tolist())

    def test_parity(self):
        ans = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
        self.assertListEqual(self.a.parity().to_list(), ans)

        # Test uint case
        ans = np.array(ans, ak.uint64)
        self.assertListEqual(self.b.parity().to_list(), ans.tolist())

        ans = [1, 0, 1]
        self.assertListEqual(ak.parity(self.edgeCases).to_list(), ans)

        # Test uint case
        ans = np.array(ans, ak.uint64)
        self.assertListEqual(ak.parity(self.edgeCasesUint).to_list(), ans.tolist())

    def test_clz(self):
        ans = [64, 63, 62, 62, 61, 61, 61, 61, 60, 60]
        self.assertListEqual(self.a.clz().to_list(), ans)

        # Test uint case
        ans = np.array(ans, ak.uint64)
        self.assertListEqual(self.b.clz().to_list(), ans.tolist())

        ans = [0, 0, 1]
        self.assertListEqual(ak.clz(self.edgeCases).to_list(), ans)

        # Test uint case
        ans = np.array(ans, ak.uint64)
        self.assertListEqual(ak.clz(self.edgeCasesUint).to_list(), ans.tolist())

    def test_ctz(self):
        ans = [0, 0, 1, 0, 2, 0, 1, 0, 3, 0]
        self.assertListEqual(self.a.ctz().to_list(), ans)

        # Test uint case
        ans = np.array(ans, ak.uint64)
        self.assertListEqual(self.b.ctz().to_list(), ans.tolist())

        ans = [63, 0, 0]
        self.assertListEqual(ak.ctz(self.edgeCases).to_list(), ans)

        # Test uint case
        ans = np.array(ans, ak.uint64)
        self.assertListEqual(ak.ctz(self.edgeCasesUint).to_list(), ans.tolist())

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
