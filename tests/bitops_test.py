from base_test import ArkoudaTest
from context import arkouda as ak


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
        p = self.a.popcount()
        ans = ak.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2])
        self.assertTrue((p == ans).all())

        # Test uint case
        p = self.b.popcount()
        ans = ak.cast(ans, ak.uint64)
        self.assertTrue((p == ans).all())

        # Function invocation
        # Edge case input
        p = ak.popcount(self.edgeCases)
        ans = ak.array([1, 64, 63])
        self.assertTrue((p == ans).all())

        # Test uint case
        p = ak.popcount(self.edgeCasesUint)
        ans = ak.cast(ans, ak.uint64)
        self.assertTrue((p == ans).all())

    def test_parity(self):
        p = self.a.parity()
        ans = ak.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
        self.assertTrue((p == ans).all())

        # Test uint case
        p = self.b.parity()
        ans = ak.cast(ans, ak.uint64)
        self.assertTrue((p == ans).all())

        p = ak.parity(self.edgeCases)
        ans = ak.array([1, 0, 1])
        self.assertTrue((p == ans).all())

        # Test uint case
        p = ak.parity(self.edgeCasesUint)
        ans = ak.cast(ans, ak.uint64)
        self.assertTrue((p == ans).all())

    def test_clz(self):
        p = self.a.clz()
        ans = ak.array([64, 63, 62, 62, 61, 61, 61, 61, 60, 60])
        self.assertTrue((p == ans).all())

        # Test uint case
        p = self.b.clz()
        ans = ak.cast(ans, ak.uint64)
        self.assertTrue((p == ans).all())

        p = ak.clz(self.edgeCases)
        ans = ak.array([0, 0, 1])
        self.assertTrue((p == ans).all())

        # Test uint case
        p = ak.clz(self.edgeCasesUint)
        ans = ak.cast(ans, ak.uint64)
        self.assertTrue((p == ans).all())

    def test_ctz(self):
        p = self.a.ctz()
        ans = ak.array([0, 0, 1, 0, 2, 0, 1, 0, 3, 0])
        self.assertTrue((p == ans).all())

        # Test uint case
        p = self.b.ctz()
        ans = ak.cast(ans, ak.uint64)
        self.assertTrue((p == ans).all())

        p = ak.ctz(self.edgeCases)
        ans = ak.array([63, 0, 0])
        self.assertTrue((p == ans).all())

        # Test uint case
        p = ak.ctz(self.edgeCasesUint)
        ans = ak.cast(ans, ak.uint64)
        self.assertTrue((p == ans).all())

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
        self.assertTrue((rotated == shifted).all())

        r = ak.rotl(self.edgeCases, 1)
        ans = ak.array([1, -1, -2])
        self.assertTrue((r == ans).all())

        # vector <<< vector
        rotated = self.a.rotl(self.a)
        shifted = self.a << self.a
        # No wraparound, so these should be equal
        self.assertTrue((rotated == shifted).all())

        r = ak.rotl(self.edgeCases, ak.array([1, 1, 1]))
        ans = ak.array([1, -1, -2])
        self.assertTrue((r == ans).all())

        # scalar <<< vector
        rotated = ak.rotl(-(2**63), self.a)
        ans = ak.array([-(2**63), 1, 2, 4, 8, 16, 32, 64, 128, 256])
        self.assertTrue((rotated == ans).all())

    def test_rotr(self):
        # vector <<< scalar
        rotated = (1024 * self.a).rotr(5)
        shifted = (1024 * self.a) >> 5
        # No wraparound, so these should be equal
        self.assertTrue((rotated == shifted).all())

        r = ak.rotr(self.edgeCases, 1)
        ans = ak.array([2**62, -1, -(2**62) - 1])
        self.assertTrue((r == ans).all())

        # vector <<< vector
        rotated = (1024 * self.a).rotr(self.a)
        shifted = (1024 * self.a) >> self.a
        # No wraparound, so these should be equal
        self.assertTrue((rotated == shifted).all())

        r = ak.rotr(self.edgeCases, ak.array([1, 1, 1]))
        ans = ak.array([2**62, -1, -(2**62) - 1])
        self.assertTrue((r == ans).all())

        # scalar <<< vector
        rotated = ak.rotr(1, self.a)
        ans = ak.array(
            [1, -(2**63), 2**62, 2**61, 2**60, 2**59, 2**58, 2**57, 2**56, 2**55]
        )
        self.assertTrue((rotated == ans).all())
