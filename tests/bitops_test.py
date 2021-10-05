from context import arkouda as ak
from base_test import ArkoudaTest

class BitOpsTest(ArkoudaTest):
    def test_popcount(self):
        # Method invocation
        # Toy input
        p = ak.arange(10).popcount()
        ans = ak.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2])
        self.assertTrue((p == ans).all())
        # Function invocation
        # Edge case input
        p = ak.popcount(ak.array([-(2**63), -1, 2**63 - 1]))
        ans = ak.array([1, 64, 63])
        self.assertTrue((p == ans).all())

    def test_parity(self):
        p = ak.arange(10).parity()
        ans = ak.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
        self.assertTrue((p == ans).all())

        p = ak.parity(ak.array([-(2**63), -1, 2**63 - 1]))
        ans = ak.array([1, 0, 1])
        self.assertTrue((p == ans).all())

    def test_clz(self):
        p = ak.arange(10).clz()
        ans = ak.array([64, 63, 62, 62, 61, 61, 61, 61, 60, 60])
        self.assertTrue((p == ans).all())

        p = ak.clz(ak.array([-(2**63), -1, 2**63 - 1]))
        ans = ak.array([0, 0, 1])
        self.assertTrue((p == ans).all())

    def test_ctz(self):
        p = ak.arange(10).ctz()
        ans = ak.array([0, 0, 1, 0, 2, 0, 1, 0, 3, 0])
        self.assertTrue((p == ans).all())

        p = ak.ctz(ak.array([-(2**63), -1, 2**63 - 1]))
        ans = ak.array([63, 0, 0])
        self.assertTrue((p == ans).all())

    def test_dtypes(self):
        f = ak.zeros(10, dtype=ak.float64)
        with self.assertRaises(TypeError) as cm:
            f.popcount()
        b = ak.zeros(10, dtype=ak.bool)
        with self.assertRaises(TypeError) as cm:
            ak.popcount(f)
