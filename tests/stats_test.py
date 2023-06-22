import numpy as np
import pandas as pd
from base_test import ArkoudaTest
from context import arkouda as ak


class StatsTest(ArkoudaTest):
    def setUp(self):
        ArkoudaTest.setUp(self)
        self.x = ak.arange(10, 20)
        self.npx = np.arange(10, 20)
        self.pdx = pd.Series(self.npx)
        self.y = ak.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
        self.npy = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
        self.pdy = pd.Series(self.npy)
        self.u = ak.cast(self.y, ak.uint64)
        self.npu = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48], dtype=np.uint64)
        self.pdu = pd.Series(self.npu)
        self.b = ak.arange(10) % 2 == 0
        self.npb = np.arange(10) % 2 == 0
        self.pdb = pd.Series(self.npb)
        self.f = ak.array([i**2 for i in range(10)], dtype=ak.float64)
        self.npf = np.array([i**2 for i in range(10)], dtype=np.float64)
        self.pdf = pd.Series(self.npf)
        self.s = ak.array([f"string {i}" for i in range(10)])
        self.c = ak.Categorical(self.s)

    def test_mean(self):
        self.assertAlmostEqual(self.x.mean(), self.pdx.mean())
        self.assertAlmostEqual(self.y.mean(), self.pdy.mean())
        self.assertAlmostEqual(self.u.mean(), self.pdu.mean())
        self.assertAlmostEqual(self.b.mean(), self.pdb.mean())
        self.assertAlmostEqual(self.f.mean(), self.pdf.mean())

    def test_var(self):
        # Note the numpy and pandas var/std differ, we follow numpy by default
        self.assertAlmostEqual(self.x.var(), self.npx.var())
        self.assertAlmostEqual(self.y.var(), self.npy.var())
        self.assertAlmostEqual(self.u.var(), self.npu.var())
        self.assertAlmostEqual(self.b.var(), self.npb.var())
        self.assertAlmostEqual(self.f.var(), self.npf.var())

        # The pandas version requires ddof = 1
        self.assertAlmostEqual(self.x.var(ddof=1), self.pdx.var())
        self.assertAlmostEqual(self.y.var(ddof=1), self.pdy.var())
        self.assertAlmostEqual(self.u.var(ddof=1), self.pdu.var())
        self.assertAlmostEqual(self.b.var(ddof=1), self.pdb.var())
        self.assertAlmostEqual(self.f.var(ddof=1), self.pdf.var())

    def test_std(self):
        # Note the numpy and pandas var/std differ, we follow numpy by default
        self.assertAlmostEqual(self.x.std(), self.npx.std())
        self.assertAlmostEqual(self.y.std(), self.npy.std())
        self.assertAlmostEqual(self.u.std(), self.npu.std())
        self.assertAlmostEqual(self.b.std(), self.npb.std())
        self.assertAlmostEqual(self.f.std(), self.npf.std())

        # The pandas version requires ddof = 1
        self.assertAlmostEqual(self.x.std(ddof=1), self.pdx.std())
        self.assertAlmostEqual(self.y.std(ddof=1), self.pdy.std())
        self.assertAlmostEqual(self.u.std(ddof=1), self.pdu.std())
        self.assertAlmostEqual(self.b.std(ddof=1), self.pdb.std())
        self.assertAlmostEqual(self.f.std(ddof=1), self.pdf.std())

    def test_cov(self):
        # test that variations are equivalent
        self.assertAlmostEqual(self.x.cov(self.y), self.pdx.cov(self.pdy))
        self.assertAlmostEqual(self.x.cov(self.y), self.y.cov(self.x))
        self.assertAlmostEqual(self.x.cov(self.y), ak.cov(self.x, self.y))
        self.assertAlmostEqual(self.x.cov(self.y), ak.cov(self.y, self.x))

        # test int with other types
        self.assertAlmostEqual(self.x.cov(self.u), self.pdx.cov(self.pdu))
        self.assertAlmostEqual(self.x.cov(self.b), self.pdx.cov(self.pdb))
        self.assertAlmostEqual(self.x.cov(self.f), self.pdx.cov(self.pdf))

        # test bool with other types
        self.assertAlmostEqual(self.b.cov(self.b), self.pdb.cov(self.pdb))
        self.assertAlmostEqual(self.b.cov(self.u), self.pdb.cov(self.pdu))
        self.assertAlmostEqual(self.b.cov(self.f), self.pdb.cov(self.pdf))

        # test float with other types
        self.assertAlmostEqual(self.f.cov(self.f), self.pdf.cov(self.pdf))
        self.assertAlmostEqual(self.f.cov(self.u), self.pdf.cov(self.pdu))

        # test uint with self (other cases covered above)
        self.assertAlmostEqual(self.u.cov(self.u), self.pdu.cov(self.pdu))

    def test_corr(self):
        # test that variations are equivalent
        self.assertAlmostEqual(self.x.corr(self.y), self.pdx.corr(self.pdy))
        self.assertAlmostEqual(self.x.corr(self.y), self.y.corr(self.x))
        self.assertAlmostEqual(self.x.corr(self.y), ak.corr(self.x, self.y))
        self.assertAlmostEqual(self.x.corr(self.y), ak.corr(self.y, self.x))

        # test int with other types
        self.assertAlmostEqual(self.x.corr(self.u), self.pdx.corr(self.pdu))
        self.assertAlmostEqual(self.x.corr(self.b), self.pdx.corr(self.pdb))
        self.assertAlmostEqual(self.x.corr(self.f), self.pdx.corr(self.pdf))

        # test bool with other types
        self.assertAlmostEqual(self.b.corr(self.b), self.pdb.corr(self.pdb))
        self.assertAlmostEqual(self.b.corr(self.u), self.pdb.corr(self.pdu))
        self.assertAlmostEqual(self.b.corr(self.f), self.pdb.corr(self.pdf))

        # test float with other types
        self.assertAlmostEqual(self.f.corr(self.f), self.pdf.corr(self.pdf))
        self.assertAlmostEqual(self.f.corr(self.u), self.pdf.corr(self.pdu))

        # test uint with self (other cases covered above)
        self.assertAlmostEqual(self.u.corr(self.u), self.pdu.corr(self.pdu))

    def test_corr_matrix(self):
        ak_df = ak.DataFrame({"x": self.x, "y": self.y, "u": self.u, "b": self.b, "f": self.f}).corr()
        pd_df = pd.DataFrame(
            {"x": self.pdx, "y": self.pdy, "u": self.pdu, "b": self.pdb, "f": self.pdf}
        ).corr()

        # is there a better way to compare to pandas dataframe when the index doesn't match
        [self.assertTrue(np.allclose(ak_df[c].to_list(), pd_df[c].to_list())) for c in ak_df.columns]

        # verify this doesn't have scoping issues with numeric conversion
        ak.DataFrame(
            {"x": self.x, "y": self.y, "u": self.u, "b": self.b, "f": self.f, "s": self.s, "c": self.c}
        ).corr()

    def test_divmod(self):
        # vector-vector cases
        # int int
        ak_div, ak_mod = ak.divmod(self.x, self.y)
        np_div, np_mod = np.divmod(self.npx, self.npy)
        self.assertListEqual(ak_div.to_list(), np_div.tolist())
        self.assertListEqual(ak_mod.to_list(), np_mod.tolist())

        # int float
        ak_div, ak_mod = ak.divmod(self.x, ak.cast(self.y, ak.float64))
        np_div, np_mod = np.divmod(self.npx, self.npy.astype(float))
        self.assertListEqual(ak_div.to_list(), np_div.tolist())
        self.assertListEqual(ak_mod.to_list(), np_mod.tolist())

        # float int
        ak_div, ak_mod = ak.divmod(ak.cast(self.x, ak.float64), self.y)
        np_div, np_mod = np.divmod(self.npx.astype(float), self.npy)
        self.assertListEqual(ak_div.to_list(), np_div.tolist())
        self.assertListEqual(ak_mod.to_list(), np_mod.tolist())

        # float float
        ak_div, ak_mod = ak.divmod(ak.cast(self.x, ak.float64), ak.cast(self.y, ak.float64))
        np_div, np_mod = np.divmod(self.npx.astype(float), self.npy.astype(float))
        self.assertListEqual(ak_div.to_list(), np_div.tolist())
        self.assertListEqual(ak_mod.to_list(), np_mod.tolist())

        # float float (non-whole numbers)
        ak_div, ak_mod = ak.divmod(ak.cast(self.x + 0.5, ak.float64), ak.cast(self.y + 1.5, ak.float64))
        np_div, np_mod = np.divmod(self.npx.astype(float) + 0.5, self.npy.astype(float) + 1.5)
        self.assertListEqual(ak_div.to_list(), np_div.tolist())
        self.assertListEqual(ak_mod.to_list(), np_mod.tolist())

        # uint uint
        ak_div, ak_mod = ak.divmod(ak.arange(10, 20, dtype=ak.uint64), self.u)
        np_div, np_mod = np.divmod(np.arange(10, 20, dtype=np.uint64), self.npu)
        self.assertListEqual(ak_div.to_list(), np_div.tolist())
        self.assertListEqual(ak_mod.to_list(), np_mod.tolist())

        # scalar-vector cases
        # int int
        ak_div, ak_mod = ak.divmod(30, self.y)
        np_div, np_mod = np.divmod(30, self.npy)
        self.assertListEqual(ak_div.to_list(), np_div.tolist())
        self.assertListEqual(ak_mod.to_list(), np_mod.tolist())

        # int float
        ak_div, ak_mod = ak.divmod(30, ak.cast(self.y, ak.float64))
        np_div, np_mod = np.divmod(30, self.npy.astype(float))
        self.assertListEqual(ak_div.to_list(), np_div.tolist())
        self.assertListEqual(ak_mod.to_list(), np_mod.tolist())

        # float int
        ak_div, ak_mod = ak.divmod(30.0, self.y)
        np_div, np_mod = np.divmod(30.0, self.npy)
        self.assertListEqual(ak_div.to_list(), np_div.tolist())
        self.assertListEqual(ak_mod.to_list(), np_mod.tolist())

        # float float
        ak_div, ak_mod = ak.divmod(30.0, ak.cast(self.y, ak.float64))
        np_div, np_mod = np.divmod(30.0, self.npy.astype(float))
        self.assertListEqual(ak_div.to_list(), np_div.tolist())
        self.assertListEqual(ak_mod.to_list(), np_mod.tolist())

        # float float (non-whole numbers)
        ak_div, ak_mod = ak.divmod(30.5, ak.cast(self.y + 1.5, ak.float64))
        np_div, np_mod = np.divmod(30.5, self.npy.astype(float) + 1.5)
        self.assertListEqual(ak_div.to_list(), np_div.tolist())
        self.assertListEqual(ak_mod.to_list(), np_mod.tolist())

        # vector-scalar cases
        # int int
        ak_div, ak_mod = ak.divmod(self.x, 3)
        np_div, np_mod = np.divmod(self.npx, 3)
        self.assertListEqual(ak_div.to_list(), np_div.tolist())
        self.assertListEqual(ak_mod.to_list(), np_mod.tolist())

        # int float
        ak_div, ak_mod = ak.divmod(self.x, 3.0)
        np_div, np_mod = np.divmod(self.npx, 3.0)
        self.assertListEqual(ak_div.to_list(), np_div.tolist())
        self.assertListEqual(ak_mod.to_list(), np_mod.tolist())

        # float int
        ak_div, ak_mod = ak.divmod(ak.cast(self.x, ak.float64), 3)
        np_div, np_mod = np.divmod(self.npx.astype(float), 3)
        self.assertListEqual(ak_div.to_list(), np_div.tolist())
        self.assertListEqual(ak_mod.to_list(), np_mod.tolist())

        # float float
        ak_div, ak_mod = ak.divmod(ak.cast(self.x, ak.float64), 3.0)
        np_div, np_mod = np.divmod(self.npx.astype(float), 3.0)
        self.assertListEqual(ak_div.to_list(), np_div.tolist())
        self.assertListEqual(ak_mod.to_list(), np_mod.tolist())

        # float float (non-whole numbers)
        ak_div, ak_mod = ak.divmod(ak.cast(self.x + 0.5, ak.float64), 4.5)
        np_div, np_mod = np.divmod(self.npx.astype(float) + 0.5, 4.5)
        self.assertListEqual(ak_div.to_list(), np_div.tolist())
        self.assertListEqual(ak_mod.to_list(), np_mod.tolist())

        # Boolean where argument
        truth = ak.arange(10) % 2 == 0
        ak_div_truth, ak_mod_truth = ak.divmod(self.x, self.y, where=truth)
        self.assertListEqual(
            ak_div_truth.to_list(),
            [(self.x[i] // self.y[i]) if truth[i] else self.x[i] for i in range(10)],
        )
        self.assertListEqual(
            ak_mod_truth.to_list(),
            [(self.x[i] % self.y[i]) if truth[i] else self.x[i] for i in range(10)],
        )

        # Edge cases in the numerator
        edge_case = [-np.inf, -7.0, -0.0, np.nan, 0.0, 7.0, np.inf]
        np_edge_case = np.array(edge_case)
        ak_edge_case = ak.array(np_edge_case)
        np_ind = np.arange(1, len(edge_case) + 1)
        ak_ind = ak.arange(1, len(edge_case) + 1)
        ak_div, ak_mod = ak.divmod(ak_edge_case, ak_ind)
        np_div, np_mod = np.divmod(np_edge_case, np_ind)
        self.assertTrue(np.allclose(ak_div.to_ndarray(), np_div, equal_nan=True))
        self.assertTrue(np.allclose(ak_mod.to_ndarray(), np_mod, equal_nan=True))

        # Edge cases in the denominator
        edge_case = [-np.inf, -7.0, np.nan, 7.0, np.inf]
        np_edge_case = np.array(edge_case)
        ak_edge_case = ak.array(np_edge_case)
        np_ind = np.arange(1, len(edge_case)+1)
        ak_ind = ak.arange(1, len(edge_case)+1)
        ak_div, ak_mod = ak.divmod(ak_ind, ak_edge_case)
        np_div, np_mod = np.divmod(np_ind, np_edge_case)
        self.assertTrue(np.allclose(ak_div.to_ndarray(), np_div, equal_nan=True))
        self.assertTrue(np.allclose(ak_mod.to_ndarray(), np_mod, equal_nan=True))
