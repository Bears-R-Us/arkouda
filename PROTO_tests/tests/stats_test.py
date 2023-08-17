import numpy as np
import pandas as pd
import arkouda as ak
import pytest


class TestStats():
    def setup_class(cls):
        cls.x = ak.arange(10, 20)
        cls.npx = np.arange(10, 20)
        cls.pdx = pd.Series(cls.npx)
        cls.y = ak.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
        cls.npy = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
        cls.pdy = pd.Series(cls.npy)
        cls.u = ak.cast(cls.y, ak.uint64)
        cls.npu = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48], dtype=np.uint64)
        cls.pdu = pd.Series(cls.npu)
        cls.b = ak.arange(10) % 2 == 0
        cls.npb = np.arange(10) % 2 == 0
        cls.pdb = pd.Series(cls.npb)
        cls.f = ak.array([i**2 for i in range(10)], dtype=ak.float64)
        cls.npf = np.array([i**2 for i in range(10)], dtype=np.float64)
        cls.pdf = pd.Series(cls.npf)
        cls.s = ak.array([f"string {i}" for i in range(10)])
        cls.c = ak.Categorical(cls.s)
        cls.arks = [cls.x, cls.y, cls.u, cls.b, cls.f]
        cls.npys = [cls.npx, cls.npy, cls.npu, cls.npb, cls.npf]
        cls.pands = [cls.pdx, cls.pdy, cls.pdu, cls.pdb, cls.pdf]

    def test_mean_var_and_std(self):
        for ark, npy in zip(self.arks, self.npys):
            assert ark.var() == pytest.approx(npy.var())
            assert ark.std() == pytest.approx(npy.std())

        for ark, pand in zip(self.arks, self.pands):
            assert ark.mean() == pytest.approx(pand.mean())
            assert ark.var(ddof=1) == pytest.approx(pand.var())
            assert ark.std(ddof=1) == pytest.approx(pand.std())

    def test_cov(self):
        # test that variations are equivalent
        for var in [
            (self.pdx.cov(self.pdy)),
            (self.y.cov(self.x)),
            (ak.cov(self.x, self.y)),
            (ak.cov(self.y, self.x)),
        ]:
            assert self.x.cov(self.y) == pytest.approx(var)

        # test int with other types
        assert self.x.cov(self.u) == pytest.approx(self.pdx.cov(self.pdu))
        assert self.x.cov(self.b) == pytest.approx(self.pdx.cov(self.pdb))
        assert self.x.cov(self.f) == pytest.approx(self.pdx.cov(self.pdf))

        # test bool with other types
        assert self.b.cov(self.b) == pytest.approx(self.pdb.cov(self.pdb))
        assert self.b.cov(self.u) == pytest.approx(self.pdb.cov(self.pdu))
        assert self.b.cov(self.f) == pytest.approx(self.pdb.cov(self.pdf))

        # test float with other types
        assert self.f.cov(self.f) == pytest.approx(self.pdf.cov(self.pdf))
        assert self.f.cov(self.u) == pytest.approx(self.pdf.cov(self.pdu))

        # test uint with self (other cases covered above)
        assert self.u.cov(self.u) == pytest.approx(self.pdu.cov(self.pdu))

    def test_corr(self):
        # test that variations are equivalent
        for var in [
            (self.pdx.corr(self.pdy)),
            (self.y.corr(self.x)),
            (ak.corr(self.x, self.y)),
            (ak.corr(self.y, self.x)),
        ]:
            assert self.x.corr(self.y) == pytest.approx(var)

        # test int with other types
        assert self.x.corr(self.u) == pytest.approx(self.pdx.corr(self.pdu))
        assert self.x.corr(self.b) == pytest.approx(self.pdx.corr(self.pdb))
        assert self.x.corr(self.f) == pytest.approx(self.pdx.corr(self.pdf))

        # test bool with other types
        assert self.b.corr(self.b) == pytest.approx(self.pdb.corr(self.pdb))
        assert self.b.corr(self.u) == pytest.approx(self.pdb.corr(self.pdu))
        assert self.b.corr(self.f) == pytest.approx(self.pdb.corr(self.pdf))

        # test float with other types
        assert self.f.corr(self.f) == pytest.approx(self.pdf.corr(self.pdf))
        assert self.f.corr(self.u) == pytest.approx(self.pdf.corr(self.pdu))

        # test uint with self (other cases covered above)
        assert self.u.corr(self.u) == pytest.approx(self.pdu.corr(self.pdu))

    def test_corr_matrix(self):
        ak_df = ak.DataFrame({"x": self.x, "y": self.y, "u": self.u, "b": self.b, "f": self.f}).corr()
        pd_df = pd.DataFrame(
            {"x": self.pdx, "y": self.pdy, "u": self.pdu, "b": self.pdb, "f": self.pdf}
        ).corr()

        # is there a better way to compare to pandas dataframe when the index doesn't match
        for c in ak_df.columns:
            assert (np.allclose(ak_df[c].to_list(), pd_df[c].to_list()))

        # verify this doesn't have scoping issues with numeric conversion
        ak.DataFrame(
            {"x": self.x, "y": self.y, "u": self.u, "b": self.b, "f": self.f, "s": self.s, "c": self.c}
        ).corr()

    def test_divmod(self):
        # vector-vector cases
        # int int
        ak_div, ak_mod = ak.divmod(self.x, self.y)
        np_div, np_mod = np.divmod(self.npx, self.npy)
        assert (ak_div.to_list() == np_div.tolist())
        assert (ak_mod.to_list() == np_mod.tolist())

        # int float
        ak_div, ak_mod = ak.divmod(self.x, ak.cast(self.y, ak.float64))
        np_div, np_mod = np.divmod(self.npx, self.npy.astype(float))
        assert (ak_div.to_list() == np_div.tolist())
        assert (ak_mod.to_list() == np_mod.tolist())

        # float int
        ak_div, ak_mod = ak.divmod(ak.cast(self.x, ak.float64), self.y)
        np_div, np_mod = np.divmod(self.npx.astype(float), self.npy)
        assert (ak_div.to_list() == np_div.tolist())
        assert (ak_mod.to_list() == np_mod.tolist())

        # float float
        ak_div, ak_mod = ak.divmod(ak.cast(self.x, ak.float64), ak.cast(self.y, ak.float64))
        np_div, np_mod = np.divmod(self.npx.astype(float), self.npy.astype(float))
        assert (ak_div.to_list() == np_div.tolist())
        assert (ak_mod.to_list() == np_mod.tolist())

        # float float (non-whole numbers)
        ak_div, ak_mod = ak.divmod(ak.cast(self.x + 0.5, ak.float64), ak.cast(self.y + 1.5, ak.float64))
        np_div, np_mod = np.divmod(self.npx.astype(float) + 0.5, self.npy.astype(float) + 1.5)
        assert (ak_div.to_list() == np_div.tolist())
        assert (ak_mod.to_list() == np_mod.tolist())

        # uint uint
        ak_div, ak_mod = ak.divmod(ak.arange(10, 20, dtype=ak.uint64), self.u)
        np_div, np_mod = np.divmod(np.arange(10, 20, dtype=np.uint64), self.npu)
        assert (ak_div.to_list() == np_div.tolist())
        assert (ak_mod.to_list() == np_mod.tolist())

        # scalar-vector cases
        # int int
        ak_div, ak_mod = ak.divmod(30, self.y)
        np_div, np_mod = np.divmod(30, self.npy)
        assert (ak_div.to_list() == np_div.tolist())
        assert (ak_mod.to_list() == np_mod.tolist())

        # int float
        ak_div, ak_mod = ak.divmod(30, ak.cast(self.y, ak.float64))
        np_div, np_mod = np.divmod(30, self.npy.astype(float))
        assert (ak_div.to_list() == np_div.tolist())
        assert (ak_mod.to_list() == np_mod.tolist())

        # float int
        ak_div, ak_mod = ak.divmod(30.0, self.y)
        np_div, np_mod = np.divmod(30.0, self.npy)
        assert (ak_div.to_list() == np_div.tolist())
        assert (ak_mod.to_list() == np_mod.tolist())

        # float float
        ak_div, ak_mod = ak.divmod(30.0, ak.cast(self.y, ak.float64))
        np_div, np_mod = np.divmod(30.0, self.npy.astype(float))
        assert (ak_div.to_list() == np_div.tolist())
        assert (ak_mod.to_list() == np_mod.tolist())

        # float float (non-whole numbers)
        ak_div, ak_mod = ak.divmod(30.5, ak.cast(self.y + 1.5, ak.float64))
        np_div, np_mod = np.divmod(30.5, self.npy.astype(float) + 1.5)
        assert (ak_div.to_list() == np_div.tolist())
        assert (ak_mod.to_list() == np_mod.tolist())

        # vector-scalar cases
        # int int
        ak_div, ak_mod = ak.divmod(self.x, 3)
        np_div, np_mod = np.divmod(self.npx, 3)
        assert (ak_div.to_list() == np_div.tolist())
        assert (ak_mod.to_list() == np_mod.tolist())

        # int float
        ak_div, ak_mod = ak.divmod(self.x, 3.0)
        np_div, np_mod = np.divmod(self.npx, 3.0)
        assert (ak_div.to_list() == np_div.tolist())
        assert (ak_mod.to_list() == np_mod.tolist())

        # float int
        ak_div, ak_mod = ak.divmod(ak.cast(self.x, ak.float64), 3)
        np_div, np_mod = np.divmod(self.npx.astype(float), 3)
        assert (ak_div.to_list() == np_div.tolist())
        assert (ak_mod.to_list() == np_mod.tolist())

        # float float
        ak_div, ak_mod = ak.divmod(ak.cast(self.x, ak.float64), 3.0)
        np_div, np_mod = np.divmod(self.npx.astype(float), 3.0)
        assert (ak_div.to_list() == np_div.tolist())
        assert (ak_mod.to_list() == np_mod.tolist())

        # float float (non-whole numbers)
        ak_div, ak_mod = ak.divmod(ak.cast(self.x + 0.5, ak.float64), 4.5)
        np_div, np_mod = np.divmod(self.npx.astype(float) + 0.5, 4.5)
        assert (ak_div.to_list() == np_div.tolist())
        assert (ak_mod.to_list() == np_mod.tolist())

        # Boolean where argument
        truth = ak.arange(10) % 2 == 0
        ak_div_truth, ak_mod_truth = ak.divmod(self.x, self.y, where=truth)
        assert (
            ak_div_truth.to_list(),
            [(self.x[i] // self.y[i]) if truth[i] else self.x[i] for i in range(10)],
        )
        assert (
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
        assert (np.allclose(ak_div.to_ndarray(), np_div, equal_nan=True))
        assert (np.allclose(ak_mod.to_ndarray(), np_mod, equal_nan=True))

        # Edge cases in the denominator
        edge_case = [-np.inf, -7.0, np.nan, 7.0, np.inf]
        np_edge_case = np.array(edge_case)
        ak_edge_case = ak.array(np_edge_case)
        np_ind = np.arange(1, len(edge_case)+1)
        ak_ind = ak.arange(1, len(edge_case)+1)
        ak_div, ak_mod = ak.divmod(ak_ind, ak_edge_case)
        np_div, np_mod = np.divmod(np_ind, np_edge_case)
        assert (np.allclose(ak_div.to_ndarray(), np_div, equal_nan=True))
        assert (np.allclose(ak_mod.to_ndarray(), np_mod, equal_nan=True))
