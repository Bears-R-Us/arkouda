import math

import numpy as np
from base_test import ArkoudaTest
from context import arkouda as ak


class StatsTest(ArkoudaTest):
    def setUp(self):
        ArkoudaTest.setUp(self)

    def create_stat_test_pairs(self):
        pairs = [
            (
                ak.array([10000000, 20000000, 30000000, 40000000, 50000000, 60000000, 70000000]),
                ak.array([10000000, 20000000, 30000000, 40000001, 50000000, 60000000, 70000000]),
            ),
            (ak.array([10000000, 20000000, 30000000, 40000000, 50000000, 60000000, 70000000]), None),
            (ak.array([44, 24, 29, 3]) / 100 * 189, ak.array([43, 52, 54, 40])),
        ]
        return pairs

    def test_power_divergence(self):
        from scipy.stats import power_divergence as scipy_power_divergence

        from arkouda.scipy import power_divergence as ak_power_divergence

        pairs = self.create_stat_test_pairs()

        lambdas = [
            "pearson",
            "log-likelihood",
            "freeman-tukey",
            "mod-log-likelihood",
            "neyman",
            "cressie-read",
        ]

        ddofs = [0, 1, 2, 3, 4, 5]

        for f_obs, f_exp in pairs:
            for lambda0 in lambdas:
                for ddof in ddofs:
                    ak_power_div = ak_power_divergence(f_obs, f_exp, ddof=ddof, lambda_=lambda0)

                    np_f_obs = f_obs.to_ndarray()
                    np_f_exp = None
                    if f_exp is not None:
                        np_f_exp = f_exp.to_ndarray()

                    scipy_power_div = scipy_power_divergence(
                        np_f_obs, np_f_exp, ddof=ddof, axis=0, lambda_=lambda0
                    )

                    assert np.allclose(ak_power_div, scipy_power_div, equal_nan=True)

    def test_chisquare(self):
        from scipy.stats import chisquare as scipy_chisquare

        from arkouda.scipy import chisquare as ak_chisquare

        pairs = self.create_stat_test_pairs()

        ddofs = [0, 1, 2, 3, 4, 5]

        for f_obs, f_exp in pairs:
            for ddof in ddofs:
                ak_chisq = ak_chisquare(f_obs, f_exp, ddof=ddof)

                np_f_obs = f_obs.to_ndarray()
                np_f_exp = None
                if f_exp is not None:
                    np_f_exp = f_exp.to_ndarray()

                scipy_chisq = scipy_chisquare(np_f_obs, np_f_exp, ddof=ddof, axis=0)

                assert np.allclose(ak_chisq, scipy_chisq, equal_nan=True)
