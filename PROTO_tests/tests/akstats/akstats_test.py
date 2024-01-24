import math

from scipy.stats import power_divergence as scipy_power_divergence

import arkouda as ak
from arkouda.stats import power_divergence as ak_power_divergence


class TestStats:
    @staticmethod
    def create_stat_test_pairs():
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

                    if math.isnan(ak_power_div.statistic):
                        assert math.isnan(scipy_power_div.statistic)
                    else:
                        assert abs(ak_power_div.statistic - scipy_power_div.statistic) < 0.1 / 10**6

                    if math.isnan(ak_power_div.pvalue):
                        assert math.isnan(scipy_power_div.pvalue)
                    else:
                        assert abs(ak_power_div.pvalue - scipy_power_div.pvalue) < 0.1 / 10**6

    def test_chisquare(self):
        from scipy.stats import chisquare as scipy_chisquare

        from arkouda.stats import chisquare as ak_chisquare

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

                if math.isnan(ak_chisq.statistic):
                    assert math.isnan(scipy_chisq.statistic)
                else:
                    assert abs(ak_chisq.statistic - scipy_chisq.statistic) < 0.1 / 10**6

                if math.isnan(ak_chisq.pvalue):
                    assert math.isnan(scipy_chisq.pvalue)
                else:
                    assert abs(ak_chisq.pvalue - scipy_chisq.pvalue) < 0.1 / 10**6
