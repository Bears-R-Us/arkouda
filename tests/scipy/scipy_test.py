import numpy as np
import pytest
from scipy.stats import chisquare as scipy_chisquare
from scipy.stats import power_divergence as scipy_power_divergence

import arkouda as ak
from arkouda.scipy import chisquare as ak_chisquare
from arkouda.scipy import power_divergence as ak_power_divergence

DDOF = [0, 1, 2, 3, 4, 5]
PAIRS = [
    (
        np.array([10000000, 20000000, 30000000, 40000000, 50000000, 60000000, 70000000]),
        np.array([10000000, 20000000, 30000000, 40000001, 50000000, 60000000, 70000000]),
    ),
    (np.array([10000000, 20000000, 30000000, 40000000, 50000000, 60000000, 70000000]), None),
    (np.array([44, 24, 29, 3]) / 100 * 189, np.array([43, 52, 54, 40])),
]


class TestStats:
    @pytest.mark.parametrize(
        "lambda_",
        [
            "pearson",
            "log-likelihood",
            "freeman-tukey",
            "mod-log-likelihood",
            "neyman",
            "cressie-read",
        ],
    )
    @pytest.mark.parametrize("ddof", DDOF)
    @pytest.mark.parametrize("pair", PAIRS)
    def test_power_divergence(self, lambda_, ddof, pair):
        np_f_obs, np_f_exp = pair
        f_obs = ak.array(np_f_obs)
        f_exp = ak.array(np_f_exp) if np_f_exp is not None else None

        ak_power_div = ak_power_divergence(f_obs, f_exp, ddof=ddof, lambda_=lambda_)
        scipy_power_div = scipy_power_divergence(np_f_obs, np_f_exp, ddof=ddof, axis=0, lambda_=lambda_)

        assert np.allclose(ak_power_div, scipy_power_div, equal_nan=True)

    @pytest.mark.parametrize("ddof", DDOF)
    @pytest.mark.parametrize("pair", PAIRS)
    def test_chisquare(self, ddof, pair):
        np_f_obs, np_f_exp = pair
        f_obs = ak.array(np_f_obs)
        f_exp = ak.array(np_f_exp) if np_f_exp is not None else None

        ak_chisq = ak_chisquare(f_obs, f_exp, ddof=ddof)
        scipy_chisq = scipy_chisquare(np_f_obs, np_f_exp, ddof=ddof, axis=0)

        assert np.allclose(ak_chisq, scipy_chisq, equal_nan=True)
