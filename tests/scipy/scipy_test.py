import numpy as np
import pytest

from scipy.stats import chisquare as scipy_chisquare
from scipy.stats import power_divergence as scipy_power_divergence

import arkouda as ak

from arkouda.client import get_array_ranks, get_max_array_rank
from arkouda.scipy import chisquare as ak_chisquare
from arkouda.scipy import power_divergence as ak_power_divergence


DDOF = [0, 1, 2, 3, 4, 5]
PAIRS = [
    (
        np.array([10000000, 20000000, 30000000, 40000000, 50000000, 60000000, 70000000]),
        np.array([10000000, 20000000, 30000000, 40000001, 50000000, 60000000, 70000000]),
    ),
    (
        np.array([10000000, 20000000, 30000000, 40000000, 50000000, 60000000, 70000000]),
        None,
    ),
    (np.array([44, 24, 29, 3]) / 100 * 189, np.array([43, 52, 54, 40])),
]

#  bumpup is useful for multi-dimensional testing.
#  Given an array of shape(m1,m2,...m), it will broadcast it to shape (2,m1,m2,...,m).
#  Called iteratively, it will broadcast 1D to 2D to 3D to 4D to ...


def bumpup(a):
    if a.ndim == 1:
        blob = (2, a.size)
    else:
        blob = list(a.shape)
        blob.insert(0, 2)
        blob = tuple(blob)
    return np.broadcast_to(a, blob)


class TestStats:
    from arkouda.scipy import _stats_py as stats

    def test_scipy_stats_py_docstrings(self):
        import doctest

        from arkouda.scipy import _stats_py as stats

        result = doctest.testmod(stats, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.skip_if_scipy_version_greater_than("1.13.1")
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

        # for the potential multi-dimensional case, create higher dim versions of the above
        # pairs.
        if get_max_array_rank() > 1:
            for n in range(2, get_max_array_rank() + 1):
                np_f_obs = np.ascontiguousarray(bumpup(np_f_obs))  #
                if np_f_exp is not None:  # conversion to pdarray
                    np_f_exp = np.ascontiguousarray(bumpup(np_f_exp))
                # Note the the "bumpup" is done whether or not this rank is in get_array_ranks
                # so that the rank at each iteration will be correct.
                # But the test is only applied for ranks that are in get_array_ranks
                if n in get_array_ranks():
                    f_obs = ak.array(np_f_obs)
                    f_exp = ak.array(np_f_exp) if np_f_exp is not None else None
                    # ak_power_div does not have an axis arg, so the comparison is made
                    # to scipy_power_div with axis=None
                    ak_power_div = ak_power_divergence(f_obs, f_exp, ddof=ddof, lambda_=lambda_)
                    scipy_power_div = scipy_power_divergence(
                        np_f_obs, np_f_exp, ddof=ddof, axis=None, lambda_=lambda_
                    )

                    assert np.allclose(ak_power_div, scipy_power_div, equal_nan=True)

    @pytest.mark.skip_if_scipy_version_greater_than("1.13.1")
    @pytest.mark.parametrize("ddof", DDOF)
    @pytest.mark.parametrize("pair", PAIRS)
    def test_chisquare(self, ddof, pair):
        np_f_obs, np_f_exp = pair
        f_obs = ak.array(np_f_obs)
        f_exp = ak.array(np_f_exp) if np_f_exp is not None else None

        ak_chisq = ak_chisquare(f_obs, f_exp, ddof=ddof)
        scipy_chisq = scipy_chisquare(np_f_obs, np_f_exp, ddof=ddof, axis=0)

        assert np.allclose(ak_chisq, scipy_chisq, equal_nan=True)

        # for the potential multi-dimensional case, create higher dim versions of the above
        # pairs.
        if get_max_array_rank() > 1:
            for n in range(2, get_max_array_rank() + 1):
                np_f_obs = np.ascontiguousarray(
                    bumpup(np_f_obs)
                )  # contiguous is needed for the conversion to pdarray
                if np_f_exp is not None:
                    np_f_exp = np.ascontiguousarray(bumpup(np_f_exp))
                # Note the the "bumpup" is done whether or not this rank is in get_array_ranks
                # so that the rank at each iteration will be correct.
                # But the test is only applied for ranks that are in get_array_ranks
                if n in get_array_ranks():
                    f_obs = ak.array(np_f_obs)
                    f_exp = ak.array(np_f_exp) if np_f_exp is not None else None
                    # ak_chisq does not have an axis arg, so the comparison is made
                    # to scipy_chisq with axis=None
                    ak_chisq = ak_chisquare(f_obs, f_exp, ddof=ddof)
                    scipy_chisq = scipy_chisquare(np_f_obs, np_f_exp, ddof=ddof, axis=None)

                    assert np.allclose(ak_chisq, scipy_chisq, equal_nan=True)
