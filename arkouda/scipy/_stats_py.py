from collections import namedtuple

import numpy as np
from numpy import asarray
from scipy.stats import chi2  # type: ignore

import arkouda as ak
from arkouda.numpy import float64
from arkouda.numpy.dtypes import float64 as akfloat64
from arkouda.scipy.special import xlogy


__all__ = ["power_divergence", "chisquare", "Power_divergenceResult"]


class Power_divergenceResult(namedtuple("Power_divergenceResult", ("statistic", "pvalue"))):
    """
    The results of a power divergence statistical test.

    Attributes
    ----------
    statistic :    float64
    pvalue :    float64
    """

    statistic: float64
    pvalue: float64


# Map from names to lambda_ values used in power_divergence().
_power_div_lambda_names = {
    "pearson": 1,
    "log-likelihood": 0,
    "freeman-tukey": -0.5,
    "mod-log-likelihood": -1,
    "neyman": -2,
    "cressie-read": 2 / 3,
}


def power_divergence(f_obs, f_exp=None, ddof=0, lambda_=None):
    """
    Computes the power divergence statistic and p-value.

    Parameters
    ----------
    f_obs : pdarray
        The observed frequency.
    f_exp : pdarray, default = None
        The expected frequency.
    ddof : int
        The delta degrees of freedom.
    lambda_ : string, default = "pearson"
        The power in the Cressie-Read power divergence statistic.
        Allowed values: "pearson", "log-likelihood", "freeman-tukey", "mod-log-likelihood",
        "neyman", "cressie-read"

        Powers correspond as follows:

        "pearson": 1

        "log-likelihood": 0

        "freeman-tukey": -0.5

        "mod-log-likelihood": -1

        "neyman": -2

        "cressie-read": 2 / 3


    Returns
    -------
    arkouda.akstats.Power_divergenceResult

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda.scipy import power_divergence
    >>> x = ak.array([10, 20, 30, 10])
    >>> y = ak.array([10, 30, 20, 10])
    >>> power_divergence(x, y, lambda_="pearson")
    Power_divergenceResult(statistic=np.float64(8.333333333333334),
        pvalue=np.float64(0.03960235520756414))
    >>> power_divergence(x, y, lambda_="log-likelihood")
    Power_divergenceResult(statistic=np.float64(8.109302162163285),
        pvalue=np.float64(0.04380595350226197))

    See Also
    --------
    scipy.stats.power_divergence
    arkouda.akstats.chisquare

    Notes
    -----
    This is a modified version of scipy.stats.power_divergence [2]
    in order to scale using arkouda pdarrays.

    References
    ----------
        [1] "scipy.stats.power_divergence",
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.power_divergence.html

        [2] Scipy contributors (2024) scipy (Version v1.12.0) [Source code].
        https://github.com/scipy/scipy

    """
    if isinstance(lambda_, str):
        if lambda_ not in _power_div_lambda_names:
            names = repr(list(_power_div_lambda_names.keys()))[1:-1]
            raise ValueError(f"invalid string for lambda_: {lambda_!r}. Valid strings are {names}")
        lambda_ = _power_div_lambda_names[lambda_]
    elif lambda_ is None:
        lambda_ = 1

    f_obs_float = f_obs.astype(akfloat64)

    if f_exp is not None:
        rtol = 1e-8
        with np.errstate(invalid="ignore"):
            f_obs_sum = f_obs_float.sum()
            f_exp_sum = f_exp.sum()
            relative_diff = np.abs(f_obs_sum - f_exp_sum) / np.minimum(f_obs_sum, f_exp_sum)
            diff_gt_tol = (relative_diff > rtol).any()
        if diff_gt_tol:
            msg = (
                f"For each axis slice, the sum of the observed "
                f"frequencies must agree with the sum of the "
                f"expected frequencies to a relative tolerance "
                f"of {rtol}, but the percent differences are:\n"
                f"{relative_diff}"
            )
            raise ValueError(msg)

    else:
        # Ignore 'invalid' errors so the edge case of a data set with length 0
        # is handled without spurious warnings.
        with np.errstate(invalid="ignore"):
            f_exp = f_obs.mean()

    # `terms` is the array of terms that are summed along `axis` to create
    # the test statistic.  We use some specialized code for a few special
    # cases of lambda_.
    if lambda_ == 1:
        # Pearson's chi-squared statistic
        terms = (f_obs_float - f_exp) ** 2 / f_exp
    elif lambda_ == 0:
        # Log-likelihood ratio (i.e. G-test)

        if f_exp is not None:
            terms = 2.0 * xlogy(f_obs, f_obs / f_exp)
        else:
            terms = ak.zeros_like(f_obs)
    elif lambda_ == -1:
        # Modified log-likelihood ratio
        if (f_obs is not None) and (f_exp is not None):
            terms = 2.0 * xlogy(f_exp, f_exp / f_obs)
        else:
            terms = ak.array([])

    else:
        # General Cressie-Read power divergence.
        terms = f_obs * ((f_obs / f_exp) ** lambda_ - 1)
        terms /= 0.5 * lambda_ * (lambda_ + 1)

    stat = terms.sum()
    num_obs = terms.size
    ddof = asarray(ddof)
    p = chi2.sf(stat, num_obs - 1 - ddof)

    return Power_divergenceResult(stat, p)


def chisquare(f_obs, f_exp=None, ddof=0):
    """
    Computes the chi square statistic and p-value.

    Parameters
    ----------
    f_obs : pdarray
        The observed frequency.
    f_exp : pdarray, default = None
        The expected frequency.
    ddof : int
        The delta degrees of freedom.

    Returns
    -------
    arkouda.akstats.Power_divergenceResult

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda.scipy import chisquare
    >>> chisquare(ak.array([10, 20, 30, 10]), ak.array([10, 30, 20, 10]))
    Power_divergenceResult(statistic=np.float64(8.333333333333334),
        pvalue=np.float64(0.03960235520756414))

    See Also
    --------
    scipy.stats.chisquare
    arkouda.akstats.power_divergence

    References
    ----------
        [1] “Chi-squared test”, https://en.wikipedia.org/wiki/Chi-squared_test

        [2] "scipy.stats.chisquare",
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html

    """
    return power_divergence(f_obs, f_exp=f_exp, ddof=ddof, lambda_="pearson")
