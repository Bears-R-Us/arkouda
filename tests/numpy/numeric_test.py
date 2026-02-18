import random
import warnings
from math import isclose, prod, sqrt

import numpy as np
import pytest

import arkouda as ak
from arkouda.client import get_array_ranks, get_max_array_rank
from arkouda.numpy.dtypes import dtype as akdtype
from arkouda.numpy.dtypes import str_
from arkouda.testing import assert_almost_equivalent as ak_assert_almost_equivalent
from arkouda.testing import assert_arkouda_array_equivalent

ARRAY_TYPES = [ak.int64, ak.float64, ak.bool_, ak.uint64, str_]
NUMERIC_TYPES = [ak.int64, ak.float64, ak.bool_, ak.uint64]
NO_BOOL = [ak.int64, ak.float64, ak.uint64]
NO_FLOAT = [ak.int64, ak.bool_, ak.uint64]
INT_FLOAT = [ak.int64, ak.float64]
INT_FLOAT_BOOL = [ak.int64, ak.float64, ak.bool_]
YES_NO = [True, False]
VOWELS_AND_SUCH = ["a", "e", "i", "o", "u", "AB", 47, 2, 3.14159]

ALLOWED_PERQUANT_METHODS = [
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "linear",
    "weibull",
    "hazen",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "midpoint",
    "higher",
]  # not supporting 'nearest' at present

#  The subset of methods is used when doing multi-locale testing during development

SUBSET_PERQUANT_METHODS = ["linear", "midpoint"]  # one continuous, one discontinuous

ALLOWED_PUTMASK_PAIRS = [
    (ak.float64, ak.float64),
    (ak.float64, ak.int64),
    (ak.float64, ak.uint64),
    (ak.float64, ak.bool_),
    (ak.int64, ak.int64),
    (ak.int64, ak.bool_),
    (ak.uint64, ak.uint64),
    (ak.uint64, ak.bool_),
    (ak.bool_, ak.bool_),
]

# There are many ways to create a vector of alternating values.
# This is a fairly fast and fairly straightforward approach.


def alternate(L, R, n):
    v = np.full(n, R)
    v[::2] = L
    return v


#  The following tuples support a simplification of the trigonometric
#  and hyperbolic testing.

TRIGONOMETRICS = (
    (np.sin, ak.sin),
    (np.cos, ak.cos),
    (np.tan, ak.tan),
    (np.arcsin, ak.arcsin),
    (np.arccos, ak.arccos),
    (np.arctan, ak.arctan),
)

HYPERBOLICS = (
    (np.sinh, ak.sinh),
    (np.cosh, ak.cosh),
    (np.tanh, ak.tanh),
    (np.arcsinh, ak.arcsinh),
    (np.arccosh, ak.arccosh),
    (np.arctanh, ak.arctanh),
)

INFINITY_EDGE_CASES = (
    (np.arctan, ak.arctan),
    (np.sinh, ak.sinh),
    (np.cosh, ak.cosh),
    (np.arcsinh, ak.arcsinh),
    (np.arccosh, ak.arccosh),
)

# as noted in registration-config.json, only these types are supported

SUPPORTED_TYPES = [ak.bool_, ak.uint64, ak.int64, ak.bigint, ak.uint8, ak.float64]


NP_TRIG_ARRAYS = {
    ak.int64: np.arange(-5, 5),
    ak.float64: np.concatenate(
        [
            np.linspace(-3.5, 3.5, 5),
            np.array([np.nan, -np.inf, -0.0, 0.0, np.inf]),
        ]
    ),
    ak.bool_: alternate(True, False, 10),
    ak.uint64: np.arange(2**64 - 10, 2**64, dtype=np.uint64),
}

DENOM_ARCTAN2_ARRAYS = {
    ak.int64: np.concatenate((np.arange(-5, 0), np.arange(1, 6))),
    ak.float64: np.concatenate(
        [
            np.linspace(-3.4, 3.5, 5),
            np.array([np.nan, -np.inf, -1.0, 1.0, np.inf]),
        ]
    ),
    ak.uint64: np.arange(2**64 - 10, 2**64, dtype=np.uint64),
}

ROUNDTRIP_CAST = [
    (ak.bool_, ak.bool_),
    (ak.int64, ak.int64),
    (ak.int64, ak.float64),
    (ak.int64, ak.str_),
    (ak.float64, ak.float64),
    (ak.float64, ak.str_),
    (ak.uint8, ak.int64),
    (ak.uint8, ak.float64),
    (ak.uint8, ak.str_),
]

#  Most of the trigonometric and hyperbolic tests are identical, so they are combined
#  into this helper utility.

#  Some of the tests trigger overflow, invalid value, or divide by zero warnings.
#  We use np.errstate to ignore those, because that's not what we're testing.  We're
#  only testing that numpy's and arkouda's results match.
#  To restore those warnings, comment out all of the lines below that invoke np.seterr.


def _trig_and_hyp_test_helper(np_func, na, ak_func, pda):
    old_settings = np.seterr(all="ignore")  # retrieve current settings
    np.seterr(over="ignore", invalid="ignore", divide="ignore")
    assert np.allclose(np_func(na), ak_func(pda).to_ndarray(), equal_nan=True)
    truth_np = alternate(True, False, len(na))
    truth_ak = ak.array(truth_np)
    assert np.allclose(np_func(na, where=True), ak_func(pda, where=True).to_ndarray(), equal_nan=True)
    assert np.allclose(na, ak_func(pda, where=False).to_ndarray(), equal_nan=True)
    assert np.allclose(
        [np_func(na[i]) if truth_np[i] else na[i] for i in range(len(na))],
        ak_func(pda, where=truth_ak).tolist(),
        equal_nan=True,
    )
    np.seterr(**old_settings)  # restore original settings


#  Similarly, the infinity case causes an invalid value in arccosh, and we don't need
#  to be told that. To restore the warnings, comment out the lines that invoke np.seterr.


def _infinity_edge_case_helper(np_func, ak_func):
    na = np.array([np.inf, -np.inf])
    pda = ak.array(na)
    old_settings = np.seterr(all="ignore")
    np.seterr(invalid="ignore")
    assert np.allclose(np_func(na), ak_func(pda).to_ndarray(), equal_nan=True)
    np.seterr(**old_settings)


class TestNumeric:
    @pytest.mark.skip_if_rank_not_compiled([1, 2, 3])
    def test_numeric_docstrings(self):
        import doctest

        from arkouda.numpy import numeric

        result = doctest.testmod(numeric, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.parametrize("numeric_type", NUMERIC_TYPES)
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_seeded_rng_typed(self, prob_size, numeric_type):
        seed = pytest.seed if pytest.seed is not None else 8675309

        # Make sure unseeded runs differ
        a = ak.randint(0, 2**32, prob_size, dtype=numeric_type)
        b = ak.randint(0, 2**32, prob_size, dtype=numeric_type)
        assert not (a == b).all()

        # Make sure seeded results are same
        a = ak.randint(0, 2**32, prob_size, dtype=numeric_type, seed=seed)
        b = ak.randint(0, 2**32, prob_size, dtype=numeric_type, seed=seed)
        assert (a == b).all()

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_seeded_rng_general(self, prob_size):
        seed = pytest.seed if pytest.seed is not None else 8675309
        # Uniform
        assert not (ak.uniform(prob_size) == ak.uniform(prob_size)).all()
        assert (ak.uniform(prob_size, seed=seed) == ak.uniform(prob_size, seed=seed)).all()

        # Standard Normal
        assert not (ak.standard_normal(prob_size) == ak.standard_normal(prob_size)).all()
        assert (
            ak.standard_normal(prob_size, seed=seed) == ak.standard_normal(prob_size, seed=seed)
        ).all()

        # Strings (uniformly distributed length)
        assert not (
            ak.random_strings_uniform(1, 10, prob_size) == ak.random_strings_uniform(1, 10, prob_size)
        ).all()

        assert (
            ak.random_strings_uniform(1, 10, prob_size, seed=seed)
            == ak.random_strings_uniform(1, 10, prob_size, seed=seed)
        ).all()

        # Strings (log-normally distributed length)
        assert not (
            ak.random_strings_lognormal(2, 1, prob_size) == ak.random_strings_lognormal(2, 1, prob_size)
        ).all()
        assert (
            ak.random_strings_lognormal(2, 1, prob_size, seed=seed)
            == ak.random_strings_lognormal(2, 1, prob_size, seed=seed)
        ).all()

    @pytest.mark.parametrize("cast_to", SUPPORTED_TYPES)
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_cast(self, prob_size, cast_to):
        seed = pytest.seed if pytest.seed is not None else 8675309
        arrays = {
            ak.int64: ak.randint(-(2**48), 2**48, prob_size, seed=seed),
            ak.uint64: ak.randint(0, 2**48, prob_size, dtype=ak.uint64, seed=seed + 1),
            ak.float64: ak.randint(0, 1, prob_size, dtype=ak.float64, seed=seed + 2),
            ak.bool_: ak.randint(0, 2, prob_size, dtype=ak.bool_, seed=seed + 3),
            ak.str_: ak.cast(ak.randint(0, 2**48, prob_size, seed=seed + 4), "str"),
        }

        for t1, orig in arrays.items():
            if (t1 == ak.float64 and cast_to == ak.bigint) or (t1 == ak.str_ and cast_to == ak.bool_):
                # we don't support casting a float to a bigint
                # we do support str to bool, but it's expected to contain "true/false" not numerics
                continue
            other = ak.cast(orig, cast_to)
            assert orig.size == other.size
            if (t1, cast_to) in ROUNDTRIP_CAST:
                roundtrip = ak.cast(other, t1)
                assert (orig == roundtrip).all()

    @pytest.mark.parametrize("num_type", NUMERIC_TYPES)
    def test_str_cast_errors(self, num_type):
        strarr = None
        ans = None
        if num_type == ak.int64:
            intNAN = -(2**63)
            strarr = ak.array(["1", "2 ", "3?", "!4", "  5", "-45", "0b101", "0x30", "N/A"])
            ans = np.array([1, 2, intNAN, intNAN, 5, -45, 0b101, 0x30, intNAN])
        elif num_type == ak.uint64:
            uintNAN = 0
            strarr = ak.array(["1", "2 ", "3?", "-4", "  5", "45", "0b101", "0x30", "N/A"])
            ans = np.array([1, 2, uintNAN, uintNAN, 5, 45, 0b101, 0x30, uintNAN])
        elif num_type == ak.float64:
            strarr = ak.array(
                [
                    "1.1",
                    "2.2 ",
                    "3?.3",
                    "4.!4",
                    "  5.5",
                    "6.6e-6",
                    "78.91E+4",
                    "6",
                    "N/A",
                ]
            )
            ans = np.array([1.1, 2.2, np.nan, np.nan, 5.5, 6.6e-6, 78.91e4, 6.0, np.nan])
        elif num_type == ak.bool_:
            strarr = ak.array(
                [
                    "True",
                    "False ",
                    "Neither",
                    "N/A",
                    "  True",
                    "true",
                    "false",
                    "TRUE",
                    "NOTTRUE",
                ]
            )
            ans = np.array([True, False, False, False, True, True, False, True, False])

        validans = ak.array([True, True, False, False, True, True, True, True, False])

        with pytest.raises(RuntimeError):
            ak.cast(strarr, num_type, errors=ak.ErrorMode.strict)
        res = ak.cast(strarr, num_type, errors=ak.ErrorMode.ignore)
        assert np.allclose(ans, res.to_ndarray(), equal_nan=True)
        res, valid = ak.cast(strarr, num_type, errors=ak.ErrorMode.return_validity)
        assert valid.tolist() == validans.tolist()
        assert np.allclose(ans, res.to_ndarray(), equal_nan=True)

    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_histogram(self, num_type):
        seed = pytest.seed if pytest.seed is not None else 8675309
        pda = ak.randint(10, 30, 40, dtype=num_type, seed=seed)
        result, bins = ak.histogram(pda, bins=20)

        assert isinstance(result, ak.pdarray)
        assert 21 == len(bins)
        assert 20 == len(result)
        assert int == result.dtype

        with pytest.raises(TypeError):
            ak.histogram(np.array([range(0, 10)]).astype(num_type), bins=1)

        with pytest.raises(TypeError):
            ak.histogram(pda, bins="1")

        with pytest.raises(TypeError):
            ak.histogram(np.array([range(0, 10)]).astype(num_type), bins="1")

        # add 'range'
        ak_result, ak_bins = ak.histogram(pda, bins=20, range=(15, 20))
        np_result, np_bins = np.histogram(np.array(pda.tolist()), bins=20, range=(15, 20))
        assert np.allclose(ak_result.tolist(), np_result.tolist())
        assert np.allclose(ak_bins.tolist(), np_bins.tolist())

    #   log and exp tests were identical, and so have been combined.

    @pytest.mark.skipif(pytest.client_host == "horizon", reason="Fails on horizon")
    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    @pytest.mark.parametrize("num_type1", NO_BOOL)
    @pytest.mark.parametrize("num_type2", NO_BOOL)
    def test_histogram_multidim(self, num_type1, num_type2):
        # test 2d histogram
        seed = pytest.seed if pytest.seed is not None else 8675309
        np.random.seed(seed)
        ak_x = ak.randint(1, 100, 1000, seed=seed, dtype=num_type1)
        ak_y = ak.randint(1, 100, 1000, seed=seed + 1, dtype=num_type2)
        np_x, np_y = ak_x.to_ndarray(), ak_y.to_ndarray()

        np_hist, np_x_edges, np_y_edges = np.histogram2d(np_x, np_y)
        ak_hist, ak_x_edges, ak_y_edges = ak.histogram2d(ak_x, ak_y)
        assert np.allclose(np_hist.tolist(), ak_hist.tolist())
        assert np.allclose(np_x_edges.tolist(), ak_x_edges.tolist())
        assert np.allclose(np_y_edges.tolist(), ak_y_edges.tolist())

        np_hist, np_x_edges, np_y_edges = np.histogram2d(np_x, np_y, bins=(10, 20))
        ak_hist, ak_x_edges, ak_y_edges = ak.histogram2d(ak_x, ak_y, bins=(10, 20))
        assert np.allclose(np_hist.tolist(), ak_hist.tolist())
        assert np.allclose(np_x_edges.tolist(), ak_x_edges.tolist())
        assert np.allclose(np_y_edges.tolist(), ak_y_edges.tolist())

        # add 'range'
        np_hist, np_x_edges, np_y_edges = np.histogram2d(np_x, np_y, range=((25, 92), (15, 86)))
        ak_hist, ak_x_edges, ak_y_edges = ak.histogram2d(ak_x, ak_y, range=((25, 92), (15, 86)))
        assert np.allclose(np_hist.tolist(), ak_hist.tolist())
        assert np.allclose(np_x_edges.tolist(), ak_x_edges.tolist())
        assert np.allclose(np_y_edges.tolist(), ak_y_edges.tolist())

        # test arbitrary dimensional histogram
        dim_list = [3, 4, 5]
        bin_list = [[2, 4, 5], [2, 4, 5, 2], [2, 4, 5, 2, 3]]
        for dim, bins in zip(dim_list, bin_list):
            if dim <= get_max_array_rank():
                np_arrs = [np.random.randint(1, 100, 1000) for _ in range(dim)]
                ak_arrs = [ak.array(a) for a in np_arrs]

                np_hist, np_bin_edges = np.histogramdd(np_arrs, bins=bins)
                ak_hist, ak_bin_edges = ak.histogramdd(ak_arrs, bins=bins)
                assert np.allclose(np_hist.tolist(), ak_hist.tolist())
                for np_edge, ak_edge in zip(np_bin_edges, ak_bin_edges):
                    assert np.allclose(np_edge.tolist(), ak_edge.tolist())

                # add 'range'
                range_arg = [(10, 80) for _ in range(dim)]
                np_hist, np_bin_edges = np.histogramdd(np_arrs, bins=bins, range=range_arg)
                ak_hist, ak_bin_edges = ak.histogramdd(ak_arrs, bins=bins, range=range_arg)
                assert np.allclose(np_hist.tolist(), ak_hist.tolist())
                for np_edge, ak_edge in zip(np_bin_edges, ak_bin_edges):
                    assert np.allclose(np_edge.tolist(), ak_edge.tolist())

    @pytest.mark.parametrize("num_type", NO_BOOL)
    @pytest.mark.parametrize("op", ["exp", "log", "expm1", "log2", "log10", "log1p"])
    def test_log_and_exp(self, num_type, op):
        na = np.linspace(1, 10, 10).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        akfunc = getattr(ak, op)
        npfunc = getattr(np, op)

        ak_assert_almost_equivalent(akfunc(pda), npfunc(na))

        with pytest.raises(TypeError):
            akfunc(np.array([range(0, 10)]).astype(num_type))

    @pytest.mark.parametrize("num_type", INT_FLOAT)
    def test_abs(self, num_type):
        na = np.linspace(1, 10, 10).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        assert np.allclose(np.abs(na), ak.abs(pda).to_ndarray())

        assert (
            ak.arange(5, 0, -1, dtype=num_type).tolist()
            == ak.abs(ak.arange(-5, 0, dtype=num_type)).tolist()
        )

        with pytest.raises(TypeError):
            ak.abs(np.array([range(0, 10)]).astype(num_type))

    @pytest.mark.parametrize("num_type", NO_BOOL)
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_square(self, prob_size, num_type):
        nda = np.arange(prob_size).astype(num_type)
        if num_type != ak.uint64:
            nda = nda - prob_size // 2
        pda = ak.array(nda)

        assert np.allclose(np.square(nda), ak.square(pda).to_ndarray())

        with pytest.raises(TypeError):
            ak.square(np.array([range(-10, 10)]).astype(ak.bool_))

    @pytest.mark.parametrize("num_type", INT_FLOAT)
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_sign(self, prob_size, num_type):
        nda = np.arange(prob_size).astype(num_type) - prob_size // 2
        pda = ak.array(nda)
        assert_arkouda_array_equivalent(np.sign(nda), ak.sign(pda))

    #   cumsum and cumprod tests were identical, and so have been combined.

    @pytest.mark.parametrize("num_type", NUMERIC_TYPES)
    def test_cumsum_and_cumprod(self, num_type):
        na = np.linspace(1, 10, 10).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        for npfunc, akfunc in ((np.cumsum, ak.cumsum), (np.cumprod, ak.cumprod)):
            assert np.allclose(npfunc(na), akfunc(pda).to_ndarray())
        with pytest.raises(TypeError):
            ak.cumsum(np.array([range(0, 10)]).astype(num_type))

    #   test_trig_and_hyp covers the testing for most trigonometric and hyperbolic
    #   functions.  The exception is arctan2.

    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_trig_and_hyp(self, num_type):
        for npfunc, akfunc in set(TRIGONOMETRICS + HYPERBOLICS):
            na = NP_TRIG_ARRAYS[num_type]
            pda = ak.array(na, dtype=num_type)
            _trig_and_hyp_test_helper(npfunc, na, akfunc, pda)
            if (npfunc, akfunc) in INFINITY_EDGE_CASES:
                _infinity_edge_case_helper(npfunc, akfunc)
            with pytest.raises(TypeError):
                akfunc(np.array([range(0, 10)]).astype(num_type))

    @pytest.mark.parametrize("num_type", NO_BOOL)
    @pytest.mark.parametrize("denom_type", NO_BOOL)
    def test_arctan2(self, num_type, denom_type):
        seed = pytest.seed if pytest.seed is not None else 8675309
        np.random.seed(seed)
        na_num = np.random.permutation(NP_TRIG_ARRAYS[num_type])
        na_denom = np.random.permutation(DENOM_ARCTAN2_ARRAYS[denom_type])

        pda_num = ak.array(na_num, dtype=num_type)
        pda_denom = ak.array(na_denom, dtype=denom_type)

        truth_np = alternate(True, False, len(na_num))
        truth_ak = ak.array(truth_np)

        assert np.allclose(
            np.arctan2(na_num, na_denom, where=True),
            ak.arctan2(pda_num, pda_denom, where=True).to_ndarray(),
            equal_nan=True,
        )

        assert np.allclose(
            np.arctan2(na_num[0], na_denom, where=True),
            ak.arctan2(pda_num[0], pda_denom, where=True).to_ndarray(),
            equal_nan=True,
        )
        assert np.allclose(
            np.arctan2(na_num, na_denom[0], where=True),
            ak.arctan2(pda_num, pda_denom[0], where=True).to_ndarray(),
            equal_nan=True,
        )

        assert np.allclose(
            na_num / na_denom,
            ak.arctan2(pda_num, pda_denom, where=False).tolist(),
            equal_nan=True,
        )
        assert np.allclose(
            na_num[0] / na_denom,
            ak.arctan2(pda_num[0], pda_denom, where=False).to_ndarray(),
            equal_nan=True,
        )
        assert np.allclose(
            na_num / na_denom[0],
            ak.arctan2(pda_num, pda_denom[0], where=False).to_ndarray(),
            equal_nan=True,
        )

        assert np.allclose(
            [
                (np.arctan2(na_num[i], na_denom[i]) if truth_np[i] else na_num[i] / na_denom[i])
                for i in range(len(na_num))
            ],
            ak.arctan2(pda_num, pda_denom, where=truth_ak).to_ndarray(),
            equal_nan=True,
        )
        assert np.allclose(
            [
                (np.arctan2(na_num[0], na_denom[i]) if truth_np[i] else na_num[0] / na_denom[i])
                for i in range(len(na_denom))
            ],
            ak.arctan2(pda_num[0], pda_denom, where=truth_ak).to_ndarray(),
            equal_nan=True,
        )
        assert np.allclose(
            [
                (np.arctan2(na_num[i], na_denom[0]) if truth_np[i] else na_num[i] / na_denom[0])
                for i in range(len(na_num))
            ],
            ak.arctan2(pda_num, pda_denom[0], where=truth_ak).to_ndarray(),
            equal_nan=True,
        )

        # Edge cases: infinities and zeros.  Doesn't use _infinity_edge_case_helper
        # because arctan2 needs two numbers (numerator and denominator) rather than one.

        na1 = np.array([np.inf, -np.inf])
        pda1 = ak.array(na1)
        na2 = np.array([1, 10])
        pda2 = ak.array(na2)

        assert np.allclose(
            np.arctan2(na1, na2),
            ak.arctan2(pda1, pda2).to_ndarray(),
            equal_nan=True,
        )

        assert np.allclose(
            np.arctan2(na2, na1),
            ak.arctan2(pda2, pda1).to_ndarray(),
            equal_nan=True,
        )
        assert np.allclose(np.arctan2(na1, 5), ak.arctan2(pda1, 5).to_ndarray(), equal_nan=True)
        assert np.allclose(np.arctan2(5, na1), ak.arctan2(5, pda1).to_ndarray(), equal_nan=True)
        assert np.allclose(np.arctan2(na1, 0), ak.arctan2(pda1, 0).to_ndarray(), equal_nan=True)
        assert np.allclose(np.arctan2(0, na1), ak.arctan2(0, pda1).to_ndarray(), equal_nan=True)

        with pytest.raises(TypeError):
            ak.arctan2(
                np.array([range(0, 10)]).astype(num_type),
                np.array([range(10, 20)]).astype(num_type),
            )
        with pytest.raises(TypeError):
            ak.arctan2(pda_num[0], np.array([range(10, 20)]).astype(num_type))
        with pytest.raises(TypeError):
            ak.arctan2(np.array([range(0, 10)]).astype(num_type), pda_denom[0])

    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_rad2deg(self, num_type):
        na = NP_TRIG_ARRAYS[num_type]
        pda = ak.array(na, dtype=num_type)
        _trig_and_hyp_test_helper(np.rad2deg, na, ak.rad2deg, pda)

        with pytest.raises(TypeError):
            ak.rad2deg(np.array([range(0, 10)]).astype(num_type))

    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_deg2rad(self, num_type):
        na = NP_TRIG_ARRAYS[num_type]
        pda = ak.array(na, dtype=num_type)
        _trig_and_hyp_test_helper(np.deg2rad, na, ak.deg2rad, pda)

        with pytest.raises(TypeError):
            ak.deg2rad(np.array([range(0, 10)]).astype(num_type))

    @pytest.mark.parametrize("num_type", NO_FLOAT)
    def test_value_counts(self, num_type):
        pda = ak.ones(100, dtype=num_type)
        result = ak.value_counts(pda)

        assert ak.array([1]) == result[0]
        assert ak.array([100]) == result[1]

    def test_value_counts_error(self):
        with pytest.raises(TypeError):
            ak.value_counts([0])

    def test_isnan(self):
        """
        Test isnan; it returns a pdarray of element-wise T/F values for whether it is NaN
        (not a number)
        """
        npa = np.array([1, 2, None, 3, 4], dtype="float64")
        ark_s_float64 = ak.array(npa)
        ark_isna_float64 = ak.isnan(ark_s_float64)
        actual = ark_isna_float64.to_ndarray()
        assert np.array_equal(np.isnan(npa), actual)

        ark_s_int64 = ak.array(np.array([1, 2, 3, 4], dtype="int64"))
        assert ak.isnan(ark_s_int64).tolist() == [False, False, False, False]

        ark_s_string = ak.array(["a", "b", "c"])
        with pytest.raises(TypeError):
            ak.isnan(ark_s_string)

    def test_isinf_isfinite(self):
        """
        Test isinf and isfinite.  These return pdarrays of T/F values as appropriate.
        """
        nda = np.array([0, 9999.9999])
        pda = ak.array(nda)
        warnings.filterwarnings("ignore")
        nda_blowup = np.exp(nda)
        warnings.filterwarnings("default")
        pda_blowup = ak.exp(pda)
        assert (np.isinf(nda_blowup) == ak.isinf(pda_blowup).to_ndarray()).all()
        assert (np.isfinite(nda_blowup) == ak.isfinite(pda_blowup).to_ndarray()).all()

    def test_str_cat_cast(self):
        test_strs = [
            ak.array([f"str {i}" for i in range(101)]),
            ak.array([f"str {i % 3}" for i in range(101)]),
        ]
        for test_str, test_cat in zip(test_strs, [ak.Categorical(s) for s in test_strs]):
            cast_str = ak.cast(test_cat, ak.Strings)
            assert (cast_str == test_str).all()
            cast_cat = ak.cast(test_str, ak.Categorical)
            assert (cast_cat == test_cat).all()

            assert isinstance(cast_str, ak.Strings)
            assert isinstance(cast_cat, ak.Categorical)

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_precision(self, prob_size):
        # See https://github.com/Bears-R-Us/arkouda/issues/964
        # Grouped sum was exacerbating floating point errors
        # This test verifies the fix
        seed = pytest.seed if pytest.seed is not None else 8675309
        G = prob_size // 10
        ub = 2**63 // prob_size
        groupnum = ak.randint(0, G, prob_size, seed=seed)
        intval = ak.randint(0, ub, prob_size, seed=seed + 1)
        floatval = ak.cast(intval, ak.float64)
        g = ak.GroupBy(groupnum)
        _, intmean = g.mean(intval)
        _, floatmean = g.mean(floatval)
        ak_mse = ak.mean((intmean - floatmean) ** 2)
        assert np.isclose(ak_mse, 0.0)

    def test_hash(self):
        h1, h2 = ak.hash(ak.arange(10))
        rev = ak.arange(9, -1, -1)
        h3, h4 = ak.hash(rev)
        assert h1.tolist() == h3[rev].tolist()
        assert h2.tolist() == h4[rev].tolist()

        h1 = ak.hash(ak.arange(10), full=False)
        h3 = ak.hash(rev, full=False)
        assert h1.tolist() == h3[rev].tolist()

        h = ak.hash(ak.linspace(0, 10, 10))
        assert h[0].dtype == ak.uint64
        assert h[1].dtype == ak.uint64

        # test strings hash
        s = ak.random_strings_uniform(4, 8, 10)
        h1, h2 = ak.hash(s)
        rh1, rh2 = ak.hash(s[rev])
        assert h1.tolist() == rh1[rev].tolist()
        assert h2.tolist() == rh2[rev].tolist()

        # verify all the ways to hash strings match
        h3, h4 = ak.hash([s])
        assert h1.tolist() == h3.tolist()
        assert h2.tolist() == h4.tolist()
        h5, h6 = s.hash()
        assert h1.tolist() == h5.tolist()
        assert h2.tolist() == h6.tolist()

        # test segarray hash with int and string values
        # along with strings, categorical, and pdarrays
        segs = ak.array([0, 3, 6, 9])
        vals = ak.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 5, 5, 5, 5])
        sa = ak.SegArray(segs, vals)
        str_vals = ak.array([f"str {i}" for i in vals.tolist()])
        str_sa = ak.SegArray(segs, str_vals)
        a = ak.array([-10, 4, -10, 17])
        bi = a + 2**200
        s = ak.array([f"str {i}" for i in a.tolist()])
        c = ak.Categorical(s)
        for h in [
            sa,
            str_sa,
            [sa, a],
            [str_sa, a],
            [str_sa, bi],
            [sa, str_sa],
            [sa, str_sa, c],
            [sa, bi, str_sa, c],
            [s, sa, str_sa],
            [str_sa, s, sa, a],
            [c, str_sa, s, sa, a],
            [bi, c, str_sa, s, sa, a],
        ]:
            h1, h2 = ak.hash(h)
            if isinstance(h, ak.SegArray):
                # verify all the ways to hash segarrays match
                h3, h4 = ak.hash([h])
                assert h1.tolist() == h3.tolist()
                assert h2.tolist() == h4.tolist()
                h5, h6 = h.hash()
                assert h1.tolist() == h5.tolist()
                assert h2.tolist() == h6.tolist()
            # the first and third position are identical and should hash to the same thing
            assert h1[0] == h1[2]
            assert h2[0] == h2[2]
            # make sure the last position didn't get zeroed out by XOR
            assert h1[3] != 0
            assert h2[3] != 0

        sa = ak.SegArray(ak.array([0, 2]), ak.array([1, 1, 2, 2]))
        h1, h2 = sa.hash()
        # verify these segments don't collide (this is why we rehash)
        assert h1[0] != h1[1]
        assert h2[0] != h2[1]

        # test categorical hash
        categories, codes = ak.array([f"str {i}" for i in range(3)]), ak.randint(0, 3, 10**5)
        my_cat = ak.Categorical.from_codes(codes=codes, categories=categories)
        h1, h2 = ak.hash(my_cat)
        rev = ak.arange(10**5)[::-1]
        rh1, rh2 = ak.hash(my_cat[rev])
        assert h1.tolist() == rh1[rev].tolist()
        assert h2.tolist() == rh2[rev].tolist()

        # verify all the ways to hash Categoricals match
        h3, h4 = ak.hash([my_cat])
        assert h1.tolist() == h3.tolist()
        assert h2.tolist() == h4.tolist()
        h5, h6 = my_cat.hash()
        assert h1.tolist() == h5.tolist()
        assert h2.tolist() == h6.tolist()

        # verify it matches hashing the categories and then indexing with codes
        sh1, sh2 = my_cat.categories.hash()
        h7, h8 = sh1[my_cat.codes], sh2[my_cat.codes]
        assert h1.tolist() == h7.tolist()
        assert h2.tolist() == h8.tolist()

        # verify all the ways to hash bigint pdarrays match
        h1, h2 = ak.hash(bi)
        h3, h4 = ak.hash([bi])
        assert h1.tolist() == h3.tolist()
        assert h2.tolist() == h4.tolist()

    # Notes about median:
    #  prob_size is either even or odd, so one of sample_e, sample_o will have an even
    #  length, and the other an odd length.  Median should be tested with both even and odd
    #  length inputs.

    #  median can be done on ints or floats

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("data_type", NUMERIC_TYPES)
    def test_median(self, prob_size, data_type):
        sample_e = np.random.permutation(prob_size).astype(data_type)
        pda_e = ak.array(sample_e)
        assert isclose(np.median(sample_e), ak.median(pda_e))
        sample_o = np.random.permutation(prob_size + 1).astype(data_type)
        pda_o = ak.array(sample_o)
        assert isclose(np.median(sample_o), ak.median(pda_o))

    #  test_count_nonzero doesn't use parameterization on data types, because
    #  the data is generated differently.

    #  counts are ints, so we test for equality, not closeness.

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_count_nonzero(self, prob_size):
        # ints, floats

        for data_type in INT_FLOAT:
            sample = np.random.randint(20, size=prob_size).astype(data_type)
            pda = ak.array(sample)
            assert np.count_nonzero(sample) == ak.count_nonzero(pda)

        # bool

        sample = np.random.randint(2, size=prob_size).astype(bool)
        pda = ak.array(sample)
        assert np.count_nonzero(sample) == ak.count_nonzero(pda)

        # string

        sample = sample.astype(str)
        for i in range(10):
            sample[np.random.randint(prob_size)] = ""  # empty some strings at random
        pda = ak.array(sample)
        assert np.count_nonzero(sample) == ak.count_nonzero(pda)

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_clip(self, prob_size):
        seed = pytest.seed if pytest.seed is not None else 8675309
        np.random.seed(seed)
        ia = np.random.randint(1, 100, prob_size)
        ilo = 25
        ihi = 75

        dtypes = ["int64", "float64"]

        # test clip.
        # array to be clipped can be integer or float
        # range limits can be integer, float, or none, and can be scalars or arrays

        # Looping over all data types, the interior loop tests using lo, hi as:

        #   None, Scalar
        #   None, Array
        #   Scalar, Scalar
        #   Scalar, Array
        #   Scalar, None
        #   Array, Scalar
        #   Array, Array
        #   Array, None

        # There is no test with lo and hi both equal to None, because that's not allowed

        for dtype1 in dtypes:
            hi = np.full(ia.shape, ihi, dtype=dtype1)
            akhi = ak.array(hi)
            for dtype2 in dtypes:
                lo = np.full(ia.shape, ilo, dtype=dtype2)
                aklo = ak.array(lo)
                for dtype3 in dtypes:
                    nd_arry = ia.astype(dtype3)
                    ak_arry = ak.array(nd_arry)
                    assert np.allclose(
                        np.clip(nd_arry, None, hi[0]),
                        ak.clip(ak_arry, None, hi[0]).to_ndarray(),
                    )
                    assert np.allclose(
                        np.clip(nd_arry, None, hi),
                        ak.clip(ak_arry, None, akhi).to_ndarray(),
                    )
                    assert np.allclose(
                        np.clip(nd_arry, lo[0], hi[0]),
                        ak.clip(ak_arry, lo[0], hi[0]).to_ndarray(),
                    )
                    assert np.allclose(
                        np.clip(nd_arry, lo[0], hi),
                        ak.clip(ak_arry, lo[0], akhi).to_ndarray(),
                    )
                    assert np.allclose(
                        np.clip(nd_arry, lo[0], None),
                        ak.clip(ak_arry, lo[0], None).to_ndarray(),
                    )
                    assert np.allclose(
                        np.clip(nd_arry, lo, hi[0]),
                        ak.clip(ak_arry, aklo, hi[0]).to_ndarray(),
                    )
                    assert np.allclose(
                        np.clip(nd_arry, lo, hi),
                        ak.clip(ak_arry, aklo, akhi).to_ndarray(),
                    )
                    assert np.allclose(
                        np.clip(nd_arry, lo, None),
                        ak.clip(ak_arry, aklo, None).to_ndarray(),
                    )

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_putmask(self, prob_size):
        for d1, d2 in ALLOWED_PUTMASK_PAIRS:
            #  several things to test: values same size as data

            nda = np.random.randint(0, 10, prob_size).astype(d1)
            pda = ak.array(nda)
            nda2 = (nda**2).astype(d2)
            pda2 = ak.array(nda2)
            hold_that_thought = nda.copy()
            np.putmask(nda, nda > 5, nda2)
            ak.putmask(pda, pda > 5, pda2)
            assert np.allclose(nda, pda.to_ndarray())

            # values potentially much shorter than data

            nda = hold_that_thought.copy()
            pda = ak.array(nda)
            npvalues = np.arange(3).astype(d2)
            akvalues = ak.array(npvalues)
            np.putmask(nda, nda > 5, npvalues)
            ak.putmask(pda, pda > 5, akvalues)
            assert np.allclose(nda, pda.to_ndarray())

            # values shorter than data, but likely not to fit on one locale in a multi-locale test

            nda = hold_that_thought.copy()
            pda = ak.array(nda)
            npvalues = np.arange(prob_size // 2 + 1).astype(d2)
            akvalues = ak.array(npvalues)
            np.putmask(nda, nda > 5, npvalues)
            ak.putmask(pda, pda > 5, akvalues)
            assert np.allclose(nda, pda.to_ndarray())

            # values longer than data

            nda = hold_that_thought.copy()
            pda = ak.array(nda)
            npvalues = np.arange(prob_size + 1000).astype(d2)
            akvalues = ak.array(npvalues)
            np.putmask(nda, nda > 5, npvalues)
            ak.putmask(pda, pda > 5, akvalues)
            assert np.allclose(nda, pda.to_ndarray())

        # finally try to raise errors

        pda = ak.random.randint(0, 10, 10).astype(ak.float64)
        mask = ak.array([True])  # wrong size error
        values = ak.arange(10).astype(ak.float64)
        with pytest.raises(RuntimeError):
            ak.putmask(pda, mask, values)

        for d2, d1 in ALLOWED_PUTMASK_PAIRS:
            if d1 != d2:  # wrong types error
                pda = ak.arange(0, 10, prob_size).astype(d1)
                pda2 = (10 - pda).astype(d2)
                with pytest.raises(RuntimeError):
                    ak.putmask(pda, pda > 5, pda2)

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_putmask_multidim(self, prob_size):
        top = get_max_array_rank()
        for d1, d2 in ALLOWED_PUTMASK_PAIRS:
            # create two non-identical shapes with same size

            the_shape = np.arange(top) + 2  # e.g. [2,3,4] or [2,3,4,5] ...
            the_shape[-1] = prob_size  # now  [2,3,100] or [2,3,4,100] e.g.
            the_shape = tuple(the_shape)  # converts to tuple
            rev_shape = the_shape[::-1]  # [100,3,2] or [100,4,3,2] or ...
            the_size = prod(the_shape)  # total # elements in either shape

            nda = np.ones(the_size).reshape(the_shape).astype(d1)
            hold_that_thought = nda.copy()
            pda = ak.array(nda)
            nmask = alternate(True, False, the_size).reshape(the_shape)
            pmask = ak.array(nmask)

            # test with values the same size as a, but not same shape

            npvalues = np.arange(the_size).reshape(rev_shape).astype(d2)
            akvalues = ak.array(npvalues)
            np.putmask(nda, nmask, npvalues)
            ak.putmask(pda, pmask, akvalues)
            assert np.allclose(nda, pda.to_ndarray())

            # test with values longer than a; note that after each use of putmask
            # nda and pda have to be restored to their original values, since putmask
            # overwrites them.

            nda = hold_that_thought[:]
            pda = ak.array(nda)
            npvalues = np.arange(2 * the_size).reshape(2, the_size).astype(d2)
            akvalues = ak.array(npvalues)
            np.putmask(nda, nmask, npvalues)
            ak.putmask(pda, pmask, akvalues)
            assert np.allclose(nda, pda.to_ndarray())

            # test with values smaller than a
            # TODO: now that all allowed dims are available, extend the test below to
            # all allowed dims, rather than just max

            nda = hold_that_thought[:]
            pda = ak.array(nda)
            npvalues = np.arange(the_size - 3).astype(d2)
            akvalues = ak.array(npvalues)
            np.putmask(nda, nmask, npvalues)
            ak.putmask(pda, pmask, akvalues)
            assert np.allclose(nda, pda.to_ndarray())

            # test with values size that will require aggregator in multi-distribution
            # The choice of the_size//2 + 5 is arbitrary.

            nda = hold_that_thought[:]
            pda = ak.array(nda)
            npvalues = np.arange(the_size // 2 + 5).astype(d2)
            akvalues = ak.array(npvalues)
            np.putmask(nda, nmask, npvalues)
            ak.putmask(pda, pmask, akvalues)
            assert np.allclose(nda, pda.to_ndarray())

    # In the tests below, the rationale for using size = math.sqrt(prob_size) is that
    # the resulting matrices are on the order of size*size.

    # tril works on ints, floats, or bool
    @pytest.mark.skip_if_rank_not_compiled(2)
    @pytest.mark.parametrize("data_type", INT_FLOAT_BOOL)
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_tril(self, data_type, prob_size):
        size = int(sqrt(prob_size))

        # ints and bools are checked for equality; floats are checked for closeness

        check = lambda a, b, t: (  # noqa: E731
            np.allclose(a.tolist(), b.tolist()) if akdtype(t) == "float64" else (a == b).all()
        )

        # test on one square and two non-square matrices

        for rows, cols in [(size, size), (size + 1, size - 1), (size - 1, size + 1)]:
            pda = ak.randint(1, 10, (rows, cols))
            nda = pda.to_ndarray()
            sweep = range(-(rows - 2), cols)  # sweeps the diagonal from LL to UR
            for diag in sweep:
                npa = np.tril(nda, diag)
                ppa = ak.tril(pda, diag).to_ndarray()
                assert check(npa, ppa, data_type)

    # triu works on ints, floats, or bool

    @pytest.mark.parametrize("data_type", INT_FLOAT_BOOL)
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.skip_if_rank_not_compiled(2)
    def test_triu(self, data_type, prob_size):
        size = int(sqrt(prob_size))

        # ints and bools are checked for equality; floats are checked for closeness

        check = lambda a, b, t: (  # noqa: E731
            np.allclose(a.tolist(), b.tolist()) if akdtype(t) == "float64" else (a == b).all()
        )

        # test on one square and two non-square matrices

        for rows, cols in [(size, size), (size + 1, size - 1), (size - 1, size + 1)]:
            pda = ak.randint(1, 10, (rows, cols))
            nda = pda.to_ndarray()
            sweep = range(-(rows - 1), cols - 1)  # sweeps the diagonal from LL to UR
            for diag in sweep:
                npa = np.triu(nda, diag)
                ppa = ak.triu(pda, diag).to_ndarray()
                assert check(npa, ppa, data_type)

    # transpose works on ints, floats, or bool
    @pytest.mark.parametrize("data_type", INT_FLOAT_BOOL)
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_transpose(self, data_type, prob_size):
        size = prob_size
        for n in get_array_ranks():
            if n == 1:
                shape = size
                array_size = size
            else:
                shape = (n - 1) * [2]
                shape.append(size)
                array_size = (2 ** (n - 1)) * size

            nda = np.arange(array_size).reshape(shape)
            pda = ak.array(nda)

            if n == 1:  # trivial case
                assert_arkouda_array_equivalent(np.transpose(nda), ak.transpose(pda))
            else:  # all permutations of 'shape' must be checked
                from itertools import permutations

                perms = set(permutations(np.arange(n).tolist()))
                for perm in perms:  # contiguous array is needed for test
                    assert_arkouda_array_equivalent(
                        np.ascontiguousarray(np.transpose(nda, perm)),
                        ak.transpose(pda, perm),
                    )

    # eye works on ints, floats, or bool
    @pytest.mark.skip_if_rank_not_compiled(2)
    @pytest.mark.parametrize("data_type", INT_FLOAT_BOOL)
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_eye(self, data_type, prob_size):
        size = int(sqrt(prob_size))

        # ints and bools are checked for equality; floats are checked for closeness

        check = lambda a, b, t: (  # noqa: E731
            np.allclose(a.tolist(), b.tolist()) if akdtype(t) == "float64" else (a == b).all()
        )

        # test on one square and two non-square matrices

        for N, M in [(size, size), (size + 1, size - 1), (size - 1, size + 1)]:
            sweep = range(-(M - 1), N)  # sweeps the diagonal from LL to UR
            for k in sweep:
                nda = np.eye(N, M, k, dtype=data_type)
                pda = ak.eye(N, M, k, dt=data_type).to_ndarray()
                assert check(nda, pda, data_type)

    # matmul works on ints, floats, or bool
    @pytest.mark.skip_if_rank_not_compiled(2)
    @pytest.mark.parametrize("data_type1", INT_FLOAT_BOOL)
    @pytest.mark.parametrize("data_type2", INT_FLOAT_BOOL)
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_matmul(self, data_type1, data_type2, prob_size):
        size = int(sqrt(prob_size))

        # ints and bools are checked for equality; floats are checked for closeness

        check = lambda a, b, t: (  # noqa: E731
            np.allclose(a.tolist(), b.tolist()) if akdtype(t) == "float64" else (a == b).all()
        )

        # test on one square and two non-square products

        for rows, cols in [(size, size), (size + 1, size - 1), (size - 1, size + 1)]:
            pdaLeft = ak.randint(0, 10, (rows, size), dtype=data_type1)
            ndaLeft = pdaLeft.to_ndarray()
            pdaRight = ak.randint(0, 10, (size, cols), dtype=data_type2)
            ndaRight = pdaRight.to_ndarray()
            akProduct = ak.matmul(pdaLeft, pdaRight)
            npProduct = np.matmul(ndaLeft, ndaRight)
            assert check(npProduct, akProduct.to_ndarray(), akProduct.dtype)

    # Notes about array_equal:
    #   Strings compared to non-strings are always not equal.
    #   nan handling is (of course) unique to floating point
    #   we deliberately test on matched and mismatched arrays

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("data_type", ARRAY_TYPES)
    @pytest.mark.parametrize("same_size", YES_NO)
    @pytest.mark.parametrize("matching", YES_NO)
    @pytest.mark.parametrize("nan_handling", YES_NO)
    def test_array_equal(self, prob_size, data_type, same_size, matching, nan_handling):
        seed = pytest.seed if pytest.seed is not None else 8675309
        if data_type is ak.str_:  # strings require special handling
            np.random.seed(seed)
            temp = np.random.choice(VOWELS_AND_SUCH, prob_size)
            pda_a = ak.array(temp)
            pda_b = ak.array(temp)
            assert ak.array_equal(pda_a, pda_b)  # matching string arrays
            pda_c = pda_b[:-1]
            assert not (ak.array_equal(pda_a, pda_c))  # matching except c is shorter by 1
            temp = np.random.choice(VOWELS_AND_SUCH, prob_size)
            pda_b = ak.array(temp)
            assert not (ak.array_equal(pda_a, pda_b))  # mismatching string arrays
            pda_b = ak.randint(0, 100, prob_size, dtype=ak.int64)
            assert not (ak.array_equal(pda_a, pda_b))  # string to int comparison
            pda_b = ak.randint(0, 2, prob_size, dtype=ak.bool_)
            assert not (ak.array_equal(pda_a, pda_b))  # string to bool comparison
        elif data_type is ak.float64:  # so do floats, because of nan
            nda_a = np.random.uniform(0, 100, prob_size)
            if nan_handling:
                nda_a[-1] = np.nan
            nda_b = nda_a.copy() if matching else np.random.uniform(0, 100, prob_size)
            pda_a = ak.array(nda_a)
            pda_b = ak.array(nda_b) if same_size else ak.array(nda_b[:-1])
            assert ak.array_equal(pda_a, pda_b, nan_handling) == (matching and same_size)
        else:  # other types have simpler tests
            pda_a = ak.random.randint(0, 100, prob_size, dtype=data_type)
            if matching:  # known to match?
                pda_b = pda_a if same_size else pda_a[:-1]
                assert ak.array_equal(pda_a, pda_b) == (matching and same_size)
            elif same_size:  # not matching, but same size?
                pda_b = ak.random.randint(0, 100, prob_size, dtype=data_type)
                assert not (ak.array_equal(pda_a, pda_b))
            else:
                pda_b = ak.random.randint(
                    0, 100, (prob_size if same_size else prob_size - 1), dtype=data_type
                )
                assert not (ak.array_equal(pda_a, pda_b))

    @pytest.mark.parametrize("func", ["floor", "ceil", "trunc"])
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_rounding_functions(self, prob_size, func):
        akfunc = getattr(ak, func)
        npfunc = getattr(np, func)

        seed = pytest.seed if pytest.seed is not None else 8675309
        np.random.seed(seed)
        for rank in get_array_ranks():
            last_dim = prob_size // (2 ** (rank - 1))  # build a dimension of (2,2,...n)
            local_shape = (rank - 1) * [2]  # such that 2*2*..*n is close to prob_size
            local_shape.append(last_dim)  # building local_shape really does take
            local_shape = tuple(local_shape)  # multiple steps because .append doesn't
            local_size = prod(local_shape)  # return a value.

            sample = np.random.uniform(0, 100, local_size)  # make the data
            if rank > 1:
                sample = sample.reshape(local_shape)  # reshape only needed if rank > 1
            aksample = ak.array(sample)
            assert np.all(npfunc(sample) == akfunc(aksample).to_ndarray())

    def test_can_cast(self):
        from arkouda.numpy.dtypes import can_cast

        assert can_cast(ak.int64, ak.int64)
        assert not can_cast(ak.int64, ak.uint64)

        assert can_cast(5, ak.uint64)
        assert not can_cast(-5, ak.uint64)
        assert not can_cast(2**200, ak.uint64)

    @pytest.mark.parametrize("dtype", [ak.int64, ak.float64, ak.uint64])
    @pytest.mark.parametrize("ndims", [0, 1, 2, 3])
    def test_nextafter_multidim(self, dtype, ndims):
        if ndims != 0 and ndims not in get_array_ranks():
            pytest.skip(f"Server not compiled for rank {ndims}")

        x1_rank = ndims
        min_value = -100 if dtype != ak.uint64 else 0
        for x2_rank in range(ndims + 1):
            bcast_shape = tuple(np.random.randint(1, 10, size=max(x1_rank, x2_rank)))
            if x1_rank != 0:
                x1_shape = bcast_shape[-x1_rank:]
                if dtype == ak.bigint:
                    x1 = ak.randint(min_value, 100, dtype=ak.int64, size=x1_shape)
                    x1 = ak.cast(x1, ak.bigint)
                else:
                    x1 = ak.randint(min_value, 100, dtype=dtype, size=x1_shape)
                n_x1 = x1.to_ndarray()
            else:
                if dtype == ak.bigint:
                    x1 = ak.randint(min_value, 100, dtype=ak.int64, size=1)
                    x1 = ak.cast(x1, ak.bigint)[0]
                else:
                    x1 = ak.randint(min_value, 100, dtype=dtype, size=1)[0]
                n_x1 = x1
            if x2_rank != 0:
                x2_shape = bcast_shape[-x2_rank:]
                if dtype == ak.bigint:
                    x2 = ak.randint(min_value, 100, dtype=ak.int64, size=x2_shape)
                    x2 = ak.cast(x2, ak.bigint)
                else:
                    x2 = ak.randint(min_value, 100, dtype=dtype, size=x2_shape)
                n_x2 = x2.to_ndarray()
            else:
                if dtype == ak.bigint:
                    x2 = ak.randint(min_value, 100, dtype=ak.int64, size=1)
                    x2 = ak.cast(x2, ak.bigint)[0]
                else:
                    x2 = ak.randint(min_value, 100, dtype=dtype, size=1)[0]
                n_x2 = x2
            if x1_rank == 0 and x2_rank == 0:
                assert ak.nextafter(x1, x2) == np.nextafter(n_x1, n_x2)
            else:
                assert_arkouda_array_equivalent(ak.nextafter(x1, x2), np.nextafter(n_x1, n_x2))

    def test_nextafter_boundary(self):
        from itertools import combinations

        boundary_values = [ak.nan, -ak.inf, -1.0, -0.0, 0.0, 1.0, ak.inf]

        for x1, x2 in combinations(boundary_values, 2):
            x1 = ak.array([x1])
            x2 = ak.array([x2])
            n_x1 = x1.to_ndarray()
            n_x2 = x2.to_ndarray()
            assert np.array_equal(
                ak.nextafter(x1, x2).to_ndarray(),
                np.nextafter(n_x1, n_x2),
                equal_nan=True,
            )
            assert np.array_equal(
                ak.nextafter(x2, x1).to_ndarray(),
                np.nextafter(n_x2, n_x1),
                equal_nan=True,
            )
            assert np.array_equal(
                ak.nextafter(x1, x1).to_ndarray(),
                np.nextafter(n_x1, n_x1),
                equal_nan=True,
            )

    #   "perquant" tests both quantile and percentile, which are basically
    #   the same functions.

    #   In all of these tests, the np version is converted to np.float64.
    #   This is because ak.quantile always returns np.float64, whereas np.quantile
    #   USUALLY does, but will preserve the input data type for certain methods.
    #   This was not implemented in ak.quantile.

    #   test scalar q, 1 dimensional pda, returns scalar

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("method_name", ALLOWED_PERQUANT_METHODS)
    @pytest.mark.parametrize("data_type", NO_BOOL)
    def test_perquant_scalar_no_axis(self, prob_size, method_name, data_type):
        nda = np.random.randint(0, prob_size, prob_size).astype(data_type)
        pda = ak.array(nda)
        q = np.random.uniform()
        pr = ak.quantile(pda, q, method=method_name)
        nr = np.quantile(nda, q, method=method_name).astype(np.float64)
        assert isclose(pr, nr)
        pr = ak.percentile(pda, 100 * q, method=method_name)
        nr = np.percentile(nda, 100 * q, method=method_name).astype(np.float64)
        assert isclose(pr, nr)

    #   Note: axis slicing tests take considerably longer, so only a
    #   subset of the methods are used when multi-locales exist.

    #   test scalar q, 2 dimensional pda with 1 slice axis, returns pdarray

    @pytest.mark.skip_if_rank_not_compiled(2)
    @pytest.mark.skipif(pytest.nl > 1, reason="Single locale test skipped.")
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("method_name", ALLOWED_PERQUANT_METHODS)
    @pytest.mark.parametrize("data_type", NO_BOOL)
    def test_perquant_scalar_with_axis_one_locale(self, prob_size, method_name, data_type):
        smaller = prob_size // 2
        nda = np.random.randint(0, smaller, (2, smaller)).astype(data_type)
        pda = ak.array(nda)
        q = np.random.uniform()
        for axis in [0, 1]:
            keepdims = bool(axis)  # first False, then True
            pr = ak.quantile(pda, q, axis=axis, keepdims=keepdims, method=method_name)
            nr = np.quantile(nda, q, axis=axis, keepdims=keepdims, method=method_name).astype(np.float64)
            ak_assert_almost_equivalent(pr, nr)
            pr = ak.percentile(pda, 100 * q, axis=axis, keepdims=keepdims, method=method_name)
            nr = np.percentile(nda, 100 * q, axis=axis, keepdims=keepdims, method=method_name).astype(
                np.float64
            )
            ak_assert_almost_equivalent(pr, nr)

    @pytest.mark.skip_if_rank_not_compiled(2)
    @pytest.mark.skipif(pytest.nl < 2, reason="Multi-locale test skipped")
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("method_name", SUBSET_PERQUANT_METHODS)
    @pytest.mark.parametrize("data_type", NO_BOOL)
    def test_perquant_scalar_with_axis_multi_locale(self, prob_size, method_name, data_type):
        smaller = prob_size // 2
        nda = np.random.randint(0, smaller, (2, smaller)).astype(data_type)
        pda = ak.array(nda)
        q = np.random.uniform()
        for axis in [0, 1]:
            keepdims = bool(axis)  # first False, then True
            pr = ak.quantile(pda, q, axis=axis, keepdims=keepdims, method=method_name)
            nr = np.quantile(nda, q, axis=axis, keepdims=keepdims, method=method_name).astype(np.float64)
            ak_assert_almost_equivalent(pr, nr)
            pr = ak.percentile(pda, 100 * q, axis=axis, keepdims=keepdims, method=method_name)
            nr = np.percentile(nda, 100 * q, axis=axis, keepdims=keepdims, method=method_name).astype(
                np.float64
            )
            ak_assert_almost_equivalent(pr, nr)

    #   test array q, 2 dimensional pda with no slice axis, returns pdarray

    @pytest.mark.skip_if_rank_not_compiled(2)
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("method_name", ALLOWED_PERQUANT_METHODS)
    @pytest.mark.parametrize("data_type", NO_BOOL)
    def test_perquant_array_no_axis(self, prob_size, method_name, data_type):
        nda = np.random.randint(0, prob_size, prob_size).astype(data_type)
        pda = ak.array(nda)
        q = np.random.uniform(0, 1, 3)
        pr = ak.quantile(pda, q, method=method_name)
        nr = np.quantile(nda, q, method=method_name).astype(np.float64)
        ak_assert_almost_equivalent(pr, nr)
        pr = ak.percentile(pda, 100 * q, method=method_name)
        nr = np.percentile(nda, 100 * q, method=method_name).astype(np.float64)
        ak_assert_almost_equivalent(pr, nr)

    #   test array 1 dimensional q, 2 dimensional pda with slice axis, returns pdarray.

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    @pytest.mark.skipif(pytest.nl > 1, reason="Single locale test skipped.")
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("method_name", ALLOWED_PERQUANT_METHODS)
    @pytest.mark.parametrize("data_type", NO_BOOL)
    def test_perquant_array_with_axis_one_locale(self, prob_size, method_name, data_type):
        smaller = prob_size // 2
        nda = np.random.randint(0, smaller, (2, smaller)).astype(data_type)
        pda = ak.array(nda)
        q = np.random.uniform(0, 1, 2)
        for axis in [0, 1]:
            keepdims = bool(axis)  # alternate False, True
            pr = ak.quantile(pda, q, axis=axis, keepdims=keepdims, method=method_name)
            nr = np.quantile(nda, q, axis=axis, keepdims=keepdims, method=method_name).astype(np.float64)
            ak_assert_almost_equivalent(pr, nr)
            pr = ak.percentile(pda, 100 * q, axis=axis, keepdims=keepdims, method=method_name)
            nr = np.percentile(nda, 100 * q, axis=axis, keepdims=keepdims, method=method_name).astype(
                np.float64
            )
            ak_assert_almost_equivalent(pr, nr)

    @pytest.mark.skip_if_rank_not_compiled([2, 3])
    @pytest.mark.skipif(pytest.nl < 2, reason="Multi locale test skipped.")
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("method_name", SUBSET_PERQUANT_METHODS)
    @pytest.mark.parametrize("data_type", NO_BOOL)
    def test_perquant_array_with_axis_multi_locale(self, prob_size, method_name, data_type):
        smaller = prob_size // 2
        nda = np.random.randint(0, smaller, (2, smaller)).astype(data_type)
        pda = ak.array(nda)
        q = np.random.uniform(0, 1, 2)
        for axis in [0, 1]:
            keepdims = bool(axis)  # alternate False, True
            pr = ak.quantile(pda, q, axis=axis, keepdims=keepdims, method=method_name)
            nr = np.quantile(nda, q, axis=axis, keepdims=keepdims, method=method_name).astype(np.float64)
            ak_assert_almost_equivalent(pr, nr)
            pr = ak.percentile(pda, 100 * q, axis=axis, keepdims=keepdims, method=method_name)
            nr = np.percentile(nda, 100 * q, axis=axis, keepdims=keepdims, method=method_name).astype(
                np.float64
            )
            ak_assert_almost_equivalent(pr, nr)

    #   test array 2 dimensional q, 2 dimensional pda with slice axis, returns pdarray.
    #      This is a case with an intermediate result of rank 4, so the test can
    #      only be run if that rank is compiled.

    @pytest.mark.skip_if_rank_not_compiled([2, 3, 4])
    @pytest.mark.skipif(pytest.nl > 1, reason="Single locale test skipped.")
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("method_name", ALLOWED_PERQUANT_METHODS)
    @pytest.mark.parametrize("data_type", NO_BOOL)
    def test_perquant_array_multi_dim_one_locale(self, prob_size, method_name, data_type):
        smaller = prob_size // 2
        nda = np.random.randint(0, smaller, (2, smaller)).astype(data_type)
        pda = ak.array(nda)
        q = np.random.uniform(0, 1, (2, 2))
        for axis in [0, 1]:
            keepdims = bool(axis)  # alternate False, True
            pr = ak.quantile(pda, q, axis=axis, keepdims=keepdims, method=method_name)
            nr = np.quantile(nda, q, axis=axis, keepdims=keepdims, method=method_name).astype(np.float64)
            ak_assert_almost_equivalent(pr, nr)
            pr = ak.percentile(pda, 100 * q, axis=axis, keepdims=keepdims, method=method_name)
            nr = np.percentile(nda, 100 * q, axis=axis, keepdims=keepdims, method=method_name).astype(
                np.float64
            )
            ak_assert_almost_equivalent(pr, nr)

    @pytest.mark.skip_if_rank_not_compiled([2, 3, 4])
    @pytest.mark.skipif(pytest.nl < 2, reason="Multi locale test skipped.")
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("method_name", SUBSET_PERQUANT_METHODS)
    @pytest.mark.parametrize("data_type", NO_BOOL)
    def test_perquant_array_multi_dim_multi_locale(self, prob_size, method_name, data_type):
        smaller = prob_size // 2
        nda = np.random.randint(0, smaller, (2, smaller)).astype(data_type)
        pda = ak.array(nda)
        q = np.random.uniform(0, 1, (2, 2))
        for axis in [0, 1]:
            keepdims = bool(axis)  # First False, then True
            pr = ak.quantile(pda, q, axis=axis, keepdims=keepdims, method=method_name)
            nr = np.quantile(nda, q, axis=axis, keepdims=keepdims, method=method_name).astype(np.float64)
            ak_assert_almost_equivalent(pr, nr)
            pr = ak.percentile(pda, 100 * q, axis=axis, keepdims=keepdims, method=method_name)
            nr = np.percentile(nda, 100 * q, axis=axis, keepdims=keepdims, method=method_name).astype(
                np.float64
            )
            ak_assert_almost_equivalent(pr, nr)

    #   test array 1 dimensional q, 3 dimensional pda with multi-axis slice, returns pdarray.
    #      This is also a case with an intermediate result of rank 4, so the test can
    #      only be run if that rank is compiled.

    @pytest.mark.skip_if_rank_not_compiled([2, 3, 4])
    @pytest.mark.skipif(pytest.nl > 1, reason="Single locale test skipped.")
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("method_name", ALLOWED_PERQUANT_METHODS)
    @pytest.mark.parametrize("data_type", NO_BOOL)
    def test_perquant_array_multi_slice_one_locale(self, prob_size, method_name, data_type):
        smaller = prob_size // 4
        nda = np.random.randint(0, smaller, (2, 2, smaller)).astype(data_type)
        pda = ak.array(nda)
        q = np.random.uniform(0, 1, 3)
        keepdims = False
        for axis in [0, 1, 2, (0, 1), (0, 2), (1, 2)]:
            pr = ak.quantile(pda, q, axis=axis, keepdims=keepdims, method=method_name)
            nr = np.quantile(nda, q, axis=axis, keepdims=keepdims, method=method_name).astype(np.float64)
            ak_assert_almost_equivalent(pr, nr)
            pr = ak.percentile(pda, 100 * q, axis=axis, keepdims=keepdims, method=method_name)
            nr = np.percentile(nda, 100 * q, axis=axis, keepdims=keepdims, method=method_name).astype(
                np.float64
            )
            ak_assert_almost_equivalent(pr, nr)
            keepdims = not keepdims  # switch this each time through the loop

    @pytest.mark.skip_if_rank_not_compiled([2, 3, 4])
    @pytest.mark.skipif(pytest.nl < 2, reason="Multi locale test skipped.")
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    @pytest.mark.parametrize("method_name", SUBSET_PERQUANT_METHODS)
    @pytest.mark.parametrize("data_type", NO_BOOL)
    def test_perquant_array_multi_slice_multi_locale(self, prob_size, method_name, data_type):
        smaller = prob_size // 4
        nda = np.random.randint(0, smaller, (2, 2, smaller)).astype(data_type)
        pda = ak.array(nda)
        q = np.random.uniform(0, 1, 3)
        keepdims = False
        for axis in [random.choice([0, 1, 2]), random.choice([(0, 1), (0, 2), (1, 2)])]:
            pr = ak.quantile(pda, q, axis=axis, keepdims=keepdims, method=method_name)
            nr = np.quantile(nda, q, axis=axis, keepdims=keepdims, method=method_name).astype(np.float64)
            ak_assert_almost_equivalent(pr, nr)
            pr = ak.percentile(pda, 100 * q, axis=axis, keepdims=keepdims, method=method_name)
            nr = np.percentile(nda, 100 * q, axis=axis, keepdims=keepdims, method=method_name).astype(
                np.float64
            )
            ak_assert_almost_equivalent(pr, nr)
            keepdims = not keepdims  # switch this each time through the loop

    @pytest.mark.parametrize("dtype", NUMERIC_TYPES)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_take_1d(self, dtype, size):
        seed = pytest.seed if pytest.seed is not None else 8675309
        if dtype == "bool":
            a = ak.randint(0, 2, size, dtype=dtype, seed=seed)
        else:
            a = ak.randint(0, 100, size, dtype=dtype, seed=seed)
        anp = a.to_ndarray()

        indices = ak.randint(0, size, size // 2, dtype="int64", seed=seed)
        indices_np = indices.to_ndarray()

        a_taken = ak.take(a, indices)
        anp_taken = np.take(anp, indices_np)

        assert np.array_equal(a_taken.to_ndarray(), anp_taken)

    @pytest.mark.parametrize("dtype", NUMERIC_TYPES)
    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("axis", [None, 0, 1, 2])
    def test_take_multidim(self, dtype, axis):
        seed = pytest.seed if pytest.seed is not None else 8675309
        if dtype == "bool":
            a = ak.randint(0, 2, (5, 6, 7), dtype=dtype, seed=seed)
        else:
            a = ak.randint(0, 100, (5, 6, 7), dtype=dtype, seed=seed)
        anp = a.to_ndarray()

        indices = ak.randint(0, 6, 3, dtype="int64", seed=seed)
        indices_np = indices.to_ndarray()

        a_taken = ak.take(a, indices, axis=axis)
        anp_taken = np.take(anp, indices_np, axis=axis)

        assert np.array_equal(a_taken.to_ndarray(), anp_taken)

    @pytest.mark.parametrize("dtype", NO_BOOL)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_vecdot_1d(self, dtype, size):
        seed = pytest.seed if pytest.seed is not None else 8675309
        a = ak.randint(0, 100, size, dtype=dtype, seed=seed)
        b = ak.randint(0, 100, size, dtype=dtype, seed=seed + 1)
        np_vecdot = np.vecdot(a.to_ndarray(), b.to_ndarray())
        ak_vecdot_f = ak.vecdot(a, b)
        ak_vecdot_r = ak.vecdot(a, b)
        assert isclose(np_vecdot, ak_vecdot_f)  # for 1D case, results are scalar
        assert isclose(ak_vecdot_f, ak_vecdot_r)

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("dtype", NO_BOOL)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("same_shape", YES_NO)
    def test_vecdot_2d(self, dtype, size, same_shape):
        seed = pytest.seed if pytest.seed is not None else 8675309
        if same_shape:
            a_shape = (2, size // 2)
            b_shape = (2, size // 2)
        else:
            a_shape = (1, size // 2)
            b_shape = (2, size // 2)
        a = ak.randint(0, 100, a_shape, dtype=dtype, seed=seed)
        b = ak.randint(0, 100, b_shape, dtype=dtype, seed=seed + 1)
        if same_shape:
            for axis in [0, 1]:
                np_vecdot = np.vecdot(a.to_ndarray(), b.to_ndarray(), axis=axis)
                ak_vecdot_f = ak.vecdot(a, b, axis=axis)
                ak_vecdot_r = ak.vecdot(b, a, axis=axis)
                ak_assert_almost_equivalent(np_vecdot, ak_vecdot_f)
                ak_assert_almost_equivalent(ak_vecdot_f, ak_vecdot_r)
        else:
            np_vecdot = np.vecdot(a.to_ndarray(), b.to_ndarray())
            ak_vecdot_f = ak.vecdot(a, b)
            ak_vecdot_r = ak.vecdot(b, a)
            ak_assert_almost_equivalent(np_vecdot, ak_vecdot_f)
            ak_assert_almost_equivalent(ak_vecdot_f, ak_vecdot_r)

    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("dtype", NO_BOOL)
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("same_shape", YES_NO)
    def test_vecdot_3d(self, dtype, size, same_shape):
        seed = pytest.seed if pytest.seed is not None else 8675309
        if same_shape:
            a_shape = (2, 2, size // 4)
            b_shape = (2, 2, size // 4)
        else:
            a_shape = (1, size // 4)
            b_shape = (2, 2, size // 4)
        a = ak.randint(0, 100, a_shape, dtype=dtype, seed=seed)
        b = ak.randint(0, 100, b_shape, dtype=dtype, seed=seed + 1)
        if same_shape:
            for axis in [0, 1, 2]:
                np_vecdot = np.vecdot(a.to_ndarray(), b.to_ndarray(), axis=axis)
                ak_vecdot_f = ak.vecdot(a, b, axis=axis)
                ak_vecdot_r = ak.vecdot(b, a, axis=axis)
                ak_assert_almost_equivalent(np_vecdot, ak_vecdot_f)
                ak_assert_almost_equivalent(ak_vecdot_f, ak_vecdot_r)
        else:
            np_vecdot = np.vecdot(a.to_ndarray(), b.to_ndarray())
            ak_vecdot_f = ak.vecdot(a, b)
            ak_vecdot_r = ak.vecdot(b, a)
            ak_assert_almost_equivalent(np_vecdot, ak_vecdot_f)
            ak_assert_almost_equivalent(ak_vecdot_f, ak_vecdot_r)

    #   The broadcast test creates compatible shapes that have to be broadcast
    #   to perform the computation

    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("dtype", NO_BOOL)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_vecdot_with_broadcast(self, dtype, size):
        seed = pytest.seed if pytest.seed is not None else 8675309
        ashape = [1, 1]
        bshape = [1, 1]
        for i in range(2):
            cointoss = random.choice([True, False])
            alteration = np.random.randint(2, 4)
            if cointoss:
                ashape[i] = alteration
            else:
                bshape[i] = alteration
        lastdim = max(2, size // (prod(ashape) * prod(bshape)))
        ashape.append(lastdim)
        bshape.append(lastdim)
        a = ak.randint(0, 100, tuple(ashape), dtype=dtype, seed=seed)
        b = ak.randint(0, 100, tuple(bshape), dtype=dtype, seed=seed + 1)
        np_vecdot = np.vecdot(a.to_ndarray(), b.to_ndarray())
        ak_vecdot_f = ak.vecdot(a, b)
        ak_vecdot_r = ak.vecdot(b, a)
        ak_assert_almost_equivalent(np_vecdot, ak_vecdot_f)
        ak_assert_almost_equivalent(ak_vecdot_f, ak_vecdot_r)

    #   The error test sends incompatible shapes to vecdot, which passes them to
    #   ak.broadcast_dims, which is where the error is raised.

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("dtype", NO_BOOL)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_vecdot_error(self, dtype, size):
        seed = pytest.seed if pytest.seed is not None else 8675309
        a_shape = (1, size // 2)
        b_shape = (2, size // 4)
        with pytest.raises(ValueError):
            a = ak.randint(0, 100, a_shape, dtype=dtype, seed=seed)
            b = ak.randint(0, 100, b_shape, dtype=dtype, seed=seed + 1)
            ak.vecdot(a, b)  # causes the ValueError
