from math import isclose, sqrt

import numpy as np
import pytest

import arkouda as ak
from arkouda.client import get_max_array_rank
from arkouda.dtypes import dtype as akdtype
from arkouda.dtypes import str_

ARRAY_TYPES = [ak.int64, ak.float64, ak.bool_, ak.uint64, str_]
NUMERIC_TYPES = [ak.int64, ak.float64, ak.bool_, ak.uint64]
NO_BOOL = [ak.int64, ak.float64, ak.uint64]
NO_FLOAT = [ak.int64, ak.bool_, ak.uint64]
INT_FLOAT = [ak.int64, ak.float64]
INT_FLOAT_BOOL = [ak.int64, ak.float64, ak.bool_]
YES_NO = [True, False]
VOWELS_AND_SUCH = ["a", "e", "i", "o", "u", "AB", 47, 2, 3.14159]

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
        ak_func(pda, where=truth_ak).to_list(),
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
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_floor_float(self, prob_size):
        from arkouda import all as akall
        from arkouda.numpy import floor as ak_floor

        a = 0.5 * ak.arange(prob_size, dtype="float64")
        a_floor = ak_floor(a)

        expected_size = np.floor((prob_size + 1) / 2).astype("int64")
        expected = ak.array(np.repeat(ak.arange(expected_size, dtype="float64").to_ndarray(), 2))
        #   To deal with prob_size as an odd number:
        expected = expected[0:prob_size]

        assert akall(a_floor == expected)

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
        np.random.seed(seed)
        arrays = {
            ak.int64: ak.randint(-(2**48), 2**48, prob_size),
            ak.uint64: ak.randint(0, 2**48, prob_size, dtype=ak.uint64),
            ak.float64: ak.randint(0, 1, prob_size, dtype=ak.float64),
            ak.bool_: ak.randint(0, 2, prob_size, dtype=ak.bool_),
            ak.str_: ak.cast(ak.randint(0, 2**48, prob_size), "str"),
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
        assert valid.to_list() == validans.to_list()
        assert np.allclose(ans, res.to_ndarray(), equal_nan=True)

    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_histogram(self, num_type):
        seed = pytest.seed if pytest.seed is not None else 8675309
        np.random.seed(seed)
        pda = ak.randint(10, 30, 40, dtype=num_type)
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

    #   log and exp tests were identical, and so have been combined.

    @pytest.mark.skipif(pytest.host == "horizon", reason="Fails on horizon")
    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("num_type1", NO_BOOL)
    @pytest.mark.parametrize("num_type2", NO_BOOL)
    def test_histogram_multidim(self, num_type1, num_type2):
        # test 2d histogram
        seed = 1
        ak_x, ak_y = ak.randint(1, 100, 1000, seed=seed, dtype=num_type1), ak.randint(1, 100, 1000, seed=seed + 1, dtype=num_type2)
        np_x, np_y = ak_x.to_ndarray(), ak_y.to_ndarray()
        np_hist, np_x_edges, np_y_edges = np.histogram2d(np_x, np_y)
        ak_hist, ak_x_edges, ak_y_edges = ak.histogram2d(ak_x, ak_y)
        assert np.allclose(np_hist.tolist(), ak_hist.to_list())
        assert np.allclose(np_x_edges.tolist(), ak_x_edges.to_list())
        assert np.allclose(np_y_edges.tolist(), ak_y_edges.to_list())

        np_hist, np_x_edges, np_y_edges = np.histogram2d(np_x, np_y, bins=(10, 20))
        ak_hist, ak_x_edges, ak_y_edges = ak.histogram2d(ak_x, ak_y, bins=(10, 20))
        assert np.allclose(np_hist.tolist(), ak_hist.to_list())
        assert np.allclose(np_x_edges.tolist(), ak_x_edges.to_list())
        assert np.allclose(np_y_edges.tolist(), ak_y_edges.to_list())

        # test arbitrary dimensional histogram
        dim_list = [3, 4, 5]
        bin_list = [[2, 4, 5], [2, 4, 5, 2], [2, 4, 5, 2, 3]]
        for dim, bins in zip(dim_list, bin_list):
            if dim <= get_max_array_rank():
                np_arrs = [np.random.randint(1, 100, 1000) for _ in range(dim)]
                ak_arrs = [ak.array(a) for a in np_arrs]
                np_hist, np_bin_edges = np.histogramdd(np_arrs, bins=bins)
                ak_hist, ak_bin_edges = ak.histogramdd(ak_arrs, bins=bins)
                assert np.allclose(np_hist.tolist(), ak_hist.to_list())
                for np_edge, ak_edge in zip(np_bin_edges, ak_bin_edges):
                    assert np.allclose(np_edge.tolist(), ak_edge.to_list())

    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_log_and_exp(self, num_type):
        na = np.linspace(1, 10, 10).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        for npfunc, akfunc in ((np.log, ak.log), (np.exp, ak.exp)):
            assert np.allclose(npfunc(na), akfunc(pda).to_ndarray())
        with pytest.raises(TypeError):
            akfunc(np.array([range(0, 10)]).astype(num_type))

    @pytest.mark.parametrize("num_type", INT_FLOAT)
    def test_abs(self, num_type):
        na = np.linspace(1, 10, 10).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        assert np.allclose(np.abs(na), ak.abs(pda).to_ndarray())

        assert (
            ak.arange(5, 0, -1, dtype=num_type).to_list()
            == ak.abs(ak.arange(-5, 0, dtype=num_type)).to_list()
        )

        with pytest.raises(TypeError):
            ak.abs(np.array([range(0, 10)]).astype(num_type))

    @pytest.mark.parametrize("num_type1", NO_BOOL)
    @pytest.mark.parametrize("num_type2", NO_BOOL)
    def test_dot(self, num_type1, num_type2):
        seed = pytest.seed if pytest.seed is not None else 8675309
        np.random.seed(seed)
        if num_type1 == ak.uint64 and num_type2 == ak.int64:
            pytest.skip()
        if num_type1 == ak.int64 and num_type2 == ak.uint64:
            pytest.skip()
        na1 = np.random.randint(0, 10, 10).astype(num_type1)
        na2 = np.random.randint(0, 10, 10).astype(num_type2)
        pda1 = ak.array(na1)
        pda2 = ak.array(na2)
        assert np.allclose(np.dot(na1, na2), ak.dot(pda1, pda2))
        assert np.allclose(np.dot(na1[0], na2), ak.dot(pda1[0], pda2).to_ndarray())
        assert np.allclose(np.dot(na1, na2[0]), ak.dot(pda1, pda2[0]).to_ndarray())

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
            ak.arctan2(pda_num, pda_denom, where=False).to_list(),
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
        Test efunc `isnan`; it returns a pdarray of element-wise T/F values for whether it is NaN
        (not a number)
        """
        npa = np.array([1, 2, None, 3, 4], dtype="float64")
        ark_s_float64 = ak.array(npa)
        ark_isna_float64 = ak.isnan(ark_s_float64)
        actual = ark_isna_float64.to_ndarray()
        assert np.array_equal(np.isnan(npa), actual)

        ark_s_int64 = ak.array(np.array([1, 2, 3, 4], dtype="int64"))
        assert ak.isnan(ark_s_int64).to_list() == [False, False, False, False]

        ark_s_string = ak.array(["a", "b", "c"])
        with pytest.raises(TypeError):
            ak.isnan(ark_s_string)

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
        G = prob_size // 10
        ub = 2**63 // prob_size
        groupnum = ak.randint(0, G, prob_size, seed=1)
        intval = ak.randint(0, ub, prob_size, seed=2)
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
        assert h1.to_list() == h3[rev].to_list()
        assert h2.to_list() == h4[rev].to_list()

        h1 = ak.hash(ak.arange(10), full=False)
        h3 = ak.hash(rev, full=False)
        assert h1.to_list() == h3[rev].to_list()

        h = ak.hash(ak.linspace(0, 10, 10))
        assert h[0].dtype == ak.uint64
        assert h[1].dtype == ak.uint64

        # test strings hash
        s = ak.random_strings_uniform(4, 8, 10)
        h1, h2 = ak.hash(s)
        rh1, rh2 = ak.hash(s[rev])
        assert h1.to_list() == rh1[rev].to_list()
        assert h2.to_list() == rh2[rev].to_list()

        # verify all the ways to hash strings match
        h3, h4 = ak.hash([s])
        assert h1.to_list() == h3.to_list()
        assert h2.to_list() == h4.to_list()
        h5, h6 = s.hash()
        assert h1.to_list() == h5.to_list()
        assert h2.to_list() == h6.to_list()

        # test segarray hash with int and string values
        # along with strings, categorical, and pdarrays
        segs = ak.array([0, 3, 6, 9])
        vals = ak.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 5, 5, 5, 5])
        sa = ak.SegArray(segs, vals)
        str_vals = ak.array([f"str {i}" for i in vals.to_list()])
        str_sa = ak.SegArray(segs, str_vals)
        a = ak.array([-10, 4, -10, 17])
        bi = a + 2**200
        s = ak.array([f"str {i}" for i in a.to_list()])
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
                assert h1.to_list() == h3.to_list()
                assert h2.to_list() == h4.to_list()
                h5, h6 = h.hash()
                assert h1.to_list() == h5.to_list()
                assert h2.to_list() == h6.to_list()
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
        assert h1.to_list() == rh1[rev].to_list()
        assert h2.to_list() == rh2[rev].to_list()

        # verify all the ways to hash Categoricals match
        h3, h4 = ak.hash([my_cat])
        assert h1.to_list() == h3.to_list()
        assert h2.to_list() == h4.to_list()
        h5, h6 = my_cat.hash()
        assert h1.to_list() == h5.to_list()
        assert h2.to_list() == h6.to_list()

        # verify it matches hashing the categories and then indexing with codes
        sh1, sh2 = my_cat.categories.hash()
        h7, h8 = sh1[my_cat.codes], sh2[my_cat.codes]
        assert h1.to_list() == h7.to_list()
        assert h2.to_list() == h8.to_list()

        # verify all the ways to hash bigint pdarrays match
        h1, h2 = ak.hash(bi)
        h3, h4 = ak.hash([bi])
        assert h1.to_list() == h3.to_list()
        assert h2.to_list() == h4.to_list()

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
                        np.clip(nd_arry, None, hi[0]), ak.clip(ak_arry, None, hi[0]).to_ndarray()
                    )
                    assert np.allclose(
                        np.clip(nd_arry, None, hi), ak.clip(ak_arry, None, akhi).to_ndarray()
                    )
                    assert np.allclose(
                        np.clip(nd_arry, lo[0], hi[0]), ak.clip(ak_arry, lo[0], hi[0]).to_ndarray()
                    )
                    assert np.allclose(
                        np.clip(nd_arry, lo[0], hi), ak.clip(ak_arry, lo[0], akhi).to_ndarray()
                    )
                    assert np.allclose(
                        np.clip(nd_arry, lo[0], None), ak.clip(ak_arry, lo[0], None).to_ndarray()
                    )
                    assert np.allclose(
                        np.clip(nd_arry, lo, hi[0]), ak.clip(ak_arry, aklo, hi[0]).to_ndarray()
                    )
                    assert np.allclose(
                        np.clip(nd_arry, lo, hi), ak.clip(ak_arry, aklo, akhi).to_ndarray()
                    )
                    assert np.allclose(
                        np.clip(nd_arry, lo, None), ak.clip(ak_arry, aklo, None).to_ndarray()
                    )

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_putmask(self, prob_size):

        for d1, d2 in ALLOWED_PUTMASK_PAIRS:

            #  three things to test: values same size as data

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

    # In the tests below, the rationale for using size = math.sqrt(prob_size) is that
    # the resulting matrices are on the order of size*size.

    # tril works on ints, floats, or bool
    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("data_type", INT_FLOAT_BOOL)
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_tril(self, data_type, prob_size):

        size = int(sqrt(prob_size))

        # ints and bools are checked for equality; floats are checked for closeness

        check = lambda a, b, t: (
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
    @pytest.mark.skip_if_max_rank_less_than(2)
    def test_triu(self, data_type, prob_size):
        size = int(sqrt(prob_size))

        # ints and bools are checked for equality; floats are checked for closeness

        check = lambda a, b, t: (
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

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("data_type", INT_FLOAT_BOOL)
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_transpose(self, data_type, prob_size):

        size = int(sqrt(prob_size))

        # ints and bools are checked for equality; floats are checked for closeness

        check = lambda a, b, t: (
            np.allclose(a.tolist(), b.tolist()) if akdtype(t) == "float64" else (a == b).all()
        )

        # test on one square and two non-square matrices

        for rows, cols in [(size, size), (size + 1, size - 1), (size - 1, size + 1)]:
            pda = ak.randint(1, 10, (rows, cols))
            nda = pda.to_ndarray()
            npa = np.transpose(nda)
            ppa = ak.transpose(pda).to_ndarray()
            assert check(npa, ppa, data_type)

    # eye works on ints, floats, or bool
    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("data_type", INT_FLOAT_BOOL)
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_eye(self, data_type, prob_size):

        size = int(sqrt(prob_size))

        # ints and bools are checked for equality; floats are checked for closeness

        check = lambda a, b, t: (
            np.allclose(a.tolist(), b.tolist()) if akdtype(t) == "float64" else (a == b).all()
        )

        # test on one square and two non-square matrices

        for rows, cols in [(size, size), (size + 1, size - 1), (size - 1, size + 1)]:
            sweep = range(-(cols - 1), rows)  # sweeps the diagonal from LL to UR
            for diag in sweep:
                nda = np.eye(rows, cols, diag, dtype=data_type)
                pda = ak.eye(rows, cols, diag, dt=data_type).to_ndarray()
                assert check(nda, pda, data_type)

    # matmul works on ints, floats, or bool
    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("data_type1", INT_FLOAT_BOOL)
    @pytest.mark.parametrize("data_type2", INT_FLOAT_BOOL)
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_matmul(self, data_type1, data_type2, prob_size):

        size = int(sqrt(prob_size))

        # ints and bools are checked for equality; floats are checked for closeness

        check = lambda a, b, t: (
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

    # vecdot works on ints, floats, or bool, with the limitation that both inputs can't
    # be bool

    @pytest.mark.skip_if_max_rank_less_than(2)
    @pytest.mark.parametrize("data_type1", INT_FLOAT_BOOL)
    @pytest.mark.parametrize("data_type2", INT_FLOAT)
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_vecdot(self, data_type1, data_type2, prob_size):

        depth = np.random.randint(2, 10)
        width = prob_size // depth

        # ints and bools are checked for equality; floats are checked for closeness

        check = lambda a, b, t: (
            np.allclose(a.tolist(), b.tolist()) if akdtype(t) == "float64" else (a == b).all()
        )

        pda_a = ak.randint(0, 10, (depth, width), dtype=data_type1)
        nda_a = pda_a.to_ndarray()
        pda_b = ak.randint(0, 10, (depth, width), dtype=data_type2)
        nda_b = pda_b.to_ndarray()
        akProduct = ak.vecdot(pda_a, pda_b)

        # there is no vecdot in numpy (and vdot doesn't do the same thing).
        # np.add.reduce does.

        npProduct = np.add.reduce(nda_a * nda_b)
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
                        np.clip(nd_arry, None, hi[0]), ak.clip(ak_arry, None, hi[0]).to_ndarray()
                    )
                    assert np.allclose(
                        np.clip(nd_arry, None, hi), ak.clip(ak_arry, None, akhi).to_ndarray()
                    )
                    assert np.allclose(
                        np.clip(nd_arry, lo[0], hi[0]), ak.clip(ak_arry, lo[0], hi[0]).to_ndarray()
                    )
                    assert np.allclose(
                        np.clip(nd_arry, lo[0], hi), ak.clip(ak_arry, lo[0], akhi).to_ndarray()
                    )
                    assert np.allclose(
                        np.clip(nd_arry, lo[0], None), ak.clip(ak_arry, lo[0], None).to_ndarray()
                    )
                    assert np.allclose(
                        np.clip(nd_arry, lo, hi[0]), ak.clip(ak_arry, aklo, hi[0]).to_ndarray()
                    )
                    assert np.allclose(
                        np.clip(nd_arry, lo, hi), ak.clip(ak_arry, aklo, akhi).to_ndarray()
                    )
                    assert np.allclose(
                        np.clip(nd_arry, lo, None), ak.clip(ak_arry, aklo, None).to_ndarray()
                    )
