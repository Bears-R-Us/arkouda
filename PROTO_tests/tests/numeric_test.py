import arkouda as ak
import pytest
import numpy as np

from arkouda.dtypes import npstr

NUMERIC_TYPES = [ak.int64, ak.float64, ak.bool, ak.uint64]
NO_BOOL = [ak.int64, ak.float64, ak.uint64]
NO_FLOAT = [ak.int64, ak.bool, ak.uint64]
INT_FLOAT = [ak.int64, ak.float64]
CAST_TYPES = [ak.dtype(t) for t in ak.DTypes]

ROUNDTRIP_CAST = [
    (ak.bool, ak.bool),
    (ak.int64, ak.int64),
    (ak.int64, ak.float64),
    (ak.int64, npstr),
    (ak.float64, ak.float64),
    (ak.float64, npstr),
    (ak.uint8, ak.int64),
    (ak.uint8, ak.float64),
    (ak.uint8, npstr),
]


class TestNumeric:
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

    @pytest.mark.parametrize("cast_to", CAST_TYPES)
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_cast(self, prob_size, cast_to):
        arrays = {
            ak.int64: ak.randint(-(2**48), 2**48, prob_size),
            ak.float64: ak.randint(0, 1, prob_size, dtype=ak.float64),
            ak.bool: ak.randint(0, 2, prob_size, dtype=ak.bool),
        }

        for t1, orig in arrays.items():
            if t1 == ak.float64 and cast_to == ak.bigint:
                # we don't support casting a float to a bigint
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
            strarr = ak.array(["1.1", "2.2 ", "3?.3", "4.!4", "  5.5", "6.6e-6", "78.91E+4", "6", "N/A"])
            ans = np.array([1.1, 2.2, np.nan, np.nan, 5.5, 6.6e-6, 78.91e4, 6.0, np.nan])
        elif num_type == ak.bool:
            strarr = ak.array(
                ["True", "False ", "Neither", "N/A", "  True", "true", "false", "TRUE", "NOTTRUE"]
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

    @pytest.mark.parametrize("num_type", INT_FLOAT)
    def test_histogram(self, num_type):
        pda = ak.randint(10, 30, 40, dtype=num_type)
        bins, result = ak.histogram(pda, bins=20)

        assert isinstance(result, ak.pdarray)
        assert 20 == len(bins)
        assert 20 == len(result)
        assert int == result.dtype

        with pytest.raises(TypeError):
            ak.histogram(np.array([range(0, 10)]).astype(num_type), bins=1)

        with pytest.raises(TypeError):
            ak.histogram(pda, bins="1")

        with pytest.raises(TypeError):
            ak.histogram(np.array([range(0, 10)]).astype(num_type), bins="1")

    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_log(self, num_type):
        na = np.linspace(1, 10, 10).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        assert np.allclose(np.log(na), ak.log(pda).to_ndarray())
        with pytest.raises(TypeError):
            ak.log(np.array([range(0, 10)]).astype(num_type))

    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_exp(self, num_type):
        na = np.linspace(1, 10, 10).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        assert np.allclose(np.exp(na), ak.exp(pda).to_ndarray())
        with pytest.raises(TypeError):
            ak.exp(np.array([range(0, 10)]).astype(num_type))

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

    @pytest.mark.parametrize("num_type", NUMERIC_TYPES)
    def test_cumsum(self, num_type):
        na = np.linspace(1, 10, 10).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        assert np.allclose(np.cumsum(na), ak.cumsum(pda).to_ndarray())
        with pytest.raises(TypeError):
            ak.cumsum(np.array([range(0, 10)]).astype(num_type))

    @pytest.mark.parametrize("num_type", NUMERIC_TYPES)
    def test_cumprod(self, num_type):
        na = np.linspace(1, 10, 10).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        assert np.allclose(np.cumprod(na), ak.cumprod(pda).to_ndarray())
        with pytest.raises(TypeError):
            ak.cumprod(np.array([range(0, 10)]).astype(num_type))

    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_sin(self, num_type):
        if num_type == ak.float64:
            na = np.linspace(0, 10).astype(num_type)
        else:
            na = np.arange(0, 10).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        assert np.allclose(np.sin(na), ak.sin(pda).to_ndarray())
        with pytest.raises(TypeError):
            ak.sin(np.array([range(0, 10)]).astype(num_type))

    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_cos(self, num_type):
        if num_type == ak.float64:
            print(num_type)
            na = np.linspace(0, 10).astype(num_type)
        else:
            na = np.arange(0, 10).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        assert np.allclose(np.cos(na), ak.cos(pda).to_ndarray())
        with pytest.raises(TypeError):
            ak.cos(np.array([range(0, 10)]).astype(num_type))

    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_tan(self, num_type):
        if num_type == ak.float64:
            na = np.linspace(0, 10).astype(num_type)
        else:
            na = np.arange(0, 10).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        assert np.allclose(np.tan(na), ak.tan(pda).to_ndarray())
        with pytest.raises(TypeError):
            ak.tan(np.array([range(0, 10)]).astype(num_type))

    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_arcsin(self, num_type):
        if num_type == ak.uint64:
            na = np.arange(0, 2).astype(num_type)
        elif num_type == ak.float64:
            na = np.linspace(-1, 1).astype(num_type)
        else:
            na = np.arange(-1, 2).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        assert np.allclose(np.arcsin(na), ak.arcsin(pda).to_ndarray())
        with pytest.raises(TypeError):
            ak.arcsin(np.array([range(0, 10)]).astype(num_type))

    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_arccos(self, num_type):
        print("mytest")
        if num_type == ak.uint64:
            na = np.arange(0, 2).astype(num_type)
        elif num_type == ak.float64:
            na = np.linspace(-1, 1).astype(num_type)
        else:
            na = np.arange(-1, 2).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        assert np.allclose(np.arccos(na), ak.arccos(pda).to_ndarray())
        with pytest.raises(TypeError):
            ak.arccos(np.array([range(0, 10)]).astype(num_type))

    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_arctan(self, num_type):
        if num_type == ak.float64:
            na = np.linspace(0, 10).astype(num_type)
        else:
            na = np.arange(0, 10).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        assert np.allclose(np.arctan(na), ak.arctan(pda).to_ndarray())
        with pytest.raises(TypeError):
            ak.arctan(np.array([range(0, 10)]).astype(num_type))

        # Edge case: infinities
        if num_type == ak.float64:
            na = np.array([np.inf, -np.inf])
            pda = ak.array(na)
            assert np.allclose(np.arctan(na), ak.arctan(pda).to_ndarray(), equal_nan=True)

    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_sinh(self, num_type):
        if num_type == ak.float64:
            na = np.linspace(0, 10).astype(num_type)
        else:
            na = np.arange(0, 10).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        assert np.allclose(np.sinh(na), ak.sinh(pda).to_ndarray())
        with pytest.raises(TypeError):
            ak.sinh(np.array([range(0, 10)]).astype(num_type))

        # Edge case: infinities
        if num_type == ak.float64:
            na = np.array([np.inf, -np.inf])
            pda = ak.array(na)
            assert np.allclose(np.sinh(na), ak.sinh(pda).to_ndarray(), equal_nan=True)

    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_cosh(self, num_type):
        if num_type == ak.float64:
            print(num_type)
            na = np.linspace(0, 10).astype(num_type)
        else:
            na = np.arange(0, 10).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        assert np.allclose(np.cosh(na), ak.cosh(pda).to_ndarray())
        with pytest.raises(TypeError):
            ak.cosh(np.array([range(0, 10)]).astype(num_type))

        # Edge case: infinities
        if num_type == ak.float64:
            na = np.array([np.inf, -np.inf])
            pda = ak.array(na)
            assert np.allclose(np.cosh(na), ak.cosh(pda).to_ndarray(), equal_nan=True)

    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_tanh(self, num_type):
        if num_type == ak.float64:
            na = np.linspace(0, 10).astype(num_type)
        else:
            na = np.arange(0, 10).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        assert np.allclose(np.tanh(na), ak.tanh(pda).to_ndarray())
        with pytest.raises(TypeError):
            ak.tanh(np.array([range(0, 10)]).astype(num_type))

        # Edge case: infinities
        if num_type == ak.float64:
            na = np.array([np.inf, -np.inf])
            pda = ak.array(na)
            assert np.allclose(np.tanh(na), ak.tanh(pda).to_ndarray(), equal_nan=True)

    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_arcsinh(self, num_type):
        if num_type == ak.float64:
            na = np.linspace(0, 10).astype(num_type)
        else:
            na = np.arange(0, 10).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        assert np.allclose(np.arcsinh(na), ak.arcsinh(pda).to_ndarray())
        with pytest.raises(TypeError):
            ak.arcsinh(np.array([range(0, 10)]).astype(num_type))

        # Edge case: infinities
        if num_type == ak.float64:
            na = np.array([np.inf, -np.inf])
            pda = ak.array(na)
            assert np.allclose(np.arcsinh(na), ak.arcsinh(pda).to_ndarray(), equal_nan=True)

    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_arccosh(self, num_type):
        if num_type == ak.float64:
            print(num_type)
            na = np.linspace(1, 10).astype(num_type)
        else:
            na = np.arange(1, 10).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        assert np.allclose(np.arccosh(na), ak.arccosh(pda).to_ndarray())
        with pytest.raises(TypeError):
            ak.arccosh(np.array([range(0, 10)]).astype(num_type))

        # Edge case: infinities
        if num_type == ak.float64:
            na = np.array([1, np.inf])
            pda = ak.array(na)
            assert np.allclose(np.arccosh(na), ak.arccosh(pda).to_ndarray(), equal_nan=True)

    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_arctanh(self, num_type):
        if num_type == ak.uint64:
            na = np.arange(0, 2).astype(num_type)
        elif num_type == ak.float64:
            na = np.linspace(-1, 1).astype(num_type)
        else:
            na = np.arange(-1, 2).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        assert np.allclose(np.arctanh(na), ak.arctanh(pda).to_ndarray())
        with pytest.raises(TypeError):
            ak.arctanh(np.array([range(0, 10)]).astype(num_type))

    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_rad2deg(self, num_type):
        if num_type == ak.float64:
            na = np.linspace(0, 10).astype(num_type)
        else:
            na = np.arange(0, 10).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        assert np.allclose(np.rad2deg(na), ak.rad2deg(pda).to_ndarray())
        with pytest.raises(TypeError):
            ak.rad2deg(np.array([range(0, 10)]).astype(num_type))

    @pytest.mark.parametrize("num_type", NO_BOOL)
    def test_deg2rad(self, num_type):
        if num_type == ak.float64:
            na = np.linspace(0, 10).astype(num_type)
        else:
            na = np.arange(0, 10).astype(num_type)
        pda = ak.array(na, dtype=num_type)

        assert np.allclose(np.deg2rad(na), ak.deg2rad(pda).to_ndarray())
        with pytest.raises(TypeError):
            ak.deg2rad(np.array([range(0, 10)]).astype(num_type))

    @pytest.mark.parametrize("num_type", NO_FLOAT)
    def test_value_counts(self, num_type):
        pda = ak.ones(100, dtype=num_type)
        result = ak.value_counts(pda)

        assert ak.array([1]) == result[0]
        assert ak.array([100]) == result[1]

    def test_value_counts_error(self):
        pda = ak.linspace(1, 10, 10)
        with pytest.raises(TypeError):
            ak.value_counts(pda)

        with pytest.raises(TypeError):
            ak.value_counts([0])

    def test_isnan(self):
        """
        Test efunc `isnan`; it returns a pdarray of element-wise T/F values for whether it is NaN
        (not a number)
        Currently we only support float based arrays since numpy doesn't support NaN in int-based arrays
        """
        npa = np.array([1, 2, None, 3, 4], dtype="float64")
        ark_s_float64 = ak.array(npa)
        ark_isna_float64 = ak.isnan(ark_s_float64)
        actual = ark_isna_float64.to_ndarray()
        assert np.array_equal(np.isnan(npa), actual)

        # Currently we can't make an int64 array with a NaN in it so verify that we throw an Exception
        ark_s_int64 = ak.array(np.array([1, 2, 3, 4], dtype="int64"))
        with pytest.raises(RuntimeError):
            ak.isnan(ark_s_int64)

    def test_precision(self):
        # See https://github.com/Bears-R-Us/arkouda/issues/964
        # Grouped sum was exacerbating floating point errors
        # This test verifies the fix
        N = 10**6  # TODO - should this be set to prob_size?
        G = N // 10
        ub = 2**63 // N
        groupnum = ak.randint(0, G, N, seed=1)
        intval = ak.randint(0, ub, N, seed=2)
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
