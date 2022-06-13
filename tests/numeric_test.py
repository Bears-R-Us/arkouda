import numpy as np
from base_test import ArkoudaTest
from context import arkouda as ak

from arkouda.dtypes import npstr

"""
Encapsulates unit tests for the numeric module with the exception
of the where method, which is in the where_test module
"""


class NumericTest(ArkoudaTest):
    def testSeededRNG(self):
        N = 100
        seed = 8675309
        numericdtypes = [ak.int64, ak.float64, ak.bool, ak.uint64]
        for dt in numericdtypes:
            # Make sure unseeded runs differ
            a = ak.randint(0, 2**32, N, dtype=dt)
            b = ak.randint(0, 2**32, N, dtype=dt)
            self.assertFalse((a == b).all())
            # Make sure seeded results are same
            a = ak.randint(0, 2**32, N, dtype=dt, seed=seed)
            b = ak.randint(0, 2**32, N, dtype=dt, seed=seed)
            self.assertTrue((a == b).all())
        # Uniform
        self.assertFalse((ak.uniform(N) == ak.uniform(N)).all())
        self.assertTrue((ak.uniform(N, seed=seed) == ak.uniform(N, seed=seed)).all())
        # Standard Normal
        self.assertFalse((ak.standard_normal(N) == ak.standard_normal(N)).all())
        self.assertTrue((ak.standard_normal(N, seed=seed) == ak.standard_normal(N, seed=seed)).all())
        # Strings (uniformly distributed length)
        self.assertFalse(
            (ak.random_strings_uniform(1, 10, N) == ak.random_strings_uniform(1, 10, N)).all()
        )
        self.assertTrue(
            (
                ak.random_strings_uniform(1, 10, N, seed=seed)
                == ak.random_strings_uniform(1, 10, N, seed=seed)
            ).all()
        )
        # Strings (log-normally distributed length)
        self.assertFalse(
            (ak.random_strings_lognormal(2, 1, N) == ak.random_strings_lognormal(2, 1, N)).all()
        )
        self.assertTrue(
            (
                ak.random_strings_lognormal(2, 1, N, seed=seed)
                == ak.random_strings_lognormal(2, 1, N, seed=seed)
            ).all()
        )

    def testCast(self):
        N = 100
        arrays = {
            ak.int64: ak.randint(-(2**48), 2**48, N),
            ak.float64: ak.randint(0, 1, N, dtype=ak.float64),
            ak.bool: ak.randint(0, 2, N, dtype=ak.bool),
        }
        roundtripable = set(
            (
                (ak.bool, ak.bool),
                (ak.int64, ak.int64),
                (ak.int64, ak.float64),
                (ak.int64, npstr),
                (ak.float64, ak.float64),
                (ak.float64, npstr),
                (ak.uint8, ak.int64),
                (ak.uint8, ak.float64),
                (ak.uint8, npstr),
            )
        )
        for t1, orig in arrays.items():
            for t2 in ak.DTypes:
                t2 = ak.dtype(t2)
                other = ak.cast(orig, t2)
                self.assertEqual(orig.size, other.size)
                if (t1, t2) in roundtripable:
                    roundtrip = ak.cast(other, t1)
                    self.assertTrue(
                        (orig == roundtrip).all(), f"{t1}: {orig[:5]}, {t2}: {roundtrip[:5]}"
                    )

        self.assertTrue((ak.array([1, 2, 3, 4, 5]) == ak.cast(ak.linspace(1, 5, 5), dt=ak.int64)).all())
        self.assertEqual(ak.cast(ak.arange(0, 5), dt=ak.float64).dtype, ak.float64)
        self.assertTrue(
            (
                ak.array([False, True, True, True, True]) == ak.cast(ak.linspace(0, 4, 5), dt=ak.bool)
            ).all()
        )

    def testStrCastErrors(self):
        intNAN = -(2**63)
        intstr = ak.array(["1", "2 ", "3?", "!4", "  5", "-45", "0b101", "0x30", "N/A"])
        intans = np.array([1, 2, intNAN, intNAN, 5, -45, 0b101, 0x30, intNAN])
        uintNAN = 0
        uintstr = ak.array(["1", "2 ", "3?", "-4", "  5", "45", "0b101", "0x30", "N/A"])
        uintans = np.array([1, 2, uintNAN, uintNAN, 5, 45, 0b101, 0x30, uintNAN])
        floatstr = ak.array(["1.1", "2.2 ", "3?.3", "4.!4", "  5.5", "6.6e-6", "78.91E+4", "6", "N/A"])
        floatans = np.array([1.1, 2.2, np.nan, np.nan, 5.5, 6.6e-6, 78.91e4, 6.0, np.nan])
        boolstr = ak.array(
            ["True", "False ", "Neither", "N/A", "  True", "true", "false", "TRUE", "NOTTRUE"]
        )
        boolans = np.array([True, False, False, False, True, True, False, True, False])
        validans = ak.array([True, True, False, False, True, True, True, True, False])
        for dt, arg, ans in [
            (ak.int64, intstr, intans),
            (ak.uint64, uintstr, uintans),
            (ak.float64, floatstr, floatans),
            (ak.bool, boolstr, boolans),
        ]:
            with self.assertRaises(RuntimeError):
                ak.cast(arg, dt, errors=ak.ErrorMode.strict)
            res = ak.cast(arg, dt, errors=ak.ErrorMode.ignore)
            self.assertTrue(np.allclose(ans, res.to_ndarray(), equal_nan=True))
            res, valid = ak.cast(arg, dt, errors=ak.ErrorMode.return_validity)
            self.assertTrue((valid == validans).all())
            self.assertTrue(np.allclose(ans, res.to_ndarray(), equal_nan=True))

    def testHistogram(self):
        pda = ak.randint(10, 30, 40)
        bins, result = ak.histogram(pda, bins=20)

        self.assertIsInstance(result, ak.pdarray)
        self.assertEqual(20, len(bins))
        self.assertEqual(20, len(result))
        self.assertEqual(int, result.dtype)

        with self.assertRaises(TypeError):
            ak.histogram([range(0,10)], bins=1)
        
        with self.assertRaises(TypeError):
            ak.histogram(pda, bins='1')
        
        with self.assertRaises(TypeError):
            ak.histogram([range(0,10)], bins='1')

    def testLog(self):
        na = np.linspace(1, 10, 10)
        pda = ak.array(na)

        self.assertTrue((np.log(na) == ak.log(pda).to_ndarray()).all())
        with self.assertRaises(TypeError):
            ak.log([range(0,10)])

    def testExp(self):
        na = np.linspace(1, 10, 10)
        pda = ak.array(na)

        self.assertTrue((np.exp(na) == ak.exp(pda).to_ndarray()).all())
        with self.assertRaises(TypeError):
            ak.exp([range(0,10)])

    def testAbs(self):
        na = np.linspace(1, 10, 10)
        pda = ak.array(na)

        self.assertTrue((np.abs(na) == ak.abs(pda).to_ndarray()).all())
        self.assertTrue((ak.arange(5, 1, -1) == ak.abs(ak.arange(-5, -1))).all())
        self.assertTrue((ak.array([5, 4, 3, 2, 1]) == ak.abs(ak.linspace(-5, -1, 5))).all())

        with self.assertRaises(TypeError):
            ak.abs([range(0, 10)])

    def testCumSum(self):
        na = np.linspace(1, 10, 10)
        pda = ak.array(na)

        self.assertTrue((np.cumsum(na) == ak.cumsum(pda).to_ndarray()).all())

        # Test uint case
        na = np.linspace(1, 10, 10, "uint64")
        pda = ak.cast(pda, ak.uint64)

        self.assertTrue((np.cumsum(na) == ak.cumsum(pda).to_ndarray()).all())

        with self.assertRaises(TypeError):
            ak.cumsum([range(0, 10)])

    def testCumProd(self):
        na = np.linspace(1, 10, 10)
        pda = ak.array(na)

        self.assertTrue((np.cumprod(na) == ak.cumprod(pda).to_ndarray()).all())
        with self.assertRaises(TypeError):
            ak.cumprod([range(0, 10)])

    def testSin(self):
        na = np.linspace(1, 10, 10)
        pda = ak.array(na)

        self.assertTrue((np.sin(na) == ak.sin(pda).to_ndarray()).all())
        with self.assertRaises(TypeError):
            ak.cos([range(0, 10)])

    def testCos(self):
        na = np.linspace(1, 10, 10)
        pda = ak.array(na)

        self.assertTrue((np.cos(na) == ak.cos(pda).to_ndarray()).all())
        with self.assertRaises(TypeError):
            ak.cos([range(0, 10)])

    def testHash(self):
        h1, h2 = ak.hash(ak.arange(10))
        rev = ak.arange(9, -1, -1)
        h3, h4 = ak.hash(rev)
        self.assertTrue((h1 == h3[rev]).all() and (h2 == h4[rev]).all())

        h1 = ak.hash(ak.arange(10), full=False)
        h3 = ak.hash(rev, full=False)
        self.assertTrue((h1 == h3[rev]).all())

        h = ak.hash(ak.linspace(0, 10, 10))
        self.assertTrue((h[0].dtype == ak.uint64) and (h[1].dtype == ak.uint64))

    def testValueCounts(self):
        pda = ak.ones(100, dtype=ak.int64)
        result = ak.value_counts(pda)

        self.assertEqual(ak.array([1]), result[0])
        self.assertEqual(ak.array([100]), result[1])

        pda = ak.linspace(1, 10, 10)
        with self.assertRaises(TypeError):
            ak.value_counts(pda)

        with self.assertRaises(TypeError):
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
        expected = np.isnan(npa)
        self.assertTrue(np.array_equal(expected, actual))

        # Currently we can't make an int64 array with a NaN in it so verify that we throw an Exception
        ark_s_int64 = ak.array(np.array([1, 2, 3, 4], dtype="int64"))
        with self.assertRaises(RuntimeError, msg="Currently isnan on int64 is not supported"):
            ak.isnan(ark_s_int64)

    def testPrecision(self):
        # See https://github.com/Bears-R-Us/arkouda/issues/964
        # Grouped sum was exacerbating floating point errors
        # This test verifies the fix
        N = 10**6
        G = N // 10
        ub = 2**63 // N
        groupnum = ak.randint(0, G, N, seed=1)
        intval = ak.randint(0, ub, N, seed=2)
        floatval = ak.cast(intval, ak.float64)
        g = ak.GroupBy(groupnum)
        _, intmean = g.mean(intval)
        _, floatmean = g.mean(floatval)
        ak_mse = ak.mean((intmean - floatmean) ** 2)
        self.assertTrue(np.isclose(ak_mse, 0.0))
