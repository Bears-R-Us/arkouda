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
            self.assertListEqual(a.to_list(), b.to_list())
        # Uniform
        self.assertFalse((ak.uniform(N) == ak.uniform(N)).all())
        self.assertListEqual(
            ak.uniform(N, seed=seed).to_list(),
            ak.uniform(N, seed=seed).to_list(),
        )
        # Standard Normal
        self.assertFalse((ak.standard_normal(N) == ak.standard_normal(N)).all())
        self.assertListEqual(
            ak.standard_normal(N, seed=seed).to_list(),
            ak.standard_normal(N, seed=seed).to_list(),
        )
        # Strings (uniformly distributed length)
        self.assertFalse(
            (ak.random_strings_uniform(1, 10, N) == ak.random_strings_uniform(1, 10, N)).all()
        )
        self.assertListEqual(
            ak.random_strings_uniform(1, 10, N, seed=seed).to_list(),
            ak.random_strings_uniform(1, 10, N, seed=seed).to_list(),
        )
        # Strings (log-normally distributed length)
        self.assertFalse(
            (ak.random_strings_lognormal(2, 1, N) == ak.random_strings_lognormal(2, 1, N)).all()
        )
        self.assertListEqual(
            ak.random_strings_lognormal(2, 1, N, seed=seed).to_list(),
            ak.random_strings_lognormal(2, 1, N, seed=seed).to_list(),
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
                if t1 == ak.float64 and t2 == ak.bigint:
                    # we don't support casting a float to a bigint
                    continue
                other = ak.cast(orig, t2)
                self.assertEqual(orig.size, other.size)
                if (t1, t2) in roundtripable:
                    roundtrip = ak.cast(other, t1)
                    self.assertTrue(
                        (orig == roundtrip).all(), f"{t1}: {orig[:5]}, {t2}: {roundtrip[:5]}"
                    )

        self.assertListEqual(
            [1, 2, 3, 4, 5],
            ak.cast(ak.linspace(1, 5, 5), dt=ak.int64).to_list(),
        )
        self.assertEqual(ak.cast(ak.arange(0, 5), dt=ak.float64).dtype, ak.float64)
        self.assertListEqual(
            [False, True, True, True, True],
            ak.cast(ak.linspace(0, 4, 5), dt=ak.bool).to_list(),
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
            self.assertListEqual(valid.to_list(), validans.to_list())
            self.assertTrue(np.allclose(ans, res.to_ndarray(), equal_nan=True))

    def testHistogram(self):
        pda = ak.randint(10, 30, 40)
        bins, result = ak.histogram(pda, bins=20)

        self.assertIsInstance(result, ak.pdarray)
        self.assertEqual(20, len(bins))
        self.assertEqual(20, len(result))
        self.assertEqual(int, result.dtype)

        with self.assertRaises(TypeError):
            ak.histogram([range(0, 10)], bins=1)

        with self.assertRaises(TypeError):
            ak.histogram(pda, bins="1")

        with self.assertRaises(TypeError):
            ak.histogram([range(0, 10)], bins="1")

    def testLog(self):
        na = np.linspace(1, 10, 10)
        pda = ak.array(na)

        self.assertTrue(np.allclose(np.log(na), ak.log(pda).to_ndarray()))
        with self.assertRaises(TypeError):
            ak.log([range(0, 10)])

    def testExp(self):
        na = np.linspace(1, 10, 10)
        pda = ak.array(na)

        self.assertTrue(np.allclose(np.exp(na), ak.exp(pda).to_ndarray()))
        with self.assertRaises(TypeError):
            ak.exp([range(0, 10)])

    def testAbs(self):
        na = np.linspace(1, 10, 10)
        pda = ak.array(na)

        self.assertTrue(np.allclose(np.abs(na), ak.abs(pda).to_ndarray()))
        self.assertListEqual(ak.arange(5, 1, -1).to_list(), ak.abs(ak.arange(-5, -1)).to_list())
        self.assertListEqual(
            [5, 4, 3, 2, 1],
            ak.abs(ak.linspace(-5, -1, 5)).to_list(),
        )

        with self.assertRaises(TypeError):
            ak.abs([range(0, 10)])

    def testCumSum(self):
        na = np.linspace(1, 10)
        pda = ak.array(na)

        self.assertTrue(np.allclose(np.cumsum(na), ak.cumsum(pda).to_ndarray()))

        # Test uint case
        na = np.arange(1, 10, dtype="uint64")
        pda = ak.array(na)

        self.assertTrue(np.allclose(np.cumsum(na), ak.cumsum(pda).to_ndarray()))

        with self.assertRaises(TypeError):
            ak.cumsum([range(0, 10)])

    def testCumProd(self):
        na = np.linspace(1, 10, 10)
        pda = ak.array(na)

        self.assertTrue(np.allclose(np.cumprod(na), ak.cumprod(pda).to_ndarray()))
        with self.assertRaises(TypeError):
            ak.cumprod([range(0, 10)])

    def testSin(self):
        na = np.arange(0, 10, dtype="int64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.sin(na), ak.sin(pda).to_ndarray()))

        na = np.arange(0, 10, dtype="uint64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.sin(na), ak.sin(pda).to_ndarray()))

        na = np.linspace(0, 10)
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.sin(na), ak.sin(pda).to_ndarray()))

        with self.assertRaises(TypeError):
            ak.sin([range(0, 10)])

    def testCos(self):
        na = np.arange(0, 10, dtype="int64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.cos(na), ak.cos(pda).to_ndarray()))

        na = np.arange(0, 10, dtype="uint64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.cos(na), ak.cos(pda).to_ndarray()))

        na = np.linspace(0, 10)
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.cos(na), ak.cos(pda).to_ndarray()))

        with self.assertRaises(TypeError):
            ak.cos([range(0, 10)])

    def testTan(self):
        na = np.arange(0, 10, dtype="int64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.tan(na), ak.tan(pda).to_ndarray()))

        na = np.arange(0, 10, dtype="uint64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.tan(na), ak.tan(pda).to_ndarray()))

        na = np.linspace(0, 10)
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.tan(na), ak.tan(pda).to_ndarray()))

        with self.assertRaises(TypeError):
            ak.tan([range(0, 10)])

    def testArcsin(self):
        na = np.arange(-1, 2, dtype="int64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.arcsin(na), ak.arcsin(pda).to_ndarray()))

        na = np.arange(0, 2, dtype="uint64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.arcsin(na), ak.arcsin(pda).to_ndarray()))

        na = np.linspace(-1, 1)
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.arcsin(na), ak.arcsin(pda).to_ndarray()))

        with self.assertRaises(TypeError):
            ak.arcsin([range(0, 10)])

    def testArccos(self):
        na = np.arange(-1, 2, dtype="int64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.arccos(na), ak.arccos(pda).to_ndarray()))

        na = np.arange(0, 2, dtype="uint64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.arccos(na), ak.arccos(pda).to_ndarray()))

        na = np.linspace(-1, 1)
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.arccos(na), ak.arccos(pda).to_ndarray()))

        with self.assertRaises(TypeError):
            ak.arccos([range(0, 10)])

    def testArctan(self):
        na = np.arange(0, 10, dtype="int64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.arctan(na), ak.arctan(pda).to_ndarray()))

        na = np.arange(0, 10, dtype="uint64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.arctan(na), ak.arctan(pda).to_ndarray()))

        na = np.linspace(0, 10)
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.arctan(na), ak.arctan(pda).to_ndarray()))

        with self.assertRaises(TypeError):
            ak.arctan([range(0, 10)])

        # Edge case: infinities
        na = np.array([np.inf, -np.inf])
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.arctan(na), ak.arctan(pda).to_ndarray(), equal_nan=True))

    def testArctan2(self):
        na1_int = np.arange(10, 0, -1, dtype="int64")
        pda1_int = ak.array(na1_int)
        # this also checks that a divide by zero error is properly handled
        na2_int = np.arange(0, 10, dtype="int64")
        pda2_int = ak.array(na2_int)

        na1_uint = np.arange(10, 0, -1, dtype="uint64")
        pda1_uint = ak.array(na1_uint)
        na2_uint = np.arange(0, 10, dtype="uint64")
        pda2_uint = ak.array(na2_uint)

        na1_float = np.linspace(0, 1, 10)
        pda1_float = ak.array(na1_float)
        na2_float = np.linspace(0, 10, 10)
        pda2_float = ak.array(na2_float)

        # vector-vector case
        self.assertTrue(
            np.allclose(np.arctan2(na1_int, na2_int), ak.arctan2(pda1_int, pda2_int).to_ndarray())
        )
        self.assertTrue(
            np.allclose(np.arctan2(na1_int, na2_uint), ak.arctan2(pda1_int, pda2_uint).to_ndarray())
        )
        self.assertTrue(
            np.allclose(np.arctan2(na1_int, na2_float), ak.arctan2(pda1_int, pda2_float).to_ndarray())
        )

        self.assertTrue(
            np.allclose(np.arctan2(na1_uint, na2_int), ak.arctan2(pda1_uint, pda2_int).to_ndarray())
        )
        self.assertTrue(
            np.allclose(np.arctan2(na1_uint, na2_uint), ak.arctan2(pda1_uint, pda2_uint).to_ndarray())
        )
        self.assertTrue(
            np.allclose(np.arctan2(na1_uint, na2_float), ak.arctan2(pda1_uint, pda2_float).to_ndarray())
        )

        self.assertTrue(
            np.allclose(np.arctan2(na1_float, na2_int), ak.arctan2(pda1_float, pda2_int).to_ndarray())
        )
        self.assertTrue(
            np.allclose(np.arctan2(na1_float, na2_uint), ak.arctan2(pda1_float, pda2_uint).to_ndarray())
        )
        self.assertTrue(
            np.allclose(
                np.arctan2(na1_float, na2_float), ak.arctan2(pda1_float, pda2_float).to_ndarray()
            )
        )

        with self.assertRaises(TypeError):
            ak.arctan2([range(0, 10)], [range(0, 10)])

        # vector-scalar case
        denom = np.array([5]).astype(np.uint)  # work around to get a scalar uint

        self.assertTrue(np.allclose(np.arctan2(na1_int, 5), ak.arctan2(pda1_int, 5).to_ndarray()))
        self.assertTrue(
            np.allclose(np.arctan2(na1_int, denom[0]), ak.arctan2(pda1_int, denom[0]).to_ndarray())
        )
        self.assertTrue(np.allclose(np.arctan2(na1_int, 5.0), ak.arctan2(pda1_int, 5.0).to_ndarray()))

        self.assertTrue(np.allclose(np.arctan2(na1_uint, 5), ak.arctan2(pda1_uint, 5).to_ndarray()))
        self.assertTrue(
            np.allclose(np.arctan2(na1_uint, denom[0]), ak.arctan2(pda1_uint, denom[0]).to_ndarray())
        )
        self.assertTrue(np.allclose(np.arctan2(na1_uint, 5.0), ak.arctan2(pda1_uint, 5.0).to_ndarray()))

        self.assertTrue(np.allclose(np.arctan2(na1_float, 5), ak.arctan2(pda1_float, 5).to_ndarray()))
        self.assertTrue(
            np.allclose(np.arctan2(na1_float, denom[0]), ak.arctan2(pda1_float, denom[0]).to_ndarray())
        )
        self.assertTrue(
            np.allclose(np.arctan2(na1_float, 5.0), ak.arctan2(pda1_float, 5.0).to_ndarray())
        )

        with self.assertRaises(TypeError):
            ak.arctan2([range(0, 10)], 5)
        with self.assertRaises(TypeError):
            ak.arctan2([range(0, 10)], denom[0])
        with self.assertRaises(TypeError):
            ak.arctan2([range(0, 10)], 5.0)

        # scalar-vector case
        num = np.array([1]).astype(np.uint)  # work around to get a scalar uint

        self.assertTrue(np.allclose(np.arctan2(1, na2_int), ak.arctan2(1, pda2_int).to_ndarray()))
        self.assertTrue(np.allclose(np.arctan2(1, na2_uint), ak.arctan2(1, pda2_uint).to_ndarray()))
        self.assertTrue(np.allclose(np.arctan2(1, na2_float), ak.arctan2(1, pda2_float).to_ndarray()))

        self.assertTrue(
            np.allclose(np.arctan2(num[0], na2_int), ak.arctan2(num[0], pda2_int).to_ndarray())
        )
        self.assertTrue(
            np.allclose(np.arctan2(num[0], na2_uint), ak.arctan2(num[0], pda2_uint).to_ndarray())
        )
        self.assertTrue(
            np.allclose(np.arctan2(num[0], na2_float), ak.arctan2(num[0], pda2_float).to_ndarray())
        )

        self.assertTrue(np.allclose(np.arctan2(1.0, na2_int), ak.arctan2(1.0, pda2_int).to_ndarray()))
        self.assertTrue(np.allclose(np.arctan2(1.0, na2_uint), ak.arctan2(1.0, pda2_uint).to_ndarray()))
        self.assertTrue(
            np.allclose(np.arctan2(1.0, na2_float), ak.arctan2(1.0, pda2_float).to_ndarray())
        )

        with self.assertRaises(TypeError):
            ak.arctan2(1, [range(0, 10)])
        with self.assertRaises(TypeError):
            ak.arctan2(num[0], [range(0, 10)])
        with self.assertRaises(TypeError):
            ak.arctan2(1.0, [range(0, 10)])

        # Edge case: Infinities and Zeros
        na1 = np.array([np.inf, -np.inf])
        pda1 = ak.array(na1)
        na2 = np.array([1, 10])
        pda2 = ak.array(na2)

        self.assertTrue(
            np.allclose(np.arctan2(na1, na2), ak.arctan2(pda1, pda2).to_ndarray(), equal_nan=True)
        )
        self.assertTrue(
            np.allclose(np.arctan2(na2, na1), ak.arctan2(pda2, pda1).to_ndarray(), equal_nan=True)
        )
        self.assertTrue(
            np.allclose(np.arctan2(na1, 5), ak.arctan2(pda1, 5).to_ndarray(), equal_nan=True)
        )
        self.assertTrue(
            np.allclose(np.arctan2(5, na1), ak.arctan2(5, pda1).to_ndarray(), equal_nan=True)
        )
        self.assertTrue(
            np.allclose(np.arctan2(na1, 0), ak.arctan2(pda1, 0).to_ndarray(), equal_nan=True)
        )
        self.assertTrue(
            np.allclose(np.arctan2(0, na1), ak.arctan2(0, pda1).to_ndarray(), equal_nan=True)
        )
    
    def testSinh(self):
        na = np.arange(0, 10, dtype="int64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.sinh(na), ak.sinh(pda).to_ndarray()))

        na = np.arange(0, 10, dtype="uint64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.sinh(na), ak.sinh(pda).to_ndarray()))

        na = np.linspace(0, 10)
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.sinh(na), ak.sinh(pda).to_ndarray()))

        with self.assertRaises(TypeError):
            ak.sinh([range(0, 10)])

        # Edge case: infinities
        na = np.array([np.inf, -np.inf])
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.sinh(na), ak.sinh(pda).to_ndarray(), equal_nan=True))


    def testCosh(self):
        na = np.arange(0, 10, dtype="int64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.cosh(na), ak.cosh(pda).to_ndarray()))

        na = np.arange(0, 10, dtype="uint64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.cosh(na), ak.cosh(pda).to_ndarray()))

        na = np.linspace(0, 10)
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.cosh(na), ak.cosh(pda).to_ndarray()))

        with self.assertRaises(TypeError):
            ak.cosh([range(0, 10)])

        # Edge case: infinities
        na = np.array([np.inf, -np.inf])
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.cosh(na), ak.cosh(pda).to_ndarray(), equal_nan=True))


    def testTanh(self):
        na = np.arange(0, 10, dtype="int64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.tanh(na), ak.tanh(pda).to_ndarray()))

        na = np.arange(0, 10, dtype="uint64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.tanh(na), ak.tanh(pda).to_ndarray()))

        na = np.linspace(0, 10)
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.tanh(na), ak.tanh(pda).to_ndarray()))

        with self.assertRaises(TypeError):
            ak.tanh([range(0, 10)])

        # Edge case: infinities
        na = np.array([np.inf, -np.inf])
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.tanh(na), ak.tanh(pda).to_ndarray(), equal_nan=True))
    

    def testArcsinh(self):
        na = np.arange(0, 10, dtype="int64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.arcsinh(na), ak.arcsinh(pda).to_ndarray()))

        na = np.arange(0, 10, dtype="uint64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.arcsinh(na), ak.arcsinh(pda).to_ndarray()))

        na = np.linspace(0, 10)
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.arcsinh(na), ak.arcsinh(pda).to_ndarray()))

        with self.assertRaises(TypeError):
            ak.arcsinh([range(0, 10)])

        # Edge case: infinities
        na = np.array([np.inf, -np.inf])
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.arcsinh(na), ak.arcsinh(pda).to_ndarray(), equal_nan=True))


    def testArccosh(self):
        na = np.arange(1, 10, dtype="int64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.arccosh(na), ak.arccosh(pda).to_ndarray()))

        na = np.arange(1, 10, dtype="uint64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.arccosh(na), ak.arccosh(pda).to_ndarray()))

        na = np.linspace(1, 10)
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.arccosh(na), ak.arccosh(pda).to_ndarray()))

        with self.assertRaises(TypeError):
            ak.arccosh([range(0, 10)])

        # Edge case: infinities
        na = np.array([1, np.inf])
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.arccosh(na).tolist(), ak.arccosh(pda).to_list(), equal_nan=True))
        self.assertTrue(np.allclose(np.arccosh(na), ak.arccosh(pda).to_ndarray(), equal_nan=True))


    def testArctanh(self):
        na = np.arange(-1, 2, dtype="int64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.arctanh(na), ak.arctanh(pda).to_ndarray()))

        na = np.arange(0, 2, dtype="uint64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.arctanh(na), ak.arctanh(pda).to_ndarray()))

        na = np.linspace(-1, 1)
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.arctanh(na), ak.arctanh(pda).to_ndarray()))

        with self.assertRaises(TypeError):
            ak.arctanh([range(0, 10)])

    def testRad2deg(self):
        na = np.arange(0, 10, dtype="int64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.rad2deg(na), ak.rad2deg(pda).to_ndarray()))

        na = np.arange(0, 10, dtype="uint64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.rad2deg(na), ak.rad2deg(pda).to_ndarray()))

        na = np.linspace(0, 10)
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.rad2deg(na), ak.rad2deg(pda).to_ndarray()))

        with self.assertRaises(TypeError):
            ak.rad2deg([range(0, 10)])

    def testDeg2rad(self):
        na = np.arange(0, 10, dtype="int64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.deg2rad(na), ak.deg2rad(pda).to_ndarray()))

        na = np.arange(0, 10, dtype="uint64")
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.deg2rad(na), ak.deg2rad(pda).to_ndarray()))

        na = np.linspace(0, 10)
        pda = ak.array(na)
        self.assertTrue(np.allclose(np.deg2rad(na), ak.deg2rad(pda).to_ndarray()))

        with self.assertRaises(TypeError):
            ak.deg2rad([range(0, 10)])

    def testHash(self):
        h1, h2 = ak.hash(ak.arange(10))
        rev = ak.arange(9, -1, -1)
        h3, h4 = ak.hash(rev)
        self.assertListEqual(h1.to_list(), h3[rev].to_list())
        self.assertListEqual(h2.to_list(), h4[rev].to_list())

        h1 = ak.hash(ak.arange(10), full=False)
        h3 = ak.hash(rev, full=False)
        self.assertListEqual(h1.to_list(), h3[rev].to_list())

        h = ak.hash(ak.linspace(0, 10, 10))
        self.assertEqual(h[0].dtype, ak.uint64)
        self.assertEqual(h[1].dtype, ak.uint64)

        # test strings hash
        s = ak.random_strings_uniform(4, 8, 10)
        h1, h2 = ak.hash(s)
        rh1, rh2 = ak.hash(s[rev])
        self.assertListEqual(h1.to_list(), rh1[rev].to_list())
        self.assertListEqual(h2.to_list(), rh2[rev].to_list())

        # verify all the ways to hash strings match
        h3, h4 = ak.hash([s])
        self.assertListEqual(h1.to_list(), h3.to_list())
        self.assertListEqual(h2.to_list(), h4.to_list())
        h5, h6 = s.hash()
        self.assertListEqual(h1.to_list(), h5.to_list())
        self.assertListEqual(h2.to_list(), h6.to_list())

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
                self.assertListEqual(h1.to_list(), h3.to_list())
                self.assertListEqual(h2.to_list(), h4.to_list())
                h5, h6 = h.hash()
                self.assertListEqual(h1.to_list(), h5.to_list())
                self.assertListEqual(h2.to_list(), h6.to_list())
            # the first and third position are identical and should hash to the same thing
            self.assertEqual(h1[0], h1[2])
            self.assertEqual(h2[0], h2[2])
            # make sure the last position didn't get zeroed out by XOR
            self.assertNotEqual(h1[3], 0)
            self.assertNotEqual(h2[3], 0)

        sa = ak.SegArray(ak.array([0, 2]), ak.array([1, 1, 2, 2]))
        h1, h2 = sa.hash()
        # verify these segments don't collide (this is why we rehash)
        self.assertNotEqual(h1[0], h1[1])
        self.assertNotEqual(h2[0], h2[1])

        # test categorical hash
        categories, codes = ak.array([f"str {i}" for i in range(3)]), ak.randint(0, 3, 10**5)
        my_cat = ak.Categorical.from_codes(codes=codes, categories=categories)
        h1, h2 = ak.hash(my_cat)
        rev = ak.arange(10**5)[::-1]
        rh1, rh2 = ak.hash(my_cat[rev])
        self.assertListEqual(h1.to_list(), rh1[rev].to_list())
        self.assertListEqual(h2.to_list(), rh2[rev].to_list())

        # verify all the ways to hash Categoricals match
        h3, h4 = ak.hash([my_cat])
        self.assertListEqual(h1.to_list(), h3.to_list())
        self.assertListEqual(h2.to_list(), h4.to_list())
        h5, h6 = my_cat.hash()
        self.assertListEqual(h1.to_list(), h5.to_list())
        self.assertListEqual(h2.to_list(), h6.to_list())

        # verify it matches hashing the categories and then indexing with codes
        sh1, sh2 = my_cat.categories.hash()
        h7, h8 = sh1[my_cat.codes], sh2[my_cat.codes]
        self.assertListEqual(h1.to_list(), h7.to_list())
        self.assertListEqual(h2.to_list(), h8.to_list())

        # verify all the ways to hash bigint pdarrays match
        h1, h2 = ak.hash(bi)
        h3, h4 = ak.hash([bi])
        self.assertListEqual(h1.to_list(), h3.to_list())
        self.assertListEqual(h2.to_list(), h4.to_list())

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
        self.assertTrue(np.array_equal(np.isnan(npa), actual))

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
