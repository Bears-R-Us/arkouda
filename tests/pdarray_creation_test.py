import datetime as dt
import math
import statistics
from collections import deque

import numpy as np
import pandas as pd
from base_test import ArkoudaTest
from context import arkouda as ak

"""
Encapsulates test cases for pdarray creation methods
"""


class PdarrayCreationTest(ArkoudaTest):
    def testArrayCreation(self):
        pda = ak.array(np.ones(100))
        self.assertIsInstance(pda, ak.pdarray)
        self.assertEqual(100, len(pda))
        self.assertEqual(float, pda.dtype)

        pda = ak.array(list(range(0, 100)))
        self.assertIsInstance(pda, ak.pdarray)
        self.assertEqual(100, len(pda))
        self.assertEqual(int, pda.dtype)

        pda = ak.array((range(5)))
        self.assertIsInstance(pda, ak.pdarray)
        self.assertEqual(5, len(pda))
        self.assertEqual(int, pda.dtype)

        pda = ak.array(deque(range(5)))
        self.assertEqual(5, len(pda))
        self.assertEqual(int, pda.dtype)

        pda = ak.array([f"{i}" for i in range(10)], dtype=ak.int64)
        self.assertIsInstance(pda, ak.pdarray)
        self.assertEqual(10, len(pda))
        self.assertEqual(int, pda.dtype)

        av = ak.array(np.array([[0, 1], [0, 1]]))
        self.assertIsInstance(av, ak.ArrayView)

        with self.assertRaises(TypeError):
            ak.array({range(0, 100)})

        with self.assertRaises(TypeError):
            ak.array("not an iterable")

        with self.assertRaises(TypeError):
            ak.array(list(list(0)))

    def test_bigint_creation(self):
        bi = 2**200

        pda_from_str = ak.array([f"{i}" for i in range(bi, bi + 10)], dtype=ak.bigint)
        pda_from_int = ak.array([i for i in range(bi, bi + 10)])
        cast_from_segstr = ak.cast(ak.array([f"{i}" for i in range(bi, bi + 10)]), ak.bigint)
        for pda in [pda_from_str, pda_from_int, cast_from_segstr]:
            self.assertIsInstance(pda, ak.pdarray)
            self.assertEqual(10, len(pda))
            self.assertEqual(ak.bigint, pda.dtype)
            self.assertEqual(pda[-1], bi + 9)

        # test array and arange infer dtype
        self.assertListEqual(
            ak.array([bi, bi + 1, bi + 2, bi + 3, bi + 4]).to_list(), ak.arange(bi, bi + 5).to_list()
        )

        # test that max_bits being set results in a mod
        self.assertListEqual(
            ak.arange(bi, bi + 5, max_bits=200).to_list(),
            ak.arange(5).to_list(),
        )

        # test ak.bigint_from_uint_arrays
        # top bits are all 1 which should be 2**64
        top_bits = ak.ones(5, ak.uint64)
        bot_bits = ak.arange(5, dtype=ak.uint64)
        two_arrays = ak.bigint_from_uint_arrays([top_bits, bot_bits])
        self.assertEqual(ak.bigint, two_arrays.dtype)
        self.assertListEqual(two_arrays.to_list(), [2**64 + i for i in range(5)])
        # top bits should represent 2**128
        mid_bits = ak.zeros(5, ak.uint64)
        three_arrays = ak.bigint_from_uint_arrays([top_bits, mid_bits, bot_bits])
        self.assertListEqual(three_arrays.to_list(), [2**128 + i for i in range(5)])

        # test round_trip of ak.bigint_to/from_uint_arrays
        t = ak.arange(bi - 1, bi + 9)
        t_dup = ak.bigint_from_uint_arrays(t.bigint_to_uint_arrays())
        self.assertListEqual(t.to_list(), t_dup.to_list())
        self.assertEqual(t_dup.max_bits, -1)

        # test setting max_bits after creation still mods
        t_dup.max_bits = 200
        self.assertListEqual(t_dup.to_list(), [bi - 1, 0, 1, 2, 3, 4, 5, 6, 7, 8])

        # test slice_bits along 64 bit boundaries matches return from bigint_to_uint_arrays
        for i, uint_bits in enumerate(t.bigint_to_uint_arrays()):
            slice_bits = t.slice_bits(64 * (4 - (i + 1)), 64 * (4 - i) - 1)
            self.assertListEqual(uint_bits.to_list(), slice_bits.to_list())

    def test_arange(self):
        self.assertListEqual([0, 1, 2, 3, 4], ak.arange(0, 5, 1).to_list())
        self.assertListEqual([5, 4, 3, 2, 1], ak.arange(5, 0, -1).to_list())
        self.assertListEqual(
            [-5, -6, -7, -8, -9],
            ak.arange(-5, -10, -1).to_list(),
        )
        self.assertListEqual([0, 2, 4, 6, 8], ak.arange(0, 10, 2).to_list())

    def test_arange_dtype(self):
        # test dtype works with optional start/stride
        uint_stop = ak.arange(3, dtype=ak.uint64)
        self.assertListEqual([0, 1, 2], uint_stop.to_list())
        self.assertEqual(ak.uint64, uint_stop.dtype)

        uint_start_stop = ak.arange(2**63 + 3, 2**63 + 7)
        ans = ak.arange(3, 7, dtype=ak.uint64) + 2**63
        self.assertListEqual(ans.to_list(), uint_start_stop.to_list())
        self.assertEqual(ak.uint64, uint_start_stop.dtype)

        uint_start_stop_stride = ak.arange(3, 7, 2, dtype=ak.uint64)
        self.assertListEqual([3, 5], uint_start_stop_stride.to_list())
        self.assertEqual(ak.uint64, uint_start_stop_stride.dtype)

        # test uint64 handles negatives correctly
        np_arange_uint = np.arange(2**64 - 5, 2**64 - 10, -1, dtype=np.uint64)
        ak_arange_uint = ak.arange(-5, -10, -1, dtype=ak.uint64)
        # np_arange_uint = array([18446744073709551611, 18446744073709551610, 18446744073709551609,
        #        18446744073709551608, 18446744073709551607], dtype=uint64)
        self.assertListEqual(np_arange_uint.tolist(), ak_arange_uint.to_list())
        self.assertEqual(ak.uint64, ak_arange_uint.dtype)

        # test correct conversion to float64
        np_arange_float = np.arange(-5, -10, -1, dtype=np.float64)
        ak_arange_float = ak.arange(-5, -10, -1, dtype=ak.float64)
        # array([-5., -6., -7., -8., -9.])
        self.assertListEqual(np_arange_float.tolist(), ak_arange_float.to_list())
        self.assertEqual(ak.float64, ak_arange_float.dtype)

        # test correct conversion to bool
        expected_bool = [False, True, True, True, True]
        ak_arange_bool = ak.arange(0, 10, 2, dtype=ak.bool_)
        self.assertListEqual(expected_bool, ak_arange_bool.to_list())
        self.assertEqual(ak.bool_, ak_arange_bool.dtype)

        # test uint64 input works
        uint_input = ak.arange(3, dtype=ak.uint64)
        self.assertListEqual([0, 1, 2], uint_input.to_list())
        self.assertEqual(ak.uint64, uint_input.dtype)

        # test int_scalars covers uint8, uint16, uint32
        ak.arange(np.uint8(1), np.uint16(1000), np.uint32(1))

    def test_randint(self):
        testArray = ak.randint(0, 10, 5)
        self.assertIsInstance(testArray, ak.pdarray)
        self.assertEqual(5, len(testArray))
        self.assertEqual(ak.int64, testArray.dtype)
        self.assertEqual([5], testArray.shape)

        testArray = ak.randint(np.int64(0), np.int64(10), np.int64(5))
        self.assertIsInstance(testArray, ak.pdarray)
        self.assertEqual(5, len(testArray))
        self.assertEqual(ak.int64, testArray.dtype)
        self.assertEqual([5], testArray.shape)

        testArray = ak.randint(np.float64(0), np.float64(10), np.int64(5))
        self.assertIsInstance(testArray, ak.pdarray)
        self.assertEqual(5, len(testArray))
        self.assertEqual(ak.int64, testArray.dtype)
        self.assertEqual([5], testArray.shape)

        test_ndarray = testArray.to_ndarray()

        for value in test_ndarray:
            self.assertTrue(0 <= value <= 10)

        test_array = ak.randint(0, 1, 3, dtype=ak.float64)
        self.assertEqual(ak.float64, test_array.dtype)

        test_array = ak.randint(0, 1, 5, dtype=ak.bool_)
        self.assertEqual(ak.bool_, test_array.dtype)

        test_ndarray = test_array.to_ndarray()

        # test resolution of modulus overflow - issue #1174
        test_array = ak.randint(-(2**63), 2**63 - 1, 10)
        to_validate = np.full(10, -(2**63))
        self.assertFalse((test_array.to_ndarray() == to_validate).all())

        for value in test_ndarray:
            self.assertTrue(value in [True, False])

        with self.assertRaises(TypeError):
            ak.randint(low=5)

        with self.assertRaises(TypeError):
            ak.randint(high=5)

        with self.assertRaises(TypeError):
            ak.randint()

        with self.assertRaises(ValueError):
            ak.randint(low=0, high=1, size=-1, dtype=ak.float64)

        with self.assertRaises(ValueError):
            ak.randint(low=1, high=0, size=1, dtype=ak.float64)

        with self.assertRaises(TypeError):
            ak.randint(0, 1, "1000")

        with self.assertRaises(TypeError):
            ak.randint("0", 1, 1000)

        with self.assertRaises(TypeError):
            ak.randint(0, "1", 1000)

        # Test that int_scalars covers uint8, uint16, uint32
        ak.randint(low=np.uint8(1), high=np.uint16(100), size=np.uint32(100))

    def test_randint_with_seed(self):
        values = ak.randint(1, 5, 10, seed=2)

        self.assertListEqual([4, 3, 1, 3, 2, 4, 4, 2, 3, 4], values.to_list())

        values = ak.randint(1, 5, 10, dtype=ak.float64, seed=2)
        self.assertListEqual(
            [
                2.9160772326374946,
                4.353429832157099,
                4.5392023718621486,
                4.4019932101126606,
                3.3745324569952304,
                1.1642002901528308,
                4.4714086874555292,
                3.7098921109084522,
                4.5939589352472314,
                4.0337935981006172,
            ],
            values.to_list(),
        )

        values = ak.randint(1, 5, 10, dtype=ak.bool_, seed=2)
        self.assertListEqual(
            [False, True, True, True, True, False, True, True, True, True],
            values.to_list(),
        )

        values = ak.randint(1, 5, 10, dtype=bool, seed=2)
        self.assertListEqual(
            [False, True, True, True, True, False, True, True, True, True],
            values.to_list(),
        )

        # Test that int_scalars covers uint8, uint16, uint32
        ak.randint(np.uint8(1), np.uint32(5), np.uint16(10), seed=np.uint8(2))

    def test_uniform(self):
        testArray = ak.uniform(3)
        self.assertIsInstance(testArray, ak.pdarray)
        self.assertEqual(ak.float64, testArray.dtype)
        self.assertEqual([3], testArray.shape)

        testArray = ak.uniform(np.int64(3))
        self.assertIsInstance(testArray, ak.pdarray)
        self.assertEqual(ak.float64, testArray.dtype)
        self.assertEqual([3], testArray.shape)

        uArray = ak.uniform(size=3, low=0, high=5, seed=0)
        self.assertListEqual(
            [0.30013431967121934, 0.47383036230759112, 1.0441791878997098],
            uArray.to_list(),
        )

        uArray = ak.uniform(size=np.int64(3), low=np.int64(0), high=np.int64(5), seed=np.int64(0))
        self.assertListEqual(
            [0.30013431967121934, 0.47383036230759112, 1.0441791878997098],
            uArray.to_list(),
        )

        with self.assertRaises(TypeError):
            ak.uniform(low="0", high=5, size=100)

        with self.assertRaises(TypeError):
            ak.uniform(low=0, high="5", size=100)

        with self.assertRaises(TypeError):
            ak.uniform(low=0, high=5, size="100")

        # Test that int_scalars covers uint8, uint16, uint32
        ak.uniform(low=np.uint8(0), high=5, size=np.uint32(100))

    def test_zeros(self):
        intZeros = ak.zeros(5, dtype=ak.int64)
        self.assertIsInstance(intZeros, ak.pdarray)
        self.assertEqual(ak.int64, intZeros.dtype)

        floatZeros = ak.zeros(5, dtype=float)
        self.assertEqual(float, floatZeros.dtype)

        floatZeros = ak.zeros(5, dtype=ak.float64)
        self.assertEqual(ak.float64, floatZeros.dtype)

        boolZeros = ak.zeros(5, dtype=bool)
        self.assertEqual(bool, boolZeros.dtype)

        boolZeros = ak.zeros(5, dtype=ak.bool_)
        self.assertEqual(ak.bool_, boolZeros.dtype)

        bigintZeros = ak.zeros(5, dtype=ak.bigint)
        self.assertEqual(ak.bigint, bigintZeros.dtype)
        self.assertEqual(0, bigintZeros[0])

        zeros = ak.zeros("5")
        self.assertEqual(5, len(zeros))

        with self.assertRaises(TypeError):
            ak.zeros(5, dtype=ak.uint8)

        with self.assertRaises(TypeError):
            ak.zeros(5, dtype=str)

        # Test that int_scalars covers uint8, uint16, uint32
        ak.zeros(np.uint8(5), dtype=ak.int64)
        ak.zeros(np.uint16(5), dtype=ak.int64)
        ak.zeros(np.uint32(5), dtype=ak.int64)

    def test_ones(self):
        intOnes = ak.ones(5, dtype=int)
        self.assertIsInstance(intOnes, ak.pdarray)
        self.assertEqual(int, intOnes.dtype)

        intOnes = ak.ones(5, dtype=ak.int64)
        self.assertEqual(ak.int64, intOnes.dtype)

        floatOnes = ak.ones(5, dtype=float)
        self.assertEqual(float, floatOnes.dtype)

        floatOnes = ak.ones(5, dtype=ak.float64)
        self.assertEqual(ak.float64, floatOnes.dtype)

        boolOnes = ak.ones(5, dtype=bool)
        self.assertEqual(bool, boolOnes.dtype)

        boolOnes = ak.ones(5, dtype=ak.bool_)
        self.assertEqual(ak.bool_, boolOnes.dtype)

        bigintOnes = ak.ones(5, dtype=ak.bigint)
        self.assertEqual(ak.bigint, bigintOnes.dtype)
        self.assertEqual(1, bigintOnes[0])

        ones = ak.ones("5")
        self.assertEqual(5, len(ones))

        with self.assertRaises(TypeError):
            ak.ones(5, dtype=ak.uint8)

        with self.assertRaises(TypeError):
            ak.ones(5, dtype=str)

        # Test that int_scalars covers uint8, uint16, uint32
        ak.ones(np.uint8(5), dtype=ak.int64)
        ak.ones(np.uint16(5), dtype=ak.int64)
        ak.ones(np.uint32(5), dtype=ak.int64)

    def test_ones_like(self):
        intOnes = ak.ones(5, dtype=ak.int64)
        intOnesLike = ak.ones_like(intOnes)

        self.assertIsInstance(intOnesLike, ak.pdarray)
        self.assertEqual(ak.int64, intOnesLike.dtype)

        floatOnes = ak.ones(5, dtype=ak.float64)
        floatOnesLike = ak.ones_like(floatOnes)

        self.assertEqual(ak.float64, floatOnesLike.dtype)

        boolOnes = ak.ones(5, dtype=ak.bool_)
        boolOnesLike = ak.ones_like(boolOnes)

        self.assertEqual(ak.bool_, boolOnesLike.dtype)

        bigintOnes = ak.ones(5, dtype=ak.bigint)
        bigintOnesLike = ak.ones_like(bigintOnes)

        self.assertEqual(ak.bigint, bigintOnesLike.dtype)

    def test_full(self):
        int_full = ak.full(5, 5, dtype=int)
        self.assertIsInstance(int_full, ak.pdarray)
        self.assertEqual(int, int_full.dtype)
        self.assertEqual(int_full[0], 5)

        int_full = ak.full(5, 5, dtype=ak.int64)
        self.assertEqual(ak.int64, int_full.dtype)

        uint_full = ak.full(5, 7, dtype=ak.uint64)
        self.assertIsInstance(uint_full, ak.pdarray)
        self.assertEqual(ak.uint64, uint_full.dtype)
        self.assertEqual(uint_full[0], 7)

        float_full = ak.full(5, 0, dtype=float)
        self.assertEqual(float, float_full.dtype)
        self.assertEqual(float_full[0], 0)

        float_full = ak.full(5, 0, dtype=ak.float64)
        self.assertEqual(ak.float64, float_full.dtype)

        bool_full = ak.full(5, -1, dtype=bool)
        self.assertEqual(bool, bool_full.dtype)
        self.assertEqual(bool_full[0], True)

        bool_full = ak.full(5, False, dtype=ak.bool_)
        self.assertEqual(ak.bool_, bool_full.dtype)
        self.assertEqual(bool_full[0], False)

        string_len_full = ak.full("5", 5)
        self.assertEqual(5, len(string_len_full))

        strings_full = ak.full(5, "test")
        self.assertIsInstance(strings_full, ak.Strings)
        self.assertEqual(5, len(strings_full))
        self.assertListEqual(strings_full.to_list(), ["test"] * 5)

        with self.assertRaises(TypeError):
            ak.full(5, 1, dtype=ak.uint8)

        with self.assertRaises(TypeError):
            ak.full(5, 8, dtype=str)

        # Test that int_scalars covers uint8, uint16, uint32
        ak.full(np.uint8(5), np.uint16(5), dtype=int)
        ak.full(np.uint8(5), np.uint32(5), dtype=int)
        ak.full(np.uint16(5), np.uint32(5), dtype=int)

    def test_full_like(self):
        int_full = ak.full(5, 6, dtype=ak.int64)
        int_full_like = ak.full_like(int_full, 6)
        self.assertIsInstance(int_full_like, ak.pdarray)
        self.assertEqual(ak.int64, int_full_like.dtype)
        self.assertEqual(int_full_like[0], 6)

        float_full = ak.full(5, 4, dtype=ak.float64)
        float_full_like = ak.full_like(float_full, 4)
        self.assertEqual(ak.float64, float_full_like.dtype)
        self.assertEqual(float_full_like[0], 4)

        bool_full = ak.full(5, True, dtype=ak.bool_)
        bool_full_like = ak.full_like(bool_full, True)
        self.assertEqual(ak.bool_, bool_full_like.dtype)
        self.assertEqual(bool_full_like[0], True)

    def test_zeros_like(self):
        intZeros = ak.zeros(5, dtype=ak.int64)
        intZerosLike = ak.zeros_like(intZeros)

        self.assertIsInstance(intZerosLike, ak.pdarray)
        self.assertEqual(ak.int64, intZerosLike.dtype)

        floatZeros = ak.ones(5, dtype=ak.float64)
        floatZerosLike = ak.ones_like(floatZeros)

        self.assertEqual(ak.float64, floatZerosLike.dtype)

        boolZeros = ak.ones(5, dtype=ak.bool_)
        boolZerosLike = ak.ones_like(boolZeros)

        self.assertEqual(ak.bool_, boolZerosLike.dtype)

    def test_linspace(self):
        pda = ak.linspace(0, 100, 1000)
        self.assertEqual(1000, len(pda))
        self.assertEqual(float, pda.dtype)
        self.assertIsInstance(pda, ak.pdarray)

        pda = ak.linspace(0.0, 100.0, 150)

        pda = ak.linspace(start=5, stop=0, length=6)
        self.assertEqual(5.0000, pda[0])
        self.assertEqual(0.0000, pda[5])

        pda = ak.linspace(start=5.0, stop=0.0, length=6)
        self.assertEqual(5.0000, pda[0])
        self.assertEqual(0.0000, pda[5])

        pda = ak.linspace(start=float(5.0), stop=float(0.0), length=np.int64(6))
        self.assertEqual(5.0000, pda[0])
        self.assertEqual(0.0000, pda[5])

        with self.assertRaises(TypeError):
            ak.linspace(0, "100", 1000)

        with self.assertRaises(TypeError):
            ak.linspace("0", 100, 1000)

        with self.assertRaises(TypeError):
            ak.linspace(0, 100, "1000")

        # Test that int_scalars covers uint8, uint16, uint32
        ak.linspace(np.uint8(0), np.uint16(100), np.uint32(1000))
        ak.linspace(np.uint32(0), np.uint16(100), np.uint8(1000 % 256))
        ak.linspace(np.uint16(0), np.uint8(100), np.uint8(1000 % 256))

    def test_standard_normal(self):
        pda = ak.standard_normal(100)
        self.assertIsInstance(pda, ak.pdarray)
        self.assertEqual(100, len(pda))
        self.assertEqual(float, pda.dtype)

        pda = ak.standard_normal(np.int64(100))
        self.assertIsInstance(pda, ak.pdarray)
        self.assertEqual(100, len(pda))
        self.assertEqual(float, pda.dtype)

        pda = ak.standard_normal(np.int64(100), np.int64(1))
        self.assertIsInstance(pda, ak.pdarray)
        self.assertEqual(100, len(pda))
        self.assertEqual(float, pda.dtype)

        npda = pda.to_ndarray()
        pda = ak.standard_normal(np.int64(100), np.int64(1))

        self.assertListEqual(npda.tolist(), pda.to_list())

        with self.assertRaises(TypeError):
            ak.standard_normal("100")

        with self.assertRaises(TypeError):
            ak.standard_normal(100.0)

        with self.assertRaises(ValueError):
            ak.standard_normal(-1)

        # Test that int_scalars covers uint8, uint16, uint32
        ak.standard_normal(np.uint8(100))
        ak.standard_normal(np.uint16(100))
        ak.standard_normal(np.uint32(100))

    def test_random_strings_uniform(self):
        pda = ak.random_strings_uniform(minlen=1, maxlen=5, size=100)
        nda = pda.to_ndarray()

        self.assertIsInstance(pda, ak.Strings)
        self.assertEqual(100, len(pda))
        self.assertEqual(str, pda.dtype)
        for string in nda:
            self.assertTrue(len(string) >= 1 and len(string) <= 5)
            self.assertTrue(string.isupper())

        pda = ak.random_strings_uniform(minlen=np.int64(1), maxlen=np.int64(5), size=np.int64(100))
        nda = pda.to_ndarray()

        self.assertIsInstance(pda, ak.Strings)
        self.assertEqual(100, len(pda))
        self.assertEqual(str, pda.dtype)
        for string in nda:
            self.assertTrue(len(string) >= 1 and len(string) <= 5)
            self.assertTrue(string.isupper())

        with self.assertRaises(ValueError):
            ak.random_strings_uniform(maxlen=1, minlen=5, size=100)

        with self.assertRaises(ValueError):
            ak.random_strings_uniform(maxlen=5, minlen=1, size=-1)

        with self.assertRaises(ValueError):
            ak.random_strings_uniform(maxlen=5, minlen=5, size=10)

        with self.assertRaises(TypeError):
            ak.random_strings_uniform(minlen="1", maxlen=5, size=10)

        with self.assertRaises(TypeError):
            ak.random_strings_uniform(minlen=1, maxlen="5", size=10)

        with self.assertRaises(TypeError):
            ak.random_strings_uniform(minlen=1, maxlen=5, size="10")

    def test_random_strings_uniform_with_seed(self):
        pda = ak.random_strings_uniform(minlen=1, maxlen=5, seed=1, size=10)

        self.assertListEqual(
            ["VW", "JEXI", "EBBX", "HG", "S", "WOVK", "U", "WL", "JCSD", "DSN"],
            pda.to_list(),
        )

        pda = ak.random_strings_uniform(
            minlen=np.int64(1), maxlen=np.int64(5), seed=np.int64(1), size=np.int64(10)
        )

        self.assertListEqual(
            ["VW", "JEXI", "EBBX", "HG", "S", "WOVK", "U", "WL", "JCSD", "DSN"],
            pda.to_list(),
        )

        pda = ak.random_strings_uniform(minlen=1, maxlen=5, seed=1, size=10, characters="printable")
        self.assertListEqual(
            ["eL", "6<OD", "o-GO", " l", "m", "PV y", "f", "}.", "b3Yc", "Kw,"],
            pda.to_list(),
        )

        # Test that int_scalars covers uint8, uint16, uint32
        pda = ak.random_strings_uniform(
            minlen=np.uint8(1),
            maxlen=np.uint32(5),
            seed=np.uint16(1),
            size=np.uint8(10),
            characters="printable",
        )

    def test_random_strings_lognormal(self):
        pda = ak.random_strings_lognormal(2, 0.25, 100, characters="printable")
        self.assertIsInstance(pda, ak.Strings)
        self.assertEqual(100, len(pda))
        self.assertEqual(str, pda.dtype)

        pda = ak.random_strings_lognormal(np.int64(2), 0.25, np.int64(100), characters="printable")
        self.assertIsInstance(pda, ak.Strings)
        self.assertEqual(100, len(pda))
        self.assertEqual(str, pda.dtype)

        pda = ak.random_strings_lognormal(
            np.int64(2), float(0.25), np.int64(100), characters="printable"
        )
        self.assertIsInstance(pda, ak.Strings)
        self.assertEqual(100, len(pda))
        self.assertEqual(str, pda.dtype)

        pda = ak.random_strings_lognormal(
            logmean=np.int64(2),
            logstd=0.25,
            size=np.int64(100),
            characters="printable",
            seed=np.int64(0),
        )
        self.assertIsInstance(pda, ak.Strings)
        self.assertEqual(100, len(pda))
        self.assertEqual(str, pda.dtype)

        pda = ak.random_strings_lognormal(
            logmean=np.float64(2),
            logstd=np.float64(0.25),
            size=np.int64(100),
            characters="printable",
            seed=np.int64(0),
        )
        self.assertIsInstance(pda, ak.Strings)
        self.assertEqual(100, len(pda))
        self.assertEqual(str, pda.dtype)

        pda = ak.random_strings_lognormal(
            np.float64(2), np.float64(0.25), np.int64(100), characters="printable", seed=np.int64(0)
        )
        self.assertIsInstance(pda, ak.Strings)
        self.assertEqual(100, len(pda))
        self.assertEqual(str, pda.dtype)

        with self.assertRaises(TypeError):
            ak.random_strings_lognormal("2", 0.25, 100)

        with self.assertRaises(TypeError):
            ak.random_strings_lognormal(2, 0.25, "100")

        with self.assertRaises(TypeError):
            ak.random_strings_lognormal(2, 0.25, 100, 1000000)

        # Test that int_scalars covers uint8, uint16, uint32
        ak.random_strings_lognormal(np.uint8(2), 0.25, np.uint16(100))

    def test_random_strings_lognormal_with_seed(self):
        pda = ak.random_strings_lognormal(2, 0.25, 10, seed=1)
        self.assertListEqual(
            [
                "VWHJEX",
                "BEBBXJHGM",
                "RWOVKBUR",
                "LNJCSDXD",
                "NKEDQC",
                "GIBAFPAVWF",
                "IJIFHGDHKA",
                "VUDYRA",
                "QHQETTEZ",
                "DJBPWJV",
            ],
            pda.to_list(),
        )

        pda = ak.random_strings_lognormal(float(2), np.float64(0.25), np.int64(10), seed=1)
        self.assertListEqual(
            [
                "VWHJEX",
                "BEBBXJHGM",
                "RWOVKBUR",
                "LNJCSDXD",
                "NKEDQC",
                "GIBAFPAVWF",
                "IJIFHGDHKA",
                "VUDYRA",
                "QHQETTEZ",
                "DJBPWJV",
            ],
            pda.to_list(),
        )

        pda = ak.random_strings_lognormal(2, 0.25, 10, seed=1, characters="printable")
        self.assertListEqual(
            [
                "eL96<O",
                ")o-GOe lR",
                ")PV yHf(",
                "._b3Yc&K",
                ",7Wjef",
                "R{lQs_g]5T",
                "E[2dk\\2a9J",
                "I*VknZ",
                "0!u~e$Lm",
                "9Q{TtHq",
            ],
            pda.to_list(),
        )

        pda = ak.random_strings_lognormal(
            np.int64(2), np.float64(0.25), np.int64(10), seed=1, characters="printable"
        )
        self.assertListEqual(
            [
                "eL96<O",
                ")o-GOe lR",
                ")PV yHf(",
                "._b3Yc&K",
                ",7Wjef",
                "R{lQs_g]5T",
                "E[2dk\\2a9J",
                "I*VknZ",
                "0!u~e$Lm",
                "9Q{TtHq",
            ],
            pda.to_list(),
        )

    def test_mulitdimensional_array_creation(self):
        av = ak.array([[0, 0], [0, 1], [1, 1]])
        self.assertIsInstance(av, ak.ArrayView)

    def test_from_series(self):
        strings = ak.from_series(pd.Series(["a", "b", "c", "d", "e"], dtype="string"))

        self.assertIsInstance(strings, ak.Strings)
        self.assertEqual(5, len(strings))

        objects = ak.from_series(pd.Series(["a", "b", "c", "d", "e"]), dtype=str)

        self.assertIsInstance(objects, ak.Strings)
        self.assertEqual(str, objects.dtype)

        objects = ak.from_series(pd.Series(["a", "b", "c", "d", "e"]))

        self.assertIsInstance(objects, ak.Strings)
        self.assertEqual(str, objects.dtype)

        p_array = ak.from_series(pd.Series(np.random.randint(0, 10, 10)))

        self.assertIsInstance(p_array, ak.pdarray)
        self.assertEqual(np.int64, p_array.dtype)

        p_i_objects_array = ak.from_series(
            pd.Series(np.random.randint(0, 10, 10), dtype="object"), dtype=np.int64
        )

        self.assertIsInstance(p_i_objects_array, ak.pdarray)
        self.assertEqual(np.int64, p_i_objects_array.dtype)

        p_array = ak.from_series(pd.Series(np.random.uniform(low=0.0, high=1.0, size=10)))

        self.assertIsInstance(p_array, ak.pdarray)
        self.assertEqual(np.float64, p_array.dtype)

        p_f_objects_array = ak.from_series(
            pd.Series(np.random.uniform(low=0.0, high=1.0, size=10), dtype="object"), dtype=np.float64
        )

        self.assertIsInstance(p_f_objects_array, ak.pdarray)
        self.assertEqual(np.float64, p_f_objects_array.dtype)

        p_array = ak.from_series(pd.Series(np.random.choice([True, False], size=10)))

        self.assertIsInstance(p_array, ak.pdarray)
        self.assertEqual(bool, p_array.dtype)

        p_b_objects_array = ak.from_series(
            pd.Series(np.random.choice([True, False], size=10), dtype="object"), dtype=bool
        )

        self.assertIsInstance(p_b_objects_array, ak.pdarray)
        self.assertEqual(bool, p_b_objects_array.dtype)

        p_array = ak.from_series(pd.Series([dt.datetime(2016, 1, 1, 0, 0, 1)]))

        self.assertIsInstance(p_array, ak.pdarray)
        self.assertEqual(np.int64, p_array.dtype)

        p_array = ak.from_series(pd.Series([np.datetime64("2018-01-01")]))

        self.assertIsInstance(p_array, ak.pdarray)
        self.assertEqual(np.int64, p_array.dtype)

        p_array = ak.from_series(
            pd.Series(pd.to_datetime(["1/1/2018", np.datetime64("2018-01-01"), dt.datetime(2018, 1, 1)]))
        )

        self.assertIsInstance(p_array, ak.pdarray)
        self.assertEqual(np.int64, p_array.dtype)

        with self.assertRaises(TypeError):
            ak.from_series(np.ones(100))

        with self.assertRaises(ValueError):
            ak.from_series(pd.Series(np.random.randint(0, 10, 10), dtype=np.int8))

    def test_fill(self):
        ones = ak.ones(100)

        ones.fill(2)
        self.assertTrue((2 == ones.to_ndarray()).all())

        ones.fill(np.int64(2))
        self.assertTrue((np.int64(2) == ones.to_ndarray()).all())

        ones.fill(float(2))
        self.assertTrue((float(2) == ones.to_ndarray()).all())

        ones.fill(np.float64(2))
        self.assertTrue((np.float64(2) == ones.to_ndarray()).all())

        # Test that int_scalars covers uint8, uint16, uint32
        ones.fill(np.uint8(2))
        ones.fill(np.uint16(2))
        ones.fill(np.uint32(2))

    def test_endian(self):
        N = 100

        a = np.random.randint(1, N, N)
        aka = ak.array(a)
        npa = aka.to_ndarray()
        self.assertTrue(np.allclose(a, npa))

        a = a.newbyteorder().byteswap()
        aka = ak.array(a)
        npa = aka.to_ndarray()
        self.assertTrue(np.allclose(a, npa))

        a = a.newbyteorder().byteswap()
        aka = ak.array(a)
        npa = aka.to_ndarray()
        self.assertTrue(np.allclose(a, npa))

    def test_clobber(self):
        N = 100
        narrs = 10

        arrs = [np.random.randint(1, N, N) for _ in range(narrs)]
        akarrs = [ak.array(arr) for arr in arrs]
        nparrs = [arr.to_ndarray() for arr in akarrs]
        for a, npa in zip(arrs, nparrs):
            self.assertTrue(np.allclose(a, npa))

        arrs = [np.full(N, i) for i in range(narrs)]
        akarrs = [ak.array(arr) for arr in arrs]
        nparrs = [arr.to_ndarray() for arr in akarrs]

        for a, npa, i in zip(arrs, nparrs, range(narrs)):
            self.assertTrue(np.all(a == i))
            self.assertTrue(np.all(npa == i))

            a += 1
            self.assertTrue(np.all(a == i + 1))
            self.assertTrue(np.all(npa == i))

            npa += 1
            self.assertTrue(np.all(a == i + 1))
            self.assertTrue(np.all(npa == i + 1))

    def test_uint_greediness(self):
        # default to uint when all supportedInt and any value > 2**63
        # to avoid loss of precision see (#1297)
        for greedy_list in ([2**63, 6, 2**63 - 1, 2**63 + 1], [2**64 - 1, 0, 2**64 - 1]):
            greedy_pda = ak.array(greedy_list)
            self.assertEqual(greedy_pda.dtype, ak.uint64)
            self.assertListEqual(greedy_list, greedy_pda.to_list())

    def randint_randomness(self):
        # THIS TEST DOES NOT RUN, see Issue #1672
        # To run rename to `test_randint_randomness`
        minVal = 0
        maxVal = 2**32
        size = 250
        passed = 0
        trials = 20

        for x in range(trials):
            l = ak.randint(minVal, maxVal, size)
            l_median = statistics.median(l.to_ndarray())

            runs, n1, n2 = 0, 0, 0

            # Checking for start of new run
            for i in range(len(l)):
                # no. of runs
                if (l[i] >= l_median > l[i - 1]) or (l[i] < l_median <= l[i - 1]):
                    runs += 1

                # no. of positive values
                if (l[i]) >= l_median:
                    n1 += 1
                # no. of negative values
                else:
                    n2 += 1

            runs_exp = ((2 * n1 * n2) / (n1 + n2)) + 1
            stan_dev = math.sqrt(
                (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
            )

            if abs((runs - runs_exp) / stan_dev) < 1.9:
                passed += 1

        self.assertGreaterEqual(passed, trials * 0.8)

    def test_inferred_type(self):
        a = ak.array([1, 2, 3])
        self.assertTrue(a.inferred_type, "integer")

        a2 = ak.array([1.0, 2, 3])
        self.assertTrue(a2.inferred_type, "floating")
