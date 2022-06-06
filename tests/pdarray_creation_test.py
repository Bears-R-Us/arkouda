import datetime as dt
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

    def test_arange(self):
        self.assertTrue((ak.array([0, 1, 2, 3, 4]) == ak.arange(0, 5, 1)).all())
        self.assertTrue((ak.array([5, 4, 3, 2, 1]) == ak.arange(5, 0, -1)).all())
        self.assertTrue((ak.array([-5, -6, -7, -8, -9]) == ak.arange(-5, -10, -1)).all())
        self.assertTrue((ak.array([0, 2, 4, 6, 8]) == ak.arange(0, 10, 2)).all())

    def test_arange_dtype(self):
        # test dtype works with optional start/stride
        expected_stop = ak.array([0, 1, 2])
        uint_stop = ak.arange(3, dtype=ak.uint64)
        self.assertListEqual(expected_stop.to_ndarray().tolist(), uint_stop.to_ndarray().tolist())
        self.assertEqual(ak.uint64, uint_stop.dtype)

        expected_start_stop = ak.array([3, 4, 5, 6])
        uint_start_stop = ak.arange(3, 7, dtype=ak.uint64)
        self.assertListEqual(
            expected_start_stop.to_ndarray().tolist(), uint_start_stop.to_ndarray().tolist()
        )
        self.assertEqual(ak.uint64, uint_start_stop.dtype)

        expected_start_stop_stride = ak.array([3, 5])
        uint_start_stop_stride = ak.arange(3, 7, 2, dtype=ak.uint64)
        self.assertListEqual(
            expected_start_stop_stride.to_ndarray().tolist(),
            uint_start_stop_stride.to_ndarray().tolist(),
        )
        self.assertEqual(ak.uint64, uint_start_stop_stride.dtype)

        # test uint64 handles negatives correctly
        np_arange_uint = np.arange(-5, -10, -1, dtype=np.uint64)
        ak_arange_uint = ak.arange(-5, -10, -1, dtype=ak.uint64)
        # np_arange_uint = array([18446744073709551611, 18446744073709551610, 18446744073709551609,
        #        18446744073709551608, 18446744073709551607], dtype=uint64)
        self.assertListEqual(np_arange_uint.tolist(), ak_arange_uint.to_ndarray().tolist())
        self.assertEqual(ak.uint64, ak_arange_uint.dtype)

        # test correct conversion to float64
        np_arange_float = np.arange(-5, -10, -1, dtype=np.float64)
        ak_arange_float = ak.arange(-5, -10, -1, dtype=ak.float64)
        # array([-5., -6., -7., -8., -9.])
        self.assertListEqual(np_arange_float.tolist(), ak_arange_float.to_ndarray().tolist())
        self.assertEqual(ak.float64, ak_arange_float.dtype)

        # test correct conversion to bool
        expected_bool = ak.array([False, True, True, True, True])
        ak_arange_bool = ak.arange(0, 10, 2, dtype=ak.bool)
        self.assertListEqual(expected_bool.to_ndarray().tolist(), ak_arange_bool.to_ndarray().tolist())
        self.assertEqual(ak.bool, ak_arange_bool.dtype)

        # test uint64 input works
        expected = ak.array([0, 1, 2])
        uint_input = ak.arange(3, dtype=ak.uint64)
        self.assertListEqual(expected.to_ndarray().tolist(), uint_input.to_ndarray().tolist())
        self.assertEqual(ak.uint64, uint_input.dtype)

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

        test_array = ak.randint(0, 1, 5, dtype=ak.bool)
        self.assertEqual(ak.bool, test_array.dtype)

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

    def test_randint_with_seed(self):
        values = ak.randint(1, 5, 10, seed=2)

        self.assertTrue((ak.array([4, 3, 1, 3, 2, 4, 4, 2, 3, 4]) == values).all())

        values = ak.randint(1, 5, 10, dtype=ak.float64, seed=2)
        self.assertTrue(
            (
                ak.array(
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
                    ]
                )
                == values
            ).all()
        )

        values = ak.randint(1, 5, 10, dtype=ak.bool, seed=2)
        self.assertTrue(
            (ak.array([False, True, True, True, True, False, True, True, True, True]) == values).all()
        )

        values = ak.randint(1, 5, 10, dtype=bool, seed=2)
        self.assertTrue(
            (ak.array([False, True, True, True, True, False, True, True, True, True]) == values).all()
        )

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
        self.assertTrue(
            (ak.array([0.30013431967121934, 0.47383036230759112, 1.0441791878997098]) == uArray).all()
        )

        uArray = ak.uniform(size=np.int64(3), low=np.int64(0), high=np.int64(5), seed=np.int64(0))
        self.assertTrue(
            (ak.array([0.30013431967121934, 0.47383036230759112, 1.0441791878997098]) == uArray).all()
        )

        with self.assertRaises(TypeError):
            ak.uniform(low="0", high=5, size=100)

        with self.assertRaises(TypeError):
            ak.uniform(low=0, high="5", size=100)

        with self.assertRaises(TypeError):
            ak.uniform(low=0, high=5, size="100")

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

        boolZeros = ak.zeros(5, dtype=ak.bool)
        self.assertEqual(ak.bool, boolZeros.dtype)

        zeros = ak.zeros("5")
        self.assertEqual(5, len(zeros))

        with self.assertRaises(TypeError):
            ak.zeros(5, dtype=ak.uint8)

        with self.assertRaises(TypeError):
            ak.zeros(5, dtype=str)

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

        boolOnes = ak.ones(5, dtype=ak.bool)
        self.assertEqual(ak.bool, boolOnes.dtype)

        ones = ak.ones("5")
        self.assertEqual(5, len(ones))

        with self.assertRaises(TypeError):
            ak.ones(5, dtype=ak.uint8)

        with self.assertRaises(TypeError):
            ak.ones(5, dtype=str)

    def test_ones_like(self):
        intOnes = ak.ones(5, dtype=ak.int64)
        intOnesLike = ak.ones_like(intOnes)

        self.assertIsInstance(intOnesLike, ak.pdarray)
        self.assertEqual(ak.int64, intOnesLike.dtype)

        floatOnes = ak.ones(5, dtype=ak.float64)
        floatOnesLike = ak.ones_like(floatOnes)

        self.assertEqual(ak.float64, floatOnesLike.dtype)

        boolOnes = ak.ones(5, dtype=ak.bool)
        boolOnesLike = ak.ones_like(boolOnes)

        self.assertEqual(ak.bool, boolOnesLike.dtype)

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

        bool_full = ak.full(5, False, dtype=ak.bool)
        self.assertEqual(ak.bool, bool_full.dtype)
        self.assertEqual(bool_full[0], False)

        string_len_full = ak.full("5", 5)
        self.assertEqual(5, len(string_len_full))

        with self.assertRaises(TypeError):
            ak.full(5, 1, dtype=ak.uint8)

        with self.assertRaises(TypeError):
            ak.full(5, 8, dtype=str)

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

        bool_full = ak.full(5, True, dtype=ak.bool)
        bool_full_like = ak.full_like(bool_full, True)
        self.assertEqual(ak.bool, bool_full_like.dtype)
        self.assertEqual(bool_full_like[0], True)

    def test_zeros_like(self):
        intZeros = ak.zeros(5, dtype=ak.int64)
        intZerosLike = ak.zeros_like(intZeros)

        self.assertIsInstance(intZerosLike, ak.pdarray)
        self.assertEqual(ak.int64, intZerosLike.dtype)

        floatZeros = ak.ones(5, dtype=ak.float64)
        floatZerosLike = ak.ones_like(floatZeros)

        self.assertEqual(ak.float64, floatZerosLike.dtype)

        boolZeros = ak.ones(5, dtype=ak.bool)
        boolZerosLike = ak.ones_like(boolZeros)

        self.assertEqual(ak.bool, boolZerosLike.dtype)

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

        self.assertTrue((npda == pda.to_ndarray()).all())

        with self.assertRaises(TypeError):
            ak.standard_normal("100")

        with self.assertRaises(TypeError):
            ak.standard_normal(100.0)

        with self.assertRaises(ValueError):
            ak.standard_normal(-1)

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

        self.assertTrue(
            (ak.array(["TV", "JTEW", "BOCO", "HF", "D", "UDMM", "T", "NK", "OQNP", "ZXV"]) == pda).all()
        )

        pda = ak.random_strings_uniform(
            minlen=np.int64(1), maxlen=np.int64(5), seed=np.int64(1), size=np.int64(10)
        )

        self.assertTrue(
            (ak.array(["TV", "JTEW", "BOCO", "HF", "D", "UDMM", "T", "NK", "OQNP", "ZXV"]) == pda).all()
        )

        pda = ak.random_strings_uniform(minlen=1, maxlen=5, seed=1, size=10, characters="printable")
        self.assertTrue(
            (ak.array(["+5", "fp-P", "3Q4k", "~H", "F", "F=`,", "E", "YD", "kBa'", "(t5"]) == pda).all()
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

    def test_random_strings_lognormal_with_seed(self):
        pda = ak.random_strings_lognormal(2, 0.25, 10, seed=1)

        self.assertTrue(
            (
                ak.array(
                    [
                        "TVKJTE",
                        "ABOCORHFM",
                        "LUDMMGTB",
                        "KWOQNPHZ",
                        "VSXRRL",
                        "AKOZOEEWTB",
                        "GOSVGEJNOW",
                        "BFWSIO",
                        "MRIEJUSA",
                        "OLUKRJK",
                    ]
                )
                == pda
            ).all()
        )

        pda = ak.random_strings_lognormal(float(2), np.float64(0.25), np.int64(10), seed=1)

        self.assertTrue(
            (
                ak.array(
                    [
                        "TVKJTE",
                        "ABOCORHFM",
                        "LUDMMGTB",
                        "KWOQNPHZ",
                        "VSXRRL",
                        "AKOZOEEWTB",
                        "GOSVGEJNOW",
                        "BFWSIO",
                        "MRIEJUSA",
                        "OLUKRJK",
                    ]
                )
                == pda
            ).all()
        )

        pda = ak.random_strings_lognormal(2, 0.25, 10, seed=1, characters="printable")

        self.assertTrue(
            (
                ak.array(
                    [
                        '+5"fp-',
                        "]3Q4kC~HF",
                        "=F=`,IE!",
                        "DjkBa'9(",
                        "5oZ1)=",
                        'T^.1@6aj";',
                        "8b2$IX!Y7.",
                        "x|Y!eQ",
                        ">1\\>2,on",
                        '&#W":C3',
                    ]
                )
                == pda
            ).all()
        )

        pda = ak.random_strings_lognormal(
            np.int64(2), np.float64(0.25), np.int64(10), seed=1, characters="printable"
        )

        self.assertTrue(
            (
                ak.array(
                    [
                        '+5"fp-',
                        "]3Q4kC~HF",
                        "=F=`,IE!",
                        "DjkBa'9(",
                        "5oZ1)=",
                        'T^.1@6aj";',
                        "8b2$IX!Y7.",
                        "x|Y!eQ",
                        ">1\\>2,on",
                        '&#W":C3',
                    ]
                )
                == pda
            ).all()
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
        for (a, npa) in zip(arrs, nparrs):
            self.assertTrue(np.allclose(a, npa))

        arrs = [np.full(N, i) for i in range(narrs)]
        akarrs = [ak.array(arr) for arr in arrs]
        nparrs = [arr.to_ndarray() for arr in akarrs]

        for (a, npa, i) in zip(arrs, nparrs, range(narrs)):
            self.assertTrue(np.all(a == i))
            self.assertTrue(np.all(npa == i))

            a += 1
            self.assertTrue(np.all(a == i + 1))
            self.assertTrue(np.all(npa == i))

            npa += 1
            self.assertTrue(np.all(a == i + 1))
            self.assertTrue(np.all(npa == i + 1))
