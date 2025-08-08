import numpy as np
from base_test import ArkoudaTest
from context import arkouda as ak

from arkouda import dtypes

"""
DtypesTest encapsulates arkouda dtypes module methods
"""


class DtypesTest(ArkoudaTest):
    def test_check_np_dtype(self):
        """
        Tests dtypes.check_np_dtype method

        :return: None
        :raise: AssertionError if 1.. test cases fail
        """
        dtypes.check_np_dtype(np.dtype(np.uint8))
        dtypes.check_np_dtype(np.dtype(np.uint16))
        dtypes.check_np_dtype(np.dtype(np.uint32))
        dtypes.check_np_dtype(np.dtype(np.uint64))
        dtypes.check_np_dtype(np.dtype(np.int8))
        dtypes.check_np_dtype(np.dtype(np.int16))
        dtypes.check_np_dtype(np.dtype(np.int32))
        dtypes.check_np_dtype(np.dtype(np.int64))
        dtypes.check_np_dtype(np.dtype(np.float32))
        dtypes.check_np_dtype(np.dtype(np.float64))
        dtypes.check_np_dtype(np.dtype(np.complex64))
        dtypes.check_np_dtype(np.dtype(np.complex128))
        dtypes.check_np_dtype(np.dtype(np.bool_))
        dtypes.check_np_dtype(np.dtype(bool))
        dtypes.check_np_dtype(np.dtype(np.str_))
        dtypes.check_np_dtype(np.dtype(str))

        with self.assertRaises(TypeError):
            dtypes.check_np_dtype("np.str")

    def test_translate_np_dtype(self):
        """
        Tests dtypes.translate_np_dtype method

        :return: None
        :raise: AssertionError if 1.. test cases fail
        """
        d_tuple = dtypes.translate_np_dtype(np.dtype(np.uint8))
        self.assertEqual(1, d_tuple[1])
        self.assertEqual("uint", d_tuple[0])
        d_tuple = dtypes.translate_np_dtype(np.dtype(np.uint16))
        self.assertEqual(2, d_tuple[1])
        self.assertEqual("uint", d_tuple[0])
        d_tuple = dtypes.translate_np_dtype(np.dtype(np.uint32))
        self.assertEqual(4, d_tuple[1])
        self.assertEqual("uint", d_tuple[0])
        d_tuple = dtypes.translate_np_dtype(np.dtype(np.uint64))
        self.assertEqual(8, d_tuple[1])
        self.assertEqual("uint", d_tuple[0])

        d_tuple = dtypes.translate_np_dtype(np.dtype(np.int8))
        self.assertEqual(1, d_tuple[1])
        self.assertEqual("int", d_tuple[0])
        d_tuple = dtypes.translate_np_dtype(np.dtype(np.int16))
        self.assertEqual(2, d_tuple[1])
        self.assertEqual("int", d_tuple[0])
        d_tuple = dtypes.translate_np_dtype(np.dtype(np.int32))
        self.assertEqual(4, d_tuple[1])
        self.assertEqual("int", d_tuple[0])
        d_tuple = dtypes.translate_np_dtype(np.dtype(np.int64))
        self.assertEqual(8, d_tuple[1])
        self.assertEqual("int", d_tuple[0])

        d_tuple = dtypes.translate_np_dtype(np.dtype(np.float32))
        self.assertEqual(4, d_tuple[1])
        self.assertEqual("float", d_tuple[0])
        d_tuple = dtypes.translate_np_dtype(np.dtype(np.float64))
        self.assertEqual(8, d_tuple[1])
        self.assertEqual("float", d_tuple[0])

        d_tuple = dtypes.translate_np_dtype(np.dtype(np.complex64))
        self.assertEqual(8, d_tuple[1])
        self.assertEqual("complex", d_tuple[0])
        d_tuple = dtypes.translate_np_dtype(np.dtype(np.complex128))
        self.assertEqual(16, d_tuple[1])
        self.assertEqual("complex", d_tuple[0])

        d_tuple = dtypes.translate_np_dtype(np.dtype(np.bool_))
        self.assertEqual(1, d_tuple[1])
        self.assertEqual("bool", d_tuple[0])
        d_tuple = dtypes.translate_np_dtype(np.dtype(bool))
        self.assertEqual(1, d_tuple[1])
        self.assertEqual("bool", d_tuple[0])
        d_tuple = dtypes.translate_np_dtype(np.dtype(np.str_))
        self.assertEqual(0, d_tuple[1])
        self.assertEqual("str", d_tuple[0])
        d_tuple = dtypes.translate_np_dtype(np.dtype(str))
        self.assertEqual(0, d_tuple[1])
        self.assertEqual("str", d_tuple[0])

        with self.assertRaises(TypeError):
            dtypes.translate_np_dtype("np.str")

    def test_resolve_scalar_dtype(self):
        """
        Tests dtypes.resolve_scalar_dtype method

        :return: None
        :raise: AssertionError if 1.. test cases fail
        """
        self.assertEqual("bool", dtypes.resolve_scalar_dtype(True))
        self.assertEqual("int64", dtypes.resolve_scalar_dtype(1))
        self.assertEqual("float64", dtypes.resolve_scalar_dtype(float(0.0)))
        self.assertEqual("str", dtypes.resolve_scalar_dtype("test"))
        self.assertEqual("int64", dtypes.resolve_scalar_dtype(np.int64(1)))
        self.assertEqual("<class 'list'>", dtypes.resolve_scalar_dtype([1]))

    def test_is_dtype_in_union(self):
        from typing import Union

        from arkouda.numpy.dtypes import _is_dtype_in_union

        float_scalars = Union[float, np.float64, np.float32]
        self.assertTrue(_is_dtype_in_union(np.float64, float_scalars))
        # Test with a type not present in the union
        self.assertFalse(_is_dtype_in_union(np.int64, float_scalars))
        # Test with a non-Union type
        self.assertFalse(_is_dtype_in_union(np.float64, float))

    def test_nbytes(self):
        from arkouda.numpy.dtypes import BigInt

        a = ak.cast(ak.array([1, 2, 3]), dt="bigint")
        self.assertEqual(a.nbytes, 3 * BigInt.itemsize)

        dtype_list = [
            ak.dtypes.uint8,
            ak.dtypes.uint64,
            ak.dtypes.int64,
            ak.dtypes.float64,
            ak.dtypes.bool_,
        ]

        for dt in dtype_list:
            a = ak.array([1, 2, 3], dtype=dt)
            self.assertEqual(a.nbytes, 3 * dt.itemsize)

        a = ak.array(["a", "b", "c"])
        c = ak.Categorical(a)
        self.assertEqual(c.nbytes, 82)

    def test_pdarrays_datatypes(self):
        self.assertEqual(dtypes.dtype("float64"), ak.ones(10).dtype)
        self.assertEqual(
            dtypes.dtype("str"), ak.array(["string {}".format(i) for i in range(0, 10)]).dtype
        )

    def test_isSupportedInt(self):
        """
        Tests for both True and False scenarios of the isSupportedInt method.
        """
        self.assertTrue(dtypes.isSupportedInt(1))
        self.assertTrue(dtypes.isSupportedInt(np.int64(1)))
        self.assertTrue(dtypes.isSupportedInt(np.int64(1.0)))
        self.assertFalse(dtypes.isSupportedInt(1.0))
        self.assertFalse(dtypes.isSupportedInt("1"))
        self.assertFalse(dtypes.isSupportedInt("1.0"))

    def test_isSupportedFloat(self):
        """
        Tests for both True and False scenarios of the isSupportedFloat method.
        """
        self.assertTrue(dtypes.isSupportedFloat(1.0))
        self.assertTrue(dtypes.isSupportedFloat(float(1)))
        self.assertTrue(dtypes.isSupportedFloat(np.float64(1.0)))
        self.assertTrue(dtypes.isSupportedFloat(np.float64(1)))
        self.assertFalse(dtypes.isSupportedFloat(np.int64(1.0)))
        self.assertFalse(dtypes.isSupportedFloat(int(1.0)))
        self.assertFalse(dtypes.isSupportedFloat("1"))
        self.assertFalse(dtypes.isSupportedFloat("1.0"))

    def test_DtypeEnum(self):
        """
        Tests for DTypeEnum, ak.DTypes, and ak.ARKOUDA_SUPPORTED_DTYPES
        """
        self.assertEqual("bool", str(dtypes.DType.BOOL))
        self.assertEqual("float32", str(dtypes.DType.FLOAT32))
        self.assertEqual("float64", str(dtypes.DType.FLOAT64))
        self.assertEqual("float", str(dtypes.DType.FLOAT))
        self.assertEqual("complex64", str(dtypes.DType.COMPLEX64))
        self.assertEqual("complex128", str(dtypes.DType.COMPLEX128))
        self.assertEqual("int8", str(dtypes.DType.INT8))
        self.assertEqual("int16", str(dtypes.DType.INT16))
        self.assertEqual("int32", str(dtypes.DType.INT32))
        self.assertEqual("int64", str(dtypes.DType.INT64))
        self.assertEqual("int", str(dtypes.DType.INT))
        self.assertEqual("uint8", str(dtypes.DType.UINT8))
        self.assertEqual("uint16", str(dtypes.DType.UINT16))
        self.assertEqual("uint32", str(dtypes.DType.UINT32))
        self.assertEqual("uint64", str(dtypes.DType.UINT64))
        self.assertEqual("uint", str(dtypes.DType.UINT))
        self.assertEqual("str", str(dtypes.DType.STR))
        self.assertEqual("bigint", str(dtypes.DType.BIGINT))
        self.assertEqual(
            frozenset(
                {
                    "float32",
                    "float64",
                    "float",
                    "complex64",
                    "complex128",
                    "int8",
                    "int16",
                    "int32",
                    "int64",
                    "int",
                    "uint8",
                    "uint16",
                    "uint32",
                    "uint64",
                    "uint",
                    "bool",
                    "str",
                    "bigint",
                }
            ),
            ak.DTypes,
        )
        self.assertEqual(
            frozenset(
                {
                    "bool_",
                    "float",
                    "float64",
                    "int",
                    "int64",
                    "uint",
                    "uint64",
                    "uint8",
                    "bigint",
                    "str",
                }
            ),
            ak.ARKOUDA_SUPPORTED_DTYPES,
        )

    def test_NumericDTypes(self):
        self.assertEqual(
            frozenset(["bool", "bool_", "float", "float64", "int", "int64", "uint64", "bigint"]),
            dtypes.NumericDTypes,
        )

    def test_SeriesDTypes(self):
        self.assertEqual(np.str_, dtypes.SeriesDTypes["string"])
        self.assertEqual(np.str_, dtypes.SeriesDTypes["<class 'str'>"])
        self.assertEqual(np.int64, dtypes.SeriesDTypes["int64"])
        self.assertEqual(np.int64, dtypes.SeriesDTypes["<class 'numpy.int64'>"])
        self.assertEqual(np.float64, dtypes.SeriesDTypes["float64"])
        self.assertEqual(np.float64, dtypes.SeriesDTypes["<class 'numpy.float64'>"])
        self.assertEqual(np.bool_, dtypes.SeriesDTypes["bool"])
        self.assertEqual(np.dtype(bool), dtypes.SeriesDTypes["bool"])
        self.assertEqual(np.bool_, dtypes.SeriesDTypes["<class 'bool'>"])
        self.assertEqual(np.dtype(bool), dtypes.SeriesDTypes["<class 'bool'>"])
        self.assertEqual(np.int64, dtypes.SeriesDTypes["datetime64[ns]"])
        self.assertEqual(np.int64, dtypes.SeriesDTypes["timedelta64[ns]"])

    def test_scalars(self):
        self.assertEqual("typing.Union[bool, numpy.bool_]", str(ak.bool_scalars))
        self.assertEqual("typing.Union[float, numpy.float64, numpy.float32]", str(ak.float_scalars))
        self.assertEqual(
            (
                "typing.Union[int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, "
                + "numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]"
            ),
            str(ak.int_scalars),
        )
        self.assertEqual(
            (
                "typing.Union[float, numpy.float64, numpy.float32, int, numpy.int8, numpy.int16, "
                + "numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]"
            ),
            str(ak.numeric_scalars),
        )
        self.assertEqual("typing.Union[str, numpy.str_]", str(ak.str_scalars))
        self.assertEqual(
            (
                "typing.Union[numpy.float64, numpy.float32, numpy.int8, numpy.int16, numpy.int32, "
                + "numpy.int64, numpy.bool_, numpy.str_, numpy.uint8, numpy.uint16, numpy.uint32, "
                + "numpy.uint64]"
            ),
            str(ak.numpy_scalars),
        )
        self.assertEqual(
            (
                "typing.Union[bool, numpy.bool_, float, numpy.float64, numpy.float32, int, numpy.int8, "
                + "numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32,"
                + " numpy.uint64, numpy.str_, str]"
            ),
            str(ak.all_scalars),
        )

    def test_number_format_strings(self):
        self.assertEqual("{}", dtypes.NUMBER_FORMAT_STRINGS["bool"])
        self.assertEqual("{:n}", dtypes.NUMBER_FORMAT_STRINGS["int64"])
        self.assertEqual("{:.17f}", dtypes.NUMBER_FORMAT_STRINGS["float64"])
        self.assertEqual("f", dtypes.NUMBER_FORMAT_STRINGS["np.float64"])
        self.assertEqual("{:n}", dtypes.NUMBER_FORMAT_STRINGS["uint8"])
