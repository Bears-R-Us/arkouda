import numpy as np
import pytest

import arkouda as ak
from arkouda import dtypes

"""
DtypesTest encapsulates arkouda dtypes module methods
"""

SUPPORTED_NP_DTYPES = [
    bool,
    int,
    float,
    str,
    np.bool_,
    np.int64,
    np.float64,
    np.uint8,
    np.uint64,
    np.str_,
]


class TestDTypes:
    @pytest.mark.parametrize("dtype", SUPPORTED_NP_DTYPES)
    def test_check_np_dtype(self, dtype):
        dtypes.check_np_dtype(np.dtype(dtype))

    @pytest.mark.parametrize("dtype", [np.dtype(np.int16), "np.str", ak.bigint])
    def test_check_np_dtype_errors(self, dtype):
        with pytest.raises(TypeError):
            dtypes.check_np_dtype(dtype)

    def test_translate_np_dtype(self):
        for b in [np.bool_, bool]:
            assert ("bool", 1) == dtypes.translate_np_dtype(np.dtype(b))

        for s in [np.str_, str]:
            assert ("str", 0) == dtypes.translate_np_dtype(np.dtype(s))

        assert ("int", 8) == dtypes.translate_np_dtype(np.dtype(np.int64))
        assert ("uint", 8) == dtypes.translate_np_dtype(np.dtype(np.uint64))
        assert ("float", 8) == dtypes.translate_np_dtype(np.dtype(np.float64))
        assert ("uint", 1) == dtypes.translate_np_dtype(np.dtype(np.uint8))

    def test_resolve_scalar_dtype(self):
        for b in True, False:
            assert "bool" == dtypes.resolve_scalar_dtype(b)

        for i in np.iinfo(np.int64).min, -1, 0, 3, np.iinfo(np.int64).max:
            assert "int64" == dtypes.resolve_scalar_dtype(i)

        floats = [
            -np.inf,
            np.finfo(np.float64).min,
            -3.14,
            -0.0,
            0.0,
            7.0,
            np.finfo(np.float64).max,
            np.inf,
            np.nan,
        ]
        for f in floats:
            assert "float64" == dtypes.resolve_scalar_dtype(f)

        for s in "test", '"', " ", "":
            assert "str" == dtypes.resolve_scalar_dtype(s)
        assert "<class 'list'>" == dtypes.resolve_scalar_dtype([1])

        assert "uint64" == dtypes.resolve_scalar_dtype(2 ** 63 + 1)
        assert "bigint" == dtypes.resolve_scalar_dtype(2 ** 64)

    def test_pdarrays_datatypes(self):
        assert dtypes.dtype("int64") == ak.array(np.arange(10)).dtype
        assert dtypes.dtype("uint64") == ak.array(np.arange(10), ak.uint64).dtype
        assert dtypes.dtype("bool") == ak.ones(10, ak.bool).dtype
        assert dtypes.dtype("float64") == ak.ones(10).dtype
        assert dtypes.dtype("str") == ak.random_strings_uniform(1, 16, size=10).dtype

        bi = ak.bigint_from_uint_arrays(
            [ak.ones(10, dtype=ak.uint64), ak.arange(10, dtype=ak.uint64)]
        ).dtype
        assert dtypes.dtype("bigint") == bi
        assert dtypes.dtype("bigint") == ak.arange(2 ** 200, 2 ** 200 + 10).dtype

    def test_isSupportedInt(self):
        for supported in -10, 1, np.int64(1), np.int64(1.0), np.uint32(1), 2 ** 63 + 1, 2 ** 200:
            assert dtypes.isSupportedInt(supported)
        for unsupported in 1.0, "1":
            assert not dtypes.isSupportedInt(unsupported)

    def test_isSupportedFloat(self):
        for supported in np.nan, -np.inf, 3.1, -0.0, float(1), np.float64(1):
            assert dtypes.isSupportedFloat(supported)
        for unsupported in np.int64(1.0), int(1.0), "1.0":
            assert not dtypes.isSupportedFloat(unsupported)

    def test_DtypeEnum(self):
        assert "bool" == str(dtypes.DType.BOOL)
        assert "float" == str(dtypes.DType.FLOAT)
        assert "float64" == str(dtypes.DType.FLOAT64)
        assert "int" == str(dtypes.DType.INT)
        assert "int64" == str(dtypes.DType.INT64)
        assert "str" == str(dtypes.DType.STR)
        assert "uint8" == str(dtypes.DType.UINT8)
        assert "bigint" == str(dtypes.DType.BIGINT)

        enum_vals = frozenset(
            {"float", "float64", "bool", "uint8", "int", "int64", "str", "uint64", "bigint"}
        )
        assert enum_vals == ak.DTypes
        assert enum_vals == ak.ARKOUDA_SUPPORTED_DTYPES

    def test_NumericDTypes(self):
        num_types = frozenset(["bool", "float", "float64", "int", "int64", "uint64", "bigint"])
        assert num_types == dtypes.NumericDTypes

    def test_SeriesDTypes(self):
        for dt in "int64", "<class 'numpy.int64'>", "datetime64[ns]", "timedelta64[ns]":
            assert dtypes.SeriesDTypes[dt] == np.int64

        for dt in "string", "<class 'str'>":
            assert dtypes.SeriesDTypes[dt] == np.str_

        for dt in "float64", "<class 'numpy.float64'>":
            assert dtypes.SeriesDTypes[dt] == np.float64

        for dt in "bool", "<class 'bool'>":
            assert dtypes.SeriesDTypes[dt] == np.bool_

    def test_scalars(self):
        assert "typing.Union[bool, numpy.bool_]" == str(ak.bool_scalars)
        assert "typing.Union[float, numpy.float64]" == str(ak.float_scalars)
        assert (
            "typing.Union[int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, "
            + "numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]"
        ) == str(ak.int_scalars)
        assert (
            "typing.Union[float, numpy.float64, int, numpy.int8, numpy.int16, numpy.int32, "
            + "numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]"
        ) == str(ak.numeric_scalars)
        assert "typing.Union[str, numpy.str_]", str(ak.str_scalars)
        assert (
            "typing.Union[numpy.float64, numpy.int8, numpy.int16, numpy.int32, "
            + "numpy.int64, numpy.bool_, numpy.str_, numpy.uint8, numpy.uint16, numpy.uint32, "
            + "numpy.uint64]"
        ) == str(ak.numpy_scalars)
        assert (
            "typing.Union[bool, numpy.bool_, float, numpy.float64, int, numpy.int8, "
            + "numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32,"
            + " numpy.uint64, numpy.str_, str]"
        ) == str(ak.all_scalars)

    def test_number_format_strings(self):
        assert "{}" == dtypes.NUMBER_FORMAT_STRINGS["bool"]
        assert "{:n}" == dtypes.NUMBER_FORMAT_STRINGS["int64"]
        assert "{:.17f}" == dtypes.NUMBER_FORMAT_STRINGS["float64"]
        assert "f" == dtypes.NUMBER_FORMAT_STRINGS["np.float64"]
        assert "{:n}" == dtypes.NUMBER_FORMAT_STRINGS["uint8"]
        assert "{:n}" == dtypes.NUMBER_FORMAT_STRINGS["uint64"]
        assert "{:n}" == dtypes.NUMBER_FORMAT_STRINGS["bigint"]
