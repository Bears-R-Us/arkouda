import numpy as np
import pytest

import arkouda as ak
from arkouda.numpy import dtypes

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
    def test_dtypes_docstrings(self):
        import doctest

        result = doctest.testmod(dtypes)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

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

        assert "uint64" == dtypes.resolve_scalar_dtype(2**63 + 1)
        assert "bigint" == dtypes.resolve_scalar_dtype(2**64)

    def test_is_dtype_in_union(self):
        from typing import Union

        from arkouda.numpy.dtypes import _is_dtype_in_union

        float_scalars = Union[float, np.float64, np.float32]
        assert _is_dtype_in_union(np.float64, float_scalars)
        # Test with a type not present in the union
        assert not _is_dtype_in_union(np.int64, float_scalars)
        # Test with a non-Union type
        assert not _is_dtype_in_union(np.float64, float)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize(
        "dtype",
        [
            ak.numpy.dtypes.uint8,
            ak.numpy.dtypes.uint64,
            ak.numpy.dtypes.int64,
            ak.numpy.dtypes.float64,
            ak.numpy.dtypes.bool_,
            ak.numpy.dtypes.bigint,
        ],
    )
    def test_nbytes(self, size, dtype):
        a = ak.array(ak.arange(size), dtype=dtype)
        assert a.nbytes == size * ak.dtype(dtype).itemsize

    def test_nbytes_str(self):
        a = ak.array(["a", "b", "c"])
        c = ak.Categorical(a)
        assert c.nbytes == 82

    def test_pdarrays_datatypes(self):
        assert dtypes.dtype("int64") == ak.array(np.arange(10)).dtype
        assert dtypes.dtype("uint64") == ak.array(np.arange(10), ak.uint64).dtype
        assert dtypes.dtype("bool") == ak.ones(10, ak.bool_).dtype
        assert dtypes.dtype("float64") == ak.ones(10).dtype
        assert dtypes.dtype("str") == ak.random_strings_uniform(1, 16, size=10).dtype

        bi = ak.bigint_from_uint_arrays(
            [ak.ones(10, dtype=ak.uint64), ak.arange(10, dtype=ak.uint64)]
        ).dtype
        assert dtypes.dtype("bigint") == bi
        assert dtypes.dtype("bigint") == ak.arange(2**200, 2**200 + 10).dtype

    def test_isSupportedInt(self):
        for supported in (
            -10,
            1,
            np.int64(1),
            np.int64(1.0),
            np.uint32(1),
            2**63 + 1,
            2**200,
        ):
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
        assert "float32" == str(dtypes.DType.FLOAT32)
        assert "float64" == str(dtypes.DType.FLOAT64)
        assert "float" == str(dtypes.DType.FLOAT)
        assert "complex64" == str(dtypes.DType.COMPLEX64)
        assert "complex128" == str(dtypes.DType.COMPLEX128)
        assert "int8" == str(dtypes.DType.INT8)
        assert "int16" == str(dtypes.DType.INT16)
        assert "int32" == str(dtypes.DType.INT32)
        assert "int64" == str(dtypes.DType.INT64)
        assert "int" == str(dtypes.DType.INT)
        assert "uint8" == str(dtypes.DType.UINT8)
        assert "uint16" == str(dtypes.DType.UINT16)
        assert "uint32" == str(dtypes.DType.UINT32)
        assert "uint64" == str(dtypes.DType.UINT64)
        assert "uint" == str(dtypes.DType.UINT)
        assert "str" == str(dtypes.DType.STR)
        assert "bigint" == str(dtypes.DType.BIGINT)

        assert (
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
            )
            == ak.DTypes
        )

        from arkouda.numpy import bigint, bool_, float64, int64, uint8, uint64

        assert (
            bool_,
            float,
            float64,
            int,
            int64,
            uint64,
            uint8,
            bigint,
            str,
        ) == ak.ARKOUDA_SUPPORTED_DTYPES

    def test_NumericDTypes(self):
        num_types = frozenset(["bool", "bool_", "float", "float64", "int", "int64", "uint64", "bigint"])
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
        assert "typing.Union[bool, numpy.bool]" == str(ak.bool_scalars)
        assert "typing.Union[bool, numpy.bool]" == str(ak.bool_scalars)
        assert "typing.Union[float, numpy.float64, numpy.float32]" == str(ak.float_scalars)
        assert (
            "typing.Union[int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, "
            + "numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]"
        ) == str(ak.int_scalars)

        assert (
            "typing.Union[float, numpy.float64, numpy.float32, int, numpy.int8, numpy.int16, "
            + "numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]"
        ) == str(ak.numeric_scalars)

        assert "typing.Union[str, numpy.str_]" == str(ak.str_scalars)
        assert (
            "typing.Union[numpy.float64, numpy.float32, numpy.int8, numpy.int16, numpy.int32, "
            + "numpy.int64, numpy.bool, numpy.str_, numpy.uint8, numpy.uint16, numpy.uint32, "
            + "numpy.uint64]"
        ) == str(ak.numpy_scalars)

        assert (
            "typing.Union[bool, numpy.bool, float, numpy.float64, numpy.float32, int, numpy.int8, "
            + "numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32,"
            + " numpy.uint64, numpy.str_, str]"
        ) == str(ak.all_scalars)

    def test_number_format_strings(self):
        assert "{}" == dtypes.NUMBER_FORMAT_STRINGS["bool"]
        assert "{:d}" == dtypes.NUMBER_FORMAT_STRINGS["int64"]
        assert "{:.17f}" == dtypes.NUMBER_FORMAT_STRINGS["float64"]
        assert "{f}" == dtypes.NUMBER_FORMAT_STRINGS["np.float64"]
        assert "{:d}" == dtypes.NUMBER_FORMAT_STRINGS["uint8"]
        assert "{:d}" == dtypes.NUMBER_FORMAT_STRINGS["uint64"]
        assert "{:d}" == dtypes.NUMBER_FORMAT_STRINGS["bigint"]

    def test_dtype_for_chapel(self):
        dtypes_for_chapel = {  # see DType
            "real": "float64",
            "real(32)": "float32",
            "real(64)": "float64",
            "complex": "complex128",
            "complex(64)": "complex64",
            "complex(128)": "complex128",
            "int": "int64",
            "int(8)": "int8",
            "int(16)": "int16",
            "int(32)": "int32",
            "int(64)": "int64",
            "uint": "uint64",
            "uint(8)": "uint8",
            "uint(16)": "uint16",
            "uint(32)": "uint32",
            "uint(64)": "uint64",
            "bool": "bool",
            "bigint": "bigint",
            "string": "str",
        }
        for chapel_name, dtype_name in dtypes_for_chapel.items():
            assert dtypes.dtype_for_chapel(chapel_name) == dtypes.dtype(dtype_name)
        # check caching in the implementation of dtype_for_chapel()
        for chapel_name, dtype_name in dtypes_for_chapel.items():
            assert dtypes.dtype_for_chapel(chapel_name) == dtypes.dtype(dtype_name)

    @pytest.mark.parametrize("dtype1", [ak.bool_, ak.uint8, ak.uint64, ak.bigint, ak.int64, ak.float64])
    @pytest.mark.parametrize("dtype2", [ak.bool_, ak.uint8, ak.uint64, ak.bigint, ak.int64, ak.float64])
    def test_result_type(self, dtype1, dtype2):
        if dtype1 == ak.bigint or dtype2 == ak.bigint:
            if dtype1 == ak.float64 or dtype2 == ak.float64:
                expected_result = ak.float64
            else:
                expected_result = ak.bigint
        else:
            expected_result = np.result_type(dtype1, dtype2)
        # pdarray vs pdarray
        a = ak.array([0, 1], dtype=dtype1)
        b = ak.array([0, 1], dtype=dtype2)
        assert ak.result_type(a, b) == expected_result

        # dtype and dtype
        assert ak.result_type(dtype1, dtype2) == expected_result

        # mixed: pdarray vs dtype
        assert ak.result_type(a, dtype2) == expected_result
        assert ak.result_type(dtype1, b) == expected_result

    def test_bool_alias(self):
        assert ak.bool == ak.bool_
        assert ak.bool == np.bool_
