import datetime as dt

import numpy as np
import pandas as pd
import pytest

import arkouda as ak


class TestPandasConversion:
    def test_conversion_docstrings(self):
        import doctest

        from arkouda.pandas import conversion

        result = doctest.testmod(conversion, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("dtype", [bool, np.float64, np.int64, str])
    def test_from_series_dtypes(self, size, dtype):
        p_array = ak.from_series(pd.Series(np.random.randint(0, 10, size)), dtype)
        assert isinstance(p_array, ak.pdarray if dtype is not str else ak.Strings)
        assert dtype == p_array.dtype

        p_objects_array = ak.from_series(
            pd.Series(np.random.randint(0, 10, size), dtype="object"), dtype=dtype
        )
        assert isinstance(p_objects_array, ak.pdarray if dtype is not str else ak.Strings)
        assert dtype == p_objects_array.dtype

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize(
        "string_dtype", ["object", "str", "string", "string[python]", "string[pyarrow]"]
    )
    def test_from_series_string_spellings(self, size, string_dtype):
        # When dtype is one of these spellings, from_series should produce Strings.
        s = pd.Series([f"x{i}" for i in range(size)], dtype=string_dtype)
        a = ak.from_series(s)

        assert isinstance(a, ak.Strings)

        # Also works when passed as dtype override
        b = ak.from_series(pd.Series([f"x{i}" for i in range(size)]), dtype=string_dtype)
        assert isinstance(b, ak.Strings)

    def test_from_series_misc(self):
        p_array = ak.from_series(pd.Series(["a", "b", "c", "d", "e"]))
        assert isinstance(p_array, ak.Strings)
        assert str == p_array.dtype

        p_array = ak.from_series(pd.Series(np.random.choice([True, False], size=10)))

        assert isinstance(p_array, ak.pdarray)
        assert bool == p_array.dtype

        # Python datetime objects -> int64 ns
        p_array = ak.from_series(pd.Series([dt.datetime(2016, 1, 1, 0, 0, 1)]))
        assert isinstance(p_array, ak.pdarray)
        assert np.int64 == p_array.dtype

        # numpy datetime64 -> int64 ns
        p_array = ak.from_series(pd.Series([np.datetime64("2018-01-01")]))
        assert isinstance(p_array, ak.pdarray)
        assert np.int64 == p_array.dtype

        # Mixed datetime inputs -> int64 ns
        p_array = ak.from_series(
            pd.Series(pd.to_datetime(["1/1/2018", np.datetime64("2018-01-01"), dt.datetime(2018, 1, 1)]))
        )
        assert isinstance(p_array, ak.pdarray)
        assert np.int64 == p_array.dtype

        # Datetime resolution variants should be accepted
        p_array = ak.from_series(pd.Series([np.datetime64("2018-01-01", "ms")]))
        assert isinstance(p_array, ak.pdarray)
        assert np.int64 == p_array.dtype

        # Timedelta with non-ns resolution should be accepted
        p_array = ak.from_series(pd.Series([np.timedelta64(123, "ms")]))
        assert isinstance(p_array, ak.pdarray)
        assert np.int64 == p_array.dtype

        with pytest.raises(TypeError):
            ak.from_series(np.ones(10))

        # Unsupported dtype override (via ak_dtype) should raise ValueError
        with pytest.raises(ValueError):
            ak.from_series(pd.Series(np.random.randint(0, 10, 10)), dtype=np.int8)

        # If the Series dtype itself is unsupported and no override is given, also ValueError
        with pytest.raises(ValueError):
            ak.from_series(pd.Series(np.random.randint(0, 10, 10), dtype=np.int8))

    def test_from_series_object_dtype_normalized_to_string(self):
        # Force dt == "object" from series.dtype.name and ensure it is treated as string.
        s = pd.Series(["a", None, "b"], dtype="object")

        a = ak.from_series(s)

        assert isinstance(a, ak.Strings)

        # dtype can surface as `str` or as a NumPy unicode dtype depending on implementation details.
        assert (a.dtype == str) or (isinstance(a.dtype, np.dtype) and a.dtype.kind in ("U", "S"))

        # Most importantly: object inputs are stringified and preserved as strings
        assert a.tolist() == ["a", "None", "b"]
