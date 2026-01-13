import datetime as dt

import numpy as np
import pandas as pd
import pytest

import arkouda as ak


class TestPandasConversion:
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

    def test_from_series_misc(self):
        p_array = ak.from_series(pd.Series(["a", "b", "c", "d", "e"]))
        assert isinstance(p_array, ak.Strings)
        assert str == p_array.dtype

        p_array = ak.from_series(pd.Series(np.random.choice([True, False], size=10)))

        assert isinstance(p_array, ak.pdarray)
        assert bool == p_array.dtype

        p_array = ak.from_series(pd.Series([dt.datetime(2016, 1, 1, 0, 0, 1)]))

        assert isinstance(p_array, ak.pdarray)
        assert np.int64 == p_array.dtype

        p_array = ak.from_series(pd.Series([np.datetime64("2018-01-01")]))

        assert isinstance(p_array, ak.pdarray)
        assert np.int64 == p_array.dtype

        p_array = ak.from_series(
            pd.Series(pd.to_datetime(["1/1/2018", np.datetime64("2018-01-01"), dt.datetime(2018, 1, 1)]))
        )

        assert isinstance(p_array, ak.pdarray)
        assert np.int64 == p_array.dtype

        with pytest.raises(TypeError):
            ak.from_series(np.ones(10))

        with pytest.raises(ValueError):
            ak.from_series(pd.Series(np.random.randint(0, 10, 10), dtype=np.int8))
