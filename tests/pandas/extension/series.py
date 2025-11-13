import numpy as np
import pandas as pd

import arkouda as ak
from arkouda.pandas.extension._arkouda_array import ArkoudaArray


class TestSeriesExtension:
    def test_series_from_strings(self):
        from arkouda.pandas.extension._arkouda_string_array import ArkoudaStringArray

        s_arr = ArkoudaStringArray(ak.array(["alpha", "beta", "gamma"]))
        s = pd.Series(s_arr)
        assert s.iloc[0] == "alpha"
        assert s.iloc[2] == "gamma"

    def test_string_series_take_with_fill(self):
        from arkouda.pandas.extension._arkouda_string_array import ArkoudaStringArray

        s_arr = ArkoudaStringArray(ak.array(["a", "b", "c"]))
        s = pd.Series(s_arr)
        taken = pd.Series(s.values.take([0, -1, 2], allow_fill=True))
        assert taken.iloc[1] == ""  # default fill
        assert taken.iloc[2] == "c"

    def test_series_from_categorical(self):
        from arkouda.pandas.extension._arkouda_categorical_array import (
            ArkoudaCategoricalArray,
        )

        s_arr = ArkoudaCategoricalArray(ak.Categorical(ak.array(["high", "low", "medium", "low"])))
        s = pd.Series(s_arr)
        assert s.iloc[1] == "low"
        assert s.iloc[3] == "low"

    def test_categorical_series_take_with_fill(self):
        from arkouda.pandas.extension._arkouda_categorical_array import (
            ArkoudaCategoricalArray,
        )

        s_arr = ArkoudaCategoricalArray(ak.Categorical(ak.array(["x", "y", "z"])))
        s = pd.Series(s_arr)
        taken = pd.Series(s.values.take([1, -1, 0], allow_fill=True, fill_value="x"))
        assert taken.iloc[0] == "y"
        assert taken.iloc[1] == "x"

    def test_series_construction(self):
        a = ArkoudaArray(ak.array([10, 20, 30]))
        s = pd.Series(a)
        assert isinstance(s, pd.Series)
        assert len(s) == 3
        assert s.iloc[1] == 20

    def test_series_repr(self):
        a = ArkoudaArray(ak.array([1, 2, 3]))
        s = pd.Series(a)
        output = repr(s)
        assert "int64" in output
        assert "1" in output

    def test_series_indexing(self):
        a = ArkoudaArray(ak.arange(5))
        s = pd.Series(a)
        assert s[0] == 0
        assert s[4] == 4
        assert s.iloc[2] == 2
        assert s.loc[3] == 3

    def test_series_slicing(self):
        a = ArkoudaArray(ak.arange(10))
        s = pd.Series(a)
        sliced = s[2:5]
        assert list(sliced.values._data.to_ndarray()) == [2, 3, 4]

    def test_series_sum_and_reductions(self):
        a = ArkoudaArray(ak.array([1, 2, 3]))
        s = pd.Series(a)
        assert s.sum() == 6
        assert s.max() == 3
        assert s.min() == 1
        assert s.all()  # nonzero
        assert pd.Series(ArkoudaArray(ak.array([0, 0, 0]))).any() is np.False_

    def test_series_equality(self):
        a1 = ArkoudaArray(ak.array([1, 2, 3]))
        a2 = ArkoudaArray(ak.array([1, 2, 3]))
        s1 = pd.Series(a1)
        s2 = pd.Series(a2)
        assert s1.equals(s2)

    def test_series_take(self):
        a = ArkoudaArray(ak.array([10, 20, 30]))
        s = pd.Series(a)
        taken = s.take([2, 0])
        assert list(taken.values._data.to_ndarray()) == [30, 10]

    def test_series_copy(self):
        a = ArkoudaArray(ak.array([5, 6]))
        s = pd.Series(a)
        copied = s.copy()
        assert s.equals(copied)
        copied.iloc[0] = 999
        assert s.iloc[0] != copied.iloc[0]

    def test_numpy_boolean_mask_on_series(self):
        arr = ArkoudaArray(ak.array([1, 2, 3]))
        s = pd.Series(arr)
        mask = np.array([False, True, True])
        result = s[mask]
        assert list(result.values.to_ndarray()) == [2, 3]
