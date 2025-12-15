import pandas as pd

import arkouda as ak

from arkouda import Series
from arkouda.numpy.strings import Strings
from arkouda.pandas.extension import ArkoudaExtensionArray, ArkoudaIndexAccessor
from arkouda.pandas.extension._series_accessor import (
    ArkoudaSeriesAccessor,
    _ak_array_to_pandas_series,
    _pandas_series_to_ak_array,
)
from arkouda.pandas.series import Series as ak_Series


def _assert_series_equal_values(s: pd.Series, values):
    """Helper: assert pandas Series values == iterable `values`."""
    assert list(s.tolist()) == list(values)


class TestArkoudaSeriesAccessor:
    def test_series_accessor_docstrings(self):
        import doctest

        from arkouda.pandas.extension import _series_accessor

        result = doctest.testmod(
            _series_accessor, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    def test_to_ak_simple_series_roundtrip(self):
        s = pd.Series([1, 2, 3], name="nums")

        # Initially not Arkouda-backed
        assert not s.ak.is_arkouda

        # Convert to Arkouda-backed pandas.Series
        s_ak = s.ak.to_ak()
        assert isinstance(s_ak, pd.Series)
        assert s_ak.name == "nums"
        assert s_ak.ak.is_arkouda

        # Underlying array is ArkoudaExtensionArray
        arr = s_ak.array
        assert isinstance(arr, ArkoudaExtensionArray)
        akcol = arr._data
        assert isinstance(akcol, ak.pdarray)
        _assert_series_equal_values(s_ak, akcol.tolist())

        # Collect back to plain NumPy-backed Series
        s_local = s_ak.ak.collect()
        assert isinstance(s_local, pd.Series)
        assert not isinstance(s_local.array, ArkoudaExtensionArray)
        assert not s_local.ak.is_arkouda
        assert s_local.name == "nums"
        assert s_local.equals(s)

    def test_collect_on_plain_series_is_noop_semantics(self):
        """collect() on a non-Arkouda Series should behave like a simple copy."""
        s = pd.Series([10, 20, 30], name="vals")
        assert not s.ak.is_arkouda

        s2 = s.ak.collect()
        assert isinstance(s2, pd.Series)
        assert not isinstance(s2.array, ArkoudaExtensionArray)
        assert not s2.ak.is_arkouda
        assert s2.equals(s)
        assert s2.name == "vals"

    def test_pandas_series_to_ak_legacy_and_back(self):
        s = pd.Series(["a", "b", "c"], name="letters")

        # pandas.Series -> legacy Arkouda array
        ak_arr = _pandas_series_to_ak_array(s)
        # For non-Arkouda-backed string input, we should get an Arkouda Strings
        assert isinstance(ak_arr, Strings)
        assert ak_arr.tolist() == ["a", "b", "c"]

        # legacy Arkouda array -> Arkouda-backed pandas.Series (EA-backed)
        s_back = _ak_array_to_pandas_series(ak_arr, name="letters")
        assert isinstance(s_back, pd.Series)
        assert isinstance(s_back.array, ArkoudaExtensionArray)
        assert s_back.name == "letters"
        _assert_series_equal_values(s_back, ["a", "b", "c"])

    def test_to_ak_legacy_simple_series_and_accessor_roundtrip(self):
        s = pd.Series([4, 5, 6], name="foo")

        # pandas.Series -> legacy Arkouda Series
        ak_s = s.ak.to_ak_legacy()
        assert isinstance(ak_s, Series)
        assert ak_s.size == len(s)
        # Assume Arkouda pandas Series exposes list-like values
        assert ak_s.tolist() == [4, 5, 6]

        # pandas.Series -> legacy Arkouda array (underlying column)
        ak_arr = _pandas_series_to_ak_array(s)

        # ArkoudaSeriesAccessor.from_ak_legacy should just wrap the legacy array via EA
        s_ea = ArkoudaSeriesAccessor.from_ak_legacy(ak_arr, name="foo")
        assert isinstance(s_ea, pd.Series)
        assert isinstance(s_ea.array, ArkoudaExtensionArray)
        assert s_ea.name == "foo"
        _assert_series_equal_values(s_ea, [4, 5, 6])

    def test_is_arkouda_flag_series(self):
        s = pd.Series([1, 2, 3], name="nums")

        # Initially false
        assert not s.ak.is_arkouda

        s_ak = s.ak.to_ak()
        assert s_ak.ak.is_arkouda

        s_local = s_ak.ak.collect()
        assert not s_local.ak.is_arkouda

    def test_to_ak_preserves_index_and_name(self):
        idx = pd.Index([100, 200, 300], name="id")
        s = pd.Series([10, 20, 30], index=idx, name="values")

        s_ak = s.ak.to_ak()
        assert s_ak.name == "values"
        assert s_ak.index.name == "id"
        assert s_ak.ak.is_arkouda

        # Collect should preserve index and name too
        s_local = s_ak.ak.collect()
        assert s_local.name == "values"
        assert s_local.index.name == "id"
        assert s_local.equals(s)

    def test_to_ak_on_already_arkouda_series_idempotent_semantics(self):
        """Calling to_ak() on an already Arkouda-backed Series should keep it Arkouda-backed."""
        s = pd.Series([1, 2, 3], name="nums")
        s_ak = s.ak.to_ak()
        assert s_ak.ak.is_arkouda

        s_ak2 = s_ak.ak.to_ak()
        assert s_ak2.ak.is_arkouda
        assert isinstance(s_ak2.array, ArkoudaExtensionArray)
        _assert_series_equal_values(s_ak2, [1, 2, 3])
        assert s_ak2.name == "nums"
        assert s_ak2.index.equals(s_ak.index)

    def test_to_ak_idempotent_identity_and_zero_copy(self):
        """
        Reproducer for review comment:

            s = self._obj
            arr = s.array
            if isinstance(arr, ArkoudaExtensionArray):
                return s
            s_ak1 = s.ak.to_ak()
            s_ak2 = s_ak1.ak.to_ak()
            assert s_ak1 is s_ak2
        """
        s = pd.Series([100, 200, 300], name="values")

        s_ak1 = s.ak.to_ak()
        assert isinstance(s_ak1.array, ArkoudaExtensionArray)
        assert s_ak1.ak.is_arkouda

        # Capture underlying Arkouda column object for zero-copy assertion
        col1 = getattr(s_ak1.array, "_data", None)
        assert col1 is not None, "Expected ArkoudaExtensionArray to expose _data"

        # Idempotence: calling to_ak again returns the exact same Series object
        s_ak2 = s_ak1.ak.to_ak()
        assert s_ak2 is s_ak1

        # Zero-copy: underlying column should be the same object too
        col2 = getattr(s_ak2.array, "_data", None)
        assert col2 is col1

    def test_to_ak_legacy_preserves_named_index(self):
        """
        Reproducer for review comment / snippet:

            idx = pd.Index([10, 20, 30], name="id")
            s = pd.Series([100, 200, 300], index=idx, name="values")
            ak_s = s.ak.to_ak_legacy()
            assert ak_s.index ... preserves name + values
        """
        idx = pd.Index([10, 20, 30], name="id")
        s = pd.Series([100, 200, 300], index=idx, name="values")

        ak_s = s.ak.to_ak_legacy()
        assert isinstance(ak_s, ak_Series)

        # Values preserved
        assert ak_s.tolist() == [100, 200, 300]

        # Index preserved (values + name)
        # (Assumes Arkouda Series index is an Arkouda Index-like object with .to_list/.tolist and .name)
        assert ak_s.index.tolist() == [10, 20, 30]
        assert getattr(ak_s.index, "name", None) == "id"
        assert ak_s.name == "values"

        # If legacy Series has a pandas roundtrip API, verify it too, but don't require it.
        if hasattr(ak_s, "to_pandas"):
            s_back = ak_s.to_pandas()
            assert isinstance(s_back, pd.Series)
            assert s_back.index.tolist() == [10, 20, 30]
            assert s_back.index.name == "id"
            assert s_back.tolist() == [100, 200, 300]
